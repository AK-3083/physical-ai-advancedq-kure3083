"""合成画像で小さなCNNを学習し、重み・ONNX・分類評価を保存する。"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn

from sos.markers import COLOR_NEGATIVES, color_training_crop
from sos.vision import preprocess


def dataset(seed, count, return_categories=False):
    """正例・負例を交互に生成し、画像・ラベルと任意のカテゴリ名を返す。"""
    rng = np.random.default_rng(seed)
    labels = np.arange(count) % 2
    categories = np.array(["marker" if label else
                           COLOR_NEGATIVES[(index // 2) % len(COLOR_NEGATIVES)]
                           for index, label in enumerate(labels)])
    images = np.concatenate([
        preprocess(color_training_crop(rng, bool(label), category))
        for label, category in zip(labels, categories)
    ])
    result = (torch.from_numpy(images), torch.from_numpy(labels.astype(np.int64)))
    return result + (categories,) if return_categories else result


def main():
    """CPU学習、閾値選択、独立画像評価、ONNX出力と一致検証を実行する。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("models/color"))
    output = parser.parse_args().output
    torch.manual_seed(14)
    torch.set_num_threads(4)
    output.mkdir(parents=True, exist_ok=True)
    model = nn.Sequential(
        nn.Conv2d(1, 8, 5, padding=2), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Flatten(), nn.Linear(16 * 16 * 16, 32), nn.ReLU(), nn.Linear(32, 2),
    )
    train_count, validation_count, test_count, epochs = 6000, 1000, 1000, 24
    x, y = dataset(101, train_count)
    val_x, val_y = dataset(202, validation_count)
    test_x, test_y, categories = dataset(303, test_count, True)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(epochs):
        model.train()
        permutation = torch.randperm(len(y))
        for indices in permutation.split(64):
            logits = model(x[indices])
            loss = nn.functional.cross_entropy(logits, y[indices])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            accuracy = (model(val_x).argmax(1) == val_y).float().mean().item()
        print(f"epoch={epoch + 1} validation_accuracy={accuracy:.3f}", flush=True)
    model.eval()
    with torch.no_grad():
        val_scores = model(val_x).softmax(1)[:, 1].numpy()
        test_scores = model(test_x).softmax(1)[:, 1].numpy()
    options = []
    for threshold in np.linspace(0.5, 0.95, 10):
        predictions = val_scores >= threshold
        tp = int(np.sum(predictions & (val_y.numpy() == 1)))
        fp = int(np.sum(predictions & (val_y.numpy() == 0)))
        options.append((tp - 3 * fp, -fp, float(threshold)))
    threshold = max(options)[2]
    predictions = test_scores >= threshold
    labels = test_y.numpy()
    confusion = [[int(np.sum((labels == actual) & (predictions == predicted)))
                  for predicted in range(2)] for actual in range(2)]
    torch.save(model.state_dict(), output / "marker.pt")
    torch.onnx.export(
        model, torch.zeros(1, 1, 64, 64), str(output / "marker.onnx"),
        opset_version=11, input_names=["image"], output_names=["logits"],
    )
    network = cv2.dnn.readNetFromONNX(str(output / "marker.onnx"))
    differences = []
    for sample in test_x[:10]:
        network.setInput(sample.numpy()[None])
        with torch.no_grad():
            expected = model(sample[None]).numpy()
        differences.append(float(np.max(np.abs(network.forward() - expected))))
    if max(differences) > 1e-4:
        raise RuntimeError(f"ONNX conversion changed logits: {max(differences)}")
    metrics = {
        "threshold": threshold, "test_confusion": confusion,
        "onnx_max_logit_difference": max(differences),
        "scheme": "color",
        "train_count": train_count, "validation_count": validation_count,
        "test_count": test_count, "epochs": epochs,
        "seeds": [101, 202, 303],
        "scope": "Procedural marker crops; Gazebo evaluation is separate",
    }
    metrics["negative_categories"] = {
        category: {
            "count": int(np.sum(categories == category)),
            "false_positives": int(np.sum(predictions & (categories == category))),
            "rejection_rate": float(np.mean(~predictions[categories == category])),
        } for category in COLOR_NEGATIVES
    }
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
