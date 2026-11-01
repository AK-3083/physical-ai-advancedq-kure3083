"""マゼンタ色の候補を抽出し、ONNXで内部パターンを分類する。"""

import json
from pathlib import Path

import cv2
import numpy as np


def candidates(image):
    """BGR画像からマゼンタのHSV色候補を抽出する。"""

    if image is None or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Expected a BGR image")
    mask = cv2.inRange(cv2.cvtColor(image, cv2.COLOR_BGR2HSV),
                       (135, 70, 60), (170, 255, 255))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    result = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if min(w, h) < 24 or w * h > image.shape[0] * image.shape[1] * 0.5:
            continue
        if x == 0 or y == 0 or x + w >= image.shape[1] or y + h >= image.shape[0]:
            continue
        result.append((x, y, w, h))
    return result


def annotate_candidates(image, records, threshold):
    """Draw color candidates and CNN threshold matches without changing the input."""
    annotated = image.copy()
    for record in records:
        x, y, w, h = record["pixel"]
        cv2.rectangle(annotated, (x, y), (x+w, y+h), (0, 200, 255), 2)
    for record in records:
        if record["score"] >= threshold:
            x, y, w, h = record["pixel"]
            cv2.rectangle(annotated, (x, y), (x+w, y+h), (0, 255, 80), 2)
    return annotated


def preprocess(image):
    """候補画像をCNN入力のグレースケールテンソルへ変換する。"""

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray, (64, 64)).astype(np.float32)[None, None] / 255.


class MarkerClassifier:
    """指定したマーカーと負例の2クラスCNNをOpenCV DNNで実行する。"""

    def __init__(self, model, threshold=0.8):
        """学習済みONNXモデルと採用閾値を読み込む。"""

        if not Path(model).is_file():
            raise FileNotFoundError(model)
        if not 0 < threshold < 1:
            raise ValueError("Threshold must lie between zero and one")
        metadata = Path(model).with_name("metrics.json")
        if not metadata.is_file():
            raise ValueError("Color model requires scheme metadata")
        try:
            metrics = json.loads(metadata.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            raise ValueError("Invalid color model metadata") from error
        if not isinstance(metrics, dict) or metrics.get("scheme") != "color":
            raise ValueError("Model scheme must be 'color'")
        self.net = cv2.dnn.readNetFromONNX(str(model))
        self.threshold = threshold
        self.scheme = "color"

    def score(self, crop):
        """候補画像のマーカークラスのsoftmaxスコアを返す。"""

        self.net.setInput(preprocess(crop))
        logits = self.net.forward().reshape(-1)
        if logits.shape != (2,) or not np.isfinite(logits).all():
            raise ValueError("Model must output two finite logits")
        values = np.exp(logits - logits.max())
        return float(values[1] / values.sum())

    def detect(self, image):
        """画像内の候補のうち、CNNスコアが閾値以上のものを返す。"""

        detections = []
        for x, y, w, h in candidates(image):
            score = self.score(image[y:y+h, x:x+w])
            if score >= self.threshold:
                detections.append((x, y, w, h, score))
        return detections
