"""生検出の重複を固定基準で統合し、JSON・CSVの地点報告を保存する。"""

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Detection:
    """1回の画像認識で得られた板の推定位置と証拠を保持する。"""

    x: float
    y: float
    time: float
    score: float
    image: str
    z: float


def consolidate(detections, radius=0.8):
    """近い検出を、最初の推定位置を基準とする地点にまとめる。"""

    if not np.isfinite(radius) or radius <= 0:
        raise ValueError("Merge radius must be positive")
    sites = []
    for index, detection in enumerate(detections):
        if detection.z is None:
            raise ValueError("Detection requires finite XYZ values")
        if not np.isfinite([detection.x, detection.y, detection.z,
                            detection.time, detection.score]).all():
            raise ValueError("Detection contains non-finite values")
        distances = [
            np.sqrt((detection.x - site["x"])**2 + (detection.y - site["y"])**2
                    + (detection.z - site["z"])**2)
            for site in sites
        ]
        if distances and min(distances) <= radius:
            site = sites[int(np.argmin(distances))]
            site["observations"] += 1
            site["detection_indices"].append(index)
            if detection.score > site["score"]:
                site["score"] = detection.score
                site["image"] = detection.image
        else:
            sites.append({
                "id": len(sites) + 1, "x": detection.x, "y": detection.y, "z": detection.z,
                "first_seen": detection.time, "observations": 1,
                "score": detection.score, "image": detection.image,
                "detection_indices": [index],
            })
    return sites


def write_report(directory, detections, status, detail, radius=0.8):
    """生検出と統合地点を含む試行レポートをJSON・CSVへ保存する。"""

    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    sites = consolidate(detections, radius)
    data = {
        "status": status, "detail": detail,
        "coordinate_frame": "Gazebo world (metres), not GPS",
        "detections": [asdict(item) for item in detections],
        "sites": sites,
    }
    report_text = json.dumps(data, indent=2, allow_nan=False)
    fields = ["id", "x", "y", "z", "first_seen", "observations", "score", "image", "detection_indices"]
    with (path / "sites.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(sites)
    temporary = path / "report.json.tmp"
    temporary.write_text(report_text, encoding="utf-8")
    temporary.replace(path / "report.json")
    return data
