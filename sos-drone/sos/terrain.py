"""既知の三角形地形、安全な探索・周囲撮影・直接帰還の経路を扱う。"""

import json
from pathlib import Path

import numpy as np


class Terrain:
    """表示・衝突と同じ標高グリッドを照会する。"""

    def __init__(self, data):
        """公開地形データの標高グリッドを検証して保持する。"""
        self.data = data
        self.heights = np.asarray(data["heights"], dtype=float)
        self.step = float(data["step"])
        self.origin = np.asarray(data["origin"], dtype=float)
        if (self.heights.ndim != 2 or not np.isfinite(self.heights).all()
                or self.step <= 0 or self.origin.shape != (2,)):
            raise ValueError("Invalid terrain grid")

    @classmethod
    def load(cls, path):
        """JSONの公開地形を読み込む。"""
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))

    def height(self, x, y):
        """メッシュと同じ対角線で三角形補間した標高を返す。"""
        u, v = (np.array([x, y]) - self.origin) / self.step
        rows, columns = self.heights.shape
        if not (0 <= u < columns - 1 and 0 <= v < rows - 1):
            raise ValueError("Point outside terrain")
        i, j = int(u), int(v)
        a, b = u - i, v - j
        h00, h10 = self.heights[j, i], self.heights[j, i + 1]
        h01, h11 = self.heights[j + 1, i], self.heights[j + 1, i + 1]
        if b <= a:
            return float((1 - a) * h00 + (a - b) * h10 + b * h11)
        return float((1 - b) * h00 + a * h11 + (b - a) * h01)

    def intersection(self, origin, ray, offset=0.025):
        """下向き視線と三角形地形の最初の交点を求める。"""
        origin, ray = np.asarray(origin, dtype=float), np.asarray(ray, dtype=float)
        if (origin.shape != (3,) or ray.shape != (3,) or not np.isfinite(origin).all()
                or not np.isfinite(ray).all() or ray[2] >= -1e-8):
            raise ValueError("Invalid downward terrain ray")
        norm = np.linalg.norm(ray)
        ray = ray / norm
        previous_t = 0.
        previous = origin[2] - self.height(*origin[:2]) - offset
        if previous <= 0:
            raise ValueError("Camera below terrain")
        # Sampling moves less than one grid cell, then refines the first crossing.
        for t in np.arange(0.25, 1000., 0.25):
            point = origin + t * ray
            difference = point[2] - self.height(*point[:2]) - offset
            if difference <= 0:
                low, high = previous_t, t
                for _ in range(30):
                    middle = (low + high) / 2
                    probe = origin + middle * ray
                    if probe[2] > self.height(*probe[:2]) + offset:
                        low = middle
                    else:
                        high = middle
                return origin + (low + high) / 2 * ray
            previous_t = t
        raise ValueError("No terrain intersection")

    def validate_segment(self, start, end, clearance=1.0):
        """全移動区間を細かく照会し、地形・樹冠との余裕を検証する。"""
        start, end = np.asarray(start, dtype=float), np.asarray(end, dtype=float)
        length = np.linalg.norm(end - start)
        minimum = float("inf")
        for point in np.linspace(start, end, max(2, int(length / 0.25) + 2)):
            gap = point[2] - self.height(*point[:2])
            minimum = min(minimum, gap)
            if gap < clearance:
                raise ValueError(f"Unsafe terrain clearance: {gap:.2f}m")
            for tree in self.data["trees"]:
                horizontal = np.linalg.norm(point[:2] - [tree["x"], tree["y"]])
                if (horizontal < tree["radius"] + 1.0
                        and point[2] < tree["z"] + tree["height"] + 1.0):
                    raise ValueError("Unsafe tree clearance")
        return minimum

    def direct_return(self, start, home):
        """地形・樹冠より高い一定高度で麓へ直行する帰還目標を返す。"""
        start, home = np.asarray(start, dtype=float), np.asarray(home, dtype=float)
        if (start.shape != (3,) or home.shape != (3,)
                or not np.isfinite(start).all() or not np.isfinite(home).all()):
            raise ValueError("Invalid direct return coordinates")
        clearance = 2.0
        altitude = max(start[2], home[2], float(self.heights.max())
                       + max(clearance, self.data.get("transit_agl_m", 15.)),
                       max((tree["z"] + tree["height"] + clearance
                            for tree in self.data["trees"]), default=0.))
        above_start, above_home = start.copy(), home.copy()
        above_start[2] = above_home[2] = altitude
        targets = [above_start.tolist(), above_home.tolist(), home.tolist()]
        for first, second in zip([start.tolist()] + targets, targets):
            self.validate_segment(first, second, clearance=clearance)
        return targets

    def search_route(self):
        """撮影指定に依存せず、山道の形状だけから探索移動目標を返す。"""
        targets = [{"target": [item["x"], item["y"],
                               self.height(item["x"], item["y"])
                               + self.data.get("capture_agl_m", 5.)],
                    "capture": False}
                   for item in self.data["path"]]
        for first, second in zip(targets, targets[1:]):
            self.validate_segment(first["target"], second["target"], clearance=2.)
        return targets

    def survey_targets(self, start, point):
        """推論した地点とその周囲2位置への撮影経路を検証する。"""
        point = np.asarray(point, dtype=float)
        if point.shape != (3,) or not np.isfinite(point).all():
            raise ValueError("Invalid survey coordinates")
        targets = [[float(point[0]), float(point[1] + shift),
                    self.height(point[0], point[1] + shift)
                    + self.data.get("capture_agl_m", 5.)] for shift in (0., -0.35, 0.35)]
        for first, second in zip([start] + targets, targets):
            self.validate_segment(first, second, clearance=2.)
        return targets
