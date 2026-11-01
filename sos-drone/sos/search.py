"""移動画像での継続検出と、同じ地点への繰り返し停止を管理する。"""

import numpy as np


class SearchTrigger:
    """2枚の独立画像で同じマーカーを確認して撮影を要求する。"""

    def __init__(self):
        """直前の候補と撮影済み地点を初期化する。"""
        self.previous = []
        self.visited = []

    def update(self, points):
        """未撮影で前画像にもある地点を返し、今回の候補を保持する。"""
        current = [np.asarray(point, dtype=float) for point in points]
        if any(point.shape != (3,) or not np.isfinite(point).all() for point in current):
            raise ValueError("Invalid search candidate coordinates")
        match = next((point for point in current
                      if not any(np.linalg.norm(point - old) < 1.5 for old in self.visited)
                      and any(np.linalg.norm(point - old) < 1.0 for old in self.previous)), None)
        self.previous = current
        return match

    def mark_visited(self, point):
        """調査終了地点を記憶し、前画像の履歴を消す。"""
        point = np.asarray(point, dtype=float)
        if point.shape != (3,) or not np.isfinite(point).all():
            raise ValueError("Invalid visited coordinates")
        self.visited.append(point)
        self.previous = []
