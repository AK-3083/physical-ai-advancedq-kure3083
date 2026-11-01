"""カメラの画素から、既知地形上のワールドXYZ座標を計算する。"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Pose:
    """位置推定に用いる機体姿勢と受信時刻を保持する。"""

    position: np.ndarray
    quaternion: np.ndarray
    time: float


def rotation(quaternion):
    """クォータニオンを正規化し、機体からワールドへの回転行列を返す。"""

    q = np.asarray(quaternion, dtype=float)
    if q.shape != (4,) or not np.isfinite(q).all() or np.linalg.norm(q) < 1e-8:
        raise ValueError("Invalid quaternion")
    x, y, z, w = q / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w)],
        [2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w)],
        [2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y)],
    ])


# Gazebo camera looks along local +X. Bottom joint is Ry(pi/2);
# image right = -camera Y, image down = -camera Z.
OPTICAL_TO_BODY = np.array([[0., -1., 0.], [-1., 0., 0.], [0., 0., -1.]])


def terrain_point(pixel, intrinsic, pose, terrain):
    """カメラ視線と既知地形との交点をXYZで返す。"""
    k = np.asarray(intrinsic, dtype=float).reshape(3, 3)
    uv = np.asarray(pixel, dtype=float)
    if (uv.shape != (2,) or not np.isfinite(uv).all() or not np.isfinite(k).all()
            or k[0, 0] <= 0 or k[1, 1] <= 0):
        raise ValueError("Invalid terrain camera input")
    ray = rotation(pose.quaternion) @ OPTICAL_TO_BODY @ np.linalg.solve(
        k, [uv[0], uv[1], 1.]
    )
    return terrain.intersection(pose.position, ray)
