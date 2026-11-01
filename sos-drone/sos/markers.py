"""色マーカー・同色負例と、分類学習用の変形画像を生成する。"""

import cv2
import numpy as np


COLOR_NEGATIVES = ("flower", "clothing", "plain", "wrong", "checker")
COLOR_PATTERN = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [1, 0, 0, 0]], dtype=np.uint8)


def color_board(kind="marker", brightness=240, saturation=230, variant=0):
    """マゼンタ枠の非対称チェッカー、または同色の妨害物を描く。"""

    if kind not in ("marker",) + COLOR_NEGATIVES:
        raise ValueError(f"Unknown color board kind: {kind}")
    if not 0 <= brightness <= 255 or not 0 <= saturation <= 255:
        raise ValueError("Brightness and saturation must lie between 0 and 255")
    color = cv2.cvtColor(np.uint8([[[150, saturation, brightness]]]),
                         cv2.COLOR_HSV2BGR)[0, 0].tolist()
    image = np.full((512, 768, 3), color, np.uint8)
    if kind in ("marker", "wrong", "checker"):
        pattern = COLOR_PATTERN.copy()
        if kind == "wrong":
            pattern = 1 - pattern
        elif kind == "checker":
            pattern = np.indices((3, 4)).sum(0) % 2
        for row, column in np.argwhere(pattern):
            x = 96 + int(column) * 144
            y = 64 + int(row) * 128
            image[y:y + 128, x:x + 144] = 15
    elif kind in ("flower", "clothing"):
        image[:] = 105
        rng = np.random.default_rng(variant)
        if kind == "flower":
            petals = int(rng.integers(5, 9))
            phase = float(rng.uniform(0, 2 * np.pi))
            axes = (int(rng.integers(95, 116)), int(rng.integers(70, 86)))
            for angle in np.linspace(0, 2 * np.pi, petals, endpoint=False) + phase:
                center = (int(384 + 155 * np.cos(angle)),
                          int(256 + 120 * np.sin(angle)))
                cv2.ellipse(image, center, axes, np.degrees(angle), 0, 360, color, -1)
            cv2.circle(image, (384, 256), 95, color, -1)
        else:
            sleeve = int(rng.integers(25, 70))
            points = np.array([[280, 65], [335, 90], [433, 90], [488, 65],
                               [660, 170 + sleeve], [585, 260], [495, 195],
                               [510, 450], [258, 450], [273, 195],
                               [183, 260], [108, 170 + sleeve]], np.int32)
            cv2.fillPoly(image, [points], color)
            cv2.ellipse(image, (384, 80), (50, 35), 0, 0, 180, (105,) * 3, -1)
    return image


def color_training_crop(rng, positive, negative_kind=None):
    """色候補と同じ切り出しで回転・明度・彩度・縮尺を変えた画像を作る。"""

    kind = "marker" if positive else (
        negative_kind if negative_kind is not None else str(rng.choice(COLOR_NEGATIVES)))
    image = color_board(kind, int(rng.integers(110, 256)),
                        int(rng.integers(105, 256)), int(rng.integers(0, 100000)))
    width = int(rng.integers(40, 115))
    height = max(27, int(width * rng.uniform(0.60, 0.75)))
    image = cv2.resize(image, (width, height))
    background = (int(rng.integers(25, 100)),) * 3
    canvas = np.full((180, 180, 3), background, np.uint8)
    x, y = (180 - width) // 2, (180 - height) // 2
    canvas[y:y + height, x:x + width] = image
    transform = cv2.getRotationMatrix2D((90, 90), float(rng.uniform(-180, 180)), 1)
    canvas = cv2.warpAffine(canvas, transform, (180, 180), borderValue=background)
    hsv = cv2.cvtColor(canvas, cv2.COLOR_BGR2HSV)
    points = cv2.findNonZero(cv2.inRange(hsv, (135, 70, 60), (170, 255, 255)))
    if points is None:
        raise ValueError("Generated color board has no foreground")
    x, y, width, height = cv2.boundingRect(points)
    crop = cv2.resize(canvas[y:y + height, x:x + width], (64, 64))
    if rng.random() < 0.25:
        crop = cv2.GaussianBlur(crop, (3, 3), 0)
    noise = rng.normal(0, rng.uniform(0, 4), crop.shape)
    return np.clip(crop.astype(float) + noise, 0, 255).astype(np.uint8)
