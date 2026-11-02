"""Reversible image geometry and opt-in preprocessing profiles."""

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class PreparedImage:
    image: np.ndarray
    inverse: np.ndarray
    original_width: int
    original_height: int

    def original_box(self, box) -> tuple[float, float, float, float]:
        x0, y0, x1, y1 = box
        corners = np.array([[x0, y0, 1], [x1, y0, 1], [x1, y1, 1], [x0, y1, 1]]).T
        restored = self.inverse @ corners
        xs, ys = restored[0] / self.original_width, restored[1] / self.original_height
        return tuple(float(np.clip(v, 0, 1)) for v in (xs.min(), ys.min(), xs.max(), ys.max()))


def grayscale(image: np.ndarray) -> np.ndarray:
    return image.copy() if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


def resize_image(image: np.ndarray, max_width: int = 2000) -> np.ndarray:
    if image.shape[1] <= max_width:
        return image.copy()
    ratio = max_width / image.shape[1]
    return cv2.resize(image, None, fx=ratio, fy=ratio, interpolation=cv2.INTER_AREA)


def is_blank(image: np.ndarray) -> bool:
    gray = grayscale(image)
    return float(np.mean(gray < 220)) < 0.0003


def rotate_image(image: np.ndarray, degrees: float, expand: bool = False):
    height, width = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), degrees, 1)
    out_w, out_h = width, height
    if expand:
        c, s = abs(matrix[0, 0]), abs(matrix[0, 1])
        out_w, out_h = round(height * s + width * c), round(height * c + width * s)
        matrix[0, 2] += (out_w - width) / 2
        matrix[1, 2] += (out_h - height) / 2
    border = 255 if image.ndim == 2 else (255, 255, 255)
    rotated = cv2.warpAffine(image, matrix, (out_w, out_h), borderValue=border)
    forward = np.vstack([matrix, [0, 0, 1]])
    return rotated, forward


def estimate_skew(image: np.ndarray) -> float:
    """Projection-profile search restricted to small skew; not orientation detection."""
    if is_blank(image):
        return 0.0
    gray = grayscale(resize_image(image, 900))
    binary = (gray < 160).astype(np.uint8) * 255
    candidates = np.arange(-7.0, 7.01, 0.5)
    scores = []
    for angle in candidates:
        rotated, _ = rotate_image(255 - binary, angle)
        scores.append(np.var(np.sum(rotated < 128, axis=1)))
    return float(candidates[int(np.argmax(scores))])


def preprocess(image: np.ndarray, profile: str = "raw", rotation: float = 0) -> PreparedImage:
    height, width = image.shape[:2]
    processed = image.copy()
    forward = np.eye(3)
    if rotation:
        processed, transform = rotate_image(processed, rotation, expand=True)
        forward = transform @ forward
    if profile == "contrast":
        processed = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(grayscale(processed))
    elif profile == "adaptive":
        gray = cv2.medianBlur(grayscale(processed), 3)
        processed = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 12
        )
    elif profile == "deskew":
        angle = estimate_skew(processed)
        processed, transform = rotate_image(processed, angle)
        forward = transform @ forward
    elif profile != "raw":
        raise ValueError("Unknown preprocessing profile.")
    return PreparedImage(processed, np.linalg.inv(forward), width, height)


def document_region(image: np.ndarray) -> tuple[int, int, int, int] | None:
    """A conservative region proposal; never crop automatically on this heuristic."""
    edges = cv2.Canny(grayscale(image), 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
    return x, y, x + w, y + h
