"""Conservative contour-based checkbox detection; ambiguous marks remain null."""

import cv2
import numpy as np

from pharmadoc_ai.preprocessing import grayscale
from pharmadoc_ai.schemas import Checkbox, Word


def detect_checkboxes(image: np.ndarray, words: list[Word]) -> list[Checkbox]:
    gray = grayscale(image)
    height, width = gray.shape
    _, binary = cv2.threshold(gray, 160, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        minimum = max(8, min(height, width) * 0.009)
        maximum = max(32, min(height, width) * 0.05)
        if not (minimum <= w <= maximum and minimum <= h <= maximum and 0.8 < w / h < 1.2):
            continue
        approx = cv2.approxPolyDP(contour, 0.035 * cv2.arcLength(contour, True), True)
        if len(approx) != 4:
            continue
        box = (x / width, y / height, (x + w) / width, (y + h) / height)
        # Suppress inner/outer duplicates and letter-shaped candidates inside text words.
        if any(
            abs(box[0] - old.bbox[0]) < w / width * 0.5
            and abs(box[1] - old.bbox[1]) < h / height * 0.5
            for old in boxes
        ):
            continue
        if any(
            word.bbox[0] < box[0]
            and word.bbox[2] > box[2]
            and word.bbox[1] <= box[1]
            and word.bbox[3] >= box[3]
            for word in words
        ):
            continue
        border = max(2, round(min(w, h) * 0.22))
        interior = binary[y + border : y + h - border, x + border : x + w - border]
        if not interior.size:
            continue
        ink = float(np.mean(interior > 0))
        dark_y, dark_x = np.where(interior > 0)
        compact_mark = (
            bool(len(dark_x))
            and (dark_x.max() - dark_x.min() + 1) < interior.shape[1] * 0.75
            and (dark_y.max() - dark_y.min() + 1) < interior.shape[0] * 0.75
        )
        state = "unchecked" if ink < 0.025 else ("checked" if ink > 0.14 else "ambiguous")
        if 0.025 <= ink < 0.3 and compact_mark:
            state = "ambiguous"
        center = (box[1] + box[3]) / 2
        nearby = [
            word
            for word in words
            if word.bbox[0] >= box[2] - 0.002
            and word.bbox[0] - box[2] < 0.42
            and abs((word.bbox[1] + word.bbox[3]) / 2 - center) < (box[3] - box[1]) * 0.7
        ]
        label = " ".join(word.text for word in sorted(nearby, key=lambda w: w.bbox[0])) or None
        boxes.append(
            Checkbox(
                bbox=box,
                label=label,
                state=state,
                value=True if state == "checked" else (False if state == "unchecked" else None),
                ink_ratio=ink,
            )
        )
    return sorted(boxes, key=lambda box: (box.bbox[1], box.bbox[0]))
