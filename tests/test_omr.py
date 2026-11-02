import cv2
import numpy as np

from pharmadoc_ai.omr import detect_checkboxes
from pharmadoc_ai.schemas import Word


def test_empty_tick_dot_and_filled_boxes():
    image = np.full((300, 600, 3), 255, dtype=np.uint8)
    words = []
    for i, label in enumerate(("Empty", "Tick", "Dot", "Filled")):
        y = 30 + i * 60
        cv2.rectangle(image, (30, y), (54, y + 24), (0, 0, 0), 2)
        words.append(Word(text=label, bbox=(65 / 600, y / 300, 140 / 600, (y + 24) / 300)))
        if i == 1:
            cv2.line(image, (35, y + 12), (41, y + 19), (0, 0, 0), 3)
            cv2.line(image, (41, y + 19), (50, y + 5), (0, 0, 0), 3)
        elif i == 2:
            cv2.circle(image, (42, y + 12), 3, (0, 0, 0), -1)
        elif i == 3:
            cv2.rectangle(image, (34, y + 4), (50, y + 20), (0, 0, 0), -1)
    boxes = detect_checkboxes(image, words)
    assert [box.label for box in boxes] == ["Empty", "Tick", "Dot", "Filled"]
    assert [box.state for box in boxes] == ["unchecked", "checked", "ambiguous", "checked"]
