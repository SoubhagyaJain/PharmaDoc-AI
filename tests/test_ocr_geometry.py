import time

import numpy as np
import pytest

from pharmadoc_ai.ocr.paddle_worker import restore_polygon
from pharmadoc_ai.ocr.tesseract import TesseractEngine


def test_paddle_orientation_restores_original_coordinates():
    # Original 200x100 page rotated 90 degrees has dimensions 100x200.
    polygon = [[0, 0], [100, 0], [100, 200], [0, 200]]
    restored = restore_polygon(polygon, 90, 200, 100)
    assert restored[:, 0].min() == pytest.approx(0)
    assert restored[:, 0].max() == pytest.approx(200)
    assert restored[:, 1].min() == pytest.approx(0)
    assert restored[:, 1].max() == pytest.approx(100)


def test_tesseract_keeps_reading_order_after_rotation(settings, monkeypatch):
    monkeypatch.setattr("pharmadoc_ai.ocr.tesseract.available", lambda settings: True)
    monkeypatch.setattr("pharmadoc_ai.ocr.tesseract.is_blank", lambda image: False)
    monkeypatch.setattr(
        "pharmadoc_ai.ocr.tesseract.pytesseract.image_to_osd",
        lambda *a, **k: {"rotate": 90, "orientation_conf": 10},
    )
    monkeypatch.setattr(
        "pharmadoc_ai.ocr.tesseract.pytesseract.image_to_data",
        lambda *a, **k: {
            "text": ["first", "second"],
            "left": [10, 10],
            "top": [10, 50],
            "width": [20, 20],
            "height": [10, 10],
            "conf": [90, 90],
        },
    )
    words = TesseractEngine().recognize(
        np.zeros((100, 200, 3), dtype=np.uint8), settings, time.monotonic() + 5
    )
    assert [word.text for word in words] == ["first", "second"]
    assert [word.line for word in words] == [0, 1]
