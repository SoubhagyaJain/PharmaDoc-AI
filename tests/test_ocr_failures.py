import time

import numpy as np
import pytest

from pharmadoc_ai.errors import EngineUnavailable
from pharmadoc_ai.ocr.tesseract import TesseractEngine


def test_broken_configured_binary_is_a_clear_error(settings, monkeypatch):
    monkeypatch.setattr("pharmadoc_ai.ocr.tesseract.available", lambda settings: True)
    monkeypatch.setattr("pharmadoc_ai.ocr.tesseract.is_blank", lambda image: False)
    monkeypatch.setattr("pharmadoc_ai.ocr.tesseract.binary_version", lambda command: "test")

    def fail(*args, **kwargs):
        raise OSError("Synthetic executable failure")

    monkeypatch.setattr("pharmadoc_ai.ocr.tesseract.pytesseract.image_to_osd", fail)
    with pytest.raises(EngineUnavailable):
        TesseractEngine().recognize(
            np.zeros((100, 100, 3), dtype=np.uint8), settings, time.monotonic() + 5
        )
