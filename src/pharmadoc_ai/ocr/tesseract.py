import shutil
import subprocess
from functools import lru_cache
from pathlib import Path

import numpy as np
import pytesseract

from pharmadoc_ai.config import Settings
from pharmadoc_ai.errors import DocumentError, EngineUnavailable
from pharmadoc_ai.ocr.base import geometric_order, remaining
from pharmadoc_ai.preprocessing import is_blank, preprocess
from pharmadoc_ai.schemas import Word


def available(settings: Settings) -> bool:
    return bool(shutil.which(settings.tesseract_cmd) or Path(settings.tesseract_cmd).is_file())


@lru_cache(maxsize=4)
def binary_version(command: str) -> str:
    try:
        response = subprocess.run(
            [command, "--version"], capture_output=True, text=True, timeout=3, check=True
        )
        return response.stdout.splitlines()[0]
    except (OSError, subprocess.SubprocessError, IndexError):
        return "unavailable"


class TesseractEngine:
    def recognize(self, image: np.ndarray, settings: Settings, deadline: float) -> list[Word]:
        if not available(settings):
            raise EngineUnavailable("Tesseract")
        if is_blank(image):
            return []
        pytesseract.pytesseract.tesseract_cmd = settings.tesseract_cmd
        self.metadata = {
            "tesseract_binary_version": binary_version(settings.tesseract_cmd),
            "ocr_language": "eng",
        }
        rotation = 0
        try:
            osd = pytesseract.image_to_osd(
                image, output_type=pytesseract.Output.DICT, timeout=remaining(deadline, 5)
            )
            if float(osd.get("orientation_conf", 0)) >= 2:
                rotation = -float(osd.get("rotate", 0))
        except (pytesseract.TesseractNotFoundError, OSError) as exc:
            raise EngineUnavailable("Tesseract") from exc
        except (pytesseract.TesseractError, RuntimeError):
            pass  # Short documents may not have enough characters for orientation detection.
        prepared = preprocess(image, settings.preprocessing, rotation)
        try:
            data = pytesseract.image_to_data(
                prepared.image,
                lang="eng",
                config="--psm 3",
                output_type=pytesseract.Output.DICT,
                timeout=remaining(deadline, settings.ocr_timeout_seconds),
            )
        except pytesseract.TesseractNotFoundError as exc:
            raise EngineUnavailable("Tesseract") from exc
        except (pytesseract.TesseractError, RuntimeError) as exc:
            raise DocumentError("OCR failed or exceeded its timeout.", "ocr_failed") from exc
        words = []
        height, width = prepared.image.shape[:2]
        for i, text in enumerate(data["text"]):
            if not text.strip():
                continue
            x, y, w, h = (data[key][i] for key in ("left", "top", "width", "height"))
            words.append(
                Word(
                    text=text,
                    bbox=(x / width, y / height, (x + w) / width, (y + h) / height),
                    score=float(data["conf"][i]),
                    score_source="tesseract_0_100_uncalibrated",
                )
            )
        # Read in corrected-image order, then restore evidence coordinates.
        # Sorting restored boxes would scramble reading order on rotated scans.
        return [
            word.model_copy(
                update={
                    "bbox": prepared.original_box(
                        (
                            word.bbox[0] * width,
                            word.bbox[1] * height,
                            word.bbox[2] * width,
                            word.bbox[3] * height,
                        )
                    )
                }
            )
            for word in geometric_order(words)
        ]
