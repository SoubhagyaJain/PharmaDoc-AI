"""Isolate native Paddle dependencies and terminate the worker on timeout."""

import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from pharmadoc_ai.config import ROOT, Settings
from pharmadoc_ai.errors import DocumentError, EngineUnavailable
from pharmadoc_ai.ocr.base import remaining
from pharmadoc_ai.preprocessing import preprocess
from pharmadoc_ai.schemas import Word


class PaddleEngine:
    def recognize(self, image: np.ndarray, settings: Settings, deadline: float) -> list[Word]:
        if not settings.paddle_python.is_file():
            raise EngineUnavailable("PaddleOCR")
        prepared = preprocess(image, settings.preprocessing)
        with tempfile.TemporaryDirectory(prefix="pharmadoc-paddle-") as temp:
            root = Path(temp)
            Image.fromarray(prepared.image).save(root / "page.png")
            command = [
                str(settings.paddle_python),
                str(ROOT / "src/pharmadoc_ai/ocr/paddle_worker.py"),
                str(root / "page.png"),
                str(root / "result.json"),
                settings.paddle_det_model,
                settings.paddle_rec_model,
            ]
            try:
                subprocess.run(
                    command,
                    capture_output=True,
                    timeout=remaining(deadline, settings.ocr_timeout_seconds),
                    check=True,
                )
                payload = json.loads((root / "result.json").read_text(encoding="utf-8"))
            except subprocess.TimeoutExpired as exc:
                raise DocumentError("PaddleOCR exceeded its timeout.", "ocr_timeout") from exc
            except (subprocess.CalledProcessError, OSError, ValueError) as exc:
                raise DocumentError(
                    "PaddleOCR worker failed; check its installation.", "ocr_failed"
                ) from exc
        self.metadata = {
            f"version_{key}": value for key, value in payload.get("versions", {}).items()
        }
        return [
            Word(
                text=w["text"],
                bbox=prepared.original_box(w["bbox"]),
                line=w.get("line", 0),
                score=w["score"],
                score_source="paddle_0_1_uncalibrated",
            )
            for w in payload["words"]
        ]
