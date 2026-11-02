import json
import subprocess
import time
from pathlib import Path

import numpy as np
import pytest

from pharmadoc_ai.errors import DocumentError
from pharmadoc_ai.ocr.paddle import PaddleEngine


def test_worker_contract_and_cleanup(tmp_path, settings, monkeypatch):
    executable = tmp_path / "worker-python.exe"
    executable.touch()
    configured = settings.model_copy(update={"paddle_python": executable})
    paths = []

    def run(command, **kwargs):
        output = Path(command[3])
        paths.append(output)
        output.write_text(
            json.dumps(
                {
                    "words": [
                        {
                            "text": "Patient name: Synthetic Person",
                            "bbox": [10, 20, 90, 40],
                            "score": 0.9,
                            "line": 0,
                        }
                    ],
                    "versions": {"paddleocr": "test"},
                }
            )
        )

    monkeypatch.setattr("pharmadoc_ai.ocr.paddle.subprocess.run", run)
    engine = PaddleEngine()
    words = engine.recognize(
        np.zeros((100, 100, 3), dtype=np.uint8), configured, time.monotonic() + 10
    )
    assert words[0].bbox == (0.1, 0.2, 0.9, 0.4)
    assert words[0].score == 0.9
    assert engine.metadata["version_paddleocr"] == "test"
    assert not paths[0].exists()


def test_worker_timeout_is_reported(tmp_path, settings, monkeypatch):
    executable = tmp_path / "worker-python.exe"
    executable.touch()
    paths = []

    def fail(command, **kwargs):
        paths.append(Path(command[2]))
        raise subprocess.TimeoutExpired(command, kwargs["timeout"])

    monkeypatch.setattr("pharmadoc_ai.ocr.paddle.subprocess.run", fail)
    with pytest.raises(DocumentError) as error:
        PaddleEngine().recognize(
            np.zeros((100, 100, 3), dtype=np.uint8),
            settings.model_copy(update={"paddle_python": executable}),
            time.monotonic() + 5,
        )
    assert error.value.code == "ocr_timeout"
    assert not paths[0].exists()
