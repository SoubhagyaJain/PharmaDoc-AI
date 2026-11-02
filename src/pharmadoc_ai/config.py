"""Environment-configurable resource limits; no inference at import time."""

import shutil
from pathlib import Path
from typing import Literal
from urllib.parse import urlparse

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[2]


def tesseract_path() -> str:
    installed = shutil.which("tesseract")
    windows = Path("C:/Program Files/Tesseract-OCR/tesseract.exe")
    return installed or (str(windows) if windows.is_file() else "tesseract")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="PHARMADOC_", env_file=".env", extra="ignore")
    max_upload_bytes: int = Field(default=20 * 1024 * 1024, gt=0)
    max_pages: int = Field(default=10, ge=1, le=100)
    max_pixels: int = Field(default=25_000_000, gt=0)
    render_dpi: int = Field(default=200, ge=72, le=400)
    ocr_engine: Literal["tesseract", "paddle"] = "tesseract"
    extraction_method: Literal["rules", "ner", "llm"] = "rules"
    preprocessing: Literal["raw", "contrast", "adaptive", "deskew"] = "raw"
    force_ocr: bool = False
    tesseract_cmd: str = Field(default_factory=tesseract_path)
    ocr_timeout_seconds: float = Field(default=30, gt=0)
    document_timeout_seconds: float = Field(default=180, gt=0)
    minimum_ocr_score: float = Field(default=55, ge=0, le=100)
    paddle_python: Path = ROOT / ".venv-paddle/Scripts/python.exe"
    paddle_det_model: str = "PP-OCRv5_mobile_det"
    paddle_rec_model: str = "en_PP-OCRv5_mobile_rec"
    classifier_path: Path = ROOT / "artifacts/classifier.joblib"
    ner_path: Path = ROOT / "artifacts/ner"
    ollama_host: str = "http://127.0.0.1:11434"
    ollama_model: str = "qwen3.5:4b"
    llm_timeout_seconds: float = Field(default=30, gt=0)
    llm_context_tokens: int = Field(default=4096, ge=2048, le=8192)
    api_url: str = "http://127.0.0.1:8000"

    @field_validator("ollama_host", "api_url")
    @classmethod
    def local_endpoint(cls, value: str) -> str:
        parsed = urlparse(value)
        if (
            parsed.scheme not in {"http", "https"}
            or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}
            or parsed.username
            or parsed.password
        ):
            raise ValueError("Model and application endpoints must use loopback URLs.")
        return value.rstrip("/")
