"""Cheap signature prechecks followed by real decoding and per-page routing."""

import math
from contextlib import contextmanager
from pathlib import Path
from typing import Literal

import numpy as np
import pymupdf
from PIL import Image, UnidentifiedImageError

from pharmadoc_ai.config import Settings
from pharmadoc_ai.errors import DocumentError
from pharmadoc_ai.schemas import PageResult, Word


class InvalidDocumentError(DocumentError):
    pass


def detect_file_type(
    path: Path, max_bytes: int = 20 * 1024 * 1024
) -> Literal["pdf", "png", "jpeg"]:
    """Read at most eight content bytes; this does not validate full structure."""
    if max_bytes <= 0:
        raise InvalidDocumentError("The upload limit must be positive.")
    signatures = {
        ".pdf": ("pdf", b"%PDF-"),
        ".png": ("png", b"\x89PNG\r\n\x1a\n"),
        ".jpg": ("jpeg", b"\xff\xd8\xff"),
        ".jpeg": ("jpeg", b"\xff\xd8\xff"),
    }
    try:
        if not path.is_file():
            raise InvalidDocumentError("The input must be an existing regular file.")
        size = path.stat().st_size
        if size == 0:
            raise InvalidDocumentError("The input is empty.")
        if size > max_bytes:
            raise InvalidDocumentError(
                "The upload exceeds the byte limit.", "upload_too_large", 413
            )
        expected = signatures.get(path.suffix.lower())
        if expected is None:
            raise InvalidDocumentError("Unsupported file extension.", "unsupported_format", 415)
        with path.open("rb") as stream:
            prefix = stream.read(8)
        kind, signature = expected
        if not prefix.startswith(signature):
            raise InvalidDocumentError("The signature does not match the file extension.")
        return kind
    except OSError as exc:
        raise InvalidDocumentError("The input could not be read.") from exc


@contextmanager
def open_document(path: Path, settings: Settings):
    kind = detect_file_type(path, settings.max_upload_bytes)
    try:
        if kind != "pdf":
            with Image.open(path) as image:
                if image.width * image.height > settings.max_pixels:
                    raise InvalidDocumentError("Image exceeds the pixel limit.", "pixel_limit", 413)
                image.verify()
        # A bounded memory stream avoids native decoder handles retaining a
        # malformed file while Windows is cleaning up its temporary directory.
        doc = pymupdf.open(stream=path.read_bytes(), filetype=kind)
    except (
        UnidentifiedImageError,
        Image.DecompressionBombError,
        OSError,
        RuntimeError,
        ValueError,
    ) as exc:
        if isinstance(exc, DocumentError):
            raise
        raise InvalidDocumentError("The file could not be decoded.") from exc
    try:
        if doc.needs_pass:
            raise InvalidDocumentError(
                "Encrypted documents are not supported.", "encrypted_document"
            )
        if not 1 <= doc.page_count <= settings.max_pages:
            raise InvalidDocumentError("Document exceeds the page limit.", "page_limit", 413)
        if not doc.is_pdf:
            converted = pymupdf.open("pdf", doc.convert_to_pdf())
            doc.close()
            doc = converted
        yield doc
    finally:
        doc.close()


def normalized_box(box, width: float, height: float):
    return tuple(
        max(0.0, min(1.0, float(v) / d))
        for v, d in zip(box, (width, height, width, height), strict=True)
    )


def inspect_page(page, force_ocr: bool = False) -> PageResult:
    width, height = page.rect.width, page.rect.height
    if width <= 0 or height <= 0:
        raise InvalidDocumentError("Page has invalid dimensions.")
    native = page.get_text("words", sort=True)
    words = [
        Word(
            text=w[4],
            bbox=normalized_box(pymupdf.Rect(w[:4]) * page.rotation_matrix, width, height),
            line=w[5] * 1000 + w[6],
        )
        for w in native
        if str(w[4]).strip()
    ]
    images = page.get_image_info()
    coverage = min(
        1.0,
        sum(max(0, (r["bbox"][2] - r["bbox"][0]) * (r["bbox"][3] - r["bbox"][1])) for r in images)
        / (width * height),
    )
    text = words_to_text(words)
    usable = (
        len(words) >= 3
        and sum(c.isprintable() for c in text) / max(1, len(text)) > 0.9
        and "\ufffd" not in text
    )
    route = "ocr" if force_ocr or not usable else ("hybrid" if coverage > 0.08 else "digital")
    return PageResult(
        page_number=page.number + 1,
        width=width,
        height=height,
        route=route,
        text=text,
        words=words,
        image_coverage=coverage,
    )


def words_to_text(words: list[Word]) -> str:
    lines: dict[int, list[str]] = {}
    for word in words:
        lines.setdefault(word.line, []).append(word.text)
    return "\n".join(" ".join(parts) for parts in lines.values())


def render_page(page, settings: Settings, dpi: int | None = None) -> np.ndarray:
    scale = (dpi or settings.render_dpi) / 72
    width, height = math.ceil(page.rect.width * scale), math.ceil(page.rect.height * scale)
    if width * height > settings.max_pixels:
        raise InvalidDocumentError("Rendered page exceeds the pixel limit.", "pixel_limit", 413)
    pix = page.get_pixmap(
        matrix=pymupdf.Matrix(scale, scale), colorspace=pymupdf.csRGB, alpha=False
    )
    return np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3).copy()


def inspect_document(path: Path, settings: Settings | None = None) -> list[PageResult]:
    settings = settings or Settings()
    with open_document(path, settings) as doc:
        return [inspect_page(page, settings.force_ocr) for page in doc]
