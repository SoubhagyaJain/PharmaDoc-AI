"""One local pipeline shared by CLI, evaluation, and HTTP. Pages are sequential."""

import hashlib
import importlib.metadata
import json
import logging
import time
from pathlib import Path

from pharmadoc_ai.classification import classify
from pharmadoc_ai.config import Settings
from pharmadoc_ai.errors import DocumentError, EngineUnavailable, ProcessingTimeout
from pharmadoc_ai.extraction.rules import extract_rules
from pharmadoc_ai.ingestion import inspect_page, open_document, render_page, words_to_text
from pharmadoc_ai.ocr.base import geometric_order
from pharmadoc_ai.ocr.tesseract import TesseractEngine
from pharmadoc_ai.omr import detect_checkboxes
from pharmadoc_ai.preprocessing import is_blank
from pharmadoc_ai.schemas import DocumentResult, Evidence, FieldResult
from pharmadoc_ai.validation import validate_result

logger = logging.getLogger("pharmadoc")


def merge_words(native, recognized):
    merged = list(native)
    for word in recognized:
        cx, cy = (word.bbox[0] + word.bbox[2]) / 2, (word.bbox[1] + word.bbox[3]) / 2
        if not any(
            w.bbox[0] - 0.002 <= cx <= w.bbox[2] + 0.002
            and w.bbox[1] - 0.002 <= cy <= w.bbox[3] + 0.002
            for w in native
        ):
            merged.append(word)
    return geometric_order(merged)


def process_document(path: Path, settings: Settings | None = None) -> DocumentResult:
    settings = settings or Settings()
    started = time.monotonic()
    deadline = started + settings.document_timeout_seconds
    # Hash the already bounded input in chunks; never load another full copy into RAM.
    from pharmadoc_ai.ingestion import detect_file_type

    detect_file_type(path, settings.max_upload_bytes)
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(65536), b""):
            digest.update(chunk)
    result = DocumentResult(document_id=digest.hexdigest()[:24])
    result.metadata = {
        "ocr_engine": settings.ocr_engine,
        "extraction_method": settings.extraction_method,
        "preprocessing": settings.preprocessing,
        "render_dpi": settings.render_dpi,
        "force_ocr": settings.force_ocr,
        "scores_are_calibrated": False,
        "minimum_ocr_score": settings.minimum_ocr_score,
    }
    for package in ("pymupdf", "pytesseract", "scikit-learn", "spacy"):
        result.metadata[f"version_{package}"] = importlib.metadata.version(package)
    engine = TesseractEngine()
    if settings.ocr_engine == "paddle":
        from pharmadoc_ai.ocr.paddle import PaddleEngine

        engine = PaddleEngine()
    with open_document(path, settings) as document:
        for page in document:
            item = inspect_page(page, settings.force_ocr)
            page_started = time.monotonic()
            try:
                if time.monotonic() >= deadline:
                    raise ProcessingTimeout()
                image = render_page(page, settings)
                if item.route in {"ocr", "hybrid"}:
                    if is_blank(image):
                        item.route, item.words, item.text = "blank", [], ""
                    else:
                        recognized = engine.recognize(image, settings, deadline)
                        result.metadata.update(getattr(engine, "metadata", {}))
                        item.words = (
                            merge_words(item.words, recognized)
                            if item.route == "hybrid"
                            else recognized
                        )
                        item.text = words_to_text(item.words)
                        if not item.words:
                            item.issues.append("no_text_recognized")
                        scores = [w.score for w in item.words if w.score is not None]
                        scale = 100 if settings.ocr_engine == "paddle" else 1
                        if (
                            scores
                            and sum(scores) / len(scores) * scale < settings.minimum_ocr_score
                        ):
                            item.issues.append("low_ocr_score")
                item.checkboxes = detect_checkboxes(image, item.words)
            except EngineUnavailable:
                raise
            except DocumentError as exc:
                item.route = "failed"
                item.issues.append(exc.code)
            result.pages.append(item)
            result.stage_timings[f"page_{item.page_number}"] = time.monotonic() - page_started
    failed = sum(p.route == "failed" for p in result.pages)
    if failed:
        result.status = "failed" if failed == len(result.pages) else "partial"
    category, method, margin = classify(
        "\n".join(p.text for p in result.pages if p.route != "failed"), settings.classifier_path
    )
    result.document_category, result.classification_method, result.classification_margin = (
        category,
        method,
        margin,
    )
    extraction_started = time.monotonic()
    try:
        if time.monotonic() >= deadline:
            raise ProcessingTimeout()
        if settings.extraction_method == "ner":
            from pharmadoc_ai.extraction.ner import extract_ner

            result.fields, result.medications = extract_ner(
                result.pages, category, settings.ner_path
            )
        elif settings.extraction_method == "llm" and category == "prescription":
            from pharmadoc_ai.extraction.llm import extract_llm

            result.fields, result.medications, metadata = extract_llm(
                result.pages, category, settings, deadline
            )
            result.metadata.update(metadata)
        else:
            result.fields, result.medications = extract_rules(result.pages, category)
    except EngineUnavailable:
        raise
    except DocumentError as exc:
        result.status = "partial"
        result.pages[0].issues.append(exc.code)
    if category == "clinical_form":
        for page in result.pages:
            for i, box in enumerate(page.checkboxes):
                key = "checkbox_" + "_".join(
                    (box.label or f"unlabeled_{page.page_number}_{i}").lower().split()
                )
                result.fields[key] = FieldResult(
                    value=box.value,
                    method="omr",
                    status="ambiguous" if box.value is None else "extracted",
                    evidence=[
                        Evidence(page_number=page.page_number, text=box.label or "", bbox=box.bbox)
                    ],
                    raw_score=box.ink_ratio,
                    score_source="checkbox_interior_ink_ratio",
                )
    result.stage_timings["extraction"] = time.monotonic() - extraction_started
    validate_result(result)
    result.processing_duration_seconds = time.monotonic() - started
    logger.info(
        json.dumps(
            {
                "document_id": result.document_id,
                "stage": "complete",
                "status": result.status,
                "duration_seconds": round(result.processing_duration_seconds, 3),
            }
        )
    )
    return result
