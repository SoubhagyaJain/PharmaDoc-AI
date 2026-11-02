import logging

import pytest

from pharmadoc_ai.errors import DocumentError, EngineUnavailable
from pharmadoc_ai.pipeline import process_document


def test_digital_path_does_not_run_ocr(dataset, settings, monkeypatch, caplog):
    root, records = dataset

    def forbidden(*args):
        raise AssertionError("Digital text must not require OCR")

    monkeypatch.setattr("pharmadoc_ai.pipeline.TesseractEngine.recognize", forbidden)
    record = next(r for r in records if r["category"] == "prescription")
    with caplog.at_level(logging.INFO, logger="pharmadoc"):
        result = process_document(root / record["path"], settings)
    assert result.document_category == "prescription"
    assert result.pages[0].route == "digital"
    assert record["fields"]["patient_name"] not in caplog.text
    assert record["path"] not in caplog.text
    assert result.processing_duration_seconds > 0


def test_ocr_failure_and_deadline_are_explicit(dataset, settings, monkeypatch):
    root, records = dataset

    def fail(*args):
        raise DocumentError("Synthetic timeout", "ocr_timeout")

    monkeypatch.setattr("pharmadoc_ai.pipeline.TesseractEngine.recognize", fail)
    result = process_document(
        root / records[0]["path"], settings.model_copy(update={"force_ocr": True})
    )
    assert result.status == "failed"
    assert "ocr_timeout" in result.review_reasons
    result = process_document(
        root / records[0]["path"], settings.model_copy(update={"document_timeout_seconds": 1e-9})
    )
    assert result.human_review_required
    assert "processing_timeout" in result.review_reasons


def test_missing_ner_is_not_silent_fallback(dataset, settings):
    root, records = dataset
    with pytest.raises(EngineUnavailable):
        process_document(
            root / records[0]["path"], settings.model_copy(update={"extraction_method": "ner"})
        )


def test_render_pixel_limit(dataset, settings):
    root, records = dataset
    result = process_document(
        root / records[0]["path"], settings.model_copy(update={"max_pixels": 100})
    )
    assert result.status == "failed"
    assert "pixel_limit" in result.review_reasons
