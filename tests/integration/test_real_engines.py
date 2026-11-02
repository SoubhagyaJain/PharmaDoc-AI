import pytest

from pharmadoc_ai.config import ROOT, Settings
from pharmadoc_ai.ocr.tesseract import available
from pharmadoc_ai.pipeline import process_document


@pytest.mark.integration
def test_tesseract_scanned_prescription():
    settings = Settings()
    if not available(settings):
        pytest.skip("Tesseract is not installed")
    path = ROOT / "data/examples/prescription_scan.pdf"
    if not path.is_file():
        pytest.skip("Generate curated examples first")
    result = process_document(path, settings)
    assert result.document_category == "prescription"
    assert result.medications[0].strength.value == "500 mg"
    assert result.medications[0].administered_dose.value == "1 tablet"
    assert result.medications[0].frequency.value == "once daily"


@pytest.mark.integration
def test_mixed_digital_and_scanned_pages():
    settings = Settings()
    path = ROOT / "data/examples/mixed_pages.pdf"
    if not available(settings) or not path.is_file():
        pytest.skip("Requires Tesseract and generated examples")
    result = process_document(path, settings)
    assert [page.route for page in result.pages] == ["digital", "ocr"]
    assert all(page.text for page in result.pages)
