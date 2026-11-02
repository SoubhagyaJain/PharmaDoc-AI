import pymupdf
import pytest
from PIL import Image

from pharmadoc_ai.ingestion import InvalidDocumentError, inspect_document, open_document


@pytest.mark.parametrize("suffix", ["png", "jpg", "jpeg"])
def test_real_image_decoding(tmp_path, settings, suffix):
    path = tmp_path / f"synthetic.{suffix}"
    Image.new("RGB", (200, 300), "white").save(path)
    with open_document(path, settings) as doc:
        assert len(doc) == 1
    assert inspect_document(path, settings)[0].route == "ocr"


def test_image_pixel_limit(tmp_path, settings):
    path = tmp_path / "synthetic.png"
    Image.new("RGB", (200, 300), "white").save(path)
    with pytest.raises(InvalidDocumentError, match="pixel limit"):
        with open_document(path, settings.model_copy(update={"max_pixels": 100})):
            pass


def test_rotated_pdf_native_evidence_is_normalized(tmp_path, settings):
    path = tmp_path / "rotated.pdf"
    with pymupdf.open() as doc:
        page = doc.new_page(width=400, height=600)
        page.insert_text((50, 100), "Patient name: Synthetic Example")
        page.set_rotation(90)
        doc.save(path)
    result = inspect_document(path, settings)[0]
    assert result.width == 600 and result.height == 400
    assert all(0 <= v <= 1 for word in result.words for v in word.bbox)
    assert any(word.bbox[0] > 0.7 for word in result.words)
