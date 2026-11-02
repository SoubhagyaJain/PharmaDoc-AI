import pytest
from pydantic import ValidationError

from pharmadoc_ai.schemas import DocumentResult, Evidence, FieldResult


def test_roundtrip_and_null():
    result = DocumentResult(document_id="synthetic", fields={"patient_name": FieldResult()})
    restored = DocumentResult.model_validate_json(result.model_dump_json())
    assert restored.fields["patient_name"].value is None
    assert restored.reviewed_fields == {}


@pytest.mark.parametrize("box", [(-0.1, 0, 1, 1), (0.8, 0, 0.2, 1), (0, 0, 2, 1)])
def test_bad_boxes(box):
    with pytest.raises(ValidationError):
        Evidence(page_number=1, text="synthetic", bbox=box)


def test_forbid_extra_and_page_zero():
    with pytest.raises(ValidationError):
        Evidence(page_number=0, text="synthetic")
    with pytest.raises(ValidationError):
        FieldResult(confidence=0.99)
