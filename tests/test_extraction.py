from pharmadoc_ai.extraction.rules import extract_rules
from pharmadoc_ai.schemas import DocumentResult, PageResult, ReviewCorrection
from pharmadoc_ai.validation import review_result, validate_result


def page(text):
    return PageResult(page_number=1, width=595, height=842, route="digital", text=text)


def test_strength_dose_frequency_are_separate():
    pages = [
        page(
            "Patient name: Example Person\nDocument date: 2025-01-02\nPrescriber: Dr. Example\nMedication: Amoxicillin | Strength: 500 mg\nDirections: Take 1 capsule three times daily."
        )
    ]
    fields, meds = extract_rules(pages, "prescription")
    result = validate_result(
        DocumentResult(
            document_id="synthetic",
            document_category="prescription",
            pages=pages,
            fields=fields,
            medications=meds,
        )
    )
    assert meds[0].strength.value == "500 mg"
    assert meds[0].administered_dose.value == "1 capsule"
    assert meds[0].frequency.value == "three times daily"
    assert not result.human_review_required


def test_missing_dose_is_not_inferred_and_multiple_drugs_stay_separate():
    pages = [
        page(
            "Medication: Alpha | Strength: 5 mg\nDirections: Use as directed; once daily.\nMedication: Beta | Strength: 10 mg\nDirections: Take 2 tablets twice daily."
        )
    ]
    _, meds = extract_rules(pages, "prescription")
    assert len(meds) == 2
    assert meds[0].administered_dose.value is None
    assert meds[1].administered_dose.value == "2 tablets"


def test_conflicting_patient_names_and_ambiguous_frequency():
    pages = [
        page(
            "Patient name: Synthetic One\nPatient name: Synthetic Two\nMedication: Alpha | Strength: 5 mg\nDirections: Take 1 tablet OD."
        )
    ]
    fields, meds = extract_rules(pages, "prescription")
    result = validate_result(
        DocumentResult(
            document_id="synthetic",
            document_category="prescription",
            pages=pages,
            fields=fields,
            medications=meds,
        )
    )
    assert fields["patient_name"].value is None
    assert fields["patient_name"].status == "ambiguous"
    assert meds[0].frequency.value == "OD"
    assert "ambiguous_frequency" in result.review_reasons


def test_review_preserves_original_and_validates_date():
    pages = [page("Patient name: Synthetic Example\nDocument date: 2025-01-02\nForm ID: FORM-1")]
    fields, _ = extract_rules(pages, "clinical_form")
    original = validate_result(
        DocumentResult(
            document_id="synthetic", document_category="clinical_form", pages=pages, fields=fields
        )
    )
    reviewed = review_result(
        original, {"document_date": ReviewCorrection(value="invalid", reason="Test edit")}
    )
    assert original.fields["document_date"].value == "2025-01-02"
    assert reviewed["fields"]["document_date"]["value"] == "2025-01-02"
    assert reviewed["reviewed_fields"]["document_date"]["value"] == "invalid"
    assert reviewed["review_validation_errors"][0]["code"] == "invalid_date"
