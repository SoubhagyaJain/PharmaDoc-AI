"""Structural/evidence checks, not clinical correctness or calibrated certainty."""

import re
from datetime import date

from pharmadoc_ai.schemas import DocumentResult, FieldResult, ReviewCorrection, ValidationIssue

REQUIRED = {
    "prescription": ("patient_name", "document_date", "prescriber_name"),
    "clinical_form": ("patient_name", "document_date", "form_id"),
    "regulatory": ("title", "document_id", "revision_date"),
    "unknown": (),
}
CONTEXT = {
    "patient_name": r"patient(?: name)?\s*:",
    "prescriber_name": r"prescriber(?: name)?\s*:|doctor\s*:",
    "document_date": r"(?:document |issue )?date\s*:",
    "form_id": r"form (?:id|identifier)\s*:",
    "document_id": r"document (?:id|identifier)\s*:",
    "revision_date": r"revision date\s*:",
    "name": r"(?:medication(?: name)?|medicine|drug)\s*:",
    "strength": r"strength\s*:",
    "administered_dose": r"(?:dose|directions|instructions|sig)\s*:",
    "frequency": r"(?:frequency|directions|instructions|sig)\s*:",
    "instructions": r"(?:directions|instructions|sig)\s*:",
}


def all_fields(result: DocumentResult) -> dict[str, FieldResult]:
    fields = dict(result.fields)
    for i, medication in enumerate(result.medications):
        fields.update(
            {
                f"medications.{i}.{key}": getattr(medication, key)
                for key in type(medication).model_fields
            }
        )
    return fields


def validate_result(result: DocumentResult) -> DocumentResult:
    result.validation_errors = []
    result.missing_fields = []
    reasons = set()
    fields = all_fields(result)
    required = list(REQUIRED[result.document_category])
    if result.document_category == "prescription":
        if not result.medications:
            reasons.add("no_medications_extracted")
        for i in range(len(result.medications)):
            required += [
                f"medications.{i}.{key}"
                for key in ("name", "strength", "administered_dose", "frequency")
            ]

    def issue(path, code, message):
        result.validation_errors.append(ValidationIssue(field=path, code=code, message=message))
        reasons.add(code)

    for path in required:
        if path not in fields or fields[path].value is None:
            result.missing_fields.append(path)
            reasons.add("missing_required_fields")
    for path, field in fields.items():
        if field.status in {"ambiguous", "invalid"}:
            issue(path, f"{field.status}_field", "The field needs human review.")
        if field.value is None:
            continue
        key = path.split(".")[-1]
        if field.method not in {"human", "omr"}:
            valid = []
            for evidence in field.evidence:
                page = next(
                    (p for p in result.pages if p.page_number == evidence.page_number), None
                )
                if page and evidence.text in page.text and str(field.value) in evidence.text:
                    context = CONTEXT.get(key)
                    if context is None or re.search(context, evidence.text, re.I):
                        valid.append(evidence)
            if not valid:
                field.rejected_value = field.value
                field.value = None
                field.status = "invalid"
                issue(
                    path,
                    "unsupported_evidence",
                    "No source evidence in the required field context.",
                )
            else:
                field.evidence = valid
                scored = []
                for evidence in valid:
                    page = next(p for p in result.pages if p.page_number == evidence.page_number)
                    for word in page.words:
                        if word.score is None or not evidence.bbox:
                            continue
                        cx, cy = (
                            (word.bbox[0] + word.bbox[2]) / 2,
                            (word.bbox[1] + word.bbox[3]) / 2,
                        )
                        if (
                            evidence.bbox[0] <= cx <= evidence.bbox[2]
                            and evidence.bbox[1] <= cy <= evidence.bbox[3]
                            and (word.text in str(field.value) or str(field.value) in word.text)
                        ):
                            scored.append(word)
                if scored:
                    weakest = min(
                        scored,
                        key=lambda word: (
                            word.score * (100 if "paddle" in (word.score_source or "") else 1)
                        ),
                    )
                    field.raw_score, field.score_source = weakest.score, weakest.score_source
                    comparable = weakest.score * (
                        100 if "paddle" in (weakest.score_source or "") else 1
                    )
                    configured_threshold = result.metadata.get("minimum_ocr_score", 55)
                    threshold = (
                        configured_threshold
                        if isinstance(configured_threshold, (int, float))
                        else 55
                    )
                    if comparable < threshold:
                        field.status = "ambiguous"
                        issue(
                            path,
                            "low_field_ocr_score",
                            "A matching source span has a weak uncalibrated OCR score.",
                        )
        if (
            key == "name"
            and field.value
            and re.search(r"\?|\b(?:illegible|unclear|unknown)\b", str(field.value), re.I)
        ):
            field.status = "ambiguous"
            issue(
                path,
                "ambiguous_medication_name",
                "The source medication name is explicitly uncertain and was not corrected.",
            )
        if field.value is not None and key in {"document_date", "revision_date"}:
            try:
                date.fromisoformat(str(field.value))
            except ValueError:
                issue(
                    path,
                    "invalid_date",
                    "Expected an unambiguous ISO date; original text is preserved.",
                )
        if (
            key == "frequency"
            and field.value
            and str(field.value).upper() in {"OD", "BD", "BID", "TID", "PRN"}
        ):
            field.status = "ambiguous"
            issue(
                path, "ambiguous_frequency", "An abbreviation requires review; it was not expanded."
            )
    for i, medication in enumerate(result.medications):
        instruction = medication.instructions.value
        if instruction:
            for key in ("administered_dose", "frequency"):
                value = getattr(medication, key).value
                if value and str(value).casefold() not in str(instruction).casefold():
                    issue(
                        f"medications.{i}.{key}",
                        "instruction_conflict",
                        "Explicit value conflicts with the extracted instruction line.",
                    )
    # Recompute after evidence rejection so missing fields include rejected predictions.
    result.missing_fields = sorted(
        set(
            result.missing_fields
            + [path for path in required if path in fields and fields[path].value is None]
        )
    )
    for page in result.pages:
        reasons.update(page.issues)
        if page.route == "failed":
            reasons.add("page_processing_failed")
        if any(box.state == "ambiguous" or box.label is None for box in page.checkboxes):
            reasons.add("ambiguous_checkbox")
    if result.document_category == "unknown":
        reasons.add("unknown_document_category")
    if result.missing_fields:
        reasons.add("missing_required_fields")
    if result.status != "complete":
        reasons.add("partial_or_failed_processing")
    result.review_reasons = sorted(reasons)
    result.human_review_required = bool(reasons)
    return result


def review_result(original: DocumentResult, corrections: dict[str, ReviewCorrection]) -> dict:
    """Return an audit-preserving export; do not overwrite machine predictions."""
    effective = original.model_copy(deep=True)
    fields = all_fields(effective)
    for path, correction in corrections.items():
        if path not in fields:
            raise ValueError("Unknown correction field.")
        field = fields[path]
        field.value, field.method, field.status = (
            correction.value,
            "human",
            "missing" if correction.value is None else "extracted",
        )
    validate_result(effective)
    exported = original.model_copy(deep=True)
    exported.reviewed_fields = corrections
    return {
        **exported.model_dump(mode="json"),
        "review_validation_errors": [e.model_dump() for e in effective.validation_errors],
        "review_missing_fields": effective.missing_fields,
        "review_human_review_required": effective.human_review_required,
    }
