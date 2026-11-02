"""Contextual baselines. Never interpret an absent dosage or correct a drug name."""

import re

from pharmadoc_ai.schemas import Evidence, FieldResult, MedicationResult, PageResult

DOSE = r"\b\d+(?:\.\d+)?\s*(?:tablets?|capsules?|mL|puffs?|drops?|mg)\b"
FREQUENCY = r"\b(?:once daily|twice daily|three times daily|every\s+\d+\s+hours|OD|BD|BID|TID|PRN|as needed)\b"
MED_START = re.compile(r"(?im)^\s*(?:Medication(?: name)?|Medicine|Drug)\s*:\s*([^\n|;]+)")


def find_evidence(
    pages: list[PageResult], value: str, page_number: int | None = None
) -> list[Evidence]:
    matches = []
    for page in pages:
        if page_number is not None and page.page_number != page_number:
            continue
        # Match against canonical page lines so evidence cannot be invented by a model.
        for line in page.text.splitlines():
            if value not in line:
                continue
            line_words = [word for word in page.words if word.text in line]
            bbox = None
            if line_words:
                # Restrict duplicate words to the closest line containing the value's words.
                candidates = {}
                for word in line_words:
                    candidates.setdefault(word.line, []).append(word)
                group = max(
                    candidates.values(),
                    key=lambda ws: sum(len(w.text) for w in ws if w.text in value),
                )
                bbox = (
                    min(w.bbox[0] for w in group),
                    min(w.bbox[1] for w in group),
                    max(w.bbox[2] for w in group),
                    max(w.bbox[3] for w in group),
                )
            matches.append(Evidence(page_number=page.page_number, text=line, bbox=bbox))
    return matches


def sourced_field(value, pages, method="rules", page_number=None) -> FieldResult:
    if value is None or value == "":
        return FieldResult(method=method)
    evidence = find_evidence(pages, str(value), page_number)
    if not evidence:
        return FieldResult(method=method, status="invalid")
    return FieldResult(value=value, method=method, evidence=evidence, status="extracted")


def labeled_field(text, labels, pages) -> FieldResult:
    found = re.findall(rf"(?im)^\s*(?:{labels})\s*:\s*([^\n]+)", text)
    values = list(dict.fromkeys(value.strip() for value in found if value.strip()))
    if len(values) > 1:
        return FieldResult(
            status="ambiguous", evidence=[e for v in values for e in find_evidence(pages, v)]
        )
    return sourced_field(values[0] if values else None, pages)


def extract_medications(pages: list[PageResult], method="rules") -> list[MedicationResult]:
    text = "\n".join(page.text for page in pages)
    starts = list(MED_START.finditer(text))
    medications = []
    for i, match in enumerate(starts):
        segment = text[match.start() : starts[i + 1].start() if i + 1 < len(starts) else len(text)]
        strength = re.search(r"(?im)\bStrength\s*:\s*([^\n|;]+)", segment)
        directions = re.search(r"(?im)^\s*(?:Directions|Instructions|Sig)\s*:\s*([^\n]+)", segment)
        explicit_dose = re.search(r"(?im)^\s*Dose\s*:\s*([^\n]+)", segment)
        explicit_freq = re.search(r"(?im)^\s*Frequency\s*:\s*([^\n]+)", segment)
        instruction = directions.group(1).strip() if directions else None
        dose = explicit_dose.group(1).strip() if explicit_dose else None
        frequency = explicit_freq.group(1).strip() if explicit_freq else None
        if instruction:
            dose_match, frequency_match = (
                re.search(DOSE, instruction, re.I),
                re.search(FREQUENCY, instruction, re.I),
            )
            dose = dose or (dose_match.group() if dose_match else None)
            frequency = frequency or (frequency_match.group() if frequency_match else None)
        med = MedicationResult(
            name=sourced_field(match.group(1).strip(), pages, method),
            strength=sourced_field(strength.group(1).strip() if strength else None, pages, method),
            administered_dose=sourced_field(dose, pages, method),
            frequency=sourced_field(frequency, pages, method),
            instructions=sourced_field(instruction, pages, method),
        )
        if frequency and frequency.upper() in {"OD", "BD", "BID", "TID", "PRN"}:
            med.frequency.status = "ambiguous"
        medications.append(med)
    return medications


def extract_rules(pages: list[PageResult], category: str):
    text = "\n".join(page.text for page in pages)
    fields = {}
    if category in ("prescription", "clinical_form"):
        fields["patient_name"] = labeled_field(text, r"Patient(?: name)?|Name of patient", pages)
        fields["document_date"] = labeled_field(text, r"Document date|Issue date|Date", pages)
    if category == "prescription":
        fields["prescriber_name"] = labeled_field(text, r"Prescriber(?: name)?|Doctor", pages)
    elif category == "clinical_form":
        fields["form_id"] = labeled_field(text, r"Form ID|Form identifier", pages)
    elif category == "regulatory":
        for page in pages:
            title = next(
                (
                    line.strip()
                    for line in page.text.splitlines()
                    if "quality record" in line.lower() and "continued" not in line.lower()
                ),
                None,
            )
            if title:
                fields["title"] = sourced_field(title, pages)
                break
        fields.setdefault("title", FieldResult())
        fields["document_id"] = labeled_field(text, r"Document ID|Document identifier", pages)
        fields["revision_date"] = labeled_field(text, r"Revision date", pages)
        for i, heading in enumerate(re.findall(r"(?im)^Section:\s*(.+)$", text)):
            fields[f"section_{i + 1}"] = sourced_field(heading.strip(), pages)
    return fields, extract_medications(pages) if category == "prescription" else []
