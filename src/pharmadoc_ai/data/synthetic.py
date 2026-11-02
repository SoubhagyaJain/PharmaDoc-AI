"""Procedural PDF/scan generation; labels originate from sampled data, not extraction."""

import hashlib
import io
import json
import random
import re
from pathlib import Path

import cv2
import numpy as np
import pymupdf
from PIL import Image

from pharmadoc_ai.data.splits import SPLIT_BY_FAMILY
from pharmadoc_ai.ingestion import inspect_page

CATEGORIES = ("prescription", "clinical_form", "regulatory")
ENTITY_LABELS = {
    "patient_name": "PATIENT",
    "prescriber_name": "PRESCRIBER",
    "document_date": "DATE",
    "name": "MEDICATION",
    "strength": "STRENGTH",
    "administered_dose": "DOSE",
    "frequency": "FREQUENCY",
}
MEDICINES = [
    ("Amoxicillin", "500 mg", "1 capsule"),
    ("Metformin", "500 mg", "1 tablet"),
    ("Cetirizine", "10 mg", "1 tablet"),
    ("Paracetamol", "500 mg", "2 tablets"),
    ("Omeprazole", "20 mg", "1 capsule"),
    ("Atorvastatin", "10 mg", "1 tablet"),
    ("Amlodipine", "5 mg", "1 tablet"),
    ("Ibuprofen", "200 mg", "1 tablet"),
]
FREQUENCIES = ["once daily", "twice daily", "three times daily", "every 8 hours"]
FIRST = [
    "Asha",
    "Ravi",
    "Mira",
    "Kiran",
    "Neha",
    "Arun",
    "Leela",
    "Dev",
    "Isha",
    "Nila",
    "Vera",
    "Rohan",
    "Tara",
    "Mohan",
    "Anya",
    "Sana",
    "Vihaan",
    "Diya",
    "Om",
    "Riya",
]
LAST = [
    "Rao",
    "Shah",
    "Sen",
    "Das",
    "Bose",
    "Jain",
    "Varma",
    "Patel",
    "Mehta",
    "Nair",
    "Roy",
    "Kumar",
    "Kapoor",
    "Bhat",
    "Seth",
    "Dutta",
    "Gill",
    "Suri",
    "Pal",
    "Rana",
]


def _insert(page, x, y, text, size=11, font="helv"):
    page.insert_text((x, y), text, fontsize=size, fontname=font)


def _make_page(doc, family, title):
    landscape = family == 3
    width, height = (842, 595) if landscape else (595, 842)
    page = doc.new_page(width=width, height=height)
    # Six layout families: plain, ruled ledger, indented cards, landscape side-panel,
    # validation framed sheet, and test compact reordered sheet.
    x = [45, 65, 150, 220, 75, 45][family]
    y = [145, 170, 145, 125, 190, 155][family]
    font = "cour" if family == 1 else "helv"
    _insert(page, 35, 28, "SYNTHETIC - NOT FOR CLINICAL USE", 9)
    _insert(page, x, 72 if family != 4 else 100, title, 17)
    _insert(page, 35, height - 22, "PharmaDoc AI fictional training document", 8)
    if family in (1, 4):
        page.draw_rect(
            pymupdf.Rect(28, 110 if family == 1 else 125, width - 28, height - 45), width=0.8
        )
    if family == 2:
        page.draw_rect(pymupdf.Rect(25, 110, 125, height - 45), fill=(0.94, 0.94, 0.94), color=None)
        _insert(page, 37, 145, "Record", 12)
    if family == 3:
        page.draw_line((180, 90), (180, height - 45))
        _insert(page, 35, 125, "Document", 12)
        _insert(page, 35, 148, "information", 12)
    if family == 5:
        page.draw_line((35, 90), (width - 35, 90))
    return page, x, y, font


def _labels_from_pdf(doc, fields, medications, checkbox_labels):
    pages = []
    entities = []
    evidence = {}
    for page in doc:
        inspected = inspect_page(page)
        pages.append(
            {
                "page_number": page.number + 1,
                "text": inspected.text,
                "words": [w.model_dump() for w in inspected.words],
            }
        )
    candidates = list(fields.items())
    for i, med in enumerate(medications):
        candidates += [(f"medications.{i}.{key}", value) for key, value in med.items()]
    # Restrict each repeated medication value to its own generated block. Without
    # this, two drugs of the same strength could be annotated on the first drug.
    full_text = "\n".join(p["text"] for p in pages)
    med_starts = [m.start() for m in re.finditer(r"(?m)^Medication(?: name)?:", full_text)]
    page_offsets = []
    offset = 0
    for p in pages:
        page_offsets.append(offset)
        offset += len(p["text"]) + 1
    occupied = set()
    for key, value in candidates:
        if not isinstance(value, str) or not value:
            continue
        lower, upper = 0, len(full_text)
        prefixes = {
            "patient_name": "Patient name:",
            "document_date": "Document date:",
            "prescriber_name": "Prescriber:",
        }
        if key in prefixes and prefixes[key] in full_text:
            lower = full_text.index(prefixes[key]) + len(prefixes[key])
            newline = full_text.find("\n", lower)
            upper = newline if newline >= 0 else len(full_text)
        if key.startswith("medications."):
            med_index = int(key.split(".")[1])
            if med_index < len(med_starts):
                lower = med_starts[med_index]
                upper = (
                    med_starts[med_index + 1] if med_index + 1 < len(med_starts) else len(full_text)
                )
        for page, page_offset in zip(pages, page_offsets, strict=True):
            start = max(0, lower - page_offset)
            stop = min(len(page["text"]), upper - page_offset)
            if start >= stop:
                continue
            while (at := page["text"].find(value, start, stop)) >= 0:
                end = at + len(value)
                label = ENTITY_LABELS.get(key.split(".")[-1])
                if label and not any((page["page_number"], k) in occupied for k in range(at, end)):
                    entities.append(
                        {
                            "page_number": page["page_number"],
                            "start": at,
                            "end": end,
                            "label": label,
                        }
                    )
                    occupied.update((page["page_number"], k) for k in range(at, end))
                evidence.setdefault(key, {"page_number": page["page_number"], "text": value})
                start = end
                # Instructions overlap dose/frequency and are deliberately not NER labels.
                if key.split(".")[-1] not in {"administered_dose", "frequency"}:
                    break
            if key in evidence:
                break
    return pages, entities, evidence, checkbox_labels


def create_document(
    output: Path, category: str, family: int, index: int, rng: random.Random, name: str
) -> dict:
    split = SPLIT_BY_FAMILY[family]
    doc_id = f"{category}-{family}-{index:03d}"
    fields = {}
    meds = []
    checkboxes = []
    date = f"2025-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}"
    titles = {
        "prescription": "Medical Prescription",
        "clinical_form": "Clinical Intake and Insurance Form",
        "regulatory": "Fictional Pharmaceutical Quality Record",
    }
    with pymupdf.open() as doc:
        page, x, y, font = _make_page(doc, family, titles[category])

        def line(text, gap=23):
            nonlocal y
            _insert(page, x, y, text, 10 if family in (1, 2) else 11, font)
            y += gap

        if category in ("prescription", "clinical_form"):
            fields = {"patient_name": name, "document_date": date}
            metadata = [("Patient name", name), ("Document date", date)]
            if category == "prescription":
                fields["prescriber_name"] = rng.choice(
                    ["Dr. Tara Suri", "Dr. Arun Seth", "Dr. Nila Bhat", "Dr. Dev Pal"]
                )
                metadata.append(("Prescriber", fields["prescriber_name"]))
            else:
                fields["form_id"] = f"FORM-{index:04d}-{family}"
                metadata.append(("Form ID", fields["form_id"]))
            if family == 5:
                metadata.reverse()
            for label, value in metadata:
                line(f"{label}: {value}")
        if category == "prescription":
            y += 15
            chosen = rng.sample(MEDICINES, rng.choice([1, 1, 2, 3]))
            for mi, (drug, strength, dose) in enumerate(chosen):
                if mi == 2 or (mi == 1 and index % 10 == 0):
                    page, x, y, font = _make_page(doc, family, "Prescription - continued")
                frequency = rng.choice(FREQUENCIES)
                missing = index % 13 == 0 and mi == 0
                ambiguous = index % 17 == 1 and mi == 0
                dose_value = None if missing else dose
                freq_value = "OD" if ambiguous else frequency
                instructions = (
                    f"Take {dose} {freq_value}."
                    if dose_value
                    else f"Use as directed; {freq_value}."
                )
                meds.append(
                    {
                        "name": drug,
                        "strength": strength,
                        "administered_dose": dose_value,
                        "instructions": instructions,
                        "frequency": freq_value,
                    }
                )
                if family in (2, 5):
                    line(f"Medication name: {drug}")
                    line(f"Strength: {strength}")
                    if dose_value:
                        line(f"Dose: {dose}")
                    line(f"Frequency: {freq_value}")
                else:
                    line(f"Medication: {drug} | Strength: {strength}")
                line(f"Directions: {instructions}")
                y += 12
        elif category == "clinical_form":
            y += 25
            line("Consent and coverage selections")
            for label in ("Consent provided", "Insurance available", "Follow-up requested"):
                rect = pymupdf.Rect(x, y - 12, x + 13, y + 1)
                page.draw_rect(rect, width=1)
                checked = rng.choice([True, False])
                ambiguous = index % 11 == 0 and label == "Follow-up requested"
                if checked and not ambiguous:
                    page.draw_line((x + 3, y - 6), (x + 6, y - 2), width=1.7)
                    page.draw_line((x + 6, y - 2), (x + 11, y - 10), width=1.7)
                if ambiguous:
                    page.draw_circle((x + 6, y - 6), 1.2, color=(0, 0, 0), fill=(0, 0, 0))
                _insert(page, x + 24, y, label, 11)
                box = (
                    rect.x0 / page.rect.width,
                    rect.y0 / page.rect.height,
                    rect.x1 / page.rect.width,
                    rect.y1 / page.rect.height,
                )
                checkboxes.append(
                    {
                        "page_number": page.number + 1,
                        "label": label,
                        "bbox": box,
                        "value": None if ambiguous else checked,
                        "state": "ambiguous"
                        if ambiguous
                        else ("checked" if checked else "unchecked"),
                    }
                )
                y += 30
        else:
            fields = {
                "title": "Fictional Pharmaceutical Quality Record",
                "document_id": f"REG-{index:04d}-{family}",
                "revision_date": date,
            }
            line(f"Document ID: {fields['document_id']}")
            line(f"Revision date: {date}")
            line("Section: Scope")
            line("Fictional packaging review for demonstration only.")
            line("Section: Records")
            line("Sample batch records use invented identifiers.")
            if index % 5 == 0:
                page, x, y, font = _make_page(doc, family, "Quality record - continued")
                line("Section: Limitations")
                line("This document is not regulatory guidance.")
        doc.set_metadata(
            {
                "title": titles[category],
                "author": "PharmaDoc AI synthetic generator",
                "creationDate": "D:20250101000000Z",
            }
        )
        path = output / f"{doc_id}.pdf"
        doc.save(path, garbage=4, deflate=True, no_new_id=True)
        pages, entities, evidence, checkboxes = _labels_from_pdf(doc, fields, meds, checkboxes)
    record = {
        "document_id": doc_id,
        "category": category,
        "split": split,
        "template_family": f"{category}-layout-{family}",
        "patient_record_id": f"{split}-{name}" if category != "regulatory" else None,
        "lineage_id": doc_id,
        "path": path.name,
        "fields": fields,
        "medications": meds,
        "pages": pages,
        "entities": entities,
        "evidence": evidence,
        "checkboxes": checkboxes,
    }
    record["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    return record


def generate_dataset(output: Path, seed: int = 42, per_family: int = 10) -> list[dict]:
    """per_family=10 gives 180 prescriptions + 60 forms + 60 regulatory records."""
    if per_family < 1 or per_family > 20:
        raise ValueError("per_family must be between 1 and 20.")
    output.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    names = [f"{first} {last}" for first in FIRST for last in LAST]
    rng.shuffle(names)
    # The pool is partitioned BEFORE sampling; identities cannot cross splits.
    pools = {"train": names[:260], "validation": names[260:330], "test": names[330:]}
    counters = {split: 0 for split in pools}
    records = []
    for category in CATEGORIES:
        for family in range(6):
            split = SPLIT_BY_FAMILY[family]
            count = per_family * (3 if category == "prescription" else 1)
            for index in range(count):
                name = pools[split][counters[split] % len(pools[split])]
                if category != "regulatory":
                    counters[split] += 1
                records.append(create_document(output, category, family, index, rng, name))
    manifest = output / "manifest.json"
    manifest.write_text(
        json.dumps({"seed": seed, "per_family": per_family, "records": records}, indent=2),
        encoding="utf-8",
    )
    return records


def create_scan(source: Path, destination: Path, degradation: str = "clean", seed: int = 42):
    """Create a derivative without changing logical labels or split identity."""
    rng = np.random.default_rng(seed)
    with pymupdf.open(source) as doc, pymupdf.open() as scanned:
        for page in doc:
            pix = page.get_pixmap(dpi=150, alpha=False, colorspace=pymupdf.csRGB)
            image = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, 3).copy()
            if degradation == "noisy":
                image = np.clip(
                    image.astype(float) + rng.normal(0, 22, image.shape), 0, 255
                ).astype(np.uint8)
            elif degradation == "low_resolution":
                image = cv2.resize(image, None, fx=0.45, fy=0.45, interpolation=cv2.INTER_AREA)
            elif degradation == "rotated":
                image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
            elif degradation == "skewed":
                h, w = image.shape[:2]
                image = cv2.warpAffine(
                    image,
                    cv2.getRotationMatrix2D((w / 2, h / 2), 4, 1),
                    (w, h),
                    borderValue=(255, 255, 255),
                )
            elif degradation == "blurry":
                image = cv2.GaussianBlur(image, (7, 7), 2)
            elif degradation != "clean":
                raise ValueError("Unknown degradation profile.")
            buffer = io.BytesIO()
            Image.fromarray(image).save(buffer, format="PNG")
            target = scanned.new_page(width=page.rect.width, height=page.rect.height)
            target.insert_image(target.rect, stream=buffer.getvalue())
        destination.parent.mkdir(parents=True, exist_ok=True)
        scanned.save(destination, deflate=True, no_new_id=True)
