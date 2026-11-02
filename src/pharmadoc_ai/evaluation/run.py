"""Evaluation defaults to validation. Test execution requires an explicit opt-in."""

import hashlib
import json
import platform
import tempfile
import time
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path

from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from pharmadoc_ai.classification import classify
from pharmadoc_ai.config import Settings
from pharmadoc_ai.data.synthetic import CATEGORIES, create_scan
from pharmadoc_ai.errors import DocumentError
from pharmadoc_ai.evaluation.metrics import (
    entity_counts,
    iou,
    latency_summary,
    normalize,
    precision_recall_f1,
    rate,
    text_counts,
    text_rates,
)
from pharmadoc_ai.extraction.rules import extract_rules
from pharmadoc_ai.ingestion import open_document, render_page
from pharmadoc_ai.pipeline import process_document
from pharmadoc_ai.schemas import DocumentResult, PageResult
from pharmadoc_ai.validation import all_fields, validate_result


def select_records(records, split, allow_test=False, prescriptions_only=False, limit=0):
    if split == "test" and not allow_test:
        raise ValueError(
            "Test data is sealed. Pass --allow-test only after freezing validation choices."
        )
    selected = [
        r
        for r in records
        if r["split"] == split and (not prescriptions_only or r["category"] == "prescription")
    ]
    selected.sort(key=lambda r: hashlib.sha256(("42:" + r["document_id"]).encode()).hexdigest())
    return selected[:limit] if limit else selected


def truth_fields(record):
    result = dict(record["fields"])
    for i, med in enumerate(record["medications"]):
        result.update({f"medications.{i}.{key}": value for key, value in med.items()})
    for box in record["checkboxes"]:
        result["checkbox_" + "_".join(box["label"].lower().split())] = box["value"]
    # Regulatory section fields are part of the schema and evaluated too.
    section = 0
    for page in record["pages"]:
        for line in page["text"].splitlines():
            if line.startswith("Section:"):
                section += 1
                result[f"section_{section}"] = line.split(":", 1)[1].strip()
    return result


def gold_text_result(record, settings):
    start = time.monotonic()
    result = DocumentResult(
        document_id=record["document_id"],
        pages=[
            PageResult(
                page_number=p["page_number"],
                width=595,
                height=842,
                route="digital",
                text=p["text"],
                words=p["words"],
            )
            for p in record["pages"]
        ],
    )
    result.document_category, result.classification_method, result.classification_margin = classify(
        "\n".join(p.text for p in result.pages), settings.classifier_path
    )
    if settings.extraction_method == "ner":
        from pharmadoc_ai.extraction.ner import extract_ner

        result.fields, result.medications = extract_ner(
            result.pages, result.document_category, settings.ner_path
        )
    elif settings.extraction_method == "llm" and result.document_category == "prescription":
        from pharmadoc_ai.extraction.llm import extract_llm

        result.fields, result.medications, metadata = extract_llm(
            result.pages,
            result.document_category,
            settings,
            time.monotonic() + settings.document_timeout_seconds,
        )
        result.metadata.update(metadata)
    else:
        result.fields, result.medications = extract_rules(result.pages, result.document_category)
    validate_result(result)
    result.processing_duration_seconds = time.monotonic() - start
    return result


def evaluate(
    manifest: Path,
    output: Path,
    settings: Settings,
    split="validation",
    source="digital",
    limit=0,
    prescriptions_only=False,
    allow_test=False,
):
    records = select_records(
        json.loads(manifest.read_text(encoding="utf-8"))["records"],
        split,
        allow_test,
        prescriptions_only,
        limit,
    )
    if not records:
        raise ValueError("No evaluation records matched.")
    output.mkdir(parents=True, exist_ok=True)
    counts = Counter()
    errors = Counter()
    reasons = Counter()
    per_field = defaultdict(Counter)
    expected_categories, actual_categories, latencies = [], [], []
    examples = []
    ocr_counts = Counter()
    checkpoint = []
    with tempfile.TemporaryDirectory(prefix="pharmadoc-eval-") as temp:
        for record in records:
            started = time.monotonic()
            try:
                if source == "gold":
                    result = gold_text_result(record, settings)
                else:
                    path = manifest.parent / record["path"]
                    if source == "scan":
                        scanned = Path(temp) / record["path"]
                        create_scan(path, scanned)
                        path = scanned
                    result = process_document(path, settings)
            except DocumentError as exc:
                errors[exc.code] += 1
                result = DocumentResult(
                    document_id=record["document_id"], status="failed", review_reasons=[exc.code]
                )
            latencies.append(time.monotonic() - started)
            expected_categories.append(record["category"])
            actual_categories.append(result.document_category)
            counts["documents"] += 1
            counts["review_documents"] += result.human_review_required
            counts["failed_documents"] += result.status == "failed"
            reasons.update(result.review_reasons)
            predicted_fields = all_fields(result)
            actual = {key: field.value for key, field in predicted_fields.items()}
            gold = truth_fields(record)
            tp, fp, fn = entity_counts(gold, actual)
            counts.update({"entity_tp": tp, "entity_fp": fp, "entity_fn": fn})
            for key, value in gold.items():
                predicted = actual.get(key)
                if value is None:
                    counts["absent_fields"] += 1
                    counts["false_fills"] += predicted is not None
                else:
                    matched = predicted is not None and normalize(value) == normalize(predicted)
                    counts["present_fields"] += 1
                    counts["exact_fields"] += matched
                    per_field[key]["total"] += 1
                    per_field[key]["correct"] += matched
            for field in predicted_fields.values():
                candidate = field.value if field.value is not None else field.rejected_value
                if candidate is not None:
                    counts["candidate_fields"] += 1
                    counts["rejected_unsupported_fields"] += field.rejected_value is not None
                if field.value is not None:
                    counts["emitted_fields"] += 1
                    counts["unsupported_emitted_fields"] += (
                        field.method not in {"omr", "human"} and not field.evidence
                    )
            predicted_boxes = [(p.page_number, box) for p in result.pages for box in p.checkboxes]
            used = set()
            for box in record["checkboxes"]:
                matches = [
                    (iou(box["bbox"], predicted.bbox), i, predicted)
                    for i, (page, predicted) in enumerate(predicted_boxes)
                    if page == box["page_number"] and i not in used
                ]
                match = max(matches, key=lambda item: item[0]) if matches else None
                if match and match[0] >= 0.5:
                    _, index, predicted = match
                    used.add(index)
                    counts["checkbox_tp"] += 1
                    counts["checkbox_state_correct"] += predicted.state == box["state"]
                else:
                    counts["checkbox_fn"] += 1
                counts["checkbox_total"] += 1
            counts["checkbox_fp"] += len(predicted_boxes) - len(used)
            if source != "gold":
                reference = "\n".join(p["text"] for p in record["pages"])
                hypothesis = "\n".join(p.text for p in result.pages)
                ocr_counts.update(text_counts(reference, hypothesis))
            example = {
                "document_id": record["document_id"],
                "status": result.status,
                "review_reasons": result.review_reasons,
                "missing_fields": result.missing_fields,
            }
            if result.human_review_required:
                examples.append(example)
            (output / f"{record['document_id']}.json").write_text(
                result.model_dump_json(indent=2), encoding="utf-8"
            )
            checkpoint.append(example)
            (output / "progress.json").write_text(
                json.dumps(checkpoint, indent=2), encoding="utf-8"
            )
    report = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "split": split,
        "source": source,
        "extraction_method": settings.extraction_method,
        "ocr_engine": settings.ocr_engine,
        "preprocessing": settings.preprocessing,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "document_ids": [r["document_id"] for r in records],
        "documents": counts["documents"],
        "classification": {
            "accuracy": accuracy_score(expected_categories, actual_categories),
            "macro_f1": f1_score(
                expected_categories,
                actual_categories,
                labels=list(CATEGORIES),
                average="macro",
                zero_division=0,
            ),
            "labels": [*CATEGORIES, "unknown"],
            "confusion_matrix": confusion_matrix(
                expected_categories, actual_categories, labels=[*CATEGORIES, "unknown"]
            ).tolist(),
            "unknown_rate": actual_categories.count("unknown") / len(records),
        },
        "entities": precision_recall_f1(
            counts["entity_tp"], counts["entity_fp"], counts["entity_fn"]
        ),
        "fields": {
            "exact_match_accuracy_present": rate(counts["exact_fields"], counts["present_fields"]),
            "present_fields": counts["present_fields"],
            "correct_fields": counts["exact_fields"],
            "absent_fields": counts["absent_fields"],
            "false_fills": counts["false_fills"],
            "false_fill_rate": rate(counts["false_fills"], counts["absent_fields"]),
            "per_field": dict(per_field),
        },
        "checkboxes": {
            **precision_recall_f1(
                counts["checkbox_tp"], counts["checkbox_fp"], counts["checkbox_fn"]
            ),
            "state_accuracy_all_ground_truth": rate(
                counts["checkbox_state_correct"], counts["checkbox_total"]
            ),
            "ground_truth_boxes": counts["checkbox_total"],
        },
        "unsupported_fields": {
            "rejected_candidates": counts["rejected_unsupported_fields"],
            "candidates": counts["candidate_fields"],
            "rejected_candidate_rate": rate(
                counts["rejected_unsupported_fields"], counts["candidate_fields"]
            ),
            "emitted_unsupported_rate": rate(
                counts["unsupported_emitted_fields"], counts["emitted_fields"]
            ),
            "limitation": "Evidence checks do not detect every semantic role error; ground-truth precision and false fills remain necessary.",
        },
        "human_review_rate": counts["review_documents"] / len(records),
        "failed_documents": counts["failed_documents"],
        "errors": dict(errors),
        "review_reasons": dict(reasons),
        "latency": latency_summary(latencies),
        "failure_examples": examples[:20],
    }
    if ocr_counts:
        report["ocr"] = text_rates(ocr_counts)
    (output / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def benchmark_ocr(
    manifest,
    output,
    settings,
    split="validation",
    engines=("tesseract", "paddle"),
    profiles=("raw", "contrast", "adaptive", "deskew"),
    pages=30,
    allow_test=False,
):
    """Paired first-page scans; deterministic stratified selection and degradation."""
    import pymupdf

    from pharmadoc_ai.ingestion import words_to_text
    from pharmadoc_ai.ocr.paddle import PaddleEngine
    from pharmadoc_ai.ocr.tesseract import TesseractEngine

    records = select_records(json.loads(Path(manifest).read_text())["records"], split, allow_test)
    selected = []
    quotas = [round(pages * 0.6), round(pages * 0.2)]
    quotas.append(pages - sum(quotas))
    for category, quota in zip(CATEGORIES, quotas, strict=True):
        selected += [r for r in records if r["category"] == category][:quota]
    if not selected:
        raise ValueError("No benchmark records matched.")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    measurements = []
    degradations = ("clean", "noisy", "low_resolution", "rotated", "skewed", "blurry")
    with tempfile.TemporaryDirectory(prefix="pharmadoc-ocr-benchmark-") as temp:
        temp = Path(temp)
        for i, record in enumerate(selected):
            with (
                pymupdf.open(Path(manifest).parent / record["path"]) as doc,
                pymupdf.open() as single,
            ):
                single.insert_pdf(doc, from_page=0, to_page=0)
                single.save(temp / "single.pdf")
            create_scan(
                temp / "single.pdf",
                temp / "scan.pdf",
                degradations[i % len(degradations)],
                seed=42 + i,
            )
            with open_document(temp / "scan.pdf", settings) as doc:
                image = render_page(doc[0], settings)
            for engine_name in engines:
                engine = TesseractEngine() if engine_name == "tesseract" else PaddleEngine()
                for profile in profiles:
                    started = time.monotonic()
                    error = None
                    try:
                        configured = settings.model_copy(update={"preprocessing": profile})
                        words = engine.recognize(
                            image, configured, started + settings.document_timeout_seconds
                        )
                        text = words_to_text(words)
                    except DocumentError as exc:
                        error, text = exc.code, ""
                    measurements.append(
                        {
                            "document_id": record["document_id"],
                            "page_number": 1,
                            "degradation": degradations[i % len(degradations)],
                            "engine": engine_name,
                            "profile": profile,
                            "duration_seconds": time.monotonic() - started,
                            "error": error,
                            **text_counts(record["pages"][0]["text"], text),
                        }
                    )
                    (output / "measurements.json").write_text(
                        json.dumps(measurements, indent=2), encoding="utf-8"
                    )
    summaries = []
    for engine in engines:
        for profile in profiles:
            rows = [m for m in measurements if m["engine"] == engine and m["profile"] == profile]
            totals = Counter()
            for row in rows:
                totals.update(
                    {
                        k: row[k]
                        for k in (
                            "character_edits",
                            "reference_characters",
                            "word_edits",
                            "reference_words",
                        )
                    }
                )
            summaries.append(
                {
                    "engine": engine,
                    "profile": profile,
                    "pages": len(rows),
                    "failures": sum(r["error"] is not None for r in rows),
                    **text_rates(totals),
                    **latency_summary([r["duration_seconds"] for r in rows]),
                }
            )
    report = {
        "split": split,
        "seed": 42,
        "pages": len(selected),
        "render_dpi": settings.render_dpi,
        "worker_startup_included": True,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "summary": summaries,
        "measurements": measurements,
    }
    (output / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
