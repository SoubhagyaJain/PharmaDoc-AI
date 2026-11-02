"""Publish existing measured summaries to compact JSON and readable Markdown."""

import json
from datetime import UTC, datetime

from pharmadoc_ai.config import ROOT


def display(value, digits=3):
    return "—" if value is None else f"{value:.{digits}f}"


def main():
    summaries = {}
    for path in sorted((ROOT / "reports/runs").glob("*/summary.json")):
        measured = json.loads(path.read_text(encoding="utf-8"))
        summaries[path.parent.name] = measured
    if not summaries:
        raise RuntimeError("No measured summaries exist. Run evaluation before publishing.")
    reports = ROOT / "reports"
    (reports / "summaries").mkdir(parents=True, exist_ok=True)
    compact = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "scope": "Synthetic English documents only; no real-world clinical accuracy claim.",
        "extraction": [],
        "ocr": [],
        "runtime": None,
    }
    for name, measured in summaries.items():
        if measured.get("split") == "validation":
            measured["publication_note"] = (
                "Exploratory development run, not a frozen-source performance estimate. OCR development included a reading-order fix; use the final test rows for the released implementation."
            )
        (reports / "summaries" / f"{name}.json").write_text(
            json.dumps(measured, indent=2), encoding="utf-8"
        )
        if "extraction_method" in measured and measured.get("split") == "test":
            compact["extraction"].append(
                {
                    "run": name,
                    "split": measured["split"],
                    "source": measured["source"],
                    "method": measured["extraction_method"],
                    "documents": measured["documents"],
                    "field_exact_match": measured["fields"]["exact_match_accuracy_present"],
                    "entity_f1": measured["entities"]["f1"],
                    "classification_macro_f1": measured["classification"]["macro_f1"]
                    if all(
                        sum(row) > 0 for row in measured["classification"]["confusion_matrix"][:3]
                    )
                    else None,
                    "false_fills": measured["fields"]["false_fills"],
                    "absent_fields": measured["fields"]["absent_fields"],
                    "rejected_candidates": measured["unsupported_fields"]["rejected_candidates"],
                    "review_rate": measured["human_review_rate"],
                    "failed_documents": measured["failed_documents"],
                    "median_seconds": measured["latency"]["median_seconds"],
                    "p95_seconds": measured["latency"]["p95_seconds"],
                    "source_changed_during_run": measured.get("source_changed_during_run"),
                }
            )
        elif "measurements" in measured and measured.get("split") == "test":
            compact["ocr"].extend(
                {"run": name, "split": measured["split"], **row} for row in measured["summary"]
            )
    runtime = reports / "llm_runtime.json"
    if runtime.is_file():
        compact["runtime"] = json.loads(runtime.read_text(encoding="utf-8"))
        compact["runtime"].pop("samples", None)
    (reports / "benchmarks.json").write_text(json.dumps(compact, indent=2), encoding="utf-8")
    text = [
        "# Measured results",
        "",
        "Generated from executed local evaluation summaries. All numbers refer to controlled synthetic documents. No real clinical, handwriting, or compliance claims are supported.",
        "",
        "## Extraction",
        "",
        "All prescription comparisons use matching IDs for the same source/split. The all-category digital rules run is reported separately. Timeouts remain in the denominator. Extraction scan runs use clean synthetic scans; the OCR robustness table separately uses six scan conditions. Scan wall time includes derivative preparation; per-document JSON records core pipeline time.",
        "",
        "| Run | n | Field exact match | Entity F1 | False fills / absent | Review rate | Failed | Median s | p95 s |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in compact["extraction"]:
        text.append(
            f"| {row['run']} | {row['documents']} | {display(row['field_exact_match'])} | {display(row['entity_f1'])} | {row['false_fills']}/{row['absent_fields']} | {display(row['review_rate'])} | {row['failed_documents']} | {display(row['median_seconds'], 2)} | {display(row['p95_seconds'], 2)} |"
        )
    text += [
        "",
        "## OCR and preprocessing",
        "",
        "The final test experiment compares the same 30 pages under six declared scan conditions. Worker startup is included. Validation timings were exploratory and may include competing development workloads; final runs execute serially.",
        "",
        "| Run | Engine | Profile | Pages | CER | WER | Failures | Median s | p95 s |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in compact["ocr"]:
        text.append(
            f"| {row['run']} | {row['engine']} | {row['profile']} | {row['pages']} | {display(row['cer'])} | {display(row['wer'])} | {row['failures']} | {display(row['median_seconds'], 2)} | {display(row['p95_seconds'], 2)} |"
        )
    if "test-rules-digital" in summaries:
        full = summaries["test-rules-digital"]
        boxes = full["checkboxes"]
        text += [
            "",
            "## Classification and checkboxes",
            "",
            f"On {full['documents']} held-out synthetic digital documents, classification accuracy was {display(full['classification']['accuracy'])} and macro F1 was {display(full['classification']['macro_f1'])}. The rejection rate was {display(full['classification']['unknown_rate'])}.",
            "",
            f"Checkbox detection F1 was {display(boxes['f1'])} on {boxes['ground_truth_boxes']} ground-truth boxes. State accuracy over all ground-truth boxes was {display(boxes['state_accuracy_all_ground_truth'])}.",
        ]
    if compact["runtime"]:
        runtime = compact["runtime"]
        text += [
            "",
            "## Local LLM hardware observation",
            "",
            f"Two real synthetic-prescription calls used {runtime['model']} at {runtime['context_tokens']} context tokens, with a separate 90-second smoke-check budget. Observed sampled total system GPU memory peaked at {runtime['observed_peak_system_gpu_memory_mib']} MiB. This includes desktop processes and is not a guaranteed instantaneous or model-only peak.",
            "",
        ]
        for call in runtime["calls"]:
            text.append(
                f"- Call {call['call']}: loaded before call = {call['model_loaded_before_call']}; {call['seconds']:.2f} s; model load {call['load_seconds']:.2f} s; status {call['status']}."
            )
    text += [
        "",
        "## Reproducibility and limitations",
        "",
        "- Seeds, IDs, counts, configurations, source fingerprints, failure examples, per-field counts, unsupported-candidate diagnostics, and confusion matrices are in the linked JSON summaries.",
        "- `reports/frozen_protocol.json` records the held-out configuration and code revision before test execution. The runner rejects source changes between final stages.",
        "- Validation was used for development, including fixes to rotation order, annotation alignment, and dot-shaped checkbox handling. Those exploratory runs are not interchangeable with final frozen-source measurements.",
        "- spaCy extraction uses trained entities plus deterministic medication grouping and a rule-based instruction-line assembler. The latter is disclosed rather than counted as a separately learned instruction model.",
        "- Real quotes can still have incorrect semantic assignments. Evidence rejection rates do not replace gold-label extraction accuracy and missing-field false-fill measurements.",
        "- Only one unseen layout family per category is tested; vocabulary and labels are narrow. Strong scores are expected on these controlled layouts and cannot be generalized to real clinical documents.",
        "- Local automated tests and browser acceptance were executed. Remote GitHub Actions, Label Studio GUI, LangChain, and RAG are not claimed as executed.",
        "",
        "See [methodology](evaluation.md), [error analysis](error_analysis.md), and [raw aggregate summaries](../reports/summaries).",
        "",
    ]
    (ROOT / "docs/results.md").write_text("\n".join(text), encoding="utf-8")
    print(
        f"Published {len(compact['extraction'])} extraction and {len(compact['ocr'])} OCR summary rows from measured files."
    )


if __name__ == "__main__":
    main()
