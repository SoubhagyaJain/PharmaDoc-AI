"""Run the frozen held-out protocol serially; no selection from test performance."""

import hashlib
import json
import platform
import subprocess
import time
from datetime import UTC, datetime

from pharmadoc_ai.config import ROOT, Settings
from pharmadoc_ai.evaluation.run import benchmark_ocr, evaluate, source_fingerprint


def write_progress(stage, **extra):
    progress = {"stage": stage, "updated_at_utc": datetime.now(UTC).isoformat(), **extra}
    (ROOT / "reports/final_progress.json").write_text(
        json.dumps(progress, indent=2), encoding="utf-8"
    )
    print(stage, flush=True)


def main():
    # The validation Paddle experiment may still be completing. This runner
    # explicitly waits so final latency measurements have no competing benchmark.
    expected_validation = ROOT / "reports/runs/validation-ocr-paddle/summary.json"
    while not expected_validation.is_file():
        write_progress("waiting_for_validation_ocr")
        time.sleep(15)
    settings = Settings()
    manifest = ROOT / "data/generated/manifest.json"
    frozen = {
        "frozen_at_utc": datetime.now(UTC).isoformat(),
        "source_fingerprint": source_fingerprint(),
        "git_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "python": platform.python_version(),
        "protocol": "30 paired held-out OCR pages; raw/adaptive chosen before test. Prescription extraction: same 30 test IDs for all three methods on gold text and clean scanned documents. Rules digital: all 50 test documents.",
        "selection_basis": "Adaptive selected from the declared validation preprocessing experiment, never from test performance. Validation was exploratory during development; final runs are serial.",
        "configuration": settings.model_dump(mode="json"),
    }
    (ROOT / "reports/frozen_protocol.json").write_text(
        json.dumps(frozen, indent=2), encoding="utf-8"
    )
    write_progress("test_ocr_30_pages_two_engines_two_profiles")
    benchmark_ocr(
        manifest,
        ROOT / "reports/runs/test-ocr",
        settings,
        split="test",
        engines=("tesseract", "paddle"),
        profiles=("raw", "adaptive"),
        pages=30,
        allow_test=True,
    )
    cases = [
        ("rules", "digital", False),
        ("rules", "gold", True),
        ("ner", "gold", True),
        ("llm", "gold", True),
        ("rules", "scan", True),
        ("ner", "scan", True),
        ("llm", "scan", True),
    ]
    for method, source, prescriptions_only in cases:
        if source_fingerprint() != frozen["source_fingerprint"]:
            raise RuntimeError("Source changed after freeze; do not mix evaluation versions.")
        stage = f"test-{method}-{source}"
        write_progress(stage)
        summary = evaluate(
            manifest,
            ROOT / "reports/runs" / stage,
            settings.model_copy(update={"extraction_method": method}),
            split="test",
            source=source,
            prescriptions_only=prescriptions_only,
            allow_test=True,
        )
        print(
            json.dumps(
                {
                    "stage": stage,
                    "documents": summary["documents"],
                    "exact_match": summary["fields"]["exact_match_accuracy_present"],
                    "failed": summary["failed_documents"],
                }
            ),
            flush=True,
        )
    write_progress("complete", source_fingerprint=frozen["source_fingerprint"])


if __name__ == "__main__":
    main()
