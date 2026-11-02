"""Measure two real local calls and observed system GPU/RAM use; no invented peaks."""

import json
import subprocess
import threading
import time
from datetime import UTC, datetime

import httpx
import psutil

from pharmadoc_ai.config import ROOT, Settings
from pharmadoc_ai.pipeline import process_document


def main():
    settings = Settings(extraction_method="llm", llm_timeout_seconds=90)
    samples, calls = [], []
    stop = threading.Event()

    def sample():
        while not stop.is_set():
            try:
                completed = subprocess.run(
                    ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                    capture_output=True,
                    text=True,
                    timeout=3,
                    check=True,
                )
                gpu_mib = int(completed.stdout.strip().splitlines()[0])
            except (OSError, ValueError, subprocess.SubprocessError):
                gpu_mib = None
            samples.append(
                {
                    "monotonic_seconds": time.monotonic(),
                    "system_gpu_memory_mib": gpu_mib,
                    "system_available_ram_mib": psutil.virtual_memory().available / 1048576,
                }
            )
            stop.wait(1)

    thread = threading.Thread(target=sample, daemon=True)
    thread.start()
    try:
        for index in range(2):
            active = httpx.get(settings.ollama_host + "/api/ps", timeout=5, trust_env=False).json()
            loaded_before = any(
                m.get("name") == settings.ollama_model for m in active.get("models", [])
            )
            result = process_document(ROOT / "data/examples/prescription_digital.pdf", settings)
            calls.append(
                {
                    "call": index + 1,
                    "model_loaded_before_call": loaded_before,
                    "seconds": result.processing_duration_seconds,
                    "status": result.status,
                    "review_reasons": result.review_reasons,
                    "model_digest": result.metadata.get("llm_model_digest"),
                    "load_seconds": result.metadata.get("llm_load_seconds"),
                    "generation_seconds": result.metadata.get("llm_generation_seconds"),
                    "generated_tokens": result.metadata.get("llm_generated_tokens"),
                }
            )
    finally:
        stop.set()
        thread.join(timeout=5)
    report = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "model": settings.ollama_model,
        "context_tokens": settings.llm_context_tokens,
        "call_timeout_seconds": 90,
        "calls": calls,
        "samples": samples,
        "observed_peak_system_gpu_memory_mib": max(
            (s["system_gpu_memory_mib"] for s in samples if s["system_gpu_memory_mib"] is not None),
            default=None,
        ),
        "minimum_observed_available_system_ram_mib": min(
            (s["system_available_ram_mib"] for s in samples), default=None
        ),
        "interpretation": "Sampled total system GPU/RAM usage during two calls, including other desktop processes; not model-only allocation or a guaranteed instantaneous peak.",
    }
    output = ROOT / "reports/llm_runtime.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "samples"}, indent=2))


if __name__ == "__main__":
    main()
