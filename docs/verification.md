# Executed verification

On 2 November 2026, local verification completed with:

- **70 pytest cases passed**, no failures or skips. This includes real Tesseract scanned/mixed-page integration checks, model training/reload, input limits, cleanup, evidence rejection, review preservation, and API failures. One upstream Starlette TestClient deprecation warning remains.
- Ruff lint and formatting checks passed for source, tests, and scripts.
- A fresh environment installed 87 application packages from the lockfile and processed a digital prescription successfully.
- Dependency checks found the main environment's 94 packages and Paddle environment's 64 packages compatible.
- Headless Chromium exercised scanned upload, API processing, evidence, human correction, and both JSON downloads after the final server refresh. The original machine fields and duration remained unchanged.
- The frozen held-out protocol completed 120 paired OCR engine/profile measurements and 230 document evaluations across methods/sources, with no processing failures. These repeated evaluations come from **50 unique logical test documents**, not 230 independent test records.
- Source fingerprints remained unchanged throughout the final run. Configurations, IDs, counts, failures, and timings are saved with the summaries.

See `reports/verification.json`, `reports/frozen_protocol.json`, `reports/benchmarks.json`, and `reports/summaries/` for inspectable evidence. Detailed per-document outputs remain in ignored `reports/runs/` and can be regenerated using the CLI.

Remote GitHub Actions is configured but was not executed here. Label Studio task export is tested; its GUI was not launched. LangChain and RAG remain deferred.
