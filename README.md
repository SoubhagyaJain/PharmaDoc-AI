# PharmaDoc AI

Local document intelligence for **synthetic, printed English healthcare documents**. Upload a prescription, clinical/insurance form, or fictional regulatory record; inspect extracted fields alongside their page evidence; review uncertain values; export the original result and separate human corrections.

The project is an engineering and learning demonstration. It does not assess treatment safety, provide clinical advice, or claim HIPAA compliance. Synthetic benchmark scores do not establish performance on real clinical documents or handwriting.

## What works

- PDF, PNG, JPG/JPEG; digital, scanned, and mixed-page PDFs.
- Per-page text/OCR routing, image limits, reversible preprocessing, and checkbox recognition.
- Tesseract and isolated CPU PaddleOCR adapters with timeouts.
- Keyword baseline and trained TF-IDF/LinearSVC document classification.
- Contextual rules, genuinely trained spaCy NER, and evidence-grounded local Ollama extraction.
- Separate medication strength, administered dose, instructions, and frequency; multiple medications.
- FastAPI, Streamlit preview/evidence/review/export, and a reproducible evaluation CLI.
- A seeded 300-document generator with layout/identity/lineage grouping and machine-checkable labels.

See [architecture](docs/architecture.md), [evaluation methodology](docs/evaluation.md), [measured results](docs/results.md), [failure analysis](docs/error_analysis.md), and the [30-day learning roadmap](docs/roadmap.md).

## Windows setup

Use Python **3.11** and [uv](https://docs.astral.sh/uv/). From the repository root:

```powershell
uv sync --locked
uv run --locked pharmadoc-ai doctor
uv run --locked pharmadoc-ai generate --examples
uv run --locked pharmadoc-ai train data/generated/manifest.json --epochs 30
uv run --locked pytest -q
```

The main lockfile includes the application and development dependencies. Models and the generated corpus are ignored by Git; the commands above reproduce them locally. Training uses CPU and only the training partition. Do not load untrusted joblib model files.

Tesseract is a separate executable. The application discovers it on PATH or at `C:/Program Files/Tesseract-OCR/tesseract.exe`. Install the Windows build linked by [Tesseract's documentation](https://tesseract-ocr.github.io/tessdoc/Installation.html) if needed. `pytesseract` alone does not install it. Use `.env.example` as the template for a local `.env` when overriding settings.

The PDF package is imported as `pymupdf`; do not install the unrelated `fitz` package. Run `uv run` instead of relying on whichever Python is on PATH.

## Start the app

Use two PowerShell terminals:

```powershell
uv run --locked pharmadoc-ai serve
```

```powershell
uv run --locked pharmadoc-ai ui
```

Open **http://127.0.0.1:8501**. API documentation is at **http://127.0.0.1:8000/docs**. Both services bind to localhost. Select a provided synthetic example or upload your own synthetic file, then press **Process document**. Changing a field or switching tabs does not rerun inference.

Uploads are limited to 20 MiB, 10 pages, and 25 megapixels per rendered page. The default PDF render is 200 DPI, chosen to limit laptop memory and latency. The API accepts one processing job at a time and bounds inference calls. Configure limits through `PHARMADOC_` environment variables.

## CLI examples

```powershell
uv run --locked pharmadoc-ai inspect data/examples/mixed_pages.pdf
uv run --locked pharmadoc-ai process data/examples/prescription_scan.pdf --output output/result.json
uv run --locked pharmadoc-ai process data/examples/prescription_digital.pdf --method ner --output output/ner.json
uv run --locked pharmadoc-ai audit data/generated/manifest.json
uv run --locked pharmadoc-ai evaluate data/generated/manifest.json --source digital --output reports/runs/validation-rules
uv run --locked pharmadoc-ai evaluate data/generated/manifest.json --source gold --method ner --prescriptions-only --output reports/runs/validation-ner
uv run --locked pharmadoc-ai benchmark-ocr data/generated/manifest.json --output reports/runs/ocr-validation
```

Evaluation defaults to validation. `--split test --allow-test` is an explicit final-evaluation action, after configurations are frozen. Reports include failures, unsupported/rejected candidates, missing fields, review rate, and latency; no benchmark numbers are generated without executing the pipeline.

## Optional engines

### PaddleOCR

Keep its native dependencies in their own environment:

```powershell
uv venv .venv-paddle --python 3.11
uv pip sync --python .venv-paddle/Scripts/python.exe --require-hashes requirements/paddle.lock.txt
uv run --locked pharmadoc-ai warmup --engine paddle
uv run --locked pharmadoc-ai process data/examples/prescription_scan.pdf --engine paddle --output output/paddle.json
```

The worker explicitly uses `PP-OCRv5_mobile_det` and `en_PP-OCRv5_mobile_rec` on CPU. Initial inference downloads official model weights; subsequent inference is local. Worker startup is included in end-to-end latency. Paddle recognition spans are text-line boxes; Tesseract provides word boxes. Complex layouts and rotated checkbox labels remain limitations.

### Ollama

Install [Ollama](https://ollama.com/), start its local service, and ensure `qwen3.5:4b` is present. No paid API is used.

```powershell
ollama pull qwen3.5:4b
uv run --locked pharmadoc-ai warmup --engine llm
uv run --locked pharmadoc-ai process data/examples/prescription_digital.pdf --method llm --output output/llm.json
```

The model receives bounded OCR/native text, not page images. Requests use a Pydantic-generated JSON schema, temperature zero, a 4,096-token context, short output, and at most one schema-repair attempt within the document deadline. The model may remain loaded for two minutes to support warm inference. Missing evidence is rejected; ambiguous abbreviations are not expanded. Evidence checks cannot detect every semantic mistake, so accuracy and false-fill measurements remain necessary.

An unavailable selected extraction model produces an explicit error; it is never silently replaced by another extractor. Clinical/regulatory fields use rules even when the prescription extraction method is NER or LLM, as disclosed in the result's field methods.

## Review and privacy behavior

Field scores are labeled raw diagnostics, not calibrated probabilities. Required fields, weak OCR, ambiguous checkboxes, inconsistent instructions, unsupported evidence, and uncertain classification trigger review. The original machine result remains intact. Reviewed exports contain `reviewed_fields`, notes, and fresh deterministic validation results.

Temporary upload and worker files are cleaned on success and failure. Application event logs contain document IDs, stages, durations, and error codes. They do not include filenames, OCR text, or extracted medical fields. UI session data and downloaded exports remain local; stop the services and close the browser when finished. Use synthetic data only.

## Verification

```powershell
uv run --locked ruff check src tests scripts
uv run --locked ruff format --check src tests scripts
uv run --locked pytest -q
uv run --locked pytest -m integration -q
```

Core CI runs on Windows and Linux without downloading OCR/model weights. Local integration tests clearly skip when their external prerequisites are absent. The local validation record in [results](docs/results.md) distinguishes executed checks from optional or unexecuted paths.

RAG, LangChain, transformer fine-tuning, handwriting, multilingual documents, complex table extraction, cloud deployment, and enterprise compliance are deferred. Evaluation and failure analysis take priority over adding them.
