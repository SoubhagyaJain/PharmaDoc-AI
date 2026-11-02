# 30-day learning and implementation roadmap

Budget 4-5 hours/day. The application is implemented; use these checkpoints to rebuild your understanding progressively. Each day: 20-30 minutes of first principles, a tiny example, a focused challenge, tests/debugging, review, and interview practice. Explain the code before extending it.

## Week 1: data and ingestion

| Day | Learn / purpose | Implement or extend | Verify / debug | Measurable completion |
|---|---|---|---|---|
| 1 | Paths, bytes, exceptions: trustworthy input boundary | `ingestion.py`; follow `day_01.md` | Signature/extension mismatch, inclusive byte limit, eight-byte read | Supplied cases and bounded-read test pass |
| 2 | Native PDF text versus raster pages | Ingestion and CLI inspection | Real text, scanned, encrypted, malformed, and multipage PDFs | Explain every page's selected route |
| 3 | Pydantic schemas, nulls, coordinates | `schemas.py`, `config.py` | Round trips, rejected boxes, page numbering, preserved text offsets | Valid/invalid inputs behave predictably |
| 4 | Procedural generation and independent ground truth | `data/synthetic.py` | Compare sampled fields to source PDF text and rendered pages | Inspect 30 labeled examples |
| 5 | Grouped partitions and leakage | `data/splits.py` | Intentionally introduce a family, identity, and lineage crossing | Audit detects all three and clean data passes |
| 6 | Label quality, duplicates, annotation | `data/quality.py`, annotation export | Hashes, spans, missing values, repeated medication fields | 300-document audit passes; inspect ten training annotation tasks |
| 7 | Reproducibility and CI | CLI, lockfile, README, CI workflow | Fresh environment; regenerate seed; compare hashes | Foundation reproducible; milestone commit |

Interview pairs:

1. Why are extension/MIME checks insufficient? How do you bound a streamed upload?
2. Why can one PDF need both routes? What makes a text layer unusable?
3. Why distinguish null from an empty string? Which coordinate frame does an evidence box use?
4. Why shouldn't predictions become labels? What would handwriting fonts fail to simulate?
5. Why is a random file split unsafe? How do derivatives inherit a split?
6. How can duplicate text escape byte hashing? How do you detect incorrect span offsets?
7. What belongs in a lockfile? Which artifacts should a public repository exclude?

Acceptance: working parser and routing, repeatable synthetic corpus, disjoint groups, data-quality checks, tests, CI definition, setup documentation.

## Week 2: image processing and OCR/OMR

| Day | Learn / purpose | Implement or extend | Verify / debug | Measurable completion |
|---|---|---|---|---|
| 8 | Pixel arrays and contrast: preprocessing changes evidence | `preprocessing.py` | RGB/BGR, data types, resize aspect ratio, unchanged originals | Four declared profiles work on blank and text images |
| 9 | Skew, rotation, inverse transforms | Preprocessing and region proposals | Known-angle images and restored boxes | Whole-page boxes restore within one-pixel-scale tolerance |
| 10 | OCR boxes, scores, segmentation, reading order | Tesseract adapter and OCR contract | Executable path, segmentation mode, blank pages, timeout | A real scanned prescription extracts correctly |
| 11 | Native dependency isolation and adapters | Paddle worker/adapter and lockfile | Import shadowing, model downloads, worker exit/timeout | Same five source pages run through both engines |
| 12 | Edit distance and paired experiments | Evaluation metrics/runner | Hand-calculated CER/WER, empty references, unchanged comparison inputs | Saved raw/profile measurements with failures and latency |
| 13 | Checkbox detection versus mark recognition | `omr.py` | Borders, dots, ticks, fills, label association, distractors | Detection/state metrics reported separately |
| 14 | Multipage integration and failure reporting | Pipeline integration fixtures | Mixed pages, low resolution, rotation, missing engine | Every page has a visible route and outcome; milestone commit |

Interview pairs:

8. Why can thresholding reduce accuracy? When would resizing destroy small text?
9. Why keep inverse transforms? How is skew different from 90-degree orientation?
10. What does an OCR confidence score mean? How would you reconstruct reading order?
11. Why isolate native frameworks? What time belongs in end-to-end latency?
12. Why aggregate edit counts rather than average page CER? Can WER exceed one?
13. Why remove a checkbox border before measuring ink? How would you handle a dot?
14. How should one failed page affect a document? When should native text bypass OCR?

Acceptance: both engines execute locally, preprocessing comparison exists, checkbox states include ambiguity, coordinates remain grounded, and real-engine checks are recorded.

## Week 3: classification and extraction

| Day | Learn / purpose | Implement or extend | Verify / debug | Measurable completion |
|---|---|---|---|---|
| 15 | TF-IDF and linear classification | `classification.py` | Train-only vectorizer; validation confusion matrix; unknown input | Saved model and validation-selected margin threshold |
| 16 | Contextual pattern matching | `extraction/rules.py` | Metadata roles, multiple medications, absent fields | Every non-null field has source evidence |
| 17 | Strength, dose, frequency, business consistency | Rules and validation | Units, decimals, abbreviations, conflicting instructions | Missing values remain missing; uncertainty is flagged |
| 18 | Statistical NER and training spans | `extraction/ner.py` | Token alignment, duplicate values, train-only examples | Genuine trained model saves and reloads |
| 19 | Controlled comparison and error attribution | Evaluation runner | Gold text versus OCR; same field/document sets | Per-field rules/NER comparison and examples |
| 20 | Abstention, review reasons, evidence roles | `validation.py` | Real quote assigned to wrong field; missing required evidence | Unsupported candidates rejected and counted |
| 21 | Orchestration and profiling | `pipeline.py`, CLI | Fixed mixed-category batch, stage timings, partial outcomes | End-to-end JSON pipeline; milestone commit |

Interview pairs:

15. Why fit TF-IDF only on train? Why is an SVM margin not a probability?
16. When do rules outperform a language model? How do you avoid matching a prescriber's name as the patient?
17. How does strength differ from dose? Why shouldn't a system expand an ambiguous abbreviation?
18. What makes EntityRuler different from trained NER? How can repeated values corrupt supervision?
19. Why evaluate with gold text? How would you separate OCR from extraction failures?
20. Why isn't a real quote sufficient evidence? How could review rate be misleading?
21. Where would you profile first? How does sequential page processing bound memory?

Acceptance: trained classifier and NER, rule extraction for all categories, grounded medication groups, review/validation behavior, and quantitative comparisons.

## Week 4: LLM and app

| Day | Learn / purpose | Implement or extend | Verify / debug | Measurable completion |
|---|---|---|---|---|
| 22 | Structured generation and context budgets | `extraction/llm.py` | Local model, nullable schema, hardware usage, cold/warm behavior | Five cases produce valid outputs or explicit failures |
| 23 | Grounding, prompt injection, bounded retries | LLM and validation | Fabricated quote, wrong page/role, invalid JSON, timeout | One repair attempt maximum; unsupported values rejected |
| 24 | Accuracy/latency/complexity trade-offs | Comparison runner | Same prescription IDs and fields; failures in denominator | Three-method measured report; optional 45-minute LangChain exercise |
| 25 | HTTP uploads and service boundaries | `api.py` | Invalid file, limits, busy state, unavailable engine | API success and failure tests pass |
| 26 | Streamlit state and reviewer workflow | `ui.py` | Preview, evidence, corrections, reruns, original/reviewed exports | Complete browser workflow without repeated inference |
| 27 | Failure reliability and privacy | API, logging, cleanup | Path-like names, streamed overflow, process failure | Temporary files cleaned; captured logs contain no document contents |
| 28 | Regression and reproducible release | Lockfiles, docs, integration tests | Fresh install, unit checks, real engines, browser-to-API path | Validation choices frozen; milestone commit |

Interview pairs:

22. What does structured output guarantee? What consumes VRAM besides model weights?
23. Why treat document text as untrusted data? When should a retry stop?
24. When would you decline to use an LLM? How do timeouts affect comparison metrics?
25. Why validate request size before multipart parsing? How do you prevent concurrent model overload?
26. Why preserve machine predictions after review? How do you stop UI reruns from repeating inference?
27. What information belongs in logs? What must cleanup do after a decoder failure?
28. What makes a smoke test different from an evaluation? What remains untested after unit tests pass?

Acceptance: working local API/UI, bounded LLM integration, evidence/review/export, meaningful failure tests, reproducible setup, and an honest validation record.

## Days 29-30: portfolio

| Day | Learn / purpose | Implement or extend | Verify / debug | Measurable completion |
|---|---|---|---|---|
| 29 | Held-out evaluation and reporting | Results and failure analysis | Explicit test opt-in; all failures counted; no test-based tuning | Requested metric groups have measured results or explicit unavailable reasons |
| 30 | Technical storytelling | README, screenshots, demo, résumé bullets | Rehearse setup and two-minute demo; trace every claim to evidence | Reviewable portfolio and technically honest claims |

29. Why must the test set remain separate? What must you disclose after a test-driven bug fix?
30. How would you improve enterprise reliability? Which synthetic results should you refuse to generalize?

Defer RAG, LangChain, additional models, and visual polish before sacrificing the working pipeline, tests, evaluation, or failure analysis. The current repository includes annotation export but does not claim an executed Label Studio GUI session; optional extension exercises remain optional.
