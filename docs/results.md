# Measured results

Generated from executed local evaluation summaries. All numbers refer to controlled synthetic documents. No real clinical, handwriting, or compliance claims are supported.

## Extraction

All prescription comparisons use matching IDs for the same source/split. The all-category digital rules run is reported separately. Timeouts remain in the denominator. Extraction scan runs use clean synthetic scans; the OCR robustness table separately uses six scan conditions. Scan wall time includes derivative preparation; per-document JSON records core pipeline time.

| Run | n | Field exact match | Entity F1 | False fills / absent | Review rate | Failed | Median s | p95 s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| test-llm-gold | 30 | 1.000 | 1.000 | 0/3 | 0.167 | 0 | 8.68 | 22.55 |
| test-llm-scan | 30 | 1.000 | 1.000 | 0/3 | 0.233 | 0 | 9.49 | 19.11 |
| test-ner-gold | 30 | 1.000 | 1.000 | 0/3 | 0.167 | 0 | 0.03 | 0.05 |
| test-ner-scan | 30 | 1.000 | 1.000 | 0/3 | 0.233 | 0 | 1.09 | 2.15 |
| test-rules-digital | 50 | 1.000 | 1.000 | 0/4 | 0.120 | 0 | 0.05 | 0.08 |
| test-rules-gold | 30 | 1.000 | 1.000 | 0/3 | 0.167 | 0 | 0.00 | 0.02 |
| test-rules-scan | 30 | 1.000 | 1.000 | 0/3 | 0.233 | 0 | 1.13 | 2.22 |

## OCR and preprocessing

The final test experiment compares the same 30 pages under six declared scan conditions. Worker startup is included. Validation timings were exploratory and may include competing development workloads; final runs execute serially.

| Run | Engine | Profile | Pages | CER | WER | Failures | Median s | p95 s |
|---|---|---|---:|---:|---:|---:|---:|---:|
| test-ocr | tesseract | raw | 30 | 0.107 | 0.152 | 0 | 1.45 | 2.12 |
| test-ocr | tesseract | adaptive | 30 | 0.126 | 0.249 | 0 | 1.22 | 1.73 |
| test-ocr | paddle | raw | 30 | 0.004 | 0.025 | 0 | 9.45 | 10.58 |
| test-ocr | paddle | adaptive | 30 | 0.042 | 0.168 | 0 | 9.55 | 10.57 |

## Classification and checkboxes

On 50 held-out synthetic digital documents, classification accuracy was 1.000 and macro F1 was 1.000. The rejection rate was 0.000.

Checkbox detection F1 was 1.000 on 30 ground-truth boxes. State accuracy over all ground-truth boxes was 1.000.

## Local LLM hardware observation

Two real synthetic-prescription calls used qwen3.5:4b at 4096 context tokens, with a separate 90-second smoke-check budget. Observed sampled total system GPU memory peaked at 4360 MiB. This includes desktop processes and is not a guaranteed instantaneous or model-only peak.

- Call 1: loaded before call = False; 26.33 s; model load 12.56 s; status complete.
- Call 2: loaded before call = True; 14.33 s; model load 0.02 s; status complete.

## Reproducibility and limitations

- Seeds, IDs, counts, configurations, source fingerprints, failure examples, per-field counts, unsupported-candidate diagnostics, and confusion matrices are in the linked JSON summaries.
- `reports/frozen_protocol.json` records the held-out configuration and code revision before test execution. The runner rejects source changes between final stages.
- Validation was used for development, including fixes to rotation order, annotation alignment, and dot-shaped checkbox handling. Those exploratory runs are not interchangeable with final frozen-source measurements.
- spaCy extraction uses trained entities plus deterministic medication grouping and a rule-based instruction-line assembler. The latter is disclosed rather than counted as a separately learned instruction model.
- Real quotes can still have incorrect semantic assignments. Evidence rejection rates do not replace gold-label extraction accuracy and missing-field false-fill measurements.
- Only one unseen layout family per category is tested; vocabulary and labels are narrow. Strong scores are expected on these controlled layouts and cannot be generalized to real clinical documents.
- Local automated tests and browser acceptance were executed. Remote GitHub Actions, Label Studio GUI, LangChain, and RAG are not claimed as executed.

See [methodology](evaluation.md), [error analysis](error_analysis.md), and [raw aggregate summaries](../reports/summaries).
