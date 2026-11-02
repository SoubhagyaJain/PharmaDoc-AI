# Evaluation methodology

## Corpus and leakage prevention

The seeded generator creates 300 logical documents: 180 prescriptions, 60 clinical/insurance forms, and 60 fictional pharmaceutical records. Each category has six layout families: four train, one validation, one test. Counts are 200/50/50. Names are synthetic and sampled from identity pools partitioned before generation. Patient identities, template families, and logical lineages cannot cross splits. Scanned derivatives inherit the source document's identity and split.

Families vary in framing, indentation, font, landscape side-panel structure, metadata order, and explicit versus instruction-based medication fields. However, the vocabulary and writing style are intentionally narrow. One held-out family per category does not approximate the diversity of real documents. Easy synthetic classification is expected and must not be described as clinical generalization.

Labels are sampled before documents are rendered. The PDF layer locates their spans and boxes; labels are not copied from an extractor's predictions. Repeated medication values are aligned within their own medication blocks. Dataset audit checks actual hashes, label support, spans, box bounds, page counts, duplicate hashes, and group leakage. Audit checks may inspect test-label integrity but must not be used for model selection.

## Protocol

1. Generate using seed 42 and `per_family=10`.
2. Train TF-IDF/LinearSVC and spaCy NER from `train` only. Fit the vectorizer only on train.
3. Select classifier margin threshold and inspect preprocessing errors using validation.
4. Freeze extraction logic and configurations before test evaluation.
5. Run test with `--split test --allow-test`. Report failures without tuning to test performance.

When a benchmark exposes a reproducibility or implementation bug, fix it with a regression test and disclose any rerun. Do not describe a reused test set as still untouched.

## Metrics

- **CER/WER:** Unicode NFKC, casefold, collapse whitespace, retain punctuation and numerical values. Sum edit counts across pages and divide by total reference characters/words. Empty reference plus nonempty prediction is undefined (`null`), with insertions retained in counts. Errors can exceed 1.
- **Classification:** accuracy, macro F1 over the three intended categories, confusion matrix including `unknown`, and rejection rate. Rejections count against recall.

Use the all-category digital run to assess three-class classification. Prescription-only extraction subsets do not estimate performance on the two absent classes; the compact published comparison leaves their classification macro F1 unset.
- **Entity precision/recall/F1:** exact type/value matching after normalization, using multisets to prevent duplicate matching. Medication order is ignored for entity matching.
- **Field exact match:** compare each present ground-truth field at its field path, including medication position. Report missing-field false fills separately. Do not inflate accuracy by counting many absent keys as correct.
- **Checkbox detection:** greedy one-to-one matching within a page at IoU >= 0.5; report precision, recall, F1. State accuracy uses all ground-truth boxes as its denominator, so missed boxes do not disappear from the metric.
- **Unsupported candidates:** rejected non-null candidates / all non-null candidates. Also report emitted values lacking evidence and false fills of absent labels. A quote can be real but assigned to the wrong role; extraction precision still matters.
- **Review rate:** fraction of all selected documents requiring review, including failures. A low review rate alone is not success.

Ground truth preserves literal ambiguous abbreviations such as `OD`; it does not assign an interpreted dosing meaning. Copying the abbreviation with a review flag can match the raw label, while returning null is a safe abstention that reduces measured recall. Evaluate that policy behavior alongside raw exact match rather than calling every abstention an unsafe error.
- **Latency:** end-to-end median/p95, sample count, explicit failures, and OCR worker startup. LLM output includes model digest and load/generation timing diagnostics.

The extraction runner's reported wall time includes constructing a synthetic scan derivative for `source=scan`. Per-document `processing_duration_seconds` records the core pipeline alone. Gold/digital evaluation has no scan-generation step. The separate OCR benchmark times recognition after image preparation and includes worker startup. Keep these timing definitions distinct when comparing results.

Gold-text extraction isolates the extractor. Digital-PDF evaluation adds decoding/routing/OMR. Scanned evaluation adds rendering, OCR, and its errors. Compare prescription methods on the identical fixed prescription IDs. Clinical/regulatory extraction remains rules-based across method selections and is disclosed through field methods.

## OCR experiment

Choose 30 first pages with deterministic SHA-256 ordering and category quotas 18/6/6. Assign clean, noisy, low-resolution, rotated, skewed, and blurry conditions in a fixed cycle. Compare Tesseract and PaddleOCR on identical inputs for raw, contrast, adaptive, and deskew profiles. Report each engine/profile rather than assuming preprocessing improves recognition.

The API/default pipeline renders at 200 DPI. Scan derivatives are generated at 150 DPI before pipeline rendering; low-resolution derivatives are reduced further. Tesseract includes its orientation check; Paddle includes its document and text-line orientation models. Their complete configurations are compared, rather than attributing every difference to recognition alone.

Validation exploration can run alongside development workloads; its timings should be treated as exploratory. Final timing comparisons should run serially with heavy competing workloads stopped. Preserve raw measurements and the exact commands beside each aggregate report.

## Reproduce

```powershell
uv run --locked pharmadoc-ai evaluate data/generated/manifest.json --source gold --method rules --prescriptions-only --output reports/runs/rules-gold
uv run --locked pharmadoc-ai evaluate data/generated/manifest.json --source gold --method ner --prescriptions-only --output reports/runs/ner-gold
uv run --locked pharmadoc-ai evaluate data/generated/manifest.json --source gold --method llm --prescriptions-only --output reports/runs/llm-gold
uv run --locked pharmadoc-ai benchmark-ocr data/generated/manifest.json --output reports/runs/ocr-validation
```

For final reports, append `--split test --allow-test`. Keep per-document outputs in ignored `reports/runs`; publish aggregate JSON, methodology, failure examples by synthetic document ID, and honest limitations.
