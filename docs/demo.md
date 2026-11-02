# Two-minute project demonstration

Prepare the local API/UI, curated examples, trained artifacts, and saved benchmark report before recording. Use only synthetic documents. Keep a scanned result ready if the laptop is under load; disclose if displaying a previously measured result.

| Time | Show | Explain |
|---|---|---|
| 0-15 s | Title and upload area | "PharmaDoc AI turns synthetic healthcare documents into structured fields with page evidence and human review." |
| 15-40 s | Digital prescription, fields, highlighted evidence | "Selectable text bypasses OCR. Strength, dose, and frequency remain separate." |
| 40-75 s | Scanned prescription and OCR/evidence tab | "Scans use local OCR. Engine scores are diagnostics, not calibrated probabilities." |
| 75-100 s | Missing/ambiguous example, review note, correction | "Missing evidence stays null or flagged. Corrections never overwrite the original extraction." |
| 100-115 s | Saved measured comparison and JSON downloads | "All methods use the same held-out synthetic protocol; failures and latency remain visible." |
| 115-120 s | Limitations | "Printed English and controlled synthetic layouts only; handwriting and real-world clinical validation are future work." |

## Three technically honest résumé bullets

- Built a local Python healthcare-document pipeline with PDF text/OCR routing, Tesseract and PaddleOCR adapters, checkbox recognition, evidence-linked extraction, and a FastAPI/Streamlit reviewer.
- Created a reproducible 300-document synthetic corpus with layout, patient-identity, and lineage partitioning; compared contextual rules, trained spaCy NER, and local LLM extraction using documented quality and latency metrics.
- Implemented upload/resource limits, bounded inference, temporary-file cleanup, audit-preserving human corrections, and automated tests covering malformed input, evidence rejection, model interfaces, and API failure states.

Add performance numbers only by copying verified values and sample counts from `results.md`. Do not claim clinical accuracy, handwriting support, enterprise deployment, or compliance.
