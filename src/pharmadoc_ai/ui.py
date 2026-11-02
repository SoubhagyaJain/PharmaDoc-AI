"""Small local reviewer. Inference runs only when the Process button is pressed."""

import hashlib
import json
import tempfile
from pathlib import Path

import httpx
import streamlit as st
from PIL import Image, ImageDraw

from pharmadoc_ai.config import ROOT, Settings
from pharmadoc_ai.errors import DocumentError
from pharmadoc_ai.ingestion import open_document, render_page
from pharmadoc_ai.schemas import DocumentResult, ReviewCorrection
from pharmadoc_ai.validation import all_fields, review_result


def preview(data: bytes, suffix: str, page_number: int, settings: Settings):
    with tempfile.TemporaryDirectory(prefix="pharmadoc-preview-") as directory:
        path = Path(directory) / ("document" + suffix)
        path.write_bytes(data)
        with open_document(path, settings) as doc:
            if page_number > len(doc):
                raise DocumentError("Preview page is out of range.")
            image = render_page(doc[page_number - 1], settings, dpi=96)
            return Image.fromarray(image), len(doc)


def main():
    st.set_page_config(page_title="PharmaDoc AI", page_icon="📄", layout="wide")
    settings = Settings()
    st.title("PharmaDoc AI")
    st.caption("Local document intelligence · Synthetic English documents · Evidence before export")
    with st.sidebar:
        st.header("Processing")
        engine = st.selectbox("OCR engine", ["tesseract", "paddle"])
        method = st.selectbox("Extraction method", ["rules", "ner", "llm"])
        profile = st.selectbox("Preprocessing", ["raw", "contrast", "adaptive", "deskew"])
        force = st.checkbox("Force OCR on every page")
        st.caption(
            "Native PDF text is reused where possible. Model scores are uncalibrated diagnostics."
        )
        try:
            health = httpx.get(settings.api_url + "/health", timeout=2, trust_env=False).json()
            st.success("Local API ready")
            if engine == "paddle" and not health.get("paddle_interpreter_present"):
                st.warning("PaddleOCR environment is not installed.")
            if method == "ner" and not health.get("ner_available"):
                st.warning("Train the statistical NER model before using this option.")
        except (httpx.HTTPError, ValueError):
            st.warning("Start the local API to process documents.")
    uploaded = st.file_uploader(
        "Upload a synthetic document",
        type=["pdf", "png", "jpg", "jpeg"],
        max_upload_size=max(1, (settings.max_upload_bytes + 1048575) // 1048576),
    )
    examples = sorted((ROOT / "data/examples").glob("*.pdf"))
    chosen = st.selectbox(
        "Or choose a synthetic example",
        [None, *examples],
        format_func=lambda p: "Choose an example" if p is None else p.stem.replace("_", " "),
    )
    data, suffix = None, ".pdf"
    if uploaded is not None:
        data, suffix = uploaded.getvalue(), Path(uploaded.name).suffix.lower()
    elif chosen is not None:
        data = chosen.read_bytes()
    if data is None:
        st.info("Upload a file or choose an example to begin.")
        return
    fingerprint = hashlib.sha256(data + f"{engine}/{method}/{profile}/{force}".encode()).hexdigest()
    if st.session_state.get("fingerprint") != fingerprint:
        st.session_state.fingerprint = fingerprint
        st.session_state.pop("result", None)
        st.session_state.corrections = {}
    if st.button("Process document", type="primary"):
        with st.spinner("Processing locally…"):
            try:
                response = httpx.post(
                    settings.api_url + "/process",
                    files={"file": ("document" + suffix, data)},
                    params={
                        "ocr_engine": engine,
                        "extraction_method": method,
                        "preprocessing": profile,
                        "force_ocr": force,
                    },
                    timeout=settings.document_timeout_seconds + 20,
                    trust_env=False,
                )
                response.raise_for_status()
                st.session_state.result = response.json()
                st.session_state.corrections = {}
            except httpx.HTTPStatusError as exc:
                try:
                    message = exc.response.json()
                except ValueError:
                    message = {"error": "http_error", "status": exc.response.status_code}
                st.error(f"Processing failed: {message}")
            except httpx.HTTPError:
                st.error("Could not reach the local processor or the request timed out.")
    result = (
        DocumentResult.model_validate(st.session_state.result)
        if "result" in st.session_state
        else None
    )
    left, right = st.columns([1, 1.15])
    with left:
        st.subheader("Document preview")
        try:
            first, count = preview(data, suffix, 1, settings)
            page_number = st.number_input(
                "Page", min_value=1, max_value=count, value=1, step=1, key=f"page_{fingerprint}"
            )
            image = (
                first if page_number == 1 else preview(data, suffix, int(page_number), settings)[0]
            )
            if result:
                fields = all_fields(result)
                selected = st.selectbox(
                    "Highlight field evidence",
                    [None, *fields],
                    format_func=lambda key: "No highlight" if key is None else key,
                    key=f"evidence_{fingerprint}",
                )
                if selected:
                    draw = ImageDraw.Draw(image)
                    for evidence in fields[selected].evidence:
                        if evidence.page_number == page_number and evidence.bbox:
                            x0, y0, x1, y1 = evidence.bbox
                            draw.rectangle(
                                (
                                    x0 * image.width,
                                    y0 * image.height,
                                    x1 * image.width,
                                    y1 * image.height,
                                ),
                                outline="#b84d22",
                                width=3,
                            )
            st.image(image, width="stretch")
        except DocumentError as exc:
            st.error(str(exc))
    with right:
        if result is None:
            st.info("Press Process document to extract fields.")
            return
        a, b, c = st.columns(3)
        a.metric("Category", result.document_category.replace("_", " "))
        b.metric("Time", f"{result.processing_duration_seconds:.2f} s")
        c.metric("Review", "Required" if result.human_review_required else "No flags")
        if result.review_reasons:
            st.warning("Review reasons: " + ", ".join(result.review_reasons))
        fields_tab, text_tab, export_tab, metrics_tab = st.tabs(
            ["Fields & review", "OCR & evidence", "JSON export", "Evaluation"]
        )
        with fields_tab:
            fields = all_fields(result)
            st.dataframe(
                [
                    {
                        "Field": key,
                        "Value": field.value,
                        "Status": field.status,
                        "Method": field.method,
                    }
                    for key, field in fields.items()
                ],
                hide_index=True,
                width="stretch",
            )
            if fields:
                selected = st.selectbox(
                    "Field to review", list(fields), key=f"review_{fingerprint}"
                )
                current = fields[selected]
                if selected.startswith("checkbox_"):
                    state = st.selectbox(
                        "Corrected selection",
                        ["checked", "unchecked", "uncertain"],
                        key=f"state_{fingerprint}_{selected}",
                    )
                    corrected = {"checked": True, "unchecked": False, "uncertain": None}[state]
                else:
                    missing = st.checkbox(
                        "Value is missing",
                        value=current.value is None,
                        key=f"missing_{fingerprint}_{selected}",
                    )
                    corrected = st.text_input(
                        "Corrected value",
                        value=str(current.value) if current.value is not None else "",
                        disabled=missing,
                        key=f"value_{fingerprint}_{selected}",
                    )
                    corrected = None if missing else corrected
                reason = st.text_input("Review note", key=f"note_{fingerprint}_{selected}")
                if st.button("Save correction"):
                    if not reason.strip():
                        st.error("Add a review note before saving.")
                    else:
                        st.session_state.corrections[selected] = ReviewCorrection(
                            value=corrected, reason=reason
                        ).model_dump()
                        st.success("Correction saved. The original extraction is preserved.")
                if st.session_state.corrections:
                    st.write("Saved corrections", st.session_state.corrections)
        with text_tab:
            for page in result.pages:
                with st.expander(f"Page {page.page_number} · {page.route}", expanded=True):
                    st.text(page.text or "No text extracted.")
                    if page.issues:
                        st.write("Page issues", page.issues)
            for key, field in all_fields(result).items():
                if field.evidence:
                    st.write(key)
                    for evidence in field.evidence:
                        st.caption(f"Page {evidence.page_number}")
                        st.text(evidence.text)
        with export_tab:
            st.download_button(
                "Download original JSON",
                result.model_dump_json(indent=2),
                file_name=f"{result.document_id}.json",
                mime="application/json",
            )
            corrections = {
                key: ReviewCorrection.model_validate(value)
                for key, value in st.session_state.corrections.items()
            }
            reviewed = review_result(result, corrections)
            st.download_button(
                "Download reviewed JSON",
                json.dumps(reviewed, indent=2),
                file_name=f"{result.document_id}_reviewed.json",
                mime="application/json",
            )
            if corrections:
                st.write("Review validation", reviewed["review_validation_errors"])
                st.write("Remaining missing fields", reviewed["review_missing_fields"])
            with st.expander("Original structured output"):
                st.json(result.model_dump(mode="json"))
        with metrics_tab:
            path = ROOT / "reports/benchmarks.json"
            if path.is_file():
                st.json(json.loads(path.read_text(encoding="utf-8")))
            else:
                st.info("No saved measurements yet. Run the evaluation CLI to create a report.")


if __name__ == "__main__":
    main()
