"""Local HTTP interface with bounded request bodies and one inference admission slot."""

import json
import tempfile
import threading
from pathlib import Path
from typing import Annotated, Literal

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

from pharmadoc_ai.config import ROOT, Settings
from pharmadoc_ai.errors import DocumentError
from pharmadoc_ai.ocr.tesseract import available
from pharmadoc_ai.pipeline import process_document
from pharmadoc_ai.schemas import DocumentResult, Model, ReviewCorrection
from pharmadoc_ai.validation import review_result


class BodyLimitMiddleware:
    """Spool bounded bodies before multipart parsing, including chunked uploads."""

    def __init__(self, app, limit: int):
        self.app, self.limit = app, limit

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope["method"] != "POST":
            return await self.app(scope, receive, send)
        headers = dict(scope.get("headers", []))
        try:
            length = int(headers.get(b"content-length", b"0"))
        except ValueError:
            return await JSONResponse({"error": "invalid_content_length"}, 400)(
                scope, receive, send
            )
        if length > self.limit:
            return await JSONResponse({"error": "upload_too_large"}, 413)(scope, receive, send)
        with tempfile.SpooledTemporaryFile(max_size=1024 * 1024) as spool:
            total = 0
            while True:
                message = await receive()
                if message["type"] == "http.disconnect":
                    return
                chunk = message.get("body", b"")
                total += len(chunk)
                if total > self.limit:
                    return await JSONResponse({"error": "upload_too_large"}, 413)(
                        scope, receive, send
                    )
                spool.write(chunk)
                if not message.get("more_body", False):
                    break
            spool.seek(0)

            async def replay():
                chunk = spool.read(65536)
                return {"type": "http.request", "body": chunk, "more_body": spool.tell() < total}

            await self.app(scope, replay, send)


class ReviewRequest(Model):
    original: DocumentResult
    corrections: dict[str, ReviewCorrection]


def create_app(settings: Settings | None = None, processor=None) -> FastAPI:
    settings = settings or Settings()
    processor = processor or process_document
    application = FastAPI(
        title="PharmaDoc AI",
        version="0.1.0",
        description="Local synthetic-document extraction; no clinical decision support.",
    )
    application.add_middleware(BodyLimitMiddleware, limit=settings.max_upload_bytes + 65536)
    admission = threading.Lock()
    application.state.admission = admission

    @application.exception_handler(DocumentError)
    async def document_error(request, exc):
        return JSONResponse({"error": exc.code, "message": str(exc)}, status_code=exc.status_code)

    @application.get("/health")
    def health():
        return {
            "status": "ok",
            "tesseract_available": available(settings),
            "paddle_interpreter_present": settings.paddle_python.is_file(),
            "classifier_available": settings.classifier_path.is_file(),
            "ner_available": (settings.ner_path / "config.cfg").is_file(),
            "ollama_host": settings.ollama_host,
            "ollama_model": settings.ollama_model,
            "ollama_status": "not_probed",
            "max_upload_bytes": settings.max_upload_bytes,
            "max_pages": settings.max_pages,
        }

    @application.get("/benchmarks")
    def benchmarks():
        path = ROOT / "reports/benchmarks.json"
        return (
            json.loads(path.read_text(encoding="utf-8"))
            if path.is_file()
            else {"status": "not_run"}
        )

    @application.post("/process", response_model=DocumentResult)
    async def process(
        file: Annotated[UploadFile, File()],
        ocr_engine: Literal["tesseract", "paddle"] = Query("tesseract"),
        extraction_method: Literal["rules", "ner", "llm"] = Query("rules"),
        preprocessing: Literal["raw", "contrast", "adaptive", "deskew"] = Query("raw"),
        force_ocr: bool = Query(False),
    ):
        if not admission.acquire(blocking=False):
            await file.close()
            raise HTTPException(
                429, "The local processor is busy. Try again after the current document."
            )
        try:
            # Only the suffix is considered; user filenames never become filesystem paths.
            suffix = Path((file.filename or "").replace("\\", "/")).suffix.lower()
            if suffix not in {".pdf", ".png", ".jpg", ".jpeg"}:
                raise DocumentError("Unsupported file extension.", "unsupported_format", 415)
            with tempfile.TemporaryDirectory(prefix="pharmadoc-upload-") as directory:
                path = Path(directory) / ("document" + suffix)
                total = 0
                with path.open("wb") as target:
                    while chunk := await file.read(65536):
                        total += len(chunk)
                        if total > settings.max_upload_bytes:
                            raise DocumentError(
                                "The upload exceeds the byte limit.", "upload_too_large", 413
                            )
                        target.write(chunk)
                configured = settings.model_copy(
                    update={
                        "ocr_engine": ocr_engine,
                        "extraction_method": extraction_method,
                        "preprocessing": preprocessing,
                        "force_ocr": force_ocr,
                    }
                )
                return await run_in_threadpool(processor, path, configured)
        finally:
            await file.close()
            admission.release()

    @application.post("/review")
    def review(body: ReviewRequest):
        try:
            return review_result(body.original, body.corrections)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

    return application


app = create_app()
