from fastapi.testclient import TestClient

from pharmadoc_ai.api import create_app
from pharmadoc_ai.schemas import DocumentResult


def test_health_upload_and_cleanup(settings, dataset):
    root, records = dataset
    seen = []

    def processor(path, settings):
        seen.append(path)
        assert path.is_file()
        assert path.name == "document.pdf"
        return DocumentResult(document_id="synthetic-test")

    client = TestClient(create_app(settings, processor))
    assert client.get("/health").status_code == 200
    response = client.post(
        "/process", files={"file": ("../../unsafe.pdf", (root / records[0]["path"]).read_bytes())}
    )
    assert response.status_code == 200
    assert response.json()["document_id"] == "synthetic-test"
    assert not seen[0].exists()


def test_real_digital_api(settings, dataset):
    root, records = dataset
    record = next(r for r in records if r["category"] == "clinical_form")
    client = TestClient(create_app(settings))
    response = client.post(
        "/process", files={"file": ("form.pdf", (root / record["path"]).read_bytes())}
    )
    assert response.status_code == 200
    assert response.json()["document_category"] == "clinical_form"
    assert "patient_name" in response.json()["fields"]


def test_bad_files_limits_and_query_validation(settings):
    client = TestClient(create_app(settings.model_copy(update={"max_upload_bytes": 8})))
    assert client.post("/process", files={"file": ("x.txt", b"hi")}).status_code == 415
    assert client.post("/process", files={"file": ("x.pdf", b"%PDF-1.7extra")}).status_code == 413
    assert client.post("/process", files={"file": ("x.pdf", b"%PDF-1.7")}).status_code == 422
    assert (
        client.post(
            "/process?ocr_engine=unknown", files={"file": ("x.pdf", b"%PDF-1.7")}
        ).status_code
        == 422
    )
    assert (
        client.post(
            "/process",
            content=(b"x" * 70000 for _ in range(2)),
            headers={"Content-Type": "application/octet-stream"},
        ).status_code
        == 413
    )


def test_busy_and_failure_cleanup(settings):
    seen = []

    def fail(path, settings):
        from pharmadoc_ai.errors import EngineUnavailable

        seen.append(path)
        raise EngineUnavailable("test engine")

    app = create_app(settings, fail)
    client = TestClient(app)
    app.state.admission.acquire()
    try:
        assert client.post("/process", files={"file": ("x.pdf", b"%PDF-1.7")}).status_code == 429
    finally:
        app.state.admission.release()
    assert client.post("/process", files={"file": ("x.pdf", b"%PDF-1.7")}).status_code == 503
    assert not seen[0].exists()
    assert app.state.admission.acquire(blocking=False)
    app.state.admission.release()


def test_review_rejects_unknown_fields(settings):
    client = TestClient(create_app(settings))
    original = DocumentResult(document_id="synthetic").model_dump()
    response = client.post(
        "/review",
        json={"original": original, "corrections": {"unknown": {"value": "x", "reason": "test"}}},
    )
    assert response.status_code == 422
