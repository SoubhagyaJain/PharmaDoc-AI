import json
import time

import httpx

from pharmadoc_ai.extraction.llm import CitedValue, extract_llm, grounded
from pharmadoc_ai.schemas import DocumentResult, PageResult
from pharmadoc_ai.validation import validate_result


def source():
    return [
        PageResult(
            page_number=1,
            width=595,
            height=842,
            route="digital",
            text="Patient name: Synthetic One\nPrescriber: Dr. Synthetic Two\nDocument date: 2025-01-02",
        )
    ]


def test_grounding_rejects_invented_quote_and_wrong_page():
    for value in (
        CitedValue(value="Invented", page_number=1, quote="Invented"),
        CitedValue(value="Synthetic One", page_number=2, quote="Patient name: Synthetic One"),
    ):
        field = grounded(value, source())
        assert field.value is None
        assert field.status == "invalid"
        assert field.rejected_value == value.value


def test_grounded_quote_still_requires_correct_role():
    field = grounded(
        CitedValue(value="Dr. Synthetic Two", page_number=1, quote="Prescriber: Dr. Synthetic Two"),
        source(),
    )
    result = validate_result(
        DocumentResult(
            document_id="synthetic",
            document_category="prescription",
            pages=source(),
            fields={"patient_name": field},
        )
    )
    assert result.fields["patient_name"].value is None
    assert "unsupported_evidence" in result.review_reasons


def test_schema_retry_is_bounded_and_prompt_treats_input_as_data(settings, monkeypatch):
    calls = []
    null = {"value": None, "page_number": None, "quote": None}
    valid = {
        "patient_name": {
            "value": "Synthetic One",
            "page_number": 1,
            "quote": "Patient name: Synthetic One",
        },
        "document_date": null,
        "prescriber_name": null,
        "medications": [],
    }

    def transport(request):
        if request.url.path == "/api/tags":
            return httpx.Response(
                200,
                json={"models": [{"name": settings.ollama_model, "digest": "test-only-digest"}]},
            )
        payload = json.loads(request.content)
        calls.append(payload)
        content = "bad json" if len(calls) == 1 else json.dumps(valid)
        return httpx.Response(200, json={"message": {"content": content}})

    client = httpx.Client(transport=httpx.MockTransport(transport), base_url=settings.ollama_host)
    monkeypatch.setattr("pharmadoc_ai.extraction.llm.httpx.Client", lambda **kwargs: client)
    fields, meds, metadata = extract_llm(source(), "prescription", settings, time.monotonic() + 5)
    assert len(calls) == 2
    assert "untrusted DATA" in calls[0]["messages"][0]["content"]
    assert fields["patient_name"].value == "Synthetic One"
    assert metadata["llm_model_digest"] == "test-only-digest"
    assert not meds
