import copy
from collections import Counter

from pharmadoc_ai.data.quality import check_dataset
from pharmadoc_ai.data.splits import audit_splits
from pharmadoc_ai.data.synthetic import create_scan, generate_dataset
from pharmadoc_ai.ingestion import inspect_document


def test_generated_data_and_splits(dataset):
    root, records = dataset
    assert len(records) == 30
    assert Counter(r["split"] for r in records) == {"train": 20, "validation": 5, "test": 5}
    assert len({r["template_family"] for r in records}) == 18
    assert check_dataset(records, root)["passed"]


def test_reproducible_generation(dataset, tmp_path):
    _, records = dataset
    repeated = generate_dataset(tmp_path, seed=991, per_family=1)
    assert [r["sha256"] for r in records] == [r["sha256"] for r in repeated]


def test_leakage_is_detected(dataset):
    _, records = dataset
    leaked = copy.deepcopy(records)
    leaked.append({**leaked[0], "split": "test"})
    errors = audit_splits(leaked)
    assert any("template_family" in error for error in errors)
    assert any("patient_record_id" in error for error in errors)
    assert any("lineage_id" in error for error in errors)


def test_scan_contains_no_native_text(dataset, tmp_path):
    root, records = dataset
    create_scan(root / records[0]["path"], tmp_path / "scan.pdf")
    pages = inspect_document(tmp_path / "scan.pdf")
    assert all(page.route == "ocr" and not page.words for page in pages)


def test_medication_spans_stay_with_their_medication(dataset):
    _, records = dataset
    for record in records:
        for entity in record["entities"]:
            text = record["pages"][entity["page_number"] - 1]["text"]
            assert text[entity["start"] : entity["end"]].strip()
        if len(record["medications"]) > 1:
            names = [e for e in record["entities"] if e["label"] == "MEDICATION"]
            assert len(names) == len(record["medications"])
