from pharmadoc_ai.classification import classify, keyword_classify, train_classifier
from pharmadoc_ai.extraction.ner import extract_ner, train_ner
from pharmadoc_ai.ingestion import inspect_document


def test_train_classifier_uses_only_training_partition(dataset, tmp_path):
    _, records = dataset
    path = tmp_path / "classifier.joblib"
    report = train_classifier(records, path)
    assert report["training_documents"] == 20
    label, method, margin = classify(
        "Medical Prescription\nMedication: Alpha\nPrescriber: Dr. Synthetic", path
    )
    assert label == "prescription" and method == "tfidf_linear_svc"
    assert margin is not None
    assert classify("zzzzzzzz", path)[0] == "unknown"
    assert keyword_classify("") == "unknown"


def test_statistical_ner_trains_and_reloads(dataset, tmp_path):
    root, records = dataset
    path = tmp_path / "ner"
    report = train_ner(records, path, epochs=1)
    assert report["training_pages"] > 0
    assert (path / "config.cfg").is_file()
    assert (path / "ner/model").is_file()
    pages = inspect_document(root / records[0]["path"])
    fields, _ = extract_ner(pages, "prescription", path)
    assert all(field.method == "ner" for field in fields.values())
