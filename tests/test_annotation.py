from pharmadoc_ai.data.annotation import export_label_studio


def test_export_is_training_only_and_offsets_match(dataset, tmp_path):
    _, records = dataset
    tasks = export_label_studio(records, tmp_path / "tasks.json")
    training_ids = {r["document_id"] for r in records if r["split"] == "train"}
    assert len(tasks) == 10
    for task in tasks:
        assert task["data"]["document_id"] in training_ids
        for result in task["annotations"][0]["result"]:
            value = result["value"]
            assert task["data"]["text"][value["start"] : value["end"]] == value["text"]
