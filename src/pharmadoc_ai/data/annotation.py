"""Export training-only text tasks for a small Label Studio inspection exercise."""

import json
from pathlib import Path


def export_label_studio(records: list[dict], output: Path, limit: int = 10) -> list[dict]:
    tasks = []
    for record in records:
        if record["split"] != "train" or record["category"] != "prescription":
            continue
        for page in record["pages"]:
            results = []
            for entity in record["entities"]:
                if entity["page_number"] != page["page_number"]:
                    continue
                start, end = entity["start"], entity["end"]
                results.append(
                    {
                        "from_name": "entities",
                        "to_name": "text",
                        "type": "labels",
                        "value": {
                            "start": start,
                            "end": end,
                            "text": page["text"][start:end],
                            "labels": [entity["label"]],
                        },
                    }
                )
            tasks.append(
                {
                    "data": {
                        "text": page["text"],
                        "document_id": record["document_id"],
                        "page_number": page["page_number"],
                    },
                    "annotations": [{"result": results}],
                }
            )
            if len(tasks) >= limit:
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_text(json.dumps(tasks, indent=2), encoding="utf-8")
                return tasks
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(tasks, indent=2), encoding="utf-8")
    return tasks
