"""Assign partitions before generation; derivatives inherit their lineage."""

from collections import defaultdict

SPLIT_BY_FAMILY = {0: "train", 1: "train", 2: "train", 3: "train", 4: "validation", 5: "test"}


def audit_splits(records: list[dict]) -> list[str]:
    errors = []
    for key in ("template_family", "patient_record_id", "lineage_id"):
        seen = defaultdict(set)
        for record in records:
            if record.get(key):
                seen[record[key]].add(record["split"])
        for value, splits in seen.items():
            if len(splits) > 1:
                errors.append(f"{key} crosses partitions: {value}")
    return errors
