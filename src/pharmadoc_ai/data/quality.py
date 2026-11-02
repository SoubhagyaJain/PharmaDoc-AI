"""Inspectable dataset quality gates, including labels and duplicate hashes."""

import hashlib
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import pymupdf

from pharmadoc_ai.data.splits import audit_splits


def visual_duplicate_candidates(records: list[dict], root: Path, max_distance: int = 3) -> dict:
    """Perceptual warnings, never proof of a duplicate or a leakage-free corpus."""
    signatures = []
    for record in records:
        with pymupdf.open(root / record["path"]) as doc:
            pix = doc[0].get_pixmap(dpi=36, colorspace=pymupdf.csGRAY, alpha=False)
            gray = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width)
            transformed = cv2.dct(cv2.resize(gray, (32, 32)).astype(np.float32))[:8, :8]
            bits = (transformed > np.median(transformed[1:])).reshape(-1)
            signature = sum(int(bit) << i for i, bit in enumerate(bits))
            signatures.append((record, signature))
    pairs = []
    total = 0
    for i, (left, a) in enumerate(signatures):
        for right, b in signatures[i + 1 :]:
            distance = (a ^ b).bit_count()
            if left["lineage_id"] != right["lineage_id"] and distance <= max_distance:
                total += 1
                if len(pairs) < 20:
                    pairs.append(
                        {
                            "left": left["document_id"],
                            "right": right["document_id"],
                            "hamming_distance": distance,
                            "cross_partition": left["split"] != right["split"],
                        }
                    )
    return {
        "candidate_pairs": total,
        "examples": pairs,
        "interpretation": "Template similarity can create false positives; inspect candidates manually. No automatic leakage claim is made.",
    }


def check_dataset(records: list[dict], root: Path) -> dict:
    errors = audit_splits(records)
    hashes = defaultdict(set)
    for record in records:
        path = root / record["path"]
        if not path.is_file():
            errors.append(f"missing_file:{record['document_id']}")
            continue
        hashes[record["sha256"]].add(record["lineage_id"])
        if hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
            errors.append(f"hash_mismatch:{record['document_id']}")
        for entity in record["entities"]:
            text = record["pages"][entity["page_number"] - 1]["text"]
            if not 0 <= entity["start"] < entity["end"] <= len(text):
                errors.append(f"invalid_span:{record['document_id']}")
        for box in record["checkboxes"]:
            if not all(0 <= v <= 1 for v in box["bbox"]):
                errors.append(f"invalid_checkbox:{record['document_id']}")
        with pymupdf.open(path) as doc:
            if doc.page_count != len(record["pages"]):
                errors.append(f"page_count:{record['document_id']}")
        for key, value in record["fields"].items():
            if isinstance(value, str) and not any(value in p["text"] for p in record["pages"]):
                errors.append(f"unsupported_label:{record['document_id']}:{key}")
        for i, medication in enumerate(record["medications"]):
            for key, value in medication.items():
                if isinstance(value, str) and not any(value in p["text"] for p in record["pages"]):
                    errors.append(
                        f"unsupported_label:{record['document_id']}:medications.{i}.{key}"
                    )
    for lineages in hashes.values():
        if len(lineages) > 1:
            errors.append("duplicate_across_lineages")
    return {
        "documents": len(records),
        "split_counts": dict(Counter(r["split"] for r in records)),
        "category_counts": dict(Counter(r["category"] for r in records)),
        "errors": errors,
        "passed": not errors,
    }
