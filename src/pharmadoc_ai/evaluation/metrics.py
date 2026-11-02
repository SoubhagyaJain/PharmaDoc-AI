"""Transparent metric definitions with explicit empty-reference behavior."""

import re
import unicodedata
from collections import Counter

import numpy as np


def normalize(value) -> str:
    return " ".join(unicodedata.normalize("NFKC", str(value)).casefold().split())


def edit_distance(reference, hypothesis) -> int:
    if len(reference) < len(hypothesis):
        reference, hypothesis = hypothesis, reference
    previous = list(range(len(hypothesis) + 1))
    for i, a in enumerate(reference, 1):
        current = [i]
        for j, b in enumerate(hypothesis, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (a != b)))
        previous = current
    return previous[-1]


def text_counts(reference: str, hypothesis: str) -> dict:
    reference, hypothesis = normalize(reference), normalize(hypothesis)
    return {
        "character_edits": edit_distance(reference, hypothesis),
        "reference_characters": len(reference),
        "word_edits": edit_distance(reference.split(), hypothesis.split()),
        "reference_words": len(reference.split()),
    }


def rate(numerator: int, denominator: int):
    return numerator / denominator if denominator else (0.0 if numerator == 0 else None)


def text_rates(counts: dict) -> dict:
    return {
        **counts,
        "cer": rate(counts["character_edits"], counts["reference_characters"]),
        "wer": rate(counts["word_edits"], counts["reference_words"]),
    }


def precision_recall_f1(tp: int, fp: int, fn: int) -> dict:
    precision, recall = tp / max(1, tp + fp), tp / max(1, tp + fn)
    return {
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def entity_counts(reference: dict, prediction: dict) -> tuple[int, int, int]:
    # Entity matching ignores medication list order but preserves duplicates.
    # Field exact match separately checks grouping and list position.
    def entities(fields):
        return Counter(
            (re.sub(r"medications\.\d+\.", "medications.", key), normalize(value))
            for key, value in fields.items()
            if value is not None
        )

    gold, actual = entities(reference), entities(prediction)
    tp = sum((gold & actual).values())
    return tp, sum(actual.values()) - tp, sum(gold.values()) - tp


def latency_summary(values: list[float]) -> dict:
    return {
        "median_seconds": float(np.median(values)) if values else None,
        "p95_seconds": float(np.percentile(values, 95)) if values else None,
        "samples": len(values),
    }


def iou(a, b) -> float:
    intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(
        0, min(a[3], b[3]) - max(a[1], b[1])
    )
    area_a, area_b = (a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1])
    return intersection / max(1e-12, area_a + area_b - intersection)
