import pytest

from pharmadoc_ai.evaluation.metrics import (
    edit_distance,
    entity_counts,
    iou,
    text_counts,
    text_rates,
)
from pharmadoc_ai.evaluation.run import select_records


def test_hand_calculated_text_errors():
    counts = text_counts("abc def", "abc deg")
    rates = text_rates(counts)
    assert rates["character_edits"] == 1
    assert rates["cer"] == pytest.approx(1 / 7)
    assert rates["wer"] == 0.5
    assert edit_distance("kitten", "sitting") == 3


def test_empty_reference_and_unicode_normalization():
    assert text_rates(text_counts("", ""))["cer"] == 0
    assert text_rates(text_counts("", "inserted"))["cer"] is None
    assert text_rates(text_counts("ＭＧ  5", "mg 5"))["cer"] == 0


def test_duplicate_entities_are_not_double_matched():
    reference = {"medications.0.strength": "5 mg"}
    prediction = {"medications.0.strength": "5 mg", "medications.1.strength": "5 mg"}
    assert entity_counts(reference, prediction) == (1, 1, 0)
    assert iou((0, 0, 1, 1), (0, 0, 0.5, 1)) == 0.5


def test_test_partition_requires_opt_in(dataset):
    _, records = dataset
    with pytest.raises(ValueError, match="sealed"):
        select_records(records, "test")
    assert len(select_records(records, "test", allow_test=True)) == 5
