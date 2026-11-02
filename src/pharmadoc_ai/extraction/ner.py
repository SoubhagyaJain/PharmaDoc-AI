"""A real statistical spaCy NER baseline, with deterministic medication assembly."""

import random
from functools import lru_cache
from pathlib import Path

import spacy
from spacy.training import Example
from spacy.util import fix_random_seed, minibatch

from pharmadoc_ai.errors import EngineUnavailable
from pharmadoc_ai.extraction.rules import extract_rules, sourced_field
from pharmadoc_ai.schemas import FieldResult, MedicationResult


@lru_cache(maxsize=2)
def _load(path: str, modified: int):
    return spacy.load(path)


def train_ner(records: list[dict], output: Path, epochs: int = 25, seed: int = 42) -> dict:
    fix_random_seed(seed)
    rng = random.Random(seed)
    nlp = spacy.blank("en")
    ner = nlp.add_pipe("ner")
    training = []
    for record in records:
        if record["split"] != "train" or record["category"] != "prescription":
            continue
        for page in record["pages"]:
            spans = [
                (e["start"], e["end"], e["label"])
                for e in record["entities"]
                if e["page_number"] == page["page_number"]
            ]
            for _, _, label in spans:
                ner.add_label(label)
            training.append((page["text"], {"entities": spans}))
    if not training:
        raise ValueError("No training prescriptions were provided.")
    examples = [Example.from_dict(nlp.make_doc(text), labels) for text, labels in training]
    optimizer = nlp.initialize(lambda: examples)
    history = []
    for _ in range(epochs):
        rng.shuffle(examples)
        losses = {}
        for batch in minibatch(examples, size=8):
            nlp.update(batch, sgd=optimizer, drop=0.15, losses=losses)
        history.append(float(losses.get("ner", 0)))
    output.parent.mkdir(parents=True, exist_ok=True)
    nlp.to_disk(output)
    _load.cache_clear()
    return {
        "training_pages": len(training),
        "epochs": epochs,
        "seed": seed,
        "training_loss": history,
    }


def extract_ner(pages, category, model_path: Path):
    if category != "prescription":
        return extract_rules(pages, category)
    if not (model_path / "config.cfg").is_file():
        raise EngineUnavailable("spaCy NER")
    nlp = _load(str(model_path), (model_path / "config.cfg").stat().st_mtime_ns)
    fields = {
        name: FieldResult(method="ner")
        for name in ("patient_name", "document_date", "prescriber_name")
    }
    mapping = {"PATIENT": "patient_name", "DATE": "document_date", "PRESCRIBER": "prescriber_name"}
    med_mapping = {"STRENGTH": "strength", "DOSE": "administered_dose", "FREQUENCY": "frequency"}
    medications = []
    candidates = {name: [] for name in fields}
    for page in pages:
        parsed = nlp(page.text)
        current = None
        for entity in parsed.ents:
            field = sourced_field(entity.text, pages, "ner", page.page_number)
            if entity.label_ in mapping:
                candidates[mapping[entity.label_]].append(field)
            elif entity.label_ == "MEDICATION":
                current = MedicationResult(name=field)
                medications.append(current)
            elif current is not None and entity.label_ in med_mapping:
                key = med_mapping[entity.label_]
                if getattr(current, key).value is None:
                    setattr(current, key, field)
        # Instructions are a full line, containing overlapping dose/frequency entities.
        # Extracting that line is deterministic assembly, explicitly disclosed in reports.
    for name, values in candidates.items():
        unique = {v.value for v in values}
        if len(unique) == 1:
            fields[name] = values[0]
        elif unique:
            fields[name] = FieldResult(
                method="ner", status="ambiguous", evidence=[e for v in values for e in v.evidence]
            )
    _, rule_meds = extract_rules(pages, category)
    for med in medications:
        matching = next((m for m in rule_meds if m.name.value == med.name.value), None)
        if matching:
            med.instructions = matching.instructions
    return fields, medications
