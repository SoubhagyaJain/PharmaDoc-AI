"""Keyword sanity baseline and train-only TF-IDF / LinearSVC."""

from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC


def keyword_classify(text: str) -> str:
    lowered = text.lower()
    cues = {
        "prescription": ("prescription", "prescriber", "medication:"),
        "clinical_form": ("intake", "insurance", "form id:"),
        "regulatory": ("quality record", "revision date:", "section:"),
    }
    scores = {category: sum(cue in lowered for cue in words) for category, words in cues.items()}
    ranked = sorted(scores, key=scores.get, reverse=True)
    return ranked[0] if scores[ranked[0]] > scores[ranked[1]] else "unknown"


@lru_cache(maxsize=4)
def _load(path: str, modified: int):
    # Only load artifacts trained locally. joblib is unsafe for untrusted files.
    return joblib.load(path)


def classify(text: str, model_path: Path):
    if not model_path.is_file():
        return keyword_classify(text), "keywords", None
    artifact = _load(str(model_path), model_path.stat().st_mtime_ns)
    scores = artifact["model"].decision_function([text])[0]
    margin = float(np.sort(scores)[-1] - np.sort(scores)[-2])
    category = str(artifact["model"].classes_[int(np.argmax(scores))])
    # Zero known vocabulary means a large intercept is not useful evidence.
    known = artifact["model"][0].transform([text]).nnz
    if known == 0 or margin < artifact["threshold"]:
        category = "unknown"
    return category, "tfidf_linear_svc", margin


def train_classifier(records: list[dict], output: Path) -> dict:
    train = [r for r in records if r["split"] == "train"]
    validation = [r for r in records if r["split"] == "validation"]
    if len({r["category"] for r in train}) != 3 or not validation:
        raise ValueError("Training requires three classes and a separate validation partition.")

    def text(r):
        return "\n".join(p["text"] for p in r["pages"])

    model = make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True), LinearSVC(random_state=42)
    )
    model.fit([text(r) for r in train], [r["category"] for r in train])
    scores = model.decision_function([text(r) for r in validation])
    margins = np.sort(scores, axis=1)[:, -1] - np.sort(scores, axis=1)[:, -2]
    predictions = model.classes_[np.argmax(scores, axis=1)]
    labels = [r["category"] for r in validation]
    trials = []
    for threshold in (0.0, 0.1, 0.2, 0.3, 0.5):
        predicted = [
            p if margin >= threshold else "unknown"
            for p, margin in zip(predictions, margins, strict=True)
        ]
        trials.append(
            (
                f1_score(
                    labels, predicted, labels=list(model.classes_), average="macro", zero_division=0
                ),
                threshold,
            )
        )
    best_f1, threshold = max(trials, key=lambda pair: (pair[0], pair[1]))
    artifact = {
        "model": model,
        "threshold": threshold,
        "training_documents": len(train),
        "validation_macro_f1": best_f1,
        "seed": 42,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, output)
    _load.cache_clear()
    return {key: value for key, value in artifact.items() if key != "model"}
