import pytest

from pharmadoc_ai.config import Settings
from pharmadoc_ai.data.synthetic import generate_dataset


@pytest.fixture(scope="session")
def dataset(tmp_path_factory):
    root = tmp_path_factory.mktemp("synthetic")
    return root, generate_dataset(root, seed=991, per_family=1)


@pytest.fixture
def settings(tmp_path):
    # Unit tests must never accidentally load developer-trained models.
    return Settings(classifier_path=tmp_path / "absent.joblib", ner_path=tmp_path / "absent-ner")
