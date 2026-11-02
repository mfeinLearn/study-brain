from datetime import datetime, timezone

import pytest
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

from study_brain.brain import StudyBrain
from study_brain.config import Settings

NOW = datetime(2026, 11, 1, 12, 0, tzinfo=timezone.utc)


@pytest.fixture(scope="session")
def embedding_function():
    # Local ONNX MiniLM; downloaded once to ~/.cache/chroma.
    return DefaultEmbeddingFunction()


@pytest.fixture
def brain(tmp_path, embedding_function):
    return StudyBrain(Settings(home=tmp_path / "home"), embedding_function=embedding_function)


@pytest.fixture
def seeded_brain(brain):
    from study_brain.seed import seed

    seed(brain, now=NOW)
    return brain
