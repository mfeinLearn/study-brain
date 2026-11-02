"""Settings. Everything can be overridden with environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    return float(value) if value else default


@dataclass(frozen=True)
class ScoreWeights:
    """Weights for the unclear score. See README "Scoring"."""

    frequency: float = 1.0  # w_f: per question asked
    recency: float = 2.0  # w_r: per unit of decayed recency
    repeats: float = 3.0  # w_c: per repeat cluster (same confusion asked 2+ times)
    half_life_days: float = 14.0  # H: a question's recency weight halves every H days


@dataclass(frozen=True)
class Settings:
    home: Path = field(
        default_factory=lambda: Path(
            os.environ.get("STUDY_BRAIN_HOME", Path.home() / ".study_brain")
        ).expanduser()
    )
    # Cosine similarity at or above which two questions count as "the same confusion".
    similarity_threshold: float = field(
        default_factory=lambda: _env_float("STUDY_BRAIN_SIM_THRESHOLD", 0.6)
    )
    chunk_size: int = 800
    chunk_overlap: int = 120
    weights: ScoreWeights = field(default_factory=ScoreWeights)

    @property
    def db_path(self) -> Path:
        return self.home / "study_brain.sqlite3"

    @property
    def chroma_path(self) -> Path:
        return self.home / "chroma"
