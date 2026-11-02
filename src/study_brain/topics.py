"""Deterministic keyword-based topic tagging (no LLM)."""

from __future__ import annotations

import re
from functools import lru_cache
from importlib import resources
from pathlib import Path

import yaml

OTHER = "other"


def load_topic_map(path: str | Path | None = None) -> dict[str, list[str]]:
    """Load the topic -> keywords map. Defaults to the bundled topics.yaml."""
    if path is None:
        text = resources.files("study_brain").joinpath("topics.yaml").read_text(encoding="utf-8")
    else:
        text = Path(path).read_text(encoding="utf-8")
    raw = yaml.safe_load(text) or {}
    return {str(topic): [str(k).lower() for k in kws or []] for topic, kws in raw.items()}


def _pattern(keyword: str) -> re.Pattern[str]:
    # Whole-word match; tolerate an optional possessive/plural "'s" or "s".
    return re.compile(r"(?<![a-z0-9])" + re.escape(keyword) + r"(?:'s|s)?(?![a-z0-9])")


class TopicTagger:
    def __init__(self, topic_map: dict[str, list[str]] | None = None) -> None:
        self.topic_map = topic_map if topic_map is not None else load_topic_map()
        self._patterns = {t: [_pattern(k) for k in kws] for t, kws in self.topic_map.items()}

    def scores(self, text: str) -> dict[str, int]:
        lowered = text.lower()
        return {t: sum(1 for p in pats if p.search(lowered)) for t, pats in self._patterns.items()}

    def tag(self, text: str) -> str:
        best, best_hits = OTHER, 0
        for topic, hits in self.scores(text).items():  # dict order == yaml order -> stable ties
            if hits > best_hits:
                best, best_hits = topic, hits
        return best


@lru_cache(maxsize=1)
def default_tagger() -> TopicTagger:
    return TopicTagger()


def tag(text: str) -> str:
    return default_tagger().tag(text)
