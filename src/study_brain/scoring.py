"""Unclear score per topic. The formula is documented in README.md ("Scoring").

    score(t) = w_f * N_t  +  w_r * sum_{q in t} 0.5 ** (age_days(q) / H)  +  w_c * C_t

N_t  = number of questions tagged t
H    = recency half-life in days
C_t  = number of repeat clusters in t (clusters with 2+ questions)
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime

from .config import ScoreWeights
from .db import Question


@dataclass
class TopicScore:
    topic: str
    score: float
    frequency: int
    recency: float
    repeat_clusters: int
    questions: list[Question] = field(default_factory=list)
    clusters: list[list[Question]] = field(default_factory=list)

    def summary(self) -> dict:
        return {
            "topic": self.topic,
            "score": round(self.score, 2),
            "frequency": self.frequency,
            "recency": round(self.recency, 2),
            "repeat_clusters": self.repeat_clusters,
        }


def recency_weight(asked_at: datetime, now: datetime, half_life_days: float) -> float:
    age_days = max((now - asked_at).total_seconds() / 86400.0, 0.0)
    return 0.5 ** (age_days / half_life_days)


def score_topics(
    questions: Sequence[Question],
    clusters: Sequence[Sequence[Question]],
    now: datetime,
    weights: ScoreWeights | None = None,
) -> list[TopicScore]:
    """Score every topic that has at least one question. Highest score first.

    `clusters` is the output of repeat detection (a partition of `questions`).
    A cluster counts toward the topic most of its questions are tagged with.
    """
    w = weights or ScoreWeights()
    by_topic: dict[str, TopicScore] = {}
    for q in questions:
        ts = by_topic.setdefault(q.topic, TopicScore(q.topic, 0.0, 0, 0.0, 0))
        ts.frequency += 1
        ts.recency += recency_weight(q.timestamp, now, w.half_life_days)
        ts.questions.append(q)

    for cluster in clusters:
        if not cluster:
            continue
        topics = [q.topic for q in cluster]
        topic = max(sorted(set(topics)), key=topics.count)
        ts = by_topic.get(topic)
        if ts is None:
            continue
        ts.clusters.append(list(cluster))
        if len(cluster) >= 2:
            ts.repeat_clusters += 1

    for ts in by_topic.values():
        ts.score = w.frequency * ts.frequency + w.recency * ts.recency + w.repeats * ts.repeat_clusters
        ts.questions.sort(key=lambda q: q.timestamp, reverse=True)
        ts.clusters.sort(key=lambda c: (-len(c), -max(q.timestamp for q in c).timestamp()))

    return sorted(by_topic.values(), key=lambda t: (-t.score, t.topic))
