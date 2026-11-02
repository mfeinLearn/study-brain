from datetime import timedelta

from study_brain.config import ScoreWeights
from study_brain.db import Question
from study_brain.sample_data import EXPECTED_TOP_TOPICS
from study_brain.scoring import recency_weight, score_topics

from .conftest import NOW


def q(i, topic, days_ago):
    return Question(i, f"q{i}", topic, "test", NOW - timedelta(days=days_ago))


def test_recency_half_life():
    assert recency_weight(NOW, NOW, 14) == 1.0
    assert abs(recency_weight(NOW - timedelta(days=14), NOW, 14) - 0.5) < 1e-9


def test_frequency_wins_when_everything_else_equal():
    qs = [q(1, "a", 1), q(2, "a", 1), q(3, "b", 1)]
    ranked = score_topics(qs, [[x] for x in qs], NOW)
    assert [t.topic for t in ranked] == ["a", "b"]


def test_recent_beats_old_at_same_frequency():
    qs = [q(1, "old", 60), q(2, "new", 0)]
    ranked = score_topics(qs, [[x] for x in qs], NOW)
    assert [t.topic for t in ranked] == ["new", "old"]


def test_repeat_cluster_beats_scattered_questions():
    # Same count and same ages; "stuck" asks one thing twice, "browsing" asks two different things.
    qs = [q(1, "stuck", 5), q(2, "stuck", 5), q(3, "browsing", 5), q(4, "browsing", 5)]
    clusters = [[qs[0], qs[1]], [qs[2]], [qs[3]]]
    ranked = score_topics(qs, clusters, NOW)
    assert ranked[0].topic == "stuck"
    assert ranked[0].repeat_clusters == 1 and ranked[1].repeat_clusters == 0


def test_formula_matches_readme():
    w = ScoreWeights(frequency=1, recency=2, repeats=3, half_life_days=14)
    qs = [q(1, "t", 0), q(2, "t", 14)]
    [t] = score_topics(qs, [qs], NOW, w)
    assert abs(t.score - (1 * 2 + 2 * (1 + 0.5) + 3 * 1)) < 1e-9


def test_seeded_top_topics_match_ground_truth(seeded_brain):
    top = seeded_brain.unclear_topics(limit=3, now=NOW)
    assert [t["topic"] for t in top] == EXPECTED_TOP_TOPICS
    # Each flagged topic comes with past questions and note chunks only after ingest.
    assert all(t["questions"] for t in top)


def test_unclear_topics_include_related_notes(seeded_brain):
    seeded_brain.ingest("notes")
    [top] = seeded_brain.unclear_topics(limit=1, now=NOW)
    assert top["related_notes"]
    assert top["related_notes"][0]["source"] == "math/eigenvectors.md"
