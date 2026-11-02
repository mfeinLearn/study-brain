from collections import defaultdict

from study_brain.clustering import cluster_indices
from study_brain.sample_data import SAMPLE_QUESTIONS


def test_cluster_indices_with_synthetic_vectors():
    vecs = [[1, 0, 0], [0.95, 0.05, 0], [0, 1, 0], [0, 0.97, 0.1], [0, 0, 1]]
    assert cluster_indices(vecs, threshold=0.9) == [[0, 1], [2, 3], [4]]


def test_single_linkage_chains():
    # a~b and b~c are above threshold, a~c is not: still one cluster.
    vecs = [[1, 0], [0.8, 0.6], [0.28, 0.96]]
    assert cluster_indices(vecs, threshold=0.75) == [[0, 1, 2]]


def test_empty():
    assert cluster_indices([], threshold=0.5) == []


def test_paraphrases_cluster_with_real_embeddings(embedding_function):
    """The same confusion asked different ways lands in one cluster, and nothing else does."""
    texts = [q[0] for q in SAMPLE_QUESTIONS]
    groups = [q[4] for q in SAMPLE_QUESTIONS]
    clusters = cluster_indices(embedding_function(texts), threshold=0.6)

    expected = defaultdict(set)
    for i, g in enumerate(groups):
        expected[g if g else f"single-{i}"].add(i)
    assert sorted(map(sorted, expected.values())) == sorted(map(sorted, clusters))


def test_brain_clusters_and_similar_questions(seeded_brain):
    sizes = sorted((len(c) for c in seeded_brain.clusters()), reverse=True)
    assert sizes[:7] == [3, 2, 2, 2, 2, 2, 1]  # 6 labelled paraphrase groups, rest singletons

    hits = seeded_brain.similar_questions("what is an eigenvector, intuitively?", k=3)
    assert hits[0]["is_repeat"]
    assert "eigenvector" in hits[0]["text"].lower()
    assert seeded_brain.similar_questions("how do I bake sourdough bread?", k=1)[0]["is_repeat"] is False


def test_log_question_reports_asked_before(seeded_brain):
    result = seeded_brain.log_question("Why do we use n - 1 when computing sample variance?", source="test")
    assert result["question"]["topic"] == "probability_stats"
    assert any("n-1" in s["text"] or "n minus 1" in s["text"] for s in result["asked_before"])
