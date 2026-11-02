import pytest

from study_brain.sample_data import SAMPLE_QUESTIONS
from study_brain.topics import OTHER, TopicTagger, tag


@pytest.mark.parametrize(
    "text, expected",
    [
        ("What does an eigenvector represent?", "linear_algebra"),
        ("Why is the determinant zero for a singular matrix?", "linear_algebra"),
        ("How does the chain rule work?", "calculus"),
        ("When do I use Bayes' theorem?", "probability_stats"),
        ("Why does L2 regularization reduce overfitting?", "machine_learning"),
        ("How do I use groupby on a pandas DataFrame?", "python_pandas"),
        ("Why does my useEffect run twice in React?", "react"),
        ("Is this a DP problem or greedy?", "dynamic_programming"),
        ("BFS or DFS for shortest path in a grid graph?", "graphs"),
        ("What should I eat for lunch?", OTHER),
        ("", OTHER),
    ],
)
def test_tagging(text, expected):
    assert tag(text) == expected


def test_whole_word_matching():
    # "dp" must not match inside other words; "rag" must not match "drag".
    assert tag("How do I set the dpi of a plot?") == OTHER
    assert tag("How do I drag a window?") == OTHER


def test_ties_go_to_first_topic_in_map():
    tagger = TopicTagger({"a": ["shared"], "b": ["shared"]})
    assert tagger.tag("a shared word") == "a"


def test_sample_questions_are_tagged_as_labelled():
    wrong = [(t, tag(t), exp) for t, _s, _d, exp, _g in SAMPLE_QUESTIONS if tag(t) != exp]
    assert wrong == []
