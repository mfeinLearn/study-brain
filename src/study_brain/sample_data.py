"""SAMPLE DATA: made-up study questions used for the demo, tests, and eval.

These are not real logged questions. Each entry is
(text, source, days_ago, expected_topic, paraphrase_group).

`paraphrase_group` labels questions that are the same confusion asked in
different words. The eval uses it to check repeat clustering, and the
expected top topics below are the ground truth for `my_unclear_topics`.
"""

from __future__ import annotations

SAMPLE_QUESTIONS: list[tuple[str, str, float, str, str | None]] = [
    # Linear algebra: the most repeated confusion (eigen-stuff).
    ("What does an eigenvector actually represent geometrically?", "deeplearning.ai", 1, "linear_algebra", "eigvec"),
    ("Why does an eigenvector keep its direction when the matrix is applied to it?", "deeplearning.ai", 3, "linear_algebra", "eigvec"),
    ("Intuitively, what is an eigenvector of a matrix?", "deeplearning.ai", 9, "linear_algebra", "eigvec"),
    ("How do I compute the eigenvalues of a 2x2 matrix by hand?", "deeplearning.ai", 5, "linear_algebra", "eigval"),
    ("What are the steps to find the eigenvalues of a 2 by 2 matrix?", "deeplearning.ai", 12, "linear_algebra", "eigval"),
    ("What is the determinant telling me about a linear transformation?", "deeplearning.ai", 20, "linear_algebra", None),
    ("When is a matrix not invertible?", "deeplearning.ai", 2, "linear_algebra", None),
    # Dynamic programming.
    ("How do I know when a LeetCode problem needs dynamic programming?", "leetcode", 2, "dynamic_programming", "dp_when"),
    ("How can I tell that a problem should be solved with dynamic programming?", "leetcode", 6, "dynamic_programming", "dp_when"),
    ("What is the difference between memoization and tabulation?", "leetcode", 4, "dynamic_programming", "memo"),
    ("Memoization vs bottom-up tabulation: which one should I use?", "leetcode", 10, "dynamic_programming", "memo"),
    ("How do I define the DP state for the coin change problem?", "leetcode", 1, "dynamic_programming", None),
    ("Why does the house robber recurrence take the max of two options?", "leetcode", 8, "dynamic_programming", None),
    # Probability / statistics.
    ("When do I use Bayes theorem instead of plain conditional probability?", "deeplearning.ai", 3, "probability_stats", "bayes"),
    ("In what situations should I reach for Bayes' theorem rather than just conditional probability?", "deeplearning.ai", 7, "probability_stats", "bayes"),
    ("What is the difference between variance and standard deviation?", "deeplearning.ai", 15, "probability_stats", None),
    ("Why do we divide by n-1 for the sample variance?", "deeplearning.ai", 5, "probability_stats", "n_minus_1"),
    ("Why is sample variance divided by n minus 1 instead of n?", "deeplearning.ai", 11, "probability_stats", "n_minus_1"),
    # Machine learning (one-offs).
    ("What is the difference between L1 and L2 regularization?", "flatiron", 6, "machine_learning", None),
    ("What does the learning rate control when training a model?", "flatiron", 18, "machine_learning", None),
    # Calculus (one-offs).
    ("How do I take the partial derivative of x^2 * y with respect to y?", "deeplearning.ai", 9, "calculus", None),
    ("How do I apply the chain rule to a composite function?", "deeplearning.ai", 25, "calculus", None),
    # Python / pandas (one-offs).
    ("How do I group a pandas DataFrame by two columns and sum another column?", "flatiron", 4, "python_pandas", None),
    ("What is the difference between loc and iloc in pandas?", "flatiron", 30, "python_pandas", None),
    # React, graphs, other.
    ("When should I use useEffect vs useMemo in React?", "flatiron", 13, "react", None),
    ("When do I use BFS instead of DFS on a graph?", "leetcode", 16, "graphs", None),
    ("How should I split my study time across the week?", "flatiron", 14, "other", None),
]

# Ground truth for the eval: the three topics the demo user is most unclear on.
EXPECTED_TOP_TOPICS: list[str] = ["linear_algebra", "dynamic_programming", "probability_stats"]
