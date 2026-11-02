"""Repeat detection: group questions that are the same confusion asked differently.

Single-linkage clustering over cosine similarity: two questions end up in the
same cluster if there is a chain of pairs each with similarity >= threshold.
For a few hundred to a few thousand questions an O(n^2) pass is fine.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def cosine_matrix(embeddings: Sequence[Sequence[float]]) -> np.ndarray:
    m = np.asarray(embeddings, dtype=np.float64)
    if m.size == 0:
        return np.zeros((0, 0))
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    m = m / norms
    return m @ m.T


def cluster_indices(embeddings: Sequence[Sequence[float]], threshold: float) -> list[list[int]]:
    """Return clusters as lists of row indices, largest first, each sorted ascending."""
    n = len(embeddings)
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    sims = cosine_matrix(embeddings)
    for i in range(n):
        for j in range(i + 1, n):
            if sims[i, j] >= threshold:
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[rj] = ri

    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return sorted(groups.values(), key=lambda g: (-len(g), g[0]))
