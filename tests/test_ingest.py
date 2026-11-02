import json

from study_brain.ingest import chunk_text, load_chunks, read_notebook


def test_chunk_text_respects_size():
    text = "\n\n".join(f"Paragraph {i} " + "word " * 40 for i in range(20))
    chunks = chunk_text(text, size=300, overlap=50)
    assert len(chunks) > 1
    assert all(len(c) <= 300 for c in chunks)


def test_notebook_is_flattened(tmp_path):
    nb = {"cells": [{"cell_type": "markdown", "source": ["# Title"]},
                    {"cell_type": "code", "source": "x = 1"}]}
    p = tmp_path / "a.ipynb"
    p.write_text(json.dumps(nb))
    assert read_notebook(p) == "# Title\n\n```python\nx = 1\n```"


def test_load_chunks_sample_notes():
    chunks = load_chunks(__import__("pathlib").Path("notes"))
    sources = {c.source for c in chunks}
    assert {"math/eigenvectors.md", "leetcode/coin_change.py", "flatiron/pandas_groupby.ipynb",
            "flatiron/regularization.txt"} <= sources


def test_search_returns_citations(brain):
    stats = brain.ingest("notes")
    assert stats["chunks"] > 0
    again = brain.ingest("notes")  # re-ingest replaces, doesn't duplicate
    assert again["total_chunks"] == stats["total_chunks"]
    hits = brain.search_notes("when should I use memoization or tabulation", k=2)
    assert hits[0]["source"] == "leetcode/dynamic_programming.md"
    assert hits[0]["citation"].startswith("leetcode/dynamic_programming.md#chunk")
