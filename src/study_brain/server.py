"""MCP server (stdio). Run with `python -m study_brain serve` or `study-brain-mcp`."""

from __future__ import annotations

from functools import lru_cache

from mcp.server.fastmcp import FastMCP

from .brain import StudyBrain

mcp = FastMCP(
    "study-brain",
    instructions=(
        "Tracks what the user is unclear on while studying. Call log_question whenever the "
        "user asks a study question. Use search_my_notes to ground answers in their own notes "
        "and cite the returned `citation`. Use my_unclear_topics to see what to review."
    ),
)


@lru_cache(maxsize=1)
def get_brain() -> StudyBrain:
    # Created lazily so the server starts (and lists tools) before the embedding model loads.
    return StudyBrain()


@mcp.tool()
def log_question(text: str, source: str = "unknown") -> dict:
    """Record a question the user asked while studying.

    Args:
        text: The question, in the user's words.
        source: Where it came up, e.g. "flatiron", "deeplearning.ai", "leetcode".

    Returns the stored question with its auto-tagged topic, plus earlier questions
    that look like the same confusion (`asked_before`).
    """
    return get_brain().log_question(text, source=source)


@mcp.tool()
def search_my_notes(query: str, k: int = 4) -> dict:
    """Retrieve the k most relevant chunks from the user's ingested notes.

    Returns {"query", "results"}. Each result has `text`, `source` (file path), `citation` ("path#chunkN"), `topic`,
    `date`, and a cosine `score`. Cite `citation` when using a chunk in an answer.
    """
    return {"query": query, "results": get_brain().search_notes(query, k=k)}


@mcp.tool()
def my_unclear_topics(limit: int = 3) -> dict:
    """Topics the user seems most unclear on, ranked by unclear score. Returns {"topics": [...]}.

    Each entry has the score breakdown (frequency, recency, repeat_clusters), the
    repeated confusions, all past questions on the topic, and related note chunks.
    """
    return {"topics": get_brain().unclear_topics(limit=limit)}


@mcp.tool()
def similar_questions(text: str, k: int = 5) -> dict:
    """Past questions most similar to `text`, with cosine similarity. Returns {"text", "results"}.

    `is_repeat` is true when the similarity passes the repeat threshold, meaning it
    is probably the same confusion asked a different way.
    """
    return {"text": text, "results": get_brain().similar_questions(text, k=k)}


def main() -> None:
    mcp.run()  # stdio transport


if __name__ == "__main__":
    main()
