"""StudyBrain: the core service. The CLI and the MCP server are thin wrappers around it."""

from __future__ import annotations

import hashlib
from datetime import datetime
from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings as ChromaSettings
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

from .clustering import cluster_indices
from .config import Settings
from .db import Question, QuestionDB, utcnow
from .ingest import load_chunks
from .scoring import TopicScore, score_topics
from .topics import OTHER, TopicTagger, default_tagger

QUESTIONS = "questions"
NOTES = "notes"


def _qid(question_id: int) -> str:
    return f"q{question_id}"


class StudyBrain:
    def __init__(
        self,
        settings: Settings | None = None,
        embedding_function: Any | None = None,
        tagger: TopicTagger | None = None,
    ) -> None:
        self.settings = settings or Settings()
        self.settings.home.mkdir(parents=True, exist_ok=True)
        self.tagger = tagger or default_tagger()
        self.db = QuestionDB(self.settings.db_path)
        # Default: Chroma's local ONNX all-MiniLM-L6-v2. No API key; the model
        # (~80 MB) is downloaded to ~/.cache/chroma on first use.
        self.ef = embedding_function or DefaultEmbeddingFunction()
        self.client = chromadb.PersistentClient(
            path=str(self.settings.chroma_path),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self.questions = self._collection(QUESTIONS)
        self.notes = self._collection(NOTES)

    def _collection(self, name: str):
        return self.client.get_or_create_collection(
            name=name,
            embedding_function=self.ef,
            configuration={"hnsw": {"space": "cosine"}},
        )

    # ------------------------------------------------------------------ questions

    def log_question(
        self, text: str, source: str = "unknown", timestamp: datetime | None = None
    ) -> dict:
        text = text.strip()
        if not text:
            raise ValueError("question text is empty")
        topic = self.tagger.tag(text)
        repeats = [s for s in self.similar_questions(text, k=5) if s["is_repeat"]]
        q = self.db.add(text, topic, source, timestamp)
        self.questions.add(
            ids=[_qid(q.id)],
            documents=[q.text],
            metadatas=[{"question_id": q.id, "topic": q.topic, "source": q.source,
                        "timestamp": q.as_dict()["timestamp"]}],
        )
        return {"question": q.as_dict(), "asked_before": repeats}

    def similar_questions(self, text: str, k: int = 5) -> list[dict]:
        """Past questions closest to `text`. `is_repeat` marks ones past the cluster threshold."""
        n = self.questions.count()
        if n == 0:
            return []
        res = self.questions.query(query_texts=[text], n_results=min(k, n))
        out = []
        for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
            sim = 1.0 - float(dist)
            out.append({
                "id": meta["question_id"],
                "text": doc,
                "topic": meta["topic"],
                "source": meta["source"],
                "timestamp": meta["timestamp"],
                "similarity": round(sim, 3),
                "is_repeat": sim >= self.settings.similarity_threshold,
            })
        return out

    def clusters(self) -> list[list[Question]]:
        """Partition all logged questions into repeat clusters."""
        questions = self.db.all()
        if not questions:
            return []
        got = self.questions.get(ids=[_qid(q.id) for q in questions], include=["embeddings"])
        emb_by_id = dict(zip(got["ids"], got["embeddings"]))
        have = [q for q in questions if _qid(q.id) in emb_by_id]
        groups = cluster_indices([emb_by_id[_qid(q.id)] for q in have], self.settings.similarity_threshold)
        return [[have[i] for i in g] for g in groups]

    def topic_scores(self, now: datetime | None = None) -> list[TopicScore]:
        return score_topics(self.db.all(), self.clusters(), now or utcnow(), self.settings.weights)

    def unclear_topics(
        self, limit: int = 3, include_other: bool = False, notes_per_topic: int = 3,
        now: datetime | None = None,
    ) -> list[dict]:
        """Top topics by unclear score, with past questions (grouped) and related note chunks."""
        ranked = [t for t in self.topic_scores(now) if include_other or t.topic != OTHER]
        out = []
        for ts in ranked[: max(limit, 0)]:
            # Use the most-repeated confusions as retrieval queries.
            reps = [c[0].text for c in ts.clusters[:3]]
            out.append({
                **ts.summary(),
                "repeated_confusions": [
                    [q.text for q in sorted(c, key=lambda q: q.timestamp, reverse=True)]
                    for c in ts.clusters if len(c) >= 2
                ],
                "questions": [q.as_dict() for q in ts.questions],
                "related_notes": self._notes_for_queries(reps, notes_per_topic),
            })
        return out

    # ---------------------------------------------------------------------- notes

    def ingest(self, path: str | Path) -> dict:
        root = Path(path)
        if not root.exists():
            raise FileNotFoundError(root)
        chunks = load_chunks(root, self.settings.chunk_size, self.settings.chunk_overlap)
        sources = sorted({c.source for c in chunks})
        for src in sources:  # re-ingesting a file replaces its old chunks
            self.notes.delete(where={"source": src})
        if chunks:
            self.notes.add(
                ids=[hashlib.sha1(f"{c.source}#{c.index}".encode()).hexdigest() for c in chunks],
                documents=[c.text for c in chunks],
                metadatas=[{"source": c.source, "chunk": c.index, "date": c.date,
                            "topic": self.tagger.tag(f"{Path(c.source).stem.replace('_', ' ')} {c.text}")}
                           for c in chunks],
            )
        return {"files": len(sources), "chunks": len(chunks), "total_chunks": self.notes.count()}

    def search_notes(self, query: str, k: int = 4) -> list[dict]:
        n = self.notes.count()
        if n == 0:
            return []
        res = self.notes.query(query_texts=[query], n_results=min(max(k, 1), n))
        return [
            {
                "text": doc,
                "source": meta["source"],
                "chunk": meta["chunk"],
                "topic": meta["topic"],
                "date": meta["date"],
                "citation": f"{meta['source']}#chunk{meta['chunk']}",
                "score": round(1.0 - float(dist), 3),
            }
            for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0])
        ]

    def _notes_for_queries(self, queries: list[str], k: int, min_score: float = 0.3) -> list[dict]:
        best: dict[str, dict] = {}
        for q in queries:
            for hit in self.search_notes(q, k):
                if hit["score"] < min_score:
                    continue
                if hit["citation"] not in best or hit["score"] > best[hit["citation"]]["score"]:
                    best[hit["citation"]] = hit
        return sorted(best.values(), key=lambda h: -h["score"])[:k]

    # ---------------------------------------------------------------------- admin

    def reset_questions(self) -> None:
        self.db.clear()
        self.client.delete_collection(QUESTIONS)
        self.questions = self._collection(QUESTIONS)

    def stats(self) -> dict:
        return {"home": str(self.settings.home), "questions": self.db.count(),
                "note_chunks": self.notes.count()}
