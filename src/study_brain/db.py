"""SQLite storage for logged questions (the source of truth)."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS questions (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    text      TEXT NOT NULL,
    topic     TEXT NOT NULL,
    source    TEXT NOT NULL DEFAULT 'unknown',
    timestamp TEXT NOT NULL  -- ISO 8601, UTC
);
CREATE INDEX IF NOT EXISTS idx_questions_topic ON questions(topic);
"""


@dataclass(frozen=True)
class Question:
    id: int
    text: str
    topic: str
    source: str
    timestamp: datetime

    def as_dict(self) -> dict:
        return {
            "id": self.id,
            "text": self.text,
            "topic": self.topic,
            "source": self.source,
            "timestamp": self.timestamp.isoformat(timespec="seconds"),
        }


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _parse_ts(value: str) -> datetime:
    ts = datetime.fromisoformat(value)
    return ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)


class QuestionDB:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(path), check_same_thread=False)
        self.conn.executescript(SCHEMA)

    def add(self, text: str, topic: str, source: str, timestamp: datetime | None = None) -> Question:
        ts = (timestamp or utcnow()).astimezone(timezone.utc)
        cur = self.conn.execute(
            "INSERT INTO questions (text, topic, source, timestamp) VALUES (?, ?, ?, ?)",
            (text, topic, source, ts.isoformat()),
        )
        self.conn.commit()
        return Question(int(cur.lastrowid), text, topic, source, ts)

    def all(self) -> list[Question]:
        rows = self.conn.execute(
            "SELECT id, text, topic, source, timestamp FROM questions ORDER BY id"
        ).fetchall()
        return [Question(r[0], r[1], r[2], r[3], _parse_ts(r[4])) for r in rows]

    def count(self) -> int:
        return int(self.conn.execute("SELECT COUNT(*) FROM questions").fetchone()[0])

    def clear(self) -> None:
        self.conn.execute("DELETE FROM questions")
        self.conn.execute("DELETE FROM sqlite_sequence WHERE name = 'questions'")
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()
