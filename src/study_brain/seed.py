"""Load the SAMPLE questions (see sample_data.py) with timestamps relative to now."""

from __future__ import annotations

from datetime import datetime, timedelta

from .brain import StudyBrain
from .db import utcnow
from .sample_data import SAMPLE_QUESTIONS


def seed(brain: StudyBrain, reset: bool = True, now: datetime | None = None) -> int:
    now = now or utcnow()
    if reset:
        brain.reset_questions()
    # Insert oldest first so ids follow time order.
    for text, source, days_ago, _topic, _group in sorted(SAMPLE_QUESTIONS, key=lambda r: -r[2]):
        brain.log_question(text, source=source, timestamp=now - timedelta(days=days_ago))
    return len(SAMPLE_QUESTIONS)
