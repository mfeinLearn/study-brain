"""Eval: does study-brain flag the topics the (sample) user is actually unclear on?

Runs against a throwaway data dir, so it never touches your real data.
Seeds the SAMPLE questions, then scores three things against ground truth:

  1. tagging accuracy      - topic tag vs the labelled topic for each question
  2. clustering pair F1    - "same confusion" pairs found vs labelled paraphrase groups
  3. top-topic precision@3 - flagged top 3 topics vs EXPECTED_TOP_TOPICS

Usage: python eval/run_eval.py [--threshold 0.6]
Exit code is 1 if the overall score is below 0.9.
"""

from __future__ import annotations

import argparse
import itertools
import sys
import tempfile
from pathlib import Path

from study_brain.brain import StudyBrain
from study_brain.config import Settings
from study_brain.sample_data import EXPECTED_TOP_TOPICS, SAMPLE_QUESTIONS
from study_brain.seed import seed


def pair_f1(predicted: list[set[str]], truth: list[set[str]]) -> tuple[float, float, float]:
    def pairs(groups):
        return {frozenset(p) for g in groups for p in itertools.combinations(sorted(g), 2)}

    pp, tp = pairs(predicted), pairs(truth)
    hit = len(pp & tp)
    precision = hit / len(pp) if pp else 1.0
    recall = hit / len(tp) if tp else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=float, default=0.6)
    args = ap.parse_args()

    labels = {text: (topic, group) for text, _s, _d, topic, group in SAMPLE_QUESTIONS}
    with tempfile.TemporaryDirectory() as tmp:
        brain = StudyBrain(Settings(home=Path(tmp), similarity_threshold=args.threshold))
        seed(brain)
        brain.ingest(Path(__file__).resolve().parent.parent / "notes")

        questions = brain.db.all()
        tag_acc = sum(q.topic == labels[q.text][0] for q in questions) / len(questions)

        predicted = [{q.text for q in c} for c in brain.clusters() if len(c) > 1]
        truth: dict[str, set[str]] = {}
        for text, (_t, group) in labels.items():
            if group:
                truth.setdefault(group, set()).add(text)
        p, r, f1 = pair_f1(predicted, list(truth.values()))

        top = [t["topic"] for t in brain.unclear_topics(limit=3)]
        p_at_3 = len(set(top) & set(EXPECTED_TOP_TOPICS)) / len(EXPECTED_TOP_TOPICS)
        exact_order = top == EXPECTED_TOP_TOPICS

    overall = (tag_acc + f1 + p_at_3) / 3
    print(f"questions evaluated      : {len(questions)}")
    print(f"tagging accuracy         : {tag_acc:.2f}")
    print(f"clustering pair P/R/F1   : {p:.2f} / {r:.2f} / {f1:.2f}  (threshold {args.threshold})")
    print(f"top-3 flagged            : {top}")
    print(f"expected top-3           : {EXPECTED_TOP_TOPICS}")
    print(f"top-3 precision@3        : {p_at_3:.2f}  (exact order: {'yes' if exact_order else 'no'})")
    print(f"OVERALL SCORE            : {overall:.2f}  ({'PASS' if overall >= 0.9 else 'FAIL'})")
    return 0 if overall >= 0.9 else 1


if __name__ == "__main__":
    sys.exit(main())
