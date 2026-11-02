"""Command line interface: python -m study_brain <command>."""

from __future__ import annotations

import argparse
import json
import sys
import textwrap

from .brain import StudyBrain


def _short(text: str, width: int = 100) -> str:
    return textwrap.shorten(" ".join(text.split()), width=width, placeholder=" ...")


def cmd_ingest(brain: StudyBrain, args: argparse.Namespace) -> None:
    stats = brain.ingest(args.path)
    print(f"Ingested {stats['chunks']} chunks from {stats['files']} files "
          f"({stats['total_chunks']} chunks in the notes collection).")


def cmd_ask(brain: StudyBrain, args: argparse.Namespace) -> None:
    if args.no_log:
        similar = [s for s in brain.similar_questions(args.question) if s["is_repeat"]]
        print("(not logged)")
    else:
        result = brain.log_question(args.question, source=args.source)
        q = result["question"]
        similar = result["asked_before"]
        print(f"Logged question #{q['id']}  topic={q['topic']}  source={q['source']}")
    if similar:
        print("\nYou've asked something like this before:")
        for s in similar:
            print(f"  - [{s['similarity']:.2f}] {s['text']}  ({s['timestamp'][:10]})")
    hits = brain.search_notes(args.question, k=args.k)
    if hits:
        print("\nFrom your notes:")
        for h in hits:
            print(f"  [{h['score']:.2f}] {h['citation']}\n      {_short(h['text'])}")
    else:
        print("\nNo notes ingested yet. Run: python -m study_brain ingest ./notes")


def cmd_report(brain: StudyBrain, args: argparse.Namespace) -> None:
    topics = brain.unclear_topics(limit=args.limit, include_other=args.include_other)
    if args.json:
        print(json.dumps(topics, indent=2))
        return
    stats = brain.stats()
    print(f"study-brain report  ({stats['questions']} questions, {stats['note_chunks']} note chunks)")
    if not topics:
        print("No questions logged yet. Try: python -m study_brain seed")
        return
    for rank, t in enumerate(topics, 1):
        print(f"\n{rank}. {t['topic']}  score={t['score']:.2f}  "
              f"(asked {t['frequency']}x, recency {t['recency']:.2f}, "
              f"{t['repeat_clusters']} repeated confusion(s))")
        for group in t["repeated_confusions"]:
            print(f"   repeated {len(group)}x:")
            for text in group:
                print(f"     - {text}")
        singles = {q for g in t["repeated_confusions"] for q in g}
        others = [q for q in t["questions"] if q["text"] not in singles][:3]
        if others:
            print("   other recent questions:")
            for q in others:
                print(f"     - {q['text']}  ({q['timestamp'][:10]})")
        if t["related_notes"]:
            print("   review these notes:")
            for n in t["related_notes"]:
                print(f"     * {n['citation']}  [{n['score']:.2f}]  {_short(n['text'], 70)}")


def cmd_similar(brain: StudyBrain, args: argparse.Namespace) -> None:
    for s in brain.similar_questions(args.text, k=args.k):
        flag = "REPEAT" if s["is_repeat"] else "      "
        print(f"{flag} [{s['similarity']:.2f}] {s['text']}  ({s['topic']})")


def cmd_search(brain: StudyBrain, args: argparse.Namespace) -> None:
    for h in brain.search_notes(args.query, k=args.k):
        print(f"[{h['score']:.2f}] {h['citation']}\n{textwrap.indent(h['text'], '    ')}\n")


def cmd_seed(brain: StudyBrain, args: argparse.Namespace) -> None:
    from .seed import seed

    n = seed(brain, reset=not args.keep)
    print(f"Seeded {n} SAMPLE questions into {brain.settings.home} "
          f"({'appended' if args.keep else 'replaced existing questions'}).")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="study_brain", description="Learn what you're unclear on from the questions you ask.")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("ingest", help="chunk and embed a notes folder (.md/.txt/.py/.ipynb)")
    s.add_argument("path")
    s.set_defaults(func=cmd_ingest)

    s = sub.add_parser("ask", help="log a question and show related notes and past questions")
    s.add_argument("question")
    s.add_argument("--source", default="cli")
    s.add_argument("-k", type=int, default=3)
    s.add_argument("--no-log", action="store_true", help="search only; don't record the question")
    s.set_defaults(func=cmd_ask)

    s = sub.add_parser("report", help="show the topics you're most unclear on")
    s.add_argument("--limit", type=int, default=3)
    s.add_argument("--include-other", action="store_true")
    s.add_argument("--json", action="store_true")
    s.set_defaults(func=cmd_report)

    s = sub.add_parser("similar", help="find past questions similar to a text")
    s.add_argument("text")
    s.add_argument("-k", type=int, default=5)
    s.set_defaults(func=cmd_similar)

    s = sub.add_parser("search", help="search your notes (RAG retrieval only)")
    s.add_argument("query")
    s.add_argument("-k", type=int, default=4)
    s.set_defaults(func=cmd_search)

    s = sub.add_parser("seed", help="load the ~27 SAMPLE questions for a demo")
    s.add_argument("--keep", action="store_true", help="append instead of replacing existing questions")
    s.set_defaults(func=cmd_seed)

    sub.add_parser("serve", help="run the MCP server over stdio")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "serve":
        from .server import main as serve

        serve()
        return 0
    brain = StudyBrain()
    try:
        args.func(brain, args)
    except (FileNotFoundError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    return 0
