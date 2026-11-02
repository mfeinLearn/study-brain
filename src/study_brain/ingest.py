"""Load and chunk notes (.md, .txt, .py, .ipynb) for the notes collection."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

SUPPORTED_SUFFIXES = {".md", ".txt", ".py", ".ipynb"}
SKIP_DIRS = {".git", ".venv", "venv", "__pycache__", ".ipynb_checkpoints", "node_modules"}


@dataclass(frozen=True)
class Chunk:
    text: str
    source: str  # path relative to the ingest root, e.g. "leetcode/dynamic_programming.md"
    index: int
    date: str  # file modified date, YYYY-MM-DD


def read_notebook(path: Path) -> str:
    """Flatten a Jupyter notebook into markdown text (markdown cells + fenced code cells)."""
    nb = json.loads(path.read_text(encoding="utf-8"))
    parts: list[str] = []
    for cell in nb.get("cells", []):
        src = cell.get("source", "")
        text = "".join(src) if isinstance(src, list) else str(src)
        if not text.strip():
            continue
        if cell.get("cell_type") == "code":
            parts.append(f"```python\n{text.strip()}\n```")
        else:
            parts.append(text.strip())
    return "\n\n".join(parts)


def read_file(path: Path) -> str:
    if path.suffix == ".ipynb":
        return read_notebook(path)
    return path.read_text(encoding="utf-8", errors="replace")


def _word_tail(text: str, n: int) -> str:
    """Last ~n chars of text, starting at a word boundary."""
    if n <= 0:
        return ""
    tail = text[-n:]
    if len(tail) < len(text) and not text[-n - 1].isspace():
        cut = tail.find(" ")
        tail = tail[cut + 1 :] if cut != -1 else ""
    return tail.strip()


def chunk_text(text: str, size: int = 800, overlap: int = 120) -> list[str]:
    """Pack paragraphs into chunks of about `size` chars.

    Paragraphs longer than `size` are split on a sliding window. Each new chunk
    starts with the last `overlap` chars of the previous one so an idea that
    straddles a boundary is still retrievable.
    """
    if overlap >= size:
        raise ValueError("overlap must be smaller than size")
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    pieces: list[str] = []
    for p in paragraphs:
        if len(p) <= size:
            pieces.append(p)
        else:
            step = size - overlap
            pieces.extend(p[i : i + size] for i in range(0, len(p), step))

    chunks: list[str] = []
    current = ""
    for piece in pieces:
        if current and len(current) + 2 + len(piece) > size:
            chunks.append(current)
            tail = _word_tail(current, overlap)
            current = (tail + "\n\n" + piece) if tail and len(tail) + 2 + len(piece) <= size else piece
        else:
            current = f"{current}\n\n{piece}" if current else piece
    if current:
        chunks.append(current)
    return chunks


def iter_note_files(root: Path) -> list[Path]:
    if root.is_file():
        return [root] if root.suffix in SUPPORTED_SUFFIXES else []
    files = []
    for path in sorted(root.rglob("*")):
        if any(part in SKIP_DIRS for part in path.relative_to(root).parts):
            continue
        if path.is_file() and path.suffix in SUPPORTED_SUFFIXES:
            files.append(path)
    return files


def load_chunks(root: Path, size: int = 800, overlap: int = 120) -> list[Chunk]:
    root = root.expanduser().resolve()
    chunks: list[Chunk] = []
    for path in iter_note_files(root):
        rel = path.name if root.is_file() else path.relative_to(root).as_posix()
        date = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).date().isoformat()
        for i, text in enumerate(chunk_text(read_file(path), size, overlap)):
            chunks.append(Chunk(text=text, source=rel, index=i, date=date))
    return chunks
