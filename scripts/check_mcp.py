"""Start the MCP server as a real stdio subprocess, list its tools, and call each one.

This is what Claude Desktop / Claude Code do. Unless STUDY_BRAIN_HOME is set, it
uses a throwaway data dir loaded with the sample notes and SAMPLE questions.

Usage: python scripts/check_mcp.py
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

ROOT = Path(__file__).resolve().parent.parent


def _show(name: str, result) -> None:
    text = result.content[0].text if result.content else ""
    print(f"\n--- {name} -> {text[:300]}{' ...' if len(text) > 300 else ''}")


async def main() -> int:
    home = os.environ.get("STUDY_BRAIN_HOME")
    if not home:  # throwaway dir with the sample notes + SAMPLE questions loaded
        home = tempfile.mkdtemp(prefix="study-brain-check-")
        env = {**os.environ, "STUDY_BRAIN_HOME": home}
        for cmd in (["ingest", str(ROOT / "notes")], ["seed"]):
            subprocess.run([sys.executable, "-m", "study_brain", *cmd], env=env, check=True,
                           stdout=subprocess.DEVNULL)
    env = {**os.environ, "STUDY_BRAIN_HOME": home}
    params = StdioServerParameters(command=sys.executable, args=["-m", "study_brain", "serve"], env=env)

    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            init = await session.initialize()
            print(f"connected to {init.serverInfo.name} (protocol {init.protocolVersion})")
            tools = await session.list_tools()
            print("tools:")
            for t in tools.tools:
                print(f"  - {t.name}({', '.join(t.inputSchema.get('properties', {}))})")

            _show("log_question", await session.call_tool(
                "log_question", {"text": "Why do eigenvectors keep their direction?", "source": "check"}))
            _show("similar_questions", await session.call_tool(
                "similar_questions", {"text": "what does an eigenvector mean?"}))
            _show("search_my_notes", await session.call_tool("search_my_notes", {"query": "eigenvector", "k": 1}))
            _show("my_unclear_topics", await session.call_tool("my_unclear_topics", {"limit": 1}))
            names = {t.name for t in tools.tools}
    expected = {"log_question", "search_my_notes", "my_unclear_topics", "similar_questions"}
    ok = names == expected
    print(f"\n{'OK' if ok else 'MISMATCH'}: {sorted(names)}  (data dir: {home})")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
