"""Smoke test: talk to the FastMCP server through a real MCP client session (in memory)."""

import json

from mcp.shared.memory import create_connected_server_and_client_session

from study_brain import server

EXPECTED_TOOLS = {"log_question", "search_my_notes", "my_unclear_topics", "similar_questions"}


def _payload(result):
    assert not result.isError, result
    assert len(result.content) == 1
    return json.loads(result.content[0].text)


async def test_tools_list_and_call(seeded_brain, monkeypatch):
    seeded_brain.ingest("notes")
    monkeypatch.setattr(server, "get_brain", lambda: seeded_brain)

    async with create_connected_server_and_client_session(server.mcp._mcp_server) as client:
        tools = await client.list_tools()
        assert {t.name for t in tools.tools} == EXPECTED_TOOLS

        logged = _payload(await client.call_tool(
            "log_question", {"text": "What is an eigenvector, intuitively?", "source": "test"}))
        assert logged["question"]["topic"] == "linear_algebra"
        assert logged["asked_before"]

        notes = _payload(await client.call_tool("search_my_notes", {"query": "house robber", "k": 2}))["results"]
        assert notes and "citation" in notes[0]

        top = _payload(await client.call_tool("my_unclear_topics", {"limit": 2}))["topics"]
        assert [t["topic"] for t in top] == ["linear_algebra", "dynamic_programming"]

        similar = _payload(await client.call_tool("similar_questions", {"text": "Bayes vs conditional probability?"}))["results"]
        assert similar[0]["is_repeat"]
