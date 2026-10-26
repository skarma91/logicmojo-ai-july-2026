"""Milestone MCP server (solution): a NASA capability exposed over MCP.

The agent discovers this over the protocol (class 5.5) via mcp_bridge.py. Each
@mcp.tool() function's name, docstring, and typed signature become the tool schema
the agent reads.

Install and run (the bridge launches this automatically):
    pip install "mcp<2" sentence-transformers numpy
    python server.py
"""

from __future__ import annotations
import json
import pathlib

from mcp.server.fastmcp import FastMCP

HERE = pathlib.Path(__file__).parent
FULL = HERE / "data" / "corpus.jsonl"
SAMPLE = HERE / "data" / "example_corpus.jsonl"
PATH = FULL if FULL.exists() else SAMPLE
DOCS = [json.loads(l) for l in PATH.read_text().splitlines() if l.strip()]
BY_TITLE = {d["title"].lower(): d for d in DOCS}

mcp = FastMCP("nasa-tools")


@mcp.tool()
def nasa_facts(title: str) -> str:
    """Return the NASA center and source URL for an image by its exact title."""
    d = BY_TITLE.get(title.strip().lower())
    if not d:
        return f"No item titled {title!r} in the catalog."
    return f"{d['title']}: released by NASA {d['section']}. Source: {d['url']}"


if __name__ == "__main__":
    mcp.run()   # stdio transport by default
