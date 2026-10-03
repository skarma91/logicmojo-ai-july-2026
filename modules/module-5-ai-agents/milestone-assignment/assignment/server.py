"""Milestone MCP server (your work): expose one NASA capability over MCP.

The agent discovers this over the protocol (class 5.5) via mcp_bridge.py. A
@mcp.tool() function's name, docstring, and typed signature become the tool schema.

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


# TODO: expose a tool `nasa_facts(title: str) -> str`, decorated with @mcp.tool().
#       Look the title up in BY_TITLE and return the NASA center (the "section" field)
#       and the source URL. Handle a title not in the catalog. The docstring becomes
#       the description the agent reads, so make it clear.


if __name__ == "__main__":
    mcp.run()   # stdio transport by default
