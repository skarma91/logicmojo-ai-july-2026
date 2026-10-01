"""Class 5.5 micro (starter): a movie MCP server. Add a new capability.

One tool (search_movies) is given. Add a NEW tool, then run client.py: it will
discover and use your new tool with no change to the client. That is the point of the
protocol.

Run (the client launches this automatically):
    pip install "mcp<2"
    python client.py     # not this file directly
"""

from __future__ import annotations
import json
import pathlib

from mcp.server.fastmcp import FastMCP

DATA = pathlib.Path(__file__).parent / "data" / "movies.jsonl"
MOVIES = [json.loads(line) for line in open(DATA)]
BY_TITLE = {m["title"].lower(): m for m in MOVIES}

mcp = FastMCP("movie-tools")


# Given: a search tool.
@mcp.tool()
def search_movies(query: str) -> str:
    """Find movies by what happens in them (their plot)."""
    words = [w for w in query.lower().split() if len(w) > 3]
    scored = [(sum(m["plot"].lower().count(w) for w in words), m) for m in MOVIES]
    scored = [x for x in scored if x[0] > 0] or [(0, m) for m in MOVIES[:3]]
    scored.sort(key=lambda x: -x[0])
    return "\n".join(f"[{m['id']}] {m['title']} ({m['year']}): {m['plot']}" for _, m in scored[:3])


# TODO: add a NEW MCP tool, movie_facts(title), decorated with @mcp.tool(). Use
# BY_TITLE to look up one movie and return its year, director, and genre. Give it a
# clear docstring (it becomes the description the agent reads). Handle a title that is
# not in the catalog. Then run client.py and watch it appear in "discovered MCP tools".


if __name__ == "__main__":
    mcp.run()
