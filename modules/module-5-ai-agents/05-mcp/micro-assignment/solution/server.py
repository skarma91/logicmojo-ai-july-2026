"""Class 5.5 micro (reference solution): a movie MCP server with a new capability.

Starts with one tool (search_movies) and adds a NEW one (movie_facts). The point of
the micro: once you add a tool here, the agent discovers and uses it with NO change
to the client, that is what the protocol buys you.

Run (the client launches this automatically):
    pip install "mcp<2"
    python server.py
"""

from __future__ import annotations
import json
import pathlib

from mcp.server.fastmcp import FastMCP

DATA = pathlib.Path(__file__).parent.parent / "data" / "movies.jsonl"
MOVIES = [json.loads(line) for line in open(DATA)]
BY_TITLE = {m["title"].lower(): m for m in MOVIES}

mcp = FastMCP("movie-tools")


@mcp.tool()
def search_movies(query: str) -> str:
    """Find movies by what happens in them (their plot)."""
    words = [w for w in query.lower().split() if len(w) > 3]
    scored = [(sum(m["plot"].lower().count(w) for w in words), m) for m in MOVIES]
    scored = [x for x in scored if x[0] > 0] or [(0, m) for m in MOVIES[:3]]
    scored.sort(key=lambda x: -x[0])
    return "\n".join(f"[{m['id']}] {m['title']} ({m['year']}): {m['plot']}" for _, m in scored[:3])


# The NEW capability. Adding it here is all it takes: the client discovers it.
@mcp.tool()
def movie_facts(title: str) -> str:
    """Return the year, director, and genre for one movie by its exact title."""
    m = BY_TITLE.get(title.strip().lower())
    if not m:
        return f"No movie titled {title!r} in the catalog."
    return f"{m['title']}: year={m['year']}, director={m['director']}, genre={m['genre']}"


if __name__ == "__main__":
    mcp.run()
