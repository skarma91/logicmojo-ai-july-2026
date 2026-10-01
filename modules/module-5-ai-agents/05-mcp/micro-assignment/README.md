# Class 5.5 micro-assignment: expose a new capability over MCP

The class build connected an agent to an MCP server. Here you experience the payoff:
add a **new tool** to your server and the agent discovers and uses it with **no change
to the client**. That is what the protocol buys you.

Work in `server.py` (the movie MCP server). `client.py` is given and needs no edits.
Native tool calling supports groq and ollama.

## The dataset

`data/movies.jsonl`: 18 movies, each with a title, year, genre, director, and plot.

## What to build

1. **A new MCP tool** in `server.py`: `movie_facts(title)`, decorated with
   `@mcp.tool()`. Look the movie up (use `BY_TITLE`) and return its year, director, and
   genre. Give it a clear docstring, that becomes the description the agent reads.
   Handle a title not in the catalog.
2. **Run `client.py`** (it launches your server). Confirm your new tool appears in the
   printed `discovered MCP tools` list, and that a "who directed X" question routes to
   it while a plot question still routes to `search_movies`.

## Expected behavior

`client.py` prints `discovered MCP tools: ['search_movies', 'movie_facts']` (order may
vary). The plot question uses `search_movies`; "Who directed Inception?" uses
`movie_facts` and returns its year, director, and genre. You changed only the server,
never the client: that is the whole point of MCP.

## How this is checked

A reference solution is in the `solution/` folder. Compare your discovered-tools list
and outputs to it.
