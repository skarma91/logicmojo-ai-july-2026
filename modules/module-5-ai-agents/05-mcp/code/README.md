# Class 5.5 code: a custom MCP server and an agent client

Two sides of the Model Context Protocol, both built here:

- `server.py`: a custom **MCP server** (FastMCP, stdio) that wraps our functions as
  tools, `search_tax_docs` (the Module 4 retriever) and `tax_glossary` (a pure-Python
  lookup). Each `@mcp.tool()` function's name, docstring, and typed signature become
  the tool's schema automatically.
- `client.py`: the **agent as an MCP client**. It launches the server, does the
  handshake, **discovers** the tools, and calls them over the protocol, mapping each
  discovered tool into the same `chat_tools` spec from class 5.2 so the agent loop is
  unchanged.

## The protocol, concretely

MCP is JSON-RPC over a transport (stdio here). The client does `initialize`, then
`list_tools` (discovery), then `call_tool` (invocation). Swap in a different MCP
server and the agent uses its tools with no code change: that is the point.

## Scope

We build **tools** over **stdio**, at build-and-ship depth. MCP also defines
**resources** (read-only data) and **prompts** (reusable templates), and other
transports (HTTP and SSE); those are named in the slides as further scope. A server
you did not write is untrusted, and tool output is untrusted input (class 5.6).

## What ships here

```
server.py           the MCP server (tools over stdio)
client.py           the agent that discovers and calls them
llm.py              chat + chat_tools (from class 5.2)
retriever.py        the Module 4 retriever, used by one server tool
data/corpus.jsonl   the corpus the retriever searches
```

## Run

```
pip install "mcp<2" sentence-transformers numpy python-dotenv
# plus your provider:  pip install groq   (set PROVIDER=groq in .env), or ollama
python client.py        # the client launches server.py for you
```

Note the pin `mcp<2`: this targets the widely-documented FastMCP API. Set `PROVIDER`
/ keys in a `.env` at the repo root. Native tool calling targets groq and ollama; the
MCP layer itself is provider-independent.
