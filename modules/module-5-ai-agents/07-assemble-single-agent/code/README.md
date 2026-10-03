# Class 5.7 code: the assembled single agent

The Module 5 capstone. `main.py` puts every piece from the module onto one LangGraph
backbone and reports the run's cost, so the last thing you see is not just "it works"
but "here is what it cost".

## What is assembled

| Piece | How | From |
|---|---|---|
| Retrieval as a tool | `search_tax_docs` over the Module 4 retriever | class 4.5 |
| A second local tool | `file_request`, a **risky** side effect | class 5.2 |
| An MCP-exposed capability | `tax_glossary`, discovered over the protocol | class 5.5 |
| Long-term memory | `MemoryStore` (vector store), read at start + write at end | class 5.3 |
| Working memory | the LangGraph checkpointer persists the thread | class 5.4 |
| Guardrails | step cap, tool-call budget, validation, injection scan, approval | class 5.6 |
| Cost-and-step summary | steps, tool calls, tokens, illustrative dollars | class 5.7 |

## The MCP bridge

`mcp_bridge.py` embeds the async MCP client (class 5.5) inside this synchronous agent.
It owns one asyncio loop on a background thread and keeps a single session open for the
agent's lifetime, exposing plain `specs()` and `call(name, args)`. The MCP session's
task group must open and close in the same task, so one long-lived coroutine owns the
whole lifecycle and calls are marshaled onto it. Swap in a different MCP server and the
agent gains its tools with no change to the loop.

## The cost summary

Cost is `sum over steps of (input_tokens * price_in + output_tokens * price_out)`. The
input grows each step as memory, tool schemas, and prior turns accumulate, so a
ten-step task is not ten times one answer. `main.py` counts tokens (a rough estimate)
and prints the total with **illustrative** prices; replace `PRICE_IN` / `PRICE_OUT`
with your provider's real numbers.

## What ships here

```
main.py         the assembled agent + the cost-and-step summary
mcp_bridge.py   synchronous bridge to an MCP server
guardrails.py   Budget, scan_tool_output, validate_call (class 5.6)
memory.py       MemoryStore + Conversation (class 5.3)
server.py       the MCP server (search_tax_docs, tax_glossary) (class 5.5)
retriever.py    the Module 4 retriever
llm.py          chat + chat_tools
data/corpus.jsonl   the corpus the retriever searches
```

## Run

```
pip install langgraph "mcp<2" sentence-transformers numpy python-dotenv
# plus your provider:  pip install groq   (set PROVIDER=groq in .env), or ollama
python main.py
```

Model calls are real; native tool calling supports groq and ollama. For production, add
tracing (LangSmith or OpenTelemetry) and an agent evaluation set (trajectory and
outcome), both covered in class 5.6.
