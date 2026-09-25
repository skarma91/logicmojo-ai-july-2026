# Class 5.2 code: reliable tool use with native tool calling

`main.py` runs a ReAct agent (think, act, observe) using the provider's **native
tool-calling API** through `llm.chat_tools`, with a three-tool registry and real
error handling.

## This upgrades the class 5.1 build (same agent, made reliable)

It is the same agent as class 5.1 (`01-what-is-an-agent/code/main.py`), not a new
topic. What changed:

| | 5.1 build | 5.2 build (here) |
| --- | --- | --- |
| Transport | hand-parse one JSON object from text (`llm.chat`) | native structured tool calls (`llm.chat_tools`) |
| Tools | one (the retriever) | three: read-only, a real API, a side-effect tool |
| Reliability | step cap only | validation, failure-recovery, parallel calls, `tool_choice`, idempotency |

The loop (think, act, observe, bounded by a step cap) is identical; 5.2 changes the
transport and adds the rails. See 5.1 for the from-scratch, any-provider version.

## Native tool calling vs the 5.1 manual loop

Class 5.1 hand-parsed a JSON reply (works with any provider). Here the model returns
**structured tool calls** and `llm.chat_tools` normalizes them into `ToolCall`
objects. The loop is the same; the transport is cleaner. `chat_tools` supports
**groq and ollama** (both OpenAI-style tool calling); for **gemini**, use the
manual-JSON loop from class 5.1 (`llm.chat`), which works with every provider.

## The three tools (chosen to show the trade-offs)

- `search_tax_docs` (the Module 4 retriever): **read-only**, safe to retry.
- `convert_currency`: a **real external API** (`frankfurter.dev`) with a timeout,
  so failures (timeout, non-200, unexpected payload) are real and handled.
- `save_note`: **side-effectful**, made **idempotent** with a key so a retried
  call never writes the same note twice (a class 5.6 preview).

## What makes it reliable

`dispatch()` validates the call (does the tool exist, are required args present),
runs it, and turns any failure into an **observation fed back to the model** rather
than a crash. The loop also handles a turn that requests **several tool calls at
once** (parallel calls), and treats every tool result as **data, not instructions**.
`run_agent` is bounded by `max_steps`.

## What ships here

```
main.py             the ReAct agent (run this)
llm.py              adds chat_tools() native tool calling + message helpers
retriever.py        the Module 4 retriever, as one tool
data/corpus.jsonl   the corpus the retriever searches
```

## Run

```
pip install sentence-transformers numpy requests python-dotenv
# plus your provider:  pip install groq   (set PROVIDER=groq in .env), or ollama
python main.py
```

`main.py` also prints the raw message list for one round-trip
(`user -> assistant(tool_calls) -> tool(result) -> assistant(final)`) so the
protocol is visible, not magic. Set `PROVIDER` / keys in a `.env` at the repo root.
