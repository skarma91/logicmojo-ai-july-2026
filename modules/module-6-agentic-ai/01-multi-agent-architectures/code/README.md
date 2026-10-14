# Class 6.1 code: one worker becomes a team

`main.py` turns the single agent from class 5.7 (Assemble a single agent) into a team: a **supervisor** plus two
**workers**, on LangGraph.

```
START -> supervisor -> tax_researcher -> supervisor -> explainer -> supervisor -> END
```

- **tax_researcher** is the class 5.7 (Assemble a single agent) agent, reused as a team member (`tax_worker.py`):
  the retriever from Module 4 (LLMs and GenAI), the MCP glossary (class 5.5, MCP), and the guardrails (class 5.6, Reliability and failure modes),
  minus the approval-gated write tool (a research worker only reads: least privilege).
  It is its own compiled graph with private state; the team sees only its findings.
- **explainer** rewrites the findings in plain language for a non-expert.
- **supervisor** decides who acts next by calling a `route` tool (native tool calling,
  `tool_choice="required"`, so it always returns a structured decision), then hands
  control over with LangGraph's `Command`. A hop cap stops it from looping forever.

## What to watch when you run it

1. **The routing log.** Each hop prints the supervisor's decision and reason.
2. **Routing includes not routing.** The AGI question needs no plain-language rewrite,
   so the supervisor finishes without calling the explainer.
3. **The per-agent cost table.** Calls, input and output tokens, cached tokens, and an
   illustrative cost per agent: the team-cost double sum from the slides, measured.
4. **The honest comparison.** The same task done by the single 5.7 agent alone. The team
   makes more calls; that overhead is only worth paying when a distinct role earns it.

## Module 6 `llm.py`

Same interface as Module 5 (AI Agents: `chat`, `chat_tools`), plus what a team needs:

- **Retry with backoff on HTTP 429.** Teams hit free-tier rate limits. `llm.py` waits
  (using the provider's `retry-after` header when present) and retries.
- **Usage on every call.** `prompt_tokens`, `completion_tokens`, `cached_tokens`.
- **Per-agent attribution.** Pass `agent="supervisor"` (or `with llm.tagged(...)`), then
  read `llm.usage_by_agent()`.

## Free-tier note

A team multiplies calls. At time of writing, Groq's free plan for `openai/gpt-oss-20b`
allows about 30 requests per minute, 1,000 per day, 8K tokens per minute, and 200K
tokens per day (check your console; limits change). One run of this file makes roughly
a dozen calls. If you hit limits, `llm.py` backs off and retries; Ollama is the
unlimited but slower fallback.

## What ships here

```
main.py          the team: supervisor + two workers, routing log, cost table, comparison
tax_worker.py    worker 1: the class 5.7 (Assemble a single agent) agent as a self-contained subgraph
llm.py           Module 6 wrapper: retry on 429, usage, per-agent attribution
retriever.py     the retriever from Module 4 (LLMs and GenAI)
guardrails.py    call validation and untrusted-output scanning (class 5.6, Reliability and failure modes)
mcp_bridge.py    synchronous bridge to the MCP server (class 5.7, Assemble a single agent)
server.py        the MCP server (search_tax_docs, tax_glossary)
data/corpus.jsonl   the IRS corpus
```

## Run

```
pip install langgraph "mcp<2" sentence-transformers numpy python-dotenv groq
# set PROVIDER=groq and GROQ_API_KEY in .env (or PROVIDER=ollama with `ollama serve`)
python main.py
```

The first run downloads the embedding model (about 90 MB) and caches the corpus
vectors; later runs reuse the cache.
