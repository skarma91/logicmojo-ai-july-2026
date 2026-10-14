# Class 6.2 code: make the team coordinate

`main.py` puts four orchestration mechanics into one LangGraph team:

```
START --Send--> researcher[limits]      \
      --Send--> researcher[eligibility] -> merge -> writer -> reviewer --approve--> END
                                                      ^           |
                                                      +--revise---+  (at most 3 rounds)
```

1. **Subgraphs.** Each researcher is the class 5.7 (Assemble a single agent) agent (`tax_worker.py`), a compiled
   graph of its own, called from a team node. Its messages and tool calls stay private.
2. **Shared versus private state.** `TeamState` holds only what everyone needs. Reducers
   decide how parallel writes merge: `findings` appends (both researchers' results
   survive), `draft` overwrites (only the latest draft matters).
3. **Parallel fan-out and fan-in.** `Send` launches one researcher per focus at the same
   time; the `merge` node waits for both, then combines their findings.
4. **A bounded loop.** The writer drafts; the reviewer runs exact checks (word limit,
   citations that really appear in the notes) and an LLM verdict through a `verdict`
   tool (approve, or revise with specific feedback). Exact checks override the LLM. The
   loop ends on approval or after three rounds.

## What to watch when you run it

- **The timeline.** A text Gantt chart: the two researcher bars overlap, so they ran in
  parallel. The line under it compares the sum of their durations (sequential) with the
  span they actually took (parallel, about the maximum).
- **The review rounds.** Each round shows which exact checks failed and the reviewer's
  feedback, then the final draft.
- **Cost per agent.** Every model call is attributed to its agent (Module 6 `llm.py`).

## Free-tier note

Parallel branches send requests at the same moment, which is exactly what rate limits
punish. Two branches are comfortable on Groq's free plan (at time of writing about 30
requests and 8K tokens per minute for `gpt-oss-20b`); widen the fan-out carefully.
`llm.py` backs off and retries on HTTP 429.

## What ships here

```
main.py          the team: fan-out, merge, writer, reviewer loop, timeline, costs
tax_worker.py    the class 5.7 (Assemble a single agent) agent as a subgraph (reused from class 6.1, Multi-agent architectures)
llm.py           Module 6 wrapper: retry on 429, usage, per-agent attribution
retriever.py, guardrails.py, mcp_bridge.py, server.py, data/corpus.jsonl
```

## Run

```
pip install langgraph "mcp<2" sentence-transformers numpy python-dotenv groq
# set PROVIDER=groq and GROQ_API_KEY in .env (or PROVIDER=ollama)
python main.py
```
