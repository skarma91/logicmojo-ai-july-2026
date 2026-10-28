# Class 6.6 code: trace the team, find the bottleneck, record feedback

`main.py` runs the class 6.2 (Orchestration) team (two parallel researchers, a writer,
a reviewer loop) with every step traced, then analyzes the trace:

```
run (chain)                          one trace per question
  researcher[contribution limits]    one span per agent turn (parallel branches nest correctly)
    llm:tax_researcher (llm)         every model call: tokens, cached tokens, cost
    mcp:tax_glossary (tool)          every MCP tool call
    search_tax_docs (retriever)      every retrieval
  merge, writer, reviewer ...
```

Spans are always written to `traces/run_<time>.jsonl`. With LangSmith switched on,
they are also sent to LangSmith, nested the same way.

## Files

```
main.py           instruments the team, runs it, records feedback, prints the analysis
tracing.py        the tracer: span tree, local JSONL, optional LangSmith, redaction before recording
analyze.py        offline analysis of any trace file: tree, self time, cost share, critical path, errors
team.py           the class 6.2 (Orchestration) team, with a wrap() hook so every node gets a span
pii.py            the class 6.4 (Guardrails, routing, and compliance) redactor
traces/sample_trace.jsonl   a pre-baked trace (see below)
tax_worker.py, llm.py, retriever.py, guardrails.py, mcp_bridge.py, server.py, data/corpus.jsonl
```

## Set up LangSmith (free, about five minutes)

1. Go to [smith.langchain.com](https://smith.langchain.com) and sign up with Google,
   GitHub, or email. No credit card is needed for the free plan.
2. Open **Settings**, then **API Keys**, then **Create API Key**. Copy the key; it is
   shown once.
3. Add these lines to the `.env` file at the project root, next to your Groq key:

   ```
   LANGSMITH_TRACING=true
   LANGSMITH_API_KEY=<your key>
   LANGSMITH_PROJECT=genai-course-6-6
   ```

4. **Only if your account is outside the US region**, also set the endpoint for your
   region, with no trailing slash, for example
   `LANGSMITH_ENDPOINT=https://eu.api.smith.langchain.com`. Without it, a non-US key
   fails to authenticate. The other regions are listed in the LangSmith docs.
5. `pip install langsmith`, run `python main.py`, then open LangSmith, go to
   **Tracing**, and pick your project. Each question is one trace.

On the free plan, traces are kept for 14 days at time of writing, and a monthly trace
allowance applies. Check the pricing page for current limits.

LangGraph also reports its own runs to LangSmith once `LANGSMITH_TRACING=true` is set,
so you will see the graph's steps next to the spans this tracer adds for model calls,
tools, and retrieval.

## What the analysis shows

- **The tree:** for every span, total time, **self time** (its duration minus the time
  covered by its children, with parallel children merged so they are not subtracted
  twice), tokens, and cost.
- **The slowest span** by self time: where waiting actually happens.
- **The costliest span** and the **share of cost** per span name,
  $s_i = c_i / \sum_j c_j$. The largest share is where tuning pays.
- **The critical path:** the chain of spans that set the wall time. With parallel
  branches the wall time is shorter than the sum of all spans, and only spans on this
  chain make the run faster when you speed them up.
- **Errors**, each with its path from the root.

## Feedback

Each run gets a feedback record: a key (`passes_exact_checks`), a score (1 or 0), and
a comment (which checks failed). It is written to `traces/feedback.jsonl` and, with
LangSmith on, attached to the run with `Client.create_feedback`. Class 6.7 (Evaluating
agentic systems) builds on exactly this.

## Practice reading a failure

`python main.py --inject-failure` makes the first MCP tool call time out. The worker
recovers (the class 5.6 (Reliability and failure modes) guardrails turn the exception
into an error message for the model), and the trace marks the failed span as an error in
LangSmith and as `ERROR` in the local analysis, on the critical path of its branch.

## Privacy

A trace is a copy of your data sent to a third party. `tracing.py` passes every
recorded input and output through the class 6.4 (Guardrails, routing, and compliance) redactor first, then truncates it to
600 characters. Nothing raw reaches the file or LangSmith.

## The pre-baked trace

`traces/sample_trace.jsonl` lets the analysis run with no network and no keys:

```
python analyze.py traces/sample_trace.jsonl
```

It was **recorded with a scripted model and simulated latency at authoring time**, so
its timings and token counts are illustrative, and every root span says so in its
metadata. Before teaching, replace it with a trace from a real run (copy one of your
`traces/run_*.jsonl` files over it).

## Run

```
pip install langgraph "mcp<2" sentence-transformers numpy python-dotenv groq langsmith
python main.py                    # two questions, traced; analysis and feedback at the end
python main.py --inject-failure   # one MCP call times out
python analyze.py traces/sample_trace.jsonl
```

About 18 model calls for the two questions, well inside Groq's free tier.
