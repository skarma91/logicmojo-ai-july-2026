# Module 5 milestone: your single-agent product (NASA)

Assemble everything from Module 5 into one production-minded agent over a **NASA**
corpus (the NASA Image and Video Library, public domain). The agent searches NASA
image captions, looks up an image's facts over MCP, and saves an item to a collection
behind a human approval, all on a LangGraph backbone, and it reports its own cost.
Then you **evaluate** it, so the claim "it works" is backed by numbers.

This is a different domain from the class builds (IRS tax docs) and the
micro-assignments (movies), so it is a genuine transfer, not a re-run.

## How to approach this (read first)

1. **Get real data.** Run `python fetch_data.py` to download a few hundred NASA
   passages into `data/corpus.jsonl`. Everything falls back to the committed
   `data/example_corpus.jsonl` (18 real records) if you skip this, but the eval is
   only meaningful on the larger corpus, and so is the RAG-versus-fine-tuning
   intuition: a few dozen passages cannot settle it, a few hundred can.
2. **Assemble the agent.** Complete the five stubs (below) so the agent wraps
   retrieval, a second tool, an MCP capability, memory, and the guardrails.
3. **Run it.** `python agent.py` runs a real task end to end, pausing for approval
   before the risky save, and prints its cost-and-step summary.
4. **Evaluate it.** Build an eval set and score retrieval and answers (below). Report
   the numbers; they are the point of the milestone, not decoration.

## What you build (five stubs)

1. **`server.py`**: expose `nasa_facts(title)` as an `@mcp.tool()`. Return the NASA
   center and source URL for the image; handle a title not in the catalog.
2. **`route` (agent.py)**: the guardrail router. Stop (`END`) on the budget cap or a
   final answer; send a risky call to `"risky"`, a read-only call to `"tools"`.
3. **`build_app` (agent.py)**: compile with the checkpointer **and**
   `interrupt_before=["risky"]`, so the run pauses for approval before a side effect.
4. **`cost_summary` (agent.py)**: turn the run's counters into the cost-and-step line.
5. **`register_tools` (agent.py)**: add every MCP-discovered tool to the registry.

## The evaluation (required)

You will not trust an agent you have not measured. RAG does **not** train on the
corpus, so we do not hold documents out of the retriever. Instead `build_eval.py`
holds out a **20% slice of items** to write a fixed eval set (`data/eval.jsonl`):
answerable questions with a known gold passage, plus a few unanswerable ones to test
correct refusal.

```
python build_eval.py     # writes data/eval.jsonl (20% holdout + refusal probes)
python eval.py           # scores the assistant
```

`eval.py` reports, reusing class 4.8 (Evaluating LLM and RAG) and class 5.6 (agent
eval):

- **hit@1, hit@3** (no model needed): is the gold passage in the top-1 / top-3
  retrieved? This is retrieval quality, the foundation everything else stands on.
- **keyword recall** (needs a provider): does the grounded answer contain the
  expected fact? A cheap, deterministic outcome check.
- **faithfulness** (needs a provider): an LLM judge checks the answer is supported by
  the retrieved passages, not invented.
- **refusal accuracy** (needs a provider): on the unanswerable questions, does the
  assistant correctly decline?

Report these five numbers with your submission and say, in a sentence or two, what
they tell you: for example, high hit@3 but low faithfulness means retrieval is fine
and the prompt or model is the problem; low hit@3 means fix retrieval first.

## What ships here

```
assignment/
  agent.py            the agent, with five TODOs to complete
  server.py           an MCP server, with one tool for you to add
  nasa_retriever.py   dense retriever over the corpus (given)
  fetch_data.py       download the full NASA corpus (given)
  build_eval.py       build the eval set with a 20% holdout (given)
  eval.py             score retrieval + answers (given)
  memory.py, guardrails.py, mcp_bridge.py, llm.py   given
  data/example_corpus.jsonl   18 real NASA records, committed
solution/             worked reference for all of the above
```

## Run

```
pip install langgraph "mcp<2" sentence-transformers numpy python-dotenv requests
# plus your provider:  pip install groq   (set PROVIDER=groq in .env), or ollama
python fetch_data.py        # download the full corpus (recommended)
python agent.py             # run the assembled agent
python build_eval.py        # build the eval set
python eval.py              # score it
```

Set `PROVIDER` / keys in a `.env` at the repo root. Model calls are real; native tool
calling supports groq and ollama. The corpus is public-domain NASA content; each
record keeps its source URL so answers stay traceable.

## How this is checked

A complete reference is in `solution/`. Compare your `trajectory`, your `COST SUMMARY`,
your discovered-tools list, and your eval numbers to it.
