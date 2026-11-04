# Module 6 milestone: a deployed arXiv research team

Assemble Module 6 (Agentic AI) into one system that answers research questions from a corpus of
**arXiv abstracts**. A supervisor runs three agents: a paper finder that searches the
corpus, a summarizer that writes a short cited answer, and a critic that sends the draft
back in a bounded review loop. The team sits behind a guardrail gateway, is traced end to
end, is measured by a trajectory evaluation on a 20% holdout, and ships as a live API on
Render's free tier.

This is a different domain from the class builds (IRS tax documents) and the
micro-assignments (movies), so it is a genuine transfer, not a re-run.

```
question -> gateway (rules, PII redaction) -> supervisor
supervisor -> paper_finder (search_papers) -> supervisor
supervisor -> summarizer -> critic -> summarizer      (revise, at most MAX_REVIEWS rounds)
                            critic -> supervisor      (approved, or the cap reached)
supervisor -> FINISH -> output guard -> answer
```

| Piece | Comes from |
|---|---|
| supervisor with a structured `route` tool, workers, compact handoffs | class 6.1 (Multi-agent architectures) and class 6.3 (Context engineering) |
| critic loop with exact checks and a review cap | class 6.2 (Orchestration) |
| rules, PII redaction, output guard, redacted audit log | class 6.4 (Guardrails, routing, and compliance) |
| spans for every agent turn, retrieval, and model call; LangSmith optional | class 6.6 (Observability and tracing) |
| trajectory matching, step precision and recall, Wilson interval, regression gate | class 6.7 (Evaluating agentic systems) |
| FastAPI, SSE, rate limit, cache, model routing, budget, Docker, Render | class 6.8 (Deployment) |

## How to approach this (read first)

1. **Get real data.** `python fetch_data.py` downloads about 300 abstracts on agents,
   RAG, reasoning, and LLM evaluation from the arXiv API into `data/corpus.jsonl`,
   waiting three seconds between requests as arXiv asks. Skip it and everything falls
   back to the committed `data/example_corpus.jsonl` (20 real papers, from AutoGen and
   ReAct to SWE-bench), but the eval only means something on the larger corpus.
2. **Measure retrieval before any model call.** `python build_eval.py` then
   `python eval.py --retrieval`. This prints hit@1 and hit@3 and, more importantly, the
   gap between the lowest top score on an answerable task and the highest on an
   off-topic one. `MIN_SCORE` must sit in that gap.
3. **Complete the five TODOs** (below), then run `python main.py` on one question.
4. **Evaluate**, pin a baseline, change one thing, and gate on it.
5. **Deploy** the same code to Render and call it.

## What you build (five TODOs)

1. **`route_guard` (team.py).** The supervisor model proposes the next step; the guard
   overrules routes that cannot work. FINISH at the hop cap; send a premature
   `summarizer` to `paper_finder` (or FINISH if the search already came back empty); send
   a premature `FINISH` with findings but no draft to `summarizer`.
2. **`exact_checks` (team.py).** The critic's checks that need no model, in order: over
   `MAX_WORDS` words; no arXiv id cited; cites an id the team never retrieved. An exact
   failure overrides a model approval.
3. **`after_critic` (team.py).** The bounded loop: back to the summarizer, or on to the
   supervisor when the draft is approved or `MAX_REVIEWS` rounds are used.
4. **`admit` (gateway.py).** Length and pattern rules on the raw text first, then PII
   redaction, so the model provider, the trace, and the audit log only ever see
   placeholders such as `<EMAIL_1>`.
5. **`score_task` (eval.py).** One run's spans and answer into a result row: the
   trajectory, `path_ok` (reference steps in order and no forbidden step), step precision
   and recall, the tool-use report, and task success.

Two decisions need reasoning, not code:

- **Choose `MIN_SCORE` from evidence.** Set it from the `--retrieval` gap on your
  corpus, not by guessing. Say what goes wrong on each side: too low and the team
  summarizes unrelated papers for an off-topic question; too high and it refuses a
  question it could answer. If the two ranges overlap, say which error you would rather
  make for a research assistant, and why.
- **Find the regression your change caused.** After `--save-baseline`, change one thing
  (for example set `MAX_REVIEWS = 0`, or lower `MIN_SCORE` to 0.1) and run `--gate`.
  Name the tasks that broke and explain, from their trajectories, why the answer-only
  score could not have caught it.

## The evaluation (required)

The team retrieves; it does not train on the corpus, so no paper leaves the index.
`build_eval.py` holds out a **20% slice of papers** to write a fixed task set
(`data/eval.jsonl`), plus five off-topic questions the corpus cannot answer.

| Task kind | Reference path | Forbidden | Success |
|---|---|---|---|
| answerable | paper_finder > tool:search_papers > summarizer > critic | none | cites the gold arXiv id and mentions a key term from its abstract |
| unanswerable | paper_finder > tool:search_papers | summarizer | the answer is exactly the team's no-results message |

```
python build_eval.py               # data/eval.jsonl
python eval.py --retrieval         # hit@1, hit@3, score gap (no model)
python eval.py                     # live run of every task, traced, then scored
python eval.py --save-baseline
python eval.py --gate              # exits 1 if a task that passed in the baseline fails now
python eval.py --from-trace traces/eval_<time>.jsonl     # rescore a recorded run, no model
```

What each number means:

- **hit@k**: the gold paper is among the top k papers the retriever returns.
- **path_ok**: the reference steps appear in order and no forbidden step ran.
- **step precision**: the share of steps taken that were required (low means wasted
  steps). **Step recall**: the share of required steps taken (low means skipped steps).
- **answer-only success** versus **path-aware pass rate**: a task passes only when the
  answer succeeds **and** the path is right. The gap between the two is the point of a
  trajectory eval.
- **pass rate with a 95% Wilson interval**: about a dozen tasks is a small sample, and
  the interval says how little a single run proves.
- **cost per task**: model calls, tokens, and seconds, read from the trace.

A live run costs about 7 model calls per answerable task and 4 per off-topic one. On
Groq's free tier (about 30 requests a minute and 1,000 a day for `openai/gpt-oss-20b`)
the full set on a 300-paper corpus fits in a day's quota; use `--limit` while you
iterate, and `--from-trace` to rescore without spending calls.

## Cost controls and the deploy

`service.py` (given) wraps the same team and gateway as `main.py`. Every request passes,
in order, a per-client token bucket (429 with `Retry-After`), the response cache (keyed on
the redacted, normalized question), a concurrency limit, a model choice (`gpt-oss-120b`
only for comparison questions), a per-request budget of model calls and prompt tokens,
and one trace. `POST /chat` returns JSON with the cited papers; `POST /chat/stream`
sends one Server-Sent Event per agent turn; `GET /health` makes no model call.

```
uvicorn service:app --reload
curl -s localhost:8000/chat -H 'content-type: application/json' \
     -d '{"question": "How do several LLM agents debating improve factual accuracy?"}'
```

To deploy: push your fork, then in the Render Dashboard choose **New > Blueprint** with
the path `modules/module-6-agentic-ai/milestone-assignment/assignment/render.yaml`, and
paste `GROQ_API_KEY` when asked. The Dockerfile bakes the FastEmbed model and the corpus
vectors into the image, so the 512 MB free instance only embeds queries. `plan: free` is
written out on purpose: an omitted plan defaults to a paid instance. The free service
sleeps after 15 minutes idle and takes up to a minute to wake.

## Report with your work

1. Your corpus size, hit@1 and hit@3, the score gap, and the `MIN_SCORE` you chose, with
   one sentence of reasoning.
2. Answer-only success, path-aware pass rate with its Wilson interval, correct refusals,
   and mean model calls and tokens per task.
3. The regression you introduced, the tasks the gate flagged, and why.
4. Your live URL, one `/chat` response, and its trace (a LangSmith link if you enabled
   it, or the matching lines of `traces/`).

## What ships here

```
assignment/
  team.py          the research team, with TODOs 1 to 3
  gateway.py       rules, redaction, output guard, audit log, with TODO 4
  eval.py          trajectory evaluation and regression gate, with TODO 5
  main.py          run one question, traced (given)
  build_eval.py    the 20% holdout task set (given)
  fetch_data.py    the arXiv API fetcher, one request every three seconds (given)
  retriever.py     FastEmbed dense retrieval with cached vectors (given)
  service.py       the FastAPI service with cost controls (given)
  llm.py, tracing.py, pii.py, evals.py, controls.py    from the Module 6 (Agentic AI) classes (given)
  Dockerfile, .dockerignore, requirements.txt, render.yaml
  data/example_corpus.jsonl    20 real arXiv papers, committed
solution/          the worked reference for all of the above
```

## Run

```
pip install langgraph groq fastembed numpy python-dotenv langsmith fastapi uvicorn
python fetch_data.py         # the full corpus (recommended)
python build_eval.py
python eval.py --retrieval
python main.py               # one question, traced
python eval.py               # the full evaluation
```

Set `PROVIDER` and keys in a `.env` at the repo root. Model calls are real; native tool
calling supports groq and ollama. Set `LANGSMITH_TRACING=true` and `LANGSMITH_API_KEY`
to send traces to LangSmith as well; local JSONL traces are always written. arXiv
metadata, abstracts included, is released under CC0, and every record keeps its
`https://arxiv.org/abs/<id>` URL so each answer traces back to its source.

## How this is checked

A reference solution is in the `solution/` folder. Compare your output to it: the route
and review count from `main.py`, the eval table, and the gate result.
