# Class 6.7 code: judge the path, not just the answer

`main.py` evaluates the class 6.1 (Multi-agent architectures) (Multi-agent architectures) team on ten tax tasks.
Each run is traced with the class 6.6 (Observability and tracing) (Observability and tracing) tracer, and its
**trajectory** (the worker-agent turns and tool calls it actually made) is read back
from the trace and scored three ways:

| What | How | Why |
|---|---|---|
| Path | in-order match against the task's reference steps (extras allowed), plus exact and set match for comparison; step precision and recall; forbidden steps | a right answer by a wrong path will not repeat |
| Tool use | every call's arguments against the tool schema; expected values (for example the glossary called with `AGI`); redundant calls | right tool, right arguments, no waste |
| Outcome | a concrete check per task: cites the right source id, or mentions the right fact | success you can verify, not a vibe |

A task **passes** only if the outcome succeeds **and** the path is right. Two cases are
flagged by name: *right answer, wrong path* (luck, or a model answering from memory
instead of the tools) and *wrong answer, right path* (a fixable bug in a step).

## The numbers, defined

- **Step precision** = steps both taken and required / steps taken. Low means wasted steps.
- **Step recall** = steps both taken and required / steps required. Low means skipped steps.
  Both count with multisets, so two searches against one required search scores one
  useful and one wasted.
- **Pass rate** with two 95% confidence intervals:
  **Wald**, $\hat p \pm 1.96\sqrt{\hat p(1-\hat p)/n}$, the textbook form, and
  **Wilson**, which stays sensible for small $n$ and rates near 0 or 1 (Wald claims zero
  uncertainty at 10 out of 10). With ten tasks the interval is wide: a one-task
  change is noise.
- **Answer only** is printed next to the pass rate: what an answer-only evaluation
  would have reported. The gap between them is what judging the path buys you.

## Regression gate

```
python main.py --save-baseline     # after a run you trust
python main.py --gate              # after a change: exit 1 if anything that passed now fails
```

Every task that passed in the baseline is pinned: if it stops passing, the gate fails
and names it. Put `--gate` in CI and a regression cannot merge quietly. The production
form of this is LangSmith **datasets and experiments**: the task set becomes a dataset,
each run of the team becomes an experiment, and these scores become evaluators you can
compare across experiments in the LangSmith UI.

## Offline, with the recorded traces

```
python main.py --from-trace traces/sample_eval_trace.jsonl
python main.py --from-trace traces/sample_regressed_trace.jsonl --baseline traces/sample_eval_trace.jsonl
```

Both traces were **recorded with a scripted model at authoring time**, and say so in
their metadata and in the report. The first includes one deliberate right-answer,
wrong-path case (t03 answers "What does AGI mean?" from a search plus the model's
memory instead of the glossary). The second is deliberately regressed (t01 and t04
skip retrieval, t08 adds an explainer nobody asked for), so the gate fails on exactly
those three. Re-record both from real runs before teaching.

## With LangSmith on

If `LANGSMITH_TRACING=true` and `LANGSMITH_API_KEY` are set (signup steps are in the
class 6.6 (Observability and tracing) README), every run is traced to LangSmith and its scores are attached as
feedback: `path_ok`, `task_success`, `step_precision`, `step_recall`.

## Free-tier note

Ten tasks take roughly 40 to 50 model calls. `--pace` (default 2 seconds between tasks)
keeps a live run inside Groq's free per-minute limits; `llm.py` still backs off on any
HTTP 429.

## Files

```
main.py          live or offline evaluation, report, baseline, gate, LangSmith feedback
evals.py         trajectory extraction, matching, precision and recall, tool checks, CIs, regressions
team.py          the class 6.1 (Multi-agent architectures) team, with a wrap() hook for tracing
data/tasks.jsonl ten tasks: question, reference steps, forbidden steps, expected arguments, success rule
traces/sample_eval_trace.jsonl, traces/sample_regressed_trace.jsonl   recorded (scripted) traces
tracing.py, pii.py, tax_worker.py, llm.py, retriever.py, guardrails.py, mcp_bridge.py, server.py, data/corpus.jsonl
```

## Run

```
pip install langgraph "mcp<2" sentence-transformers numpy python-dotenv groq langsmith
python main.py                   # live
python main.py --from-trace traces/sample_eval_trace.jsonl   # offline
```
