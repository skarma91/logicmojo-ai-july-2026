"""Class 6.6 build: trace the team, find the bottleneck, record feedback.

Runs the class 6.2 (Orchestration) team on a few questions with every step traced:

  run (chain)                         one trace per question
    researcher (chain)                one span per agent turn, parallel branches included
      llm:tax_researcher (llm)        every model call: tokens, cached tokens, cost
      search_tax_docs (retriever)     every retrieval
      mcp:tax_glossary (tool)         every MCP tool call
    merge, writer, reviewer (chain) ...

The spans go to traces/run_<time>.jsonl always, and to LangSmith as well when
LANGSMITH_TRACING=true and LANGSMITH_API_KEY are set (see the README for signup).
After the run, analyze.py reports the slowest and costliest spans, each span's
share of cost, and the critical path. A feedback score is recorded on each run.

    pip install langgraph "mcp<2" sentence-transformers numpy python-dotenv groq langsmith
    python main.py                      # trace two questions, analyze, record feedback
    python main.py --inject-failure     # make one MCP tool call fail, to practice reading an error span
    python analyze.py traces/sample_trace.jsonl   # offline: the pre-baked trace, no model calls
"""

from __future__ import annotations
import argparse
import functools
import json
import logging
import pathlib
import time

import llm
import analyze
from mcp_bridge import McpBridge
from tax_worker import build_tax_worker, search_tax_docs
from team import build_team, run, deterministic_checks
from tracing import Tracer, langsmith_enabled

HERE = pathlib.Path(__file__).parent
PRICE_IN, PRICE_OUT = 0.075 / 1_000_000, 0.30 / 1_000_000   # illustrative prices
CACHE_DISCOUNT = 0.5
QUESTIONS = ["Can I contribute to an HSA, and how much can I put in for 2025?",
             "What does AGI mean, and does a higher AGI limit my charitable deduction?"]


# ---------------------------------------------------------------------------
# Instrumentation: wrap the existing functions; the team code stays unchanged.
# ---------------------------------------------------------------------------
def instrument_llm(tracer: Tracer):
    """Wrap llm.chat and llm.chat_tools so every model call becomes an llm span."""
    def traced(fn, label):
        @functools.wraps(fn)
        def wrapper(messages, *args, agent=None, **kwargs):
            user = next((m.get("content") for m in reversed(messages) if m.get("role") == "user"), "")
            with tracer.span(f"llm:{agent or 'unattributed'}", "llm", inputs=user,
                             metadata={"call": label}) as s:
                n_before = len(llm.USAGE)
                out = fn(messages, *args, agent=agent, **kwargs)
                if len(llm.USAGE) > n_before:
                    u = llm.USAGE[-1]
                    s.tokens_in, s.tokens_out, s.cached = u["prompt_tokens"], u["completion_tokens"], u["cached_tokens"]
                    s.cost = ((s.tokens_in - s.cached) * PRICE_IN + s.cached * PRICE_IN * CACHE_DISCOUNT
                              + s.tokens_out * PRICE_OUT)
                    s.metadata["model"] = u["model"]
                text = out if isinstance(out, str) else (
                    out.text or ", ".join(f"{c.name}({json.dumps(c.args)[:80]})" for c in out.tool_calls))
                tracer.set_output(s, text)
                return out
        return wrapper
    llm.chat = traced(llm.chat, "chat")
    llm.chat_tools = traced(llm.chat_tools, "chat_tools")


def traced_search(tracer: Tracer):
    def search(query: str) -> str:
        with tracer.span("search_tax_docs", "retriever", inputs=query) as s:
            out = search_tax_docs(query)
            tracer.set_output(s, out)
            return out
    return search


def instrument_bridge(tracer: Tracer, bridge: McpBridge, inject_failure: bool):
    original = bridge.call
    state = {"failed_once": False}

    def call(name, args):
        with tracer.span(f"mcp:{name}", "tool", inputs=args) as s:
            if inject_failure and not state["failed_once"]:
                state["failed_once"] = True
                time.sleep(0.3)                      # a slow call that then times out
                raise TimeoutError(f"MCP tool {name} did not answer within the deadline (injected)")
            out = original(name, args)
            tracer.set_output(s, out)
            return out
    bridge.call = call


def node_wrapper(tracer: Tracer):
    def wrap(name, fn):
        @functools.wraps(fn)
        def node(state):
            label = f"{name}[{state['focus']}]" if isinstance(state, dict) and "focus" in state else name
            with tracer.span(label, "chain", inputs=state.get("question", "")) as s:
                out = fn(state)
                tracer.set_output(s, {k: v for k, v in out.items() if k in ("draft", "approved", "feedback")}
                                  or "ok")
                return out
        return node
    return wrap


# ---------------------------------------------------------------------------
# Feedback: a key, a score, and a comment attached to a run.
# ---------------------------------------------------------------------------
def record_feedback(tracer: Tracer, root, state: dict, path: pathlib.Path):
    failed = deterministic_checks(state["draft"], state["notes"])
    fb = {"trace_id": root.trace_id, "span_id": root.span_id, "langsmith_run_id": root.langsmith_run_id,
          "key": "passes_exact_checks", "score": 0 if failed else 1,
          "comment": "; ".join(failed) or "cites known sources, within the word limit"}
    with open(path, "a") as f:
        f.write(json.dumps(fb) + "\n")
    if tracer.remote and root.langsmith_run_id:
        from langsmith import Client
        Client().create_feedback(root.langsmith_run_id, fb["key"], score=fb["score"], comment=fb["comment"])
    return fb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inject-failure", action="store_true", help="make the first MCP tool call fail")
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s | %(name)s | %(message)s")

    trace_path = HERE / "traces" / f"run_{time.strftime('%Y%m%d_%H%M%S')}.jsonl"
    tracer = Tracer(trace_path)
    print(f"tracing to {trace_path.relative_to(HERE)}"
          + ("  and to LangSmith" if langsmith_enabled() else "  (LangSmith off: set LANGSMITH_TRACING=true)"))
    instrument_llm(tracer)

    feedback_path = HERE / "traces" / "feedback.jsonl"
    with McpBridge() as bridge:
        instrument_bridge(tracer, bridge, args.inject_failure)
        team = build_team(build_tax_worker(bridge, search_fn=traced_search(tracer)), wrap=node_wrapper(tracer))
        for q in QUESTIONS:
            llm.reset_usage()
            with tracer.span("run", "chain", inputs=q, metadata={"question": q}) as root:
                state = run(team, q)
                tracer.set_output(root, state["draft"])
            fb = record_feedback(tracer, root, state, feedback_path)
            print(f"\nQ: {q}\nA: {state['draft'][:300]}\nfeedback {fb['key']}={fb['score']} ({fb['comment']})")

    analyze.report(analyze.load(str(trace_path)))


if __name__ == "__main__":
    main()
