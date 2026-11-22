"""Run the research team on one question, traced, and print what it did (given).

    python main.py
    python main.py "How do multiple LLM agents improve factual accuracy?"

Prints the supervisor's route, the papers found, the review rounds, the answer, and
the model calls and tokens per agent. Every run is traced to traces/run_<time>.jsonl
(and to LangSmith when LANGSMITH_TRACING=true), and you can analyze the trace with
the class 6.6 (Observability and tracing) analyzer.
"""

from __future__ import annotations
import functools
import logging
import pathlib
import sys
import time

import controls
import gateway
import llm
import retriever
from team import build_team, run
from tracing import Tracer, langsmith_enabled

HERE = pathlib.Path(__file__).parent
DEFAULT_Q = "How can several LLM agents working together reduce hallucinations?"


def traced_team(tracer: Tracer):
    """The team with a span around every agent turn and every retrieval. Model calls get
    their own llm spans, with real token counts, from controls.instrument()."""
    def wrap(name, fn):
        @functools.wraps(fn)
        def node(state):
            with tracer.span(name, "chain", inputs=state.get("question", "")):
                return fn(state)
        return node

    def search(query):
        with tracer.span("search_papers", "retriever", inputs=query, metadata={"args": {"query": query}}) as s:
            hits = retriever.search(query, k=4)
            tracer.set_output(s, [f"{h['id']} ({h['score']:.2f})" for h in hits])
            return hits

    return build_team(wrap=wrap, search_fn=search)


def main():
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s | %(name)s | %(message)s")
    question = " ".join(sys.argv[1:]) or DEFAULT_Q
    n = retriever.load_index()
    tracer = Tracer(HERE / "traces" / f"run_{time.strftime('%Y%m%d_%H%M%S')}.jsonl")
    controls.instrument(tracer)
    team = traced_team(tracer)
    print(f"corpus: {n} papers ({retriever.corpus_path().name})"
          + ("; tracing to LangSmith too" if langsmith_enabled() else ""))

    llm.reset_usage()
    with tracer.span("run", "chain", inputs=question) as root:      # inputs are PII-redacted; metadata is not
        result = gateway.handle(lambda q: run(team, q), question)
        tracer.set_output(root, result["answer"])

    print(f"\nQ: {question}")
    print(f"route:   {' > '.join(result.get('route', [])) or '-'}")
    print(f"papers:  {[p['id'] for p in result['papers']] or 'none'}")
    print(f"reviews: {result.get('reviews', 0)}   outcome: {result['outcome']}   pii: {result['pii'] or '-'}")
    print(f"\n{result['answer']}\n")
    print("model calls per agent")
    for agent, u in llm.usage_by_agent().items():
        print(f"  {agent:<14} calls={u['calls']:<3} in={u['prompt']:<6} out={u['completion']}")
    print(f"trace: {tracer.path.relative_to(HERE)}  (python ../../06-observability-and-tracing/code/analyze.py {tracer.path.relative_to(HERE)})")


if __name__ == "__main__":
    main()
