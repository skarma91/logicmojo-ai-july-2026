"""Score the research team on the holdout task set (class 6.7, Evaluating agentic systems).

    python eval.py --retrieval          # hit@1, hit@3 and the score gap; no model needed
    python eval.py                      # run every task live, traced, and score it
    python eval.py --limit 6            # the first 6 tasks (kinder to a free-tier quota)
    python eval.py --from-trace traces/eval_<time>.jsonl     # rescore a recorded run
    python eval.py --save-baseline      # pin this run as the baseline
    python eval.py --gate               # exit 1 if a task that passed in the baseline fails now

A task PASSES only when the answer succeeds AND the path is right: the reference steps
appear in order and no forbidden step ran. An answer that cites the right paper after
summarizing three unrelated ones is not a pass.

Metrics, in plain words:
  hit@k            the gold paper is among the top k papers the retriever returns
  path_ok          reference steps present in order, nothing forbidden
  step P / R       precision: share of steps taken that were required (low: wasted steps);
                   recall: share of required steps taken (low: skipped steps)
  success          the answer cites the gold id and mentions a key term, or, for an
                   unanswerable task, is exactly the refusal
  pass rate        passed / tasks, with a 95% Wilson interval, because 9 tasks is a small n
"""

from __future__ import annotations
import argparse
import json
import pathlib
import sys
import time
from collections import defaultdict

import evals
import retriever
from team import MIN_SCORE

HERE = pathlib.Path(__file__).parent
TASKS = HERE / "data" / "eval.jsonl"
BASELINE = HERE / "data" / "baseline.json"


def load_tasks() -> list[dict]:
    if not TASKS.exists():
        sys.exit("data/eval.jsonl not found: run  python build_eval.py  first")
    return [json.loads(l) for l in TASKS.read_text().splitlines() if l.strip()]


# ---------------------------------------------------------------------------
# TODO 5: score one task from its trace
# ---------------------------------------------------------------------------
def score_task(task: dict, spans: list[dict], answer: str) -> dict:
    """Turn one run's spans and final answer into a result row.

    Use evals.trajectory for (steps, calls). path_ok is True when the reference steps
    appear in order (evals.match_in_order) AND no forbidden step ran. Add step precision
    and recall, the tool-use report (with task["expected_args"]), and task success."""
    steps, calls = evals.trajectory(spans)
    forbidden = evals.forbidden_steps(steps, task["forbidden"])
    p, r = evals.step_precision_recall(steps, task["reference"])
    return {"steps": steps, "path_ok": evals.match_in_order(steps, task["reference"]) and not forbidden,
            "forbidden": forbidden, "precision": round(p, 2), "recall": round(r, 2),
            "tools": evals.tool_use_report(calls, task["expected_args"]),
            "success": evals.task_success(answer, task["success"])}


# ---------------------------------------------------------------------------
# Retrieval only: no model, so run this first
# ---------------------------------------------------------------------------
def retrieval_report(tasks: list[dict]):
    retriever.load_index()
    hits1 = hits3 = n = 0
    answerable_top, off_topic_top = [], []
    for t in tasks:
        top = retriever.search(t["question"], k=3)
        if t["kind"] == "answerable":
            ids = [h["id"] for h in top]
            n += 1
            hits1 += ids[:1] == [t["gold_id"]]
            hits3 += t["gold_id"] in ids
            answerable_top.append(top[0]["score"])
        else:
            off_topic_top.append(top[0]["score"])
        print(f"  {t['id']:<4} top={top[0]['id']} score={top[0]['score']:.2f}  {t['question'][:60]}")
    print(f"\nhit@1 {hits1}/{n}   hit@3 {hits3}/{n}")
    lo, hi = min(answerable_top), max(off_topic_top)
    print(f"lowest top score on an answerable task: {lo:.2f}; highest on an unanswerable one: {hi:.2f}; "
          f"MIN_SCORE is {MIN_SCORE:.2f}")
    if hi >= MIN_SCORE:
        print("  an off-topic question clears MIN_SCORE: the team will summarize unrelated papers for it")
    if lo < MIN_SCORE:
        print("  an answerable question falls below MIN_SCORE: the team will refuse a question it could answer")


# ---------------------------------------------------------------------------
# Live runs: one traced run per task, all in one trace file
# ---------------------------------------------------------------------------
def run_live(tasks: list[dict]) -> pathlib.Path:
    import controls
    import gateway
    import llm
    from main import traced_team
    from team import run
    from tracing import Tracer

    retriever.load_index()
    tracer = Tracer(HERE / "traces" / f"eval_{time.strftime('%Y%m%d_%H%M%S')}.jsonl")
    controls.instrument(tracer)
    team = traced_team(tracer)
    for t in tasks:
        llm.reset_usage()
        with tracer.span("run", "chain", inputs=t["question"], metadata={"task_id": t["id"]}) as root:
            result = gateway.handle(lambda q: run(team, q), t["question"])
            tracer.set_output(root, result["answer"])
            root.metadata.update(answer=result["answer"], outcome=result["outcome"])
        print(f"  ran {t['id']:<4} {result['outcome']:<10} {sum(u['calls'] for u in llm.usage_by_agent().values())} model calls")
    return tracer.path


def spans_by_task(path: pathlib.Path) -> dict[str, dict]:
    spans = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    by_trace = defaultdict(list)
    for s in spans:
        by_trace[s["trace_id"]].append(s)
    out = {}
    for group in by_trace.values():
        root = next((s for s in group if s["parent_id"] is None and "task_id" in s["metadata"]), None)
        if root:
            out[root["metadata"]["task_id"]] = {"root": root, "spans": [s for s in group if s is not root]}
    return out


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def score_all(tasks: list[dict], path: pathlib.Path) -> dict:
    runs = spans_by_task(path)
    results = {}
    for t in tasks:
        if t["id"] not in runs:
            continue
        r = runs[t["id"]]
        row = score_task(t, r["spans"], r["root"]["metadata"].get("answer", ""))
        llm_spans = [s for s in r["spans"] if s["kind"] == "llm"]
        row.update(kind=t["kind"], calls=len(llm_spans),
                   tokens=sum(s["tokens_in"] + s["tokens_out"] for s in llm_spans),
                   latency=round(r["root"]["end"] - r["root"]["start"], 2))
        results[t["id"]] = row
    return results


def report(results: dict):
    print(f"\n{'task':<5}{'kind':<13}{'path':<6}{'succ':<6}{'P':<6}{'R':<6}{'calls':<7}{'tokens':<8}{'pass':<6}steps")
    for tid, r in results.items():
        print(f"{tid:<5}{r['kind']:<13}{'ok' if r['path_ok'] else 'NO':<6}{'ok' if r['success'] else 'NO':<6}"
              f"{r['precision']:<6}{r['recall']:<6}{r['calls']:<7}{r['tokens']:<8}"
              f"{'PASS' if evals.passed(r) else 'fail':<6}{' > '.join(r['steps'])}")
    n = len(results)
    if not n:
        print("no scored tasks")
        return
    k_answer = sum(r["success"] for r in results.values())
    k_pass = sum(evals.passed(r) for r in results.values())
    lo, hi = evals.wilson_ci(k_pass, n)
    calls = [c for r in results.values() for c in [r["tools"]]]
    schema = sum(c["schema_ok"] for c in calls), sum(c["calls"] for c in calls)
    print(f"\nanswer-only success  {k_answer}/{n}")
    print(f"path-aware pass rate {k_pass}/{n} = {k_pass / n:.0%}  (95% Wilson interval {lo:.0%} to {hi:.0%})")
    refusals = [r for r in results.values() if r["kind"] == "unanswerable"]
    if refusals:
        print(f"correct refusals     {sum(r['success'] for r in refusals)}/{len(refusals)}")
    print(f"tool calls schema-valid {schema[0]}/{schema[1]}; "
          f"redundant searches {sum(c['redundant'] for c in calls)}")
    print(f"mean per task: {sum(r['calls'] for r in results.values()) / n:.1f} model calls, "
          f"{sum(r['tokens'] for r in results.values()) / n:.0f} tokens, "
          f"{sum(r['latency'] for r in results.values()) / n:.1f} s")


def main():
    ap = argparse.ArgumentParser(description="Evaluate the research team.")
    ap.add_argument("--retrieval", action="store_true", help="hit@k and score gap only (no model)")
    ap.add_argument("--limit", type=int, default=0, help="score only the first N tasks")
    ap.add_argument("--from-trace", type=pathlib.Path, help="rescore a recorded eval trace")
    ap.add_argument("--save-baseline", action="store_true")
    ap.add_argument("--gate", action="store_true", help="fail on regressions against the baseline")
    args = ap.parse_args()

    tasks = load_tasks()
    if args.limit:
        tasks = tasks[:args.limit]
    if args.retrieval:
        retrieval_report(tasks)
        return
    path = args.from_trace or run_live(tasks)
    results = score_all(tasks, path)
    report(results)
    print(f"trace: {path}")
    if args.save_baseline:
        BASELINE.write_text(json.dumps(results, indent=1))
        print(f"saved baseline: {len(results)} tasks")
    if args.gate:
        if not BASELINE.exists():
            sys.exit("no baseline yet: run with --save-baseline first")
        broken = evals.regressions(json.loads(BASELINE.read_text()), results)
        print(f"regression gate: {'FAIL ' + str(broken) if broken else 'pass'}")
        sys.exit(1 if broken else 0)


if __name__ == "__main__":
    main()
