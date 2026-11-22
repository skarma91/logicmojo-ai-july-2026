"""Class 6.6 micro-assignment (reference solution): diagnose a failed run from its trace.

data/failed_trace.jsonl is the trace of one movie-team run that ended with no
answer. Each line is one span (the same format the class tracer writes):
span_id, parent_id, name, kind, start, end, tokens, cost, error, metadata.

The lines are in the order the spans FINISHED, as a real tracer writes them, so a
child appears before its parent. Nothing here calls a model: diagnosing from a trace
is reading data.

Run:  python assignment.py
"""

from __future__ import annotations
import json
import pathlib
from collections import defaultdict

TRACE = pathlib.Path(__file__).parent / "data" / "failed_trace.jsonl"
EPS = 1e-6


def load(path=TRACE) -> list[dict]:
    return [json.loads(l) for l in open(path) if l.strip()]


def duration(s: dict) -> float:
    return round(s["end"] - s["start"], 6)


# Problem 1
def build_tree(spans: list[dict]) -> tuple[dict, dict]:
    """(root span, {parent_id: [children sorted by start]})."""
    kids = defaultdict(list)
    for s in spans:
        kids[s["parent_id"]].append(s)
    for v in kids.values():
        v.sort(key=lambda s: s["start"])
    return kids[None][0], dict(kids)


# Problem 2
def failed_spans(spans: list[dict]) -> list[tuple[str, str]]:
    """(span_id, error) for every span with an error, in START order."""
    return [(s["span_id"], s["error"]) for s in sorted(spans, key=lambda s: s["start"]) if s["error"]]


# Problem 3
def self_time(span: dict, kids: dict) -> float:
    """Duration minus the time covered by children. Parallel children overlap, so
    merge their intervals first; otherwise overlapping time is subtracted twice."""
    covered, cur = 0.0, None
    for a, b in sorted((c["start"], c["end"]) for c in kids.get(span["span_id"], [])):
        if cur is None or a > cur[1]:
            if cur:
                covered += cur[1] - cur[0]
            cur = [a, b]
        else:
            cur[1] = max(cur[1], b)
    if cur:
        covered += cur[1] - cur[0]
    return round(max(0.0, duration(span) - covered), 6)


# Problem 4
def cost_share(spans: list[dict]) -> list[tuple[str, float]]:
    """(span_id, share of the run's cost) for spans with a cost, in start order.
    share s_i = c_i / sum_j c_j."""
    total = sum(s["cost"] for s in spans)
    return [(s["span_id"], round(s["cost"] / total, 3))
            for s in sorted(spans, key=lambda s: s["start"]) if s["cost"] > 0]


# Problem 5
def critical_path(span: dict, kids: dict) -> list[str]:
    """span_ids of the chain that set the wall time, below `span`.
    Start from the child that finished last; repeatedly step back to the child that
    finished last before the current one started; then recurse into each one."""
    ch = kids.get(span["span_id"], [])
    if not ch:
        return []
    chain = [max(ch, key=lambda c: c["end"])]
    while True:
        before = [c for c in ch if c["end"] <= chain[0]["start"] + EPS]
        if not before:
            break
        chain.insert(0, max(before, key=lambda c: c["end"]))
    out = []
    for c in chain:
        out += [c["span_id"]] + critical_path(c, kids)
    return out


# Problem 6
def diagnose(spans: list[dict]) -> dict:
    """What went wrong, and what it cost.

    first_failure: the earliest failed span below the root (the root's own error is
                   only the symptom: it failed because its children did)
    repeats:       how many times a span with the same name and the same error ran
    wasted_cost:   cost of every supervisor model call AFTER the first failure; with
                   the error unchanged, each retry could only fail the same way
    wasted_time:   wall time from the end of the first failure to the end of the run
    """
    root, _ = build_tree(spans)
    fails = [s for s in sorted(spans, key=lambda s: s["start"]) if s["error"] and s is not root]
    first = fails[0]
    repeats = sum(1 for s in fails if s["name"] == first["name"] and s["error"] == first["error"])
    wasted = sum(s["cost"] for s in spans if s["kind"] == "llm" and s["start"] > first["end"])
    return {"first_failure": first["span_id"], "name": first["name"], "error": first["error"],
            "repeats": repeats, "wasted_cost": round(wasted, 8),
            "wasted_time": round(root["end"] - first["end"], 3)}


DIAGNOSIS = """The facts tool failed because the title is ambiguous (two films are called The
Italian Job), and the supervisor kept calling it with the same input until the hop
cap, so the run ended with no answer while each retry cost more than the last (the
context grows every hop). Fix the input, not the loop: let facts accept a year or
return all matches (class 6.1, Multi-agent architectures, did the latter), and stop
retrying a tool call that already failed with the same arguments."""


def main():
    spans = load()
    root, kids = build_tree(spans)
    print("[1] root:", root["name"], "| children of root:", [c["span_id"] for c in kids[root["span_id"]]])
    print("[2] failed spans:", failed_spans(spans))
    print("[3] self time: root", self_time(root, kids), "| supervisor hop 1",
          self_time(next(s for s in spans if s["span_id"] == "s1"), kids))
    print("[4] cost share:", cost_share(spans))
    print("[5] critical path:", critical_path(root, kids))
    print("[6] diagnosis:", diagnose(spans))
    print("\n" + DIAGNOSIS)


if __name__ == "__main__":
    main()
