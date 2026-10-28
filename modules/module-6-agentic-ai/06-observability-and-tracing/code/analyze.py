"""Read a trace file (JSONL, one span per line) and find where time and money went.

Works offline on any trace written by tracing.py, including the pre-baked sample:

    python analyze.py traces/sample_trace.jsonl

For each trace (one root span = one end-to-end run) it prints:

  - the span tree, with total time, self time, tokens, and cost per span,
  - the slowest span by SELF time (time spent in the span itself, not its children),
  - the costliest span and each span's share of cost,  s_i = c_i / sum_j c_j,
  - the CRITICAL PATH: the chain of spans that set the wall time. With parallel
    branches, wall time is set by this chain, not by the sum of all span durations,
  - every span that recorded an error, with its path from the root.
"""

from __future__ import annotations
import json
import sys
from collections import defaultdict

EPS = 1e-6


def load(path: str) -> list[dict]:
    return [json.loads(l) for l in open(path) if l.strip()]


def by_trace(spans: list[dict]) -> dict:
    out = defaultdict(list)
    for s in spans:
        out[s["trace_id"]].append(s)
    return out


def children_of(spans: list[dict]) -> dict:
    kids = defaultdict(list)
    for s in spans:
        kids[s["parent_id"]].append(s)
    for v in kids.values():
        v.sort(key=lambda s: s["start"])
    return kids


def duration(s: dict) -> float:
    return s["end"] - s["start"]


def self_time(s: dict, kids: dict) -> float:
    """Time in the span itself: its duration minus the time covered by its children.
    Overlapping (parallel) children are merged first, so they are not subtracted twice."""
    intervals = sorted((c["start"], c["end"]) for c in kids.get(s["span_id"], []))
    covered, cur_s, cur_e = 0.0, None, None
    for a, b in intervals:
        if cur_e is None or a > cur_e:
            if cur_e is not None:
                covered += cur_e - cur_s
            cur_s, cur_e = a, b
        else:
            cur_e = max(cur_e, b)
    if cur_e is not None:
        covered += cur_e - cur_s
    return max(0.0, duration(s) - covered)


def critical_path(root: dict, kids: dict) -> list[dict]:
    """Walk back from the child that finished last; at each step take the child that
    finished last before the current one started (the one it was waiting for). Then
    recurse into each chosen child. The result is the chain that sets the wall time."""
    path = [root]
    ch = kids.get(root["span_id"], [])
    if not ch:
        return path
    chain = [max(ch, key=lambda c: c["end"])]
    while True:
        before = [c for c in ch if c["end"] <= chain[0]["start"] + EPS and c is not chain[0]]
        if not before:
            break
        chain.insert(0, max(before, key=lambda c: c["end"]))
    for c in chain:
        path += critical_path(c, kids)[0:]
    return path


def cost_share(spans: list[dict]) -> list[tuple[str, float, float]]:
    """(span name, cost, share) for every span with a cost, largest share first."""
    total = sum(s["cost"] for s in spans) or 1e-12
    rows = [(s["name"], s["cost"], s["cost"] / total) for s in spans if s["cost"] > 0]
    return sorted(rows, key=lambda r: -r[2])


def ancestry(s: dict, index: dict) -> list[str]:
    names = []
    while s is not None:
        names.insert(0, s["name"])
        s = index.get(s["parent_id"])
    return names


def print_tree(span: dict, kids: dict, depth: int = 0):
    st = self_time(span, kids)
    toks = f" tok={span['tokens_in']}+{span['tokens_out']}" if span["tokens_in"] else ""
    cost = f" ${span['cost']:.6f}" if span["cost"] else ""
    err = f"  ERROR: {span['error']}" if span["error"] else ""
    print(f"  {'  ' * depth}{span['name']:<{max(8, 40 - 2 * depth)}} {span['kind']:<9} "
          f"total={duration(span):6.2f}s self={st:6.2f}s{toks}{cost}{err}")
    for c in kids.get(span["span_id"], []):
        print_tree(c, kids, depth + 1)


def report(spans: list[dict]):
    index = {s["span_id"]: s for s in spans}
    for trace_id, tspans in by_trace(spans).items():
        kids = children_of(tspans)
        root = kids[None][0]
        print(f"\n=== trace {trace_id}: {root['name']}  ({root['metadata'].get('question', '')[:60]})")
        if root["metadata"].get("recorded_with"):
            print(f"    recorded with: {root['metadata']['recorded_with']}")
        print_tree(root, kids)

        leaves = [s for s in tspans if s is not root]
        slow = max(leaves, key=lambda s: self_time(s, kids))
        print(f"\n  slowest span (self time): {slow['name']} {self_time(slow, kids):.2f}s")
        shares = cost_share(tspans)
        if shares:
            name, c, sh = shares[0]
            print(f"  costliest span: {name} ${c:.6f} = {sh:.0%} of this run's cost")
            by_name = defaultdict(float)
            for n, c, sh in shares:
                by_name[n] += sh
            print("  cost share by span name: " +
                  ", ".join(f"{n} {v:.0%}" for n, v in sorted(by_name.items(), key=lambda kv: -kv[1])))
        cp = critical_path(root, kids)
        work = sum(duration(s) for s in tspans if not kids.get(s["span_id"]))
        print(f"  critical path ({duration(root):.2f}s wall): " + " > ".join(s["name"] for s in cp[1:]))
        print(f"  sum of all leaf spans: {work:.2f}s  (more than the wall time when branches run in parallel)")
        errors = [s for s in tspans if s["error"]]
        for e in errors:
            print(f"  error: {' > '.join(ancestry(e, index))}: {e['error']}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: python analyze.py <trace.jsonl>")
    report(load(sys.argv[1]))
