"""Class 6.6 micro-assignment: diagnose a failed run from its trace.

data/failed_trace.jsonl is the trace of one movie-team run that ended with no
answer. Each line is one span (the same format the class tracer writes):
span_id, parent_id, name, kind, start, end, tokens, cost, error, metadata.

The lines are in the order the spans FINISHED, as a real tracer writes them, so a
child appears before its parent. Nothing here calls a model: diagnosing from a trace
is reading data. Fill in every TODO and write DIAGNOSIS, then compare with the
README's expected output.

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
    # TODO(1): group spans by parent_id, sort each group by start; the root has parent_id None.
    raise NotImplementedError


# Problem 2
def failed_spans(spans: list[dict]) -> list[tuple[str, str]]:
    """(span_id, error) for every span with an error, in START order."""
    # TODO(2): every span with a non-empty error, as (span_id, error), sorted by start time.
    raise NotImplementedError


# Problem 3
def self_time(span: dict, kids: dict) -> float:
    """Duration minus the time covered by children."""
    # TODO(3): duration minus the time covered by the span's children. Check what happens
    #          when two children overlap in time, and handle it.
    raise NotImplementedError


# Problem 4
def cost_share(spans: list[dict]) -> list[tuple[str, float]]:
    """(span_id, share of the run's cost) for spans with a cost, in start order.
    share s_i = c_i / sum_j c_j."""
    # TODO(4): share of the run's total cost for each span that has a cost, rounded to 3 places.
    raise NotImplementedError


# Problem 5
def critical_path(span: dict, kids: dict) -> list[str]:
    """span_ids of the chain that set the wall time, below `span`.
    Start from the child that finished last; repeatedly step back to the child that
    finished last before the current one started; then recurse into each one."""
    # TODO(5): implement the walk described above, recursing into each chosen child.
    raise NotImplementedError


# Problem 6
def diagnose(spans: list[dict]) -> dict:
    """What went wrong, and what it cost.

    Return {"first_failure", "name", "error", "repeats", "wasted_cost", "wasted_time"}:
    first_failure: span_id of the failure that CAUSED the outcome
    repeats:       how many times a span with the same name and the same error ran
    wasted_cost:   cost of the model calls that could not have helped, rounded to 8 places
    wasted_time:   seconds from the end of the first failure to the end of the run
    """
    # TODO(6): build the dict described above. Decide which failure is the cause and which
    #          is only the symptom, and which model calls could not have helped.
    raise NotImplementedError


DIAGNOSIS = """TODO(6b): in three to five sentences, say which span failed, why, why the run
never recovered, and what you would change (in the tool, and in the supervisor)."""


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
