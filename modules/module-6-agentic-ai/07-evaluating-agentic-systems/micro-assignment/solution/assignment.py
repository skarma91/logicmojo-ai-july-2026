"""Class 6.7 micro-assignment (reference solution): an eval that catches a regression.

A movie team (supervisor plus plot_search, facts, and recommender workers) was run on
eight tasks twice: data/baseline_runs.jsonl is a version everyone trusted;
data/candidate_runs.jsonl is a new version someone wants to ship. Each run records
the worker steps it took, its tool calls, and its answer. data/tasks.jsonl gives each
task's reference steps and success rule.

This solution scores both versions on path and outcome, adds a pinned rule ("a
director question must call facts before answering"), and gates the candidate.
No model calls: the runs are recorded, so every number is exact.

Run:  python assignment.py
"""

from __future__ import annotations
import json
import math
import pathlib
from collections import Counter

DATA = pathlib.Path(__file__).parent / "data"


def load(name: str) -> list[dict]:
    return [json.loads(l) for l in (DATA / name).read_text().splitlines() if l.strip()]


TASKS = {t["id"]: t for t in load("tasks.jsonl")}


def success(answer: str, rule: dict) -> bool:
    """Every phrase in mentions_all must appear in the answer (case-insensitive)."""
    return all(m.lower() in answer.lower() for m in rule["mentions_all"])


# Problem 1
def in_order(actual: list[str], reference: list[str]) -> bool:
    """Every reference step appears in actual, in the same order; extras allowed."""
    it = iter(actual)
    return all(any(step == a for a in it) for step in reference)


# Problem 2
def step_pr(actual: list[str], reference: list[str]) -> tuple[float, float]:
    """Multiset step precision and recall, rounded to 2 places. No steps taken at all
    gives precision 0.0 when steps were required (nothing taken was useful)."""
    both = sum((Counter(actual) & Counter(reference)).values())
    p = both / len(actual) if actual else (1.0 if not reference else 0.0)
    r = both / len(reference) if reference else 1.0
    return round(p, 2), round(r, 2)


# Problem 3
def director_rule(task: dict, run: dict) -> bool:
    """Pinned rule: a question that asks who directed a film must call the facts tool.
    Questions that do not ask about a director pass trivially."""
    if "directed" not in task["question"].lower():
        return True
    return any(c["tool"] == "facts" for c in run["tool_calls"])


# Problem 4
def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson 95% interval for a pass rate, rounded to 2 places."""
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return round(max(0.0, centre - half), 2), round(min(1.0, centre + half), 2)


def score(runs: list[dict]) -> dict:
    out = {}
    for run in runs:
        task = TASKS[run["task_id"]]
        out[run["task_id"]] = {
            "success": success(run["answer"], task["success"]),
            "path_ok": in_order(run["steps"], task["reference"]),
            "rule_ok": director_rule(task, run),
            "pr": step_pr(run["steps"], task["reference"]),
        }
    return out


def passed(r: dict) -> bool:
    return r["success"] and r["path_ok"] and r["rule_ok"]


# Problem 5
def gate(baseline: dict, candidate: dict) -> list[str]:
    """Task ids that passed in the baseline and fail in the candidate, sorted."""
    return sorted(t for t, r in baseline.items() if passed(r) and not passed(candidate[t]))


def summary(name: str, scored: dict):
    n = len(scored)
    k = sum(passed(r) for r in scored.values())
    ks = sum(r["success"] for r in scored.values())
    print(f"    {name:<9} pass {k}/{n}  wilson {wilson(k, n)}  |  answer-only {ks}/{n}")


def main():
    print("[1] in_order(['plot_search', 'facts'], ['facts']) ->", in_order(["plot_search", "facts"], ["facts"]),
          "| in_order(['facts', 'plot_search'], ['plot_search', 'facts']) ->",
          in_order(["facts", "plot_search"], ["plot_search", "facts"]))
    print("[2] step_pr(['recommender', 'recommender'], ['recommender']) ->", step_pr(["recommender", "recommender"], ["recommender"]),
          "| step_pr([], ['facts']) ->", step_pr([], ["facts"]))
    base, cand = score(load("baseline_runs.jsonl")), score(load("candidate_runs.jsonl"))
    print("[3] director rule broken in candidate:", sorted(t for t, r in cand.items() if not r["rule_ok"]))
    print("[4] wilson(8, 8) ->", wilson(8, 8), "| wilson(3, 8) ->", wilson(3, 8))
    summary("baseline", base)
    summary("candidate", cand)
    right_answer_wrong_path = sorted(t for t, r in cand.items() if r["success"] and not (r["path_ok"] and r["rule_ok"]))
    print("    right answer, wrong path in candidate:", right_answer_wrong_path)
    reg = gate(base, cand)
    print("[5] gate ->", "PASS" if not reg else f"FAIL {reg}")


if __name__ == "__main__":
    main()
