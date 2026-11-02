"""Trajectory, tool-use, and task-success evaluation, as plain functions.

Everything here is deterministic and model-free, so it can be unit tested and run on
any recorded trace. main.py feeds it the spans of real team runs.

A TRAJECTORY is the ordered list of work steps a run took: worker-agent turns
("tax_researcher", "explainer") and tool calls ("tool:search_tax_docs",
"tool:tax_glossary"). Supervisor turns are routing, not work, so they are left out;
the work they caused is what we judge.
"""

from __future__ import annotations
import json
import math
import re
from collections import Counter

CITE = re.compile(r"\[([a-z0-9]+(?:-[a-z0-9]+)+)\]")
WORKER_KINDS = {"chain"}
SKIP = {"run", "supervisor"}


# ---------------------------------------------------------------------------
# From a trace to a trajectory
# ---------------------------------------------------------------------------
def trajectory(spans: list[dict]) -> tuple[list[str], list[dict]]:
    """(steps, tool_calls) for one trace, in start order.

    steps:      worker-agent turns and "tool:<name>" entries
    tool_calls: [{"tool": name, "args": {...}}] read from span metadata
    """
    steps, calls = [], []
    for s in sorted(spans, key=lambda s: s["start"]):
        if s["kind"] == "tool" or s["kind"] == "retriever":
            name = s["name"].removeprefix("mcp:")
            steps.append(f"tool:{name}")
            calls.append({"tool": name, "args": s["metadata"].get("args", {}), "error": s["error"]})
        elif s["kind"] in WORKER_KINDS and s["name"] not in SKIP:
            steps.append(s["name"])
    return steps, calls


# ---------------------------------------------------------------------------
# Matching a trajectory against a reference
# ---------------------------------------------------------------------------
def match_exact(actual: list[str], reference: list[str]) -> bool:
    """Same steps, same order, nothing extra. Strict, and brittle for agents."""
    return actual == reference


def match_in_order(actual: list[str], reference: list[str]) -> bool:
    """Every reference step appears in `actual`, in the same order; extra steps are
    allowed. The usual choice for agents: required steps, flexible details."""
    it = iter(actual)
    return all(any(step == a for a in it) for step in reference)


def match_set(actual: list[str], reference: list[str]) -> bool:
    """Every reference step appears somewhere, in any order."""
    return set(reference) <= set(actual)


def step_precision_recall(actual: list[str], reference: list[str]) -> tuple[float, float]:
    """Multiset precision and recall.

    precision = steps in both / steps taken      (low: wasted steps)
    recall    = steps in both / steps required   (low: skipped steps)
    Counting with multisets means two searches against one required search counts one
    as useful and one as waste.
    """
    both = sum((Counter(actual) & Counter(reference)).values())
    p = both / len(actual) if actual else (1.0 if not reference else 0.0)
    r = both / len(reference) if reference else 1.0
    return p, r


def forbidden_steps(actual: list[str], forbidden: list[str]) -> list[str]:
    return [s for s in forbidden if s in actual]


# ---------------------------------------------------------------------------
# Tool-use correctness
# ---------------------------------------------------------------------------
TOOL_SCHEMAS = {"search_tax_docs": {"query": str}, "tax_glossary": {"term": str}}


def schema_ok(call: dict) -> bool:
    """Required arguments present, with the right types, and non-empty."""
    schema = TOOL_SCHEMAS.get(call["tool"])
    if schema is None:
        return False
    return all(isinstance(call["args"].get(k), t) and call["args"].get(k) for k, t in schema.items())


def args_ok(call: dict, expect: dict) -> bool:
    """Expected values: {"arg": {"equals": "AGI"}} (case-insensitive) or
    {"arg": {"contains_any": ["hsa", "health savings"]}}."""
    for arg, rule in expect.items():
        value = str(call["args"].get(arg, "")).lower()
        if "equals" in rule and value != rule["equals"].lower():
            return False
        if "contains_any" in rule and not any(w.lower() in value for w in rule["contains_any"]):
            return False
    return True


def tool_use_report(calls: list[dict], expected_args: dict) -> dict:
    """Schema validity for every call; expected values for the tools a task names;
    redundant calls (same tool, same arguments, twice)."""
    seen, redundant = set(), 0
    for c in calls:
        key = (c["tool"], json.dumps(c["args"], sort_keys=True))
        redundant += key in seen
        seen.add(key)
    checked = [c for c in calls if c["tool"] in expected_args]
    return {"calls": len(calls),
            "schema_ok": sum(schema_ok(c) for c in calls),
            "args_ok": sum(args_ok(c, expected_args[c["tool"]]) for c in checked),
            "args_checked": len(checked),
            "redundant": redundant}


# ---------------------------------------------------------------------------
# Task success: a concrete, checkable outcome
# ---------------------------------------------------------------------------
def task_success(answer: str, success: dict) -> bool:
    """{"cites_any": [ids]} and/or {"mentions_any": [phrases]}: all given rules must hold."""
    a = answer.lower()
    if "cites_any" in success and not set(CITE.findall(answer)) & set(success["cites_any"]):
        return False
    if "mentions_any" in success and not any(m.lower() in a for m in success["mentions_any"]):
        return False
    return True


# ---------------------------------------------------------------------------
# Honest numbers: confidence intervals for a pass rate
# ---------------------------------------------------------------------------
def wald_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """p_hat +/- z * sqrt(p_hat (1 - p_hat) / n). Simple; misleading near 0 or 1 and
    for small n (at k = n it claims zero uncertainty)."""
    p = k / n
    h = z * math.sqrt(p * (1 - p) / n)
    return max(0.0, p - h), min(1.0, p + h)


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval: better behaved for small n and rates near 0 or 1."""
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


# ---------------------------------------------------------------------------
# Regression gate
# ---------------------------------------------------------------------------
def passed(result: dict) -> bool:
    """A task passes only if the answer succeeds AND the path is right."""
    return result["success"] and result["path_ok"]


def regressions(baseline: dict, current: dict) -> list[str]:
    """Task ids that passed in the baseline and do not pass now. Pinned behaviour:
    anything the system used to get right must stay right."""
    return sorted(t for t, r in baseline.items() if passed(r) and t in current and not passed(current[t]))
