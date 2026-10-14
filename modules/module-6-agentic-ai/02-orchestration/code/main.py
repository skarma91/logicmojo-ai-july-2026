"""Class 6.2 build: make the team coordinate.

Four orchestration mechanics, all in one graph:

  1. Subgraphs. Each researcher is the class 5.7 (Assemble a single agent) agent, a compiled graph of its own,
     called from a team node. Its messages and tool calls stay PRIVATE to it.
  2. Shared versus private state. The team state holds only what everyone needs:
     the question, the findings, the draft, the verdict. Reducers decide how
     parallel writes merge (findings APPEND; the draft OVERWRITES).
  3. Parallel fan-out and fan-in. Two researchers run AT THE SAME TIME via
     LangGraph's Send (one on contribution limits, one on eligibility), and an
     append reducer merges their findings.
  4. A bounded loop. A writer drafts, a reviewer returns a structured verdict
     (approve, or revise with specific feedback), and the draft goes back until
     it is approved or MAX_ROUNDS passes.

        START --Send--> researcher[limits]      \
              --Send--> researcher[eligibility] -> merge -> writer -> reviewer --approve--> END
                                                              ^           |
                                                              +--revise---+  (at most MAX_ROUNDS)

Run (native tool calling: groq or ollama):
    pip install langgraph "mcp<2" sentence-transformers numpy python-dotenv groq
    python main.py
"""

from __future__ import annotations
import logging
import operator
import re
import time
from typing import Annotated, TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.types import Send

import llm
from mcp_bridge import McpBridge
from tax_worker import build_tax_worker, run_tax_worker

log = logging.getLogger("course.orchestration")
FOCUSES = ["contribution limits", "eligibility rules"]   # one researcher per focus
MAX_ROUNDS = 3                                          # the loop's exit (the cap from class 5.6, Reliability and failure modes, at team scale)
MAX_WORDS = 120
PRICE_IN, PRICE_OUT = 0.075 / 1_000_000, 0.30 / 1_000_000   # illustrative

WRITER_SYSTEM = ("You write a short, accurate answer for a taxpayer from research notes. "
                 f"Stay under {MAX_WORDS} words, cover every part of the question, and cite "
                 "the source ids in brackets exactly as they appear in the notes.")

REVIEWER_SYSTEM = ("You review a draft answer against research notes. Approve only if it "
                   "answers every part of the question, every claim is supported by the "
                   "notes, and it cites sources. Otherwise ask for a revision with one or "
                   "two specific, actionable fixes. Call the verdict tool exactly once.")

VERDICT_TOOL = {"type": "function", "function": {
    "name": "verdict",
    "description": "Approve the draft, or request a revision with specific feedback.",
    "parameters": {"type": "object", "properties": {
        "approve": {"type": "boolean"},
        "feedback": {"type": "string", "description": "specific fixes; empty if approved"}},
        "required": ["approve", "feedback"]}}}


# --- State: shared by the team (private worker state never appears here) -------
class TeamState(TypedDict):
    question: str
    findings: Annotated[list, operator.add]     # fan-in: parallel researchers APPEND
    notes: str                                  # merged research notes (overwrite)
    draft: str                                  # latest draft (overwrite)
    approved: bool
    feedback: str
    rounds: Annotated[int, operator.add]
    reviews: Annotated[list, operator.add]      # verdict history
    timeline: Annotated[list, operator.add]     # (node, start, end) for the Gantt view


class ResearchTask(TypedDict):                  # what Send hands each researcher
    question: str
    focus: str


_T0 = [0.0]                                     # run start time, for the timeline


def _event(node: str, start: float) -> list:
    return [{"node": node, "start": start - _T0[0], "end": time.perf_counter() - _T0[0]}]


def deterministic_checks(draft: str, notes: str) -> list[str]:
    """Cheap, exact checks that run before (and override) the LLM reviewer."""
    failed = []
    words = len(draft.split())
    if words > MAX_WORDS:
        failed.append(f"too long ({words} words; limit {MAX_WORDS})")
    cited = set(re.findall(r"\[([a-z0-9\-]+)\]", draft))
    known = set(re.findall(r"\[([a-z0-9\-]+)\]", notes))
    if not cited:
        failed.append("no source ids cited")
    elif cited - known:
        failed.append(f"cites ids not in the notes: {sorted(cited - known)}")
    return failed


def build_team(worker_app):
    def fan_out(state: TeamState) -> list[Send]:
        # One Send per focus: LangGraph runs these researcher tasks in parallel.
        return [Send("researcher", {"question": state["question"], "focus": f}) for f in FOCUSES]

    def researcher(task: ResearchTask) -> dict:
        start = time.perf_counter()
        text = run_tax_worker(worker_app, f"{task['question']}\nFocus only on: {task['focus']}.")
        return {"findings": [{"focus": task["focus"], "text": text}],
                "timeline": _event(f"researcher[{task['focus']}]", start)}

    def merge(state: TeamState) -> dict:
        start = time.perf_counter()
        notes = "\n\n".join(f"## {f['focus']}\n{f['text']}" for f in state["findings"])
        return {"notes": notes, "timeline": _event("merge", start)}

    def writer(state: TeamState) -> dict:
        start = time.perf_counter()
        ask = f"Question: {state['question']}\n\nResearch notes:\n{state['notes']}"
        if state.get("feedback"):
            ask += f"\n\nYour previous draft:\n{state['draft']}\n\nReviewer feedback to fix:\n{state['feedback']}"
        draft = llm.chat([{"role": "system", "content": WRITER_SYSTEM},
                          {"role": "user", "content": ask}], agent="writer")
        return {"draft": draft, "timeline": _event(f"writer (round {state['rounds'] + 1})", start)}

    def reviewer(state: TeamState) -> dict:
        start = time.perf_counter()
        failed = deterministic_checks(state["draft"], state["notes"])
        reply = llm.chat_tools(
            [{"role": "system", "content": REVIEWER_SYSTEM},
             {"role": "user", "content": f"Question: {state['question']}\n\nResearch notes:\n"
                                         f"{state['notes']}\n\nDraft:\n{state['draft']}"}],
            [VERDICT_TOOL], tool_choice="required", agent="reviewer")
        v = reply.tool_calls[0].args if reply.wants_tool else {"approve": False, "feedback": "no verdict returned"}
        approved = bool(v.get("approve")) and not failed          # exact checks override the LLM
        feedback = "; ".join(failed + ([v.get("feedback", "")] if v.get("feedback") else []))
        return {"approved": approved, "feedback": "" if approved else feedback, "rounds": 1,
                "reviews": [{"round": state["rounds"] + 1, "approved": approved,
                             "failed_checks": failed, "feedback": v.get("feedback", "")}],
                "timeline": _event(f"reviewer (round {state['rounds'] + 1})", start)}

    def after_review(state: TeamState) -> str:
        if state["approved"] or state["rounds"] >= MAX_ROUNDS:
            return END
        return "writer"

    g = StateGraph(TeamState)
    g.add_node("researcher", researcher)
    g.add_node("merge", merge)
    g.add_node("writer", writer)
    g.add_node("reviewer", reviewer)
    g.add_conditional_edges(START, fan_out, ["researcher"])
    g.add_edge("researcher", "merge")            # fan-in: merge waits for every researcher
    g.add_edge("merge", "writer")
    g.add_edge("writer", "reviewer")
    g.add_conditional_edges("reviewer", after_review, {"writer": "writer", END: END})
    return g.compile()


def run(team, question: str) -> dict:
    _T0[0] = time.perf_counter()
    return team.invoke({"question": question, "findings": [], "notes": "", "draft": "",
                        "approved": False, "feedback": "", "rounds": 0, "reviews": [],
                        "timeline": []}, config={"recursion_limit": 30})


def print_timeline(timeline: list, width: int = 44):
    """A text Gantt chart: overlapping bars mean the nodes ran in parallel."""
    total = max(e["end"] for e in timeline) or 1e-9
    for e in timeline:
        a, b = int(e["start"] / total * width), max(int(e["end"] / total * width), int(e["start"] / total * width) + 1)
        print(f"  {e['node']:<34}|{' ' * a}{'#' * (b - a)}{' ' * (width - b)}| {e['end'] - e['start']:.2f}s")


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
    question = "Can I contribute to an HSA, and how much can I put in for 2025?"

    with McpBridge() as bridge:
        team = build_team(build_tax_worker(bridge))
        llm.reset_usage()
        state = run(team, question)

    print(f"\n=== {question}\n\nTIMELINE (overlapping bars ran in parallel)")
    print_timeline(state["timeline"])
    res = [e for e in state["timeline"] if e["node"].startswith("researcher")]
    seq = sum(e["end"] - e["start"] for e in res)
    par = max(e["end"] for e in res) - min(e["start"] for e in res)
    print(f"\n  researchers: sequential would take ~{seq:.2f}s (sum); in parallel took {par:.2f}s (about the max)")

    print("\nREVIEW ROUNDS")
    for r in state["reviews"]:
        status = "APPROVED" if r["approved"] else "revise"
        print(f"  round {r['round']}: {status:<9} checks failed={r['failed_checks'] or 'none'}  "
              f"feedback={r['feedback'] or '-'}")
    if not state["approved"]:
        print(f"  stopped at the {MAX_ROUNDS}-round cap without approval")

    print("\nFINAL ANSWER\n ", state["draft"])
    print("\nCOST PER AGENT (illustrative prices)")
    for agent, u in llm.usage_by_agent().items():
        cost = u["prompt"] * PRICE_IN + u["completion"] * PRICE_OUT
        print(f"  {agent:<16} calls={u['calls']:<3} in={u['prompt']:<6} out={u['completion']:<5} ${cost:.6f}")


if __name__ == "__main__":
    main()
