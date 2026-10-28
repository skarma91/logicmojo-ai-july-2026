"""The team under observation: the class 6.2 (Orchestration) team.

Two researchers run in parallel (fan-out with Send), their findings merge, a writer
drafts, and a reviewer loops the draft back until it passes or MAX_ROUNDS is reached:

    START --Send--> researcher[limits]      \\
          --Send--> researcher[eligibility] -> merge -> writer -> reviewer --approve--> END
                                                         ^           |
                                                         +--revise---+

One change from class 6.2 (Orchestration): build_team takes a `wrap(name, fn)` hook and passes every
node through it. main.py uses the hook to open a span around each agent turn, so the
team's own code carries no tracing calls at all.
"""

from __future__ import annotations
import operator
import re
from typing import Annotated, TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.types import Send

import llm
from tax_worker import run_tax_worker

FOCUSES = ["contribution limits", "eligibility rules"]
MAX_ROUNDS = 3
MAX_WORDS = 120
CITE = re.compile(r"\[([a-z0-9]+(?:-[a-z0-9]+)+)\]")

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


class TeamState(TypedDict):
    question: str
    findings: Annotated[list, operator.add]
    notes: str
    draft: str
    approved: bool
    feedback: str
    rounds: Annotated[int, operator.add]
    reviews: Annotated[list, operator.add]


def deterministic_checks(draft: str, notes: str) -> list[str]:
    failed = []
    if len(draft.split()) > MAX_WORDS:
        failed.append(f"too long ({len(draft.split())} words; limit {MAX_WORDS})")
    cited, known = set(CITE.findall(draft)), set(CITE.findall(notes))
    if not cited:
        failed.append("no source ids cited")
    elif cited - known:
        failed.append(f"cites ids not in the notes: {sorted(cited - known)}")
    return failed


def build_team(worker_app, wrap=lambda name, fn: fn):
    def fan_out(state: TeamState) -> list[Send]:
        return [Send("researcher", {"question": state["question"], "focus": f}) for f in FOCUSES]

    def researcher(task: dict) -> dict:
        text = run_tax_worker(worker_app, f"{task['question']}\nFocus only on: {task['focus']}.")
        return {"findings": [{"focus": task["focus"], "text": text}]}

    def merge(state: TeamState) -> dict:
        return {"notes": "\n\n".join(f"## {f['focus']}\n{f['text']}" for f in state["findings"])}

    def writer(state: TeamState) -> dict:
        ask = f"Question: {state['question']}\n\nResearch notes:\n{state['notes']}"
        if state.get("feedback"):
            ask += f"\n\nYour previous draft:\n{state['draft']}\n\nReviewer feedback to fix:\n{state['feedback']}"
        return {"draft": llm.chat([{"role": "system", "content": WRITER_SYSTEM},
                                   {"role": "user", "content": ask}], agent="writer")}

    def reviewer(state: TeamState) -> dict:
        failed = deterministic_checks(state["draft"], state["notes"])
        reply = llm.chat_tools(
            [{"role": "system", "content": REVIEWER_SYSTEM},
             {"role": "user", "content": f"Question: {state['question']}\n\nResearch notes:\n"
                                         f"{state['notes']}\n\nDraft:\n{state['draft']}"}],
            [VERDICT_TOOL], tool_choice="required", agent="reviewer")
        v = reply.tool_calls[0].args if reply.wants_tool else {"approve": False, "feedback": "no verdict"}
        approved = bool(v.get("approve")) and not failed
        feedback = "; ".join(failed + ([v["feedback"]] if v.get("feedback") else []))
        return {"approved": approved, "feedback": "" if approved else feedback, "rounds": 1,
                "reviews": [{"approved": approved, "failed_checks": failed}]}

    def after_review(state: TeamState) -> str:
        return END if state["approved"] or state["rounds"] >= MAX_ROUNDS else "writer"

    g = StateGraph(TeamState)
    g.add_node("researcher", wrap("researcher", researcher))
    g.add_node("merge", wrap("merge", merge))
    g.add_node("writer", wrap("writer", writer))
    g.add_node("reviewer", wrap("reviewer", reviewer))
    g.add_conditional_edges(START, fan_out, ["researcher"])
    g.add_edge("researcher", "merge")
    g.add_edge("merge", "writer")
    g.add_edge("writer", "reviewer")
    g.add_conditional_edges("reviewer", after_review, {"writer": "writer", END: END})
    return g.compile()


def run(team, question: str) -> dict:
    return team.invoke({"question": question, "findings": [], "notes": "", "draft": "",
                        "approved": False, "feedback": "", "rounds": 0, "reviews": []},
                       config={"recursion_limit": 30})
