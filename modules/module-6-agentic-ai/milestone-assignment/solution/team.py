"""The research team: a supervisor, a paper finder, a summarizer, and a critic.

    START -> supervisor -> paper_finder -> supervisor
             supervisor -> summarizer -> critic
                           critic -> summarizer          (revise; at most MAX_REVIEWS rounds)
                           critic -> supervisor -> END   (approved, or the review cap reached)

- supervisor     routes with a structured `route` tool (class 6.1, Multi-agent
                 architectures) and a guard that overrules impossible routes
- paper_finder   a tool-calling worker: searches the corpus with `search_papers`, then
                 hands off a compact note of the papers it found (class 6.3, Context
                 engineering), never its raw transcript
- summarizer     writes a short answer that cites arXiv ids like [2308.08155]
- critic         exact checks first (word limit, citations present and retrieved), then
                 a `verdict` tool; exact failures override the model (class 6.2,
                 Orchestration). A bounded loop sends the draft back for revision.

If no paper is relevant (best score below MIN_SCORE), the team says so instead of
summarizing something unrelated.
"""

from __future__ import annotations
import json
import operator
import os
import re
from typing import Annotated, TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.types import Command

import llm
import retriever

MAX_HOPS = 8                 # supervisor decisions per request
MAX_SEARCHES = 3             # paper_finder tool calls per turn
MAX_REVIEWS = 2              # critic rounds before the draft is accepted with a note
MAX_WORDS = 160
MIN_SCORE = float(os.getenv("MIN_SCORE", "0.30"))   # below this, nothing in the corpus is relevant
CITE = re.compile(r"\[(\d{4}\.\d{4,5})\]")
NO_RESULTS = "I could not find a paper in the corpus that addresses this question."

ROUTE_TOOL = {"type": "function", "function": {
    "name": "route", "description": "Choose the next worker, or FINISH when the answer is ready.",
    "parameters": {"type": "object", "properties": {
        "next": {"type": "string", "enum": ["paper_finder", "summarizer", "FINISH"]},
        "reason": {"type": "string"}}, "required": ["next", "reason"]}}}
SEARCH_TOOL = {"type": "function", "function": {
    "name": "search_papers", "description": "Search arXiv abstracts in the corpus.",
    "parameters": {"type": "object", "properties": {"query": {"type": "string"}},
                   "required": ["query"]}}}
VERDICT_TOOL = {"type": "function", "function": {
    "name": "verdict", "description": "Approve the summary, or request a revision with specific feedback.",
    "parameters": {"type": "object", "properties": {
        "approve": {"type": "boolean"}, "feedback": {"type": "string"}},
        "required": ["approve", "feedback"]}}}

SUPERVISOR_SYSTEM = ("You coordinate a research team that answers questions from a corpus of "
                     "arXiv abstracts. Workers: paper_finder (searches the corpus), summarizer "
                     "(writes a cited answer from the papers found). Call route exactly once.")
FINDER_SYSTEM = ("You find papers relevant to a research question. Call search_papers with a "
                 "focused query; search again with a different query only if the results miss "
                 "the point. Then reply with one sentence saying which papers are relevant.")
SUMMARIZER_SYSTEM = (f"You write a short answer, under {MAX_WORDS} words, to a research question "
                     "using only the papers given. Cite every claim with the paper's arXiv id in "
                     "square brackets, for example [2308.08155]. If feedback is given, fix exactly that.")
CRITIC_SYSTEM = ("You review a research summary against the papers it cites. Approve only if "
                 "every claim is supported by those papers and the question is answered. "
                 "Otherwise give one specific fix. Call verdict exactly once.")


class TeamState(TypedDict):
    question: str
    findings: list                                # [{id, title, year, snippet, score}]
    no_results: bool
    draft: str
    approved: bool
    feedback: str
    reviews: Annotated[int, operator.add]
    hops: Annotated[int, operator.add]
    route_log: Annotated[list, operator.add]
    answer: str


# ---------------------------------------------------------------------------
# TODO 1: the supervisor's guard
# ---------------------------------------------------------------------------
def route_guard(decision: str, state: dict) -> str:
    """Overrule routes that cannot work, whatever the model chose:
    - at the hop cap, FINISH
    - summarizer without findings: go to paper_finder first (unless the search already
      came back empty, in which case FINISH)
    - FINISH with findings but no draft yet: go to summarizer
    """
    if state["hops"] >= MAX_HOPS:
        return "FINISH"
    if decision == "summarizer" and not state["findings"]:
        return "FINISH" if state["no_results"] else "paper_finder"
    if decision == "FINISH" and state["findings"] and not state["draft"]:
        return "summarizer"
    return decision


# ---------------------------------------------------------------------------
# TODO 2: the critic's exact checks
# ---------------------------------------------------------------------------
def exact_checks(draft: str, retrieved_ids: set) -> list[str]:
    """Checks that need no model. Returns the failures, in this order:
    too long (over MAX_WORDS words); no arXiv ids cited; cites ids that were not retrieved."""
    failed = []
    words = len(draft.split())
    if words > MAX_WORDS:
        failed.append(f"too long ({words} words; limit {MAX_WORDS})")
    cited = set(CITE.findall(draft))
    if not cited:
        failed.append("no arXiv ids cited")
    elif cited - retrieved_ids:
        failed.append(f"cites papers that were not retrieved: {sorted(cited - retrieved_ids)}")
    return failed


# ---------------------------------------------------------------------------
# TODO 3: the bounded review loop
# ---------------------------------------------------------------------------
def after_critic(state: dict) -> str:
    """Back to the summarizer for a revision, or on to the supervisor when the draft is
    approved or MAX_REVIEWS rounds have been used."""
    if state["approved"] or state["reviews"] >= MAX_REVIEWS:
        return "supervisor"
    return "summarizer"


# ---------------------------------------------------------------------------
# The graph (given)
# ---------------------------------------------------------------------------
def _notes(findings: list) -> str:
    """The compact handoff: id, title, year, and the first two sentences of each abstract."""
    return "\n".join(f"[{f['id']}] {f['title']} ({f['year']}): {f['snippet']}" for f in findings)


def build_team(wrap=lambda name, fn: fn, search_fn=None):
    search = search_fn or (lambda q: retriever.search(q, k=4))

    def supervisor(state: TeamState) -> Command:
        status = (f"Question: {state['question']}\nPapers found: "
                  f"{[f['id'] for f in state['findings']] or ('none relevant' if state['no_results'] else 'not searched yet')}\n"
                  f"Draft: {'approved' if state['approved'] else ('written' if state['draft'] else 'none')}")
        if state["hops"] >= MAX_HOPS:
            choice = "FINISH"
        else:
            reply = llm.chat_tools([{"role": "system", "content": SUPERVISOR_SYSTEM},
                                    {"role": "user", "content": status}],
                                   [ROUTE_TOOL], tool_choice="required", agent="supervisor")
            choice = reply.tool_calls[0].args.get("next", "FINISH") if reply.wants_tool else "FINISH"
        choice = route_guard(choice, state)
        update = {"hops": 1, "route_log": [choice]}
        if choice == "FINISH":
            update["answer"] = state["draft"] if state["draft"] else NO_RESULTS
            return Command(goto=END, update=update)
        return Command(goto=choice, update=update)

    def paper_finder(state: TeamState) -> Command:
        messages = [{"role": "system", "content": FINDER_SYSTEM},
                    {"role": "user", "content": state["question"]}]
        found: dict[str, dict] = {}
        for _ in range(MAX_SEARCHES):
            reply = llm.chat_tools(messages, [SEARCH_TOOL], agent="paper_finder")
            if not reply.wants_tool:
                break
            messages.append(llm.assistant_tool_call_message(reply))
            for call in reply.tool_calls:
                hits = search(call.args.get("query") or state["question"])
                for h in hits:
                    if h["score"] >= MIN_SCORE and h["id"] not in found:
                        first = " ".join(re.split(r"(?<=[.!?])\s+", h["text"])[:2])
                        found[h["id"]] = {"id": h["id"], "title": h["title"], "year": h["year"],
                                          "snippet": first, "score": round(h["score"], 3)}
                listing = "\n".join(f"[{h['id']}] {h['title']} (score {h['score']:.2f})" for h in hits)
                messages.append(llm.tool_result_message(call, listing or "no results"))
        findings = sorted(found.values(), key=lambda f: -f["score"])[:4]
        return Command(goto="supervisor", update={"findings": findings, "no_results": not findings})

    def summarizer(state: TeamState) -> dict:
        ask = f"Question: {state['question']}\n\nPapers:\n{_notes(state['findings'])}"
        if state["feedback"]:
            ask += f"\n\nYour previous summary:\n{state['draft']}\n\nFeedback to fix:\n{state['feedback']}"
        draft = llm.chat([{"role": "system", "content": SUMMARIZER_SYSTEM},
                          {"role": "user", "content": ask}], agent="summarizer")
        return {"draft": draft.strip()}

    def critic(state: TeamState) -> dict:
        failed = exact_checks(state["draft"], {f["id"] for f in state["findings"]})
        reply = llm.chat_tools(
            [{"role": "system", "content": CRITIC_SYSTEM},
             {"role": "user", "content": f"Question: {state['question']}\n\nPapers:\n{_notes(state['findings'])}"
                                         f"\n\nSummary:\n{state['draft']}"}],
            [VERDICT_TOOL], tool_choice="required", agent="critic")
        v = reply.tool_calls[0].args if reply.wants_tool else {"approve": False, "feedback": "no verdict"}
        approved = bool(v.get("approve")) and not failed
        feedback = "; ".join(failed + ([v["feedback"]] if v.get("feedback") else []))
        return {"approved": approved, "feedback": "" if approved else feedback, "reviews": 1}

    g = StateGraph(TeamState)
    g.add_node("supervisor", wrap("supervisor", supervisor), destinations=("paper_finder", "summarizer", END))
    g.add_node("paper_finder", wrap("paper_finder", paper_finder), destinations=("supervisor",))
    g.add_node("summarizer", wrap("summarizer", summarizer))
    g.add_node("critic", wrap("critic", critic))
    g.add_edge(START, "supervisor")
    g.add_edge("summarizer", "critic")
    g.add_conditional_edges("critic", after_critic, {"summarizer": "summarizer", "supervisor": "supervisor"})
    return g.compile()


def new_state(question: str) -> dict:
    return {"question": question, "findings": [], "no_results": False, "draft": "", "approved": False,
            "feedback": "", "reviews": 0, "hops": 0, "route_log": [], "answer": ""}


def run(team, question: str) -> dict:
    return team.invoke(new_state(question), config={"recursion_limit": 40})
