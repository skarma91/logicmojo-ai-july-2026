"""The team behind the gateway: the class 6.1 (Multi-agent architectures) team, unchanged.

A supervisor routes between the class 5.7 (Assemble a single agent) tax researcher and
a plain-language explainer, handing off with LangGraph's Command. Class 6.4 does not
change the team at all: every guardrail lives OUTSIDE it, in gateway.py. That is the
point of a gateway: one safety and compliance layer around the whole team, instead of
rules scattered inside each agent.
"""

from __future__ import annotations
import logging
import operator
from typing import Annotated, TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.types import Command

import llm
from tax_worker import run_tax_worker

log = logging.getLogger("course.team")
MAX_HOPS = 6                       # team-level cap: the supervisor cannot loop forever


WORKERS = {
    "tax_researcher": "finds facts in official IRS publications and cites source ids",
    "explainer": "rewrites findings in plain language for a non-expert (needs findings first)",
}

ROUTE_TOOL = {"type": "function", "function": {
    "name": "route",
    "description": "Choose which worker acts next, or FINISH when the task is fully answered.",
    "parameters": {"type": "object", "properties": {
        "next": {"type": "string", "enum": [*WORKERS, "FINISH"]},
        "reason": {"type": "string", "description": "one short sentence"}},
        "required": ["next", "reason"]}}}

SUPERVISOR_SYSTEM = (
    "You coordinate a small team. Workers:\n"
    + "\n".join(f"- {name}: {role}" for name, role in WORKERS.items())
    + "\nCall the route tool exactly once. Send work to a worker only if the task still "
      "needs it. Use the explainer only when the user asks for a simple or plain "
      "explanation, and only after findings exist. Choose FINISH as soon as the task "
      "is answered.")

EXPLAINER_SYSTEM = ("You explain tax findings to a non-expert in three or four short, "
                    "plain sentences. Keep the source ids in brackets. Add nothing that "
                    "is not in the findings.")


class TeamState(TypedDict):
    task: str
    findings: str
    explanation: str
    hops: Annotated[int, operator.add]
    route_log: Annotated[list, operator.add]


def build_team(worker_app):
    def supervisor(state: TeamState) -> Command:
        # The supervisor sees a compact status, not every worker's scratch work.
        status = (f"Task: {state['task']}\n"
                  f"Findings: {('present: ' + state['findings'][:200]) if state['findings'] else 'none yet'}\n"
                  f"Explanation: {'present' if state['explanation'] else 'none yet'}")
        if state["hops"] >= MAX_HOPS:
            decision = {"next": "FINISH", "reason": f"hop cap ({MAX_HOPS}) reached"}
        else:
            reply = llm.chat_tools([{"role": "system", "content": SUPERVISOR_SYSTEM},
                                    {"role": "user", "content": status}],
                                   [ROUTE_TOOL], tool_choice="required", agent="supervisor")
            decision = reply.tool_calls[0].args if reply.wants_tool else {
                "next": "FINISH", "reason": "no routing decision returned"}
            if decision.get("next") not in (*WORKERS, "FINISH"):
                decision = {"next": "FINISH", "reason": f"invalid route {decision.get('next')!r}"}
            # A cheap rule beats a wasted call: no findings means no explanation yet.
            if decision["next"] == "explainer" and not state["findings"]:
                decision = {"next": "tax_researcher", "reason": "explainer needs findings first"}
        log.info("supervisor -> %s (%s)", decision["next"], decision.get("reason", ""))
        goto = END if decision["next"] == "FINISH" else decision["next"]
        return Command(goto=goto, update={"hops": 1, "route_log": [decision]})

    def tax_researcher(state: TeamState) -> Command:
        findings = run_tax_worker(worker_app, state["task"])
        return Command(goto="supervisor", update={"findings": findings})

    def explainer(state: TeamState) -> Command:
        text = llm.chat([{"role": "system", "content": EXPLAINER_SYSTEM},
                         {"role": "user", "content": f"Task: {state['task']}\n\nFindings:\n{state['findings']}"}],
                        agent="explainer")
        return Command(goto="supervisor", update={"explanation": text})

    g = StateGraph(TeamState)
    g.add_node("supervisor", supervisor, destinations=(*WORKERS, END))
    g.add_node("tax_researcher", tax_researcher, destinations=("supervisor",))
    g.add_node("explainer", explainer, destinations=("supervisor",))
    g.add_edge(START, "supervisor")
    return g.compile()


def run_team(team, task: str) -> dict:
    return team.invoke({"task": task, "findings": "", "explanation": "", "hops": 0, "route_log": []},
                       config={"recursion_limit": 25})
