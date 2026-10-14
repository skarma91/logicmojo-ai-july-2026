"""Class 6.1 build: one worker becomes a team (supervisor plus two workers).

The single agent from class 5.7 (Assemble a single agent) becomes worker 1. A second worker rewrites its
findings for a non-expert. A SUPERVISOR decides who acts next and when the task is
done, and hands control over with LangGraph's Command (a handoff):

        START -> supervisor -> tax_researcher -> supervisor -> explainer -> supervisor -> END
                       (the supervisor may skip the explainer, or finish early)

What to watch when you run it:
  - the routing log: the supervisor's decision and reason at each hop,
  - that it does NOT call the explainer when the task does not need one,
  - the per-agent cost table: the team-cost double sum from the slides, measured,
  - the comparison: the same task done by the single 5.7 agent alone. The team
    costs more calls. That overhead is the price of the team; pay it only when the
    task needs distinct roles.

Run (native tool calling: groq or ollama):
    pip install langgraph "mcp<2" sentence-transformers numpy python-dotenv groq
    python main.py
"""

from __future__ import annotations
import logging
import operator
from typing import Annotated, TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.types import Command

import llm
from mcp_bridge import McpBridge
from tax_worker import build_tax_worker, run_tax_worker

log = logging.getLogger("course.team")
MAX_HOPS = 6                       # team-level cap: the supervisor cannot loop forever

# Illustrative prices (USD per token) for the cost table; check your provider's
# current pricing. On Groq's free plan you pay nothing but are rate limited.
PRICE_IN = 0.075 / 1_000_000
PRICE_OUT = 0.30 / 1_000_000

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


def cost_table(title: str) -> tuple[int, float]:
    """Print calls, tokens, and illustrative cost per agent; return the totals."""
    rows, calls, cost = llm.usage_by_agent(), 0, 0.0
    print(f"  {title}")
    print(f"  {'agent':<16}{'calls':>6}{'in tok':>9}{'out tok':>9}{'cached':>8}{'cost $':>11}")
    for agent, u in rows.items():
        c = u["prompt"] * PRICE_IN + u["completion"] * PRICE_OUT
        calls, cost = calls + u["calls"], cost + c
        print(f"  {agent:<16}{u['calls']:>6}{u['prompt']:>9}{u['completion']:>9}{u['cached']:>8}{c:>11.6f}")
    print(f"  {'TOTAL':<16}{calls:>6}{'':>26}{cost:>11.6f}  (illustrative prices)")
    return calls, cost


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
    tasks = ["Explain in plain language how much I can put into my HSA this year.",
             "What does AGI stand for in tax terms?"]            # needs no explainer

    with McpBridge() as bridge:
        worker_app = build_tax_worker(bridge)
        team = build_team(worker_app)

        totals = {}
        for task in tasks:
            llm.reset_usage()
            state = run_team(team, task)
            print(f"\n=== {task}")
            for i, d in enumerate(state["route_log"], 1):
                print(f"  hop {i}: supervisor -> {d['next']:<15} {d.get('reason', '')}")
            print("\n  ANSWER:", state["explanation"] or state["findings"], "\n")
            totals[task] = cost_table("team cost, per agent:")

        # The honest counterpoint: the same first task, done by the single agent alone.
        llm.reset_usage()
        run_tax_worker(worker_app, tasks[0])
        print(f"\n=== single agent (class 5.7, Assemble a single agent) on: {tasks[0]}")
        solo_calls, _ = cost_table("single-agent cost:")
        team_calls, _ = totals[tasks[0]]
        print(f"\n  Team: {team_calls} calls. Single agent: {solo_calls} calls. The team adds "
              "supervisor (and explainer) calls on top of the worker's own; worth it only when "
              "a distinct role, here the plain-language rewrite, earns the overhead.")


if __name__ == "__main__":
    main()
