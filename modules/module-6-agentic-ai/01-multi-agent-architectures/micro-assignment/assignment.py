"""Class 6.1 micro-assignment: add a third specialist.

A movie team: a supervisor routes each request to worker tools over the real movie
dataset (data/movies.jsonl). Two workers exist (plot_search, facts). Your job: add
a third specialist, a recommender, and the routing rules around it. Fill in every
TODO, then run and compare with the README's expected output.

    START -> supervisor -> (plot_search | facts | recommender) -> supervisor -> ... -> END

The supervisor is the only LLM: it calls a `route` tool (native tool calling) that
names the next worker, the input to give it, and, on FINISH, the final answer. The
workers are plain, deterministic functions, so routing is what you are testing.

Run:  python assignment.py      (needs PROVIDER=groq or ollama in .env)
"""

from __future__ import annotations
import json
import logging
import operator
import pathlib
import re
from typing import Annotated, TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.types import Command

import llm

log = logging.getLogger("course.movie_team")
DATA = pathlib.Path(__file__).parent / "data" / "movies.jsonl"
MOVIES = [json.loads(l) for l in DATA.read_text().splitlines() if l.strip()]
TITLES = {m["title"].lower() for m in MOVIES}
MAX_HOPS = 6


# ---------------------------------------------------------------------------
# Worker tools (deterministic, over the real dataset)
# ---------------------------------------------------------------------------
def plot_search(query: str) -> str:
    """Titles whose plot best matches a description (keyword overlap)."""
    words = set(re.findall(r"[a-z]{4,}", query.lower()))
    scored = sorted(MOVIES, key=lambda m: -len(words & set(re.findall(r"[a-z]{4,}", m["plot"].lower()))))
    return "; ".join(f"{m['title']} ({m['year']})" for m in scored[:2])


def facts(query: str) -> str:
    """Year, genre, and director for a title (lists every match: titles can repeat)."""
    hits = [m for m in MOVIES if m["title"].lower() == query.strip().lower()]
    if not hits:
        return f"no movie titled {query!r}"
    return "; ".join(f"{m['title']} ({m['year']}): {m['genre']}, dir. {m['director']}" for m in hits)


# Problem 1
def recommend(title: str) -> list[str]:
    """Up to 3 other titles: same director first, then same genre, newest first."""
    # TODO(1): find the movie with this exact title (case-insensitive); if none, return [].
    #          Rank the OTHER movies: same director first, then same genre, each group
    #          newest first, no duplicates. Return at most 3 titles.
    raise NotImplementedError


def recommender(query: str) -> str:
    recs = recommend(query)
    return ", ".join(recs) if recs else f"no recommendations for {query!r}"


# Problems 2 and 3: the team's workers, each with a role description the supervisor reads.
WORKERS = {
    "plot_search": ("finds movies whose plot matches a description; give it the description", plot_search),
    "facts": ("returns year, genre, and director for an exact title; give it the title", facts),
    # TODO(2): register the recommender worker here, as ("<role description>", recommender).
    # TODO(3): write the role description so the supervisor uses the recommender ONLY when the
    #          user asks for recommendations or similar films, and knows to pass it a title.
}


def route_tool() -> dict:
    return {"type": "function", "function": {
        "name": "route",
        "description": "Choose the next worker and its input, or FINISH with the final answer.",
        "parameters": {"type": "object", "properties": {
            "next": {"type": "string", "enum": [*WORKERS, "FINISH"]},
            "query": {"type": "string", "description": "input for the worker"},
            "answer": {"type": "string", "description": "final answer, only when next is FINISH"},
            "reason": {"type": "string"}},
            "required": ["next", "reason"]}}}


def supervisor_system() -> str:
    return ("You coordinate a movie team. Workers:\n"
            + "\n".join(f"- {n}: {role}" for n, (role, _) in WORKERS.items())
            + "\nCall route exactly once. Use only the workers the request needs. When the "
              "results answer the request, choose FINISH and write the answer from them.")


class TeamState(TypedDict):
    request: str
    results: Annotated[list, operator.add]
    hops: Annotated[int, operator.add]
    route_log: Annotated[list, operator.add]
    answer: str


# Problem 5: the recommender needs a real catalog title. If the supervisor hands it a
# description instead, resolve the title with plot_search first.
def guard(decision: dict) -> dict:
    # TODO(5): if the supervisor routes to "recommender" with a query that is not an exact
    #          catalog title (see TITLES), reroute to "plot_search" with the same query and
    #          reason "recommender needs a catalog title; resolving it first". Otherwise
    #          return the decision unchanged.
    raise NotImplementedError


def build_team():
    def supervisor(state: TeamState) -> Command:
        if state["hops"] >= MAX_HOPS:
            decision = {"next": "FINISH", "reason": "hop cap reached",
                        "answer": "; ".join(r["output"] for r in state["results"])}
        else:
            done = "\n".join(f"- {r['worker']}({r['query']!r}) -> {r['output']}" for r in state["results"]) or "none yet"
            reply = llm.chat_tools([{"role": "system", "content": supervisor_system()},
                                    {"role": "user", "content": f"Request: {state['request']}\nResults so far:\n{done}"}],
                                   [route_tool()], tool_choice="required", agent="supervisor")
            decision = reply.tool_calls[0].args if reply.wants_tool else {"next": "FINISH", "reason": "no decision"}
            if decision.get("next") not in (*WORKERS, "FINISH"):
                decision = {"next": "FINISH", "reason": "invalid route"}
            decision = guard(decision)
        if decision["next"] == "FINISH":
            return Command(goto=END, update={"hops": 1, "route_log": [decision],
                                             "answer": decision.get("answer", "")})
        return Command(goto=decision["next"], update={"hops": 1, "route_log": [decision]})

    def make_worker(name, fn):
        def node(state: TeamState) -> Command:
            query = state["route_log"][-1].get("query", "")
            return Command(goto="supervisor",
                           update={"results": [{"worker": name, "query": query, "output": fn(query)}]})
        return node

    g = StateGraph(TeamState)
    g.add_node("supervisor", supervisor, destinations=(*WORKERS, END))
    for name, (_, fn) in WORKERS.items():
        g.add_node(name, make_worker(name, fn), destinations=("supervisor",))
    g.add_edge(START, "supervisor")
    return g.compile()


# Problem 4
def workers_used(route_log: list) -> list[str]:
    """The workers the supervisor actually sent work to, in order."""
    # TODO(4): return the "next" of every routing decision except FINISH, in order.
    raise NotImplementedError


def main():
    logging.basicConfig(level=logging.WARNING)
    print("[1] recommend('Inception') ->", recommend("Inception"))
    print("    recommend('Coco')      ->", recommend("Coco"))
    print("    recommend('Avatar')    ->", recommend("Avatar"))
    print("[2] route tool workers     ->", route_tool()["function"]["parameters"]["properties"]["next"]["enum"])
    print("[5] guard on a description ->", guard({"next": "recommender", "query": "the dream heist movie", "reason": "x"})["next"])

    team = build_team()
    for req in ["Who directed Inception?",
                "Suggest movies like Inception.",
                "Which movie is about thieves who enter dreams? Recommend similar ones."]:
        llm.reset_usage()
        state = team.invoke({"request": req, "results": [], "hops": 0, "route_log": [], "answer": ""},
                            config={"recursion_limit": 25})
        used = workers_used(state["route_log"])
        calls = sum(u["calls"] for u in llm.usage_by_agent().values())
        print(f"[3/4] {req}\n      workers={used}  recommender_used={'recommender' in used}  "
              f"supervisor_calls={calls}\n      answer: {state['answer']}")


if __name__ == "__main__":
    main()
