"""Class 5.4 micro (starter): route by query type, and resume from a checkpoint.

Extend the class LangGraph agent with a ROUTER conditional edge: factual questions
go down the model-and-tools path; action requests ("add X to my watchlist") go down
a separate action node. Then show the graph RESUMING on the same thread_id, with a
watchlist that persists.

The model/tools nodes and the movie search tool are given. Your job is the router,
the action node, the watchlist state field, and the wiring.

Run:
    pip install langgraph sentence-transformers numpy python-dotenv
    # plus your provider, e.g.:  pip install groq
    python assignment.py

A worked reference is in solution/. Compare your output to it.
"""

from __future__ import annotations
import json
import logging
import operator
import pathlib
from typing import Annotated, TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

import llm

DATA = pathlib.Path(__file__).parent / "data" / "movies.jsonl"
log = logging.getLogger("course.micro")
MOVIES = [json.loads(line) for line in open(DATA)]
ACTION_WORDS = ("add", "save", "watchlist", "remember", "put")


# Given: the factual tool.
def search_movies(query: str) -> str:
    words = [w for w in query.lower().split() if len(w) > 3]
    scored = [(sum(m["plot"].lower().count(w) for w in words), m) for m in MOVIES]
    scored = [x for x in scored if x[0] > 0] or [(0, m) for m in MOVIES[:3]]
    scored.sort(key=lambda x: -x[0])
    return "\n".join(f"[{m['id']}] {m['title']} ({m['year']}): {m['plot']}" for _, m in scored[:3])


TOOLS = {"search_movies": {"fn": search_movies, "spec": {"type": "function", "function": {
    "name": "search_movies", "description": "Find movies by plot.",
    "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}}}}
TOOL_SPECS = [t["spec"] for t in TOOLS.values()]
SYSTEM = "You are a movie assistant. Use search_movies for plot questions; then answer."


# TODO 1: add a `watchlist` field to the state with an append reducer (like messages),
# plus a `route` field (plain str, overwrite) for the routing decision.
class State(TypedDict):
    messages: Annotated[list, operator.add]


# TODO 2: the router node. Read the latest user message; if it contains an action
# word (ACTION_WORDS), set route="action", else route="factual".
def classify(state: State) -> dict:
    raise NotImplementedError


def pick_route(state: State) -> str:
    return state["route"]


# Given: model and tools nodes (same as the class build).
def model_node(state: State) -> dict:
    reply = llm.chat_tools(state["messages"], TOOL_SPECS, tool_choice="auto")
    if reply.wants_tool:
        return {"messages": [llm.assistant_tool_call_message(reply)]}
    return {"messages": [{"role": "assistant", "content": reply.text or ""}]}


def tool_node(state: State) -> dict:
    last = state["messages"][-1]
    out = []
    for tc in last.get("tool_calls", []):
        name, args = tc["function"]["name"], json.loads(tc["function"]["arguments"] or "{}")
        result = TOOLS[name]["fn"](**args) if name in TOOLS else f"error: no tool {name}"
        out.append({"role": "tool", "tool_call_id": tc["id"], "name": name, "content": str(result)})
    return {"messages": out}


def should_continue(state: State) -> str:
    return "tools" if state["messages"][-1].get("tool_calls") else END


# TODO 3: the action node. Pull the title out of the latest user message, append it
# to the watchlist, and add an assistant confirmation message.
def action_node(state: State) -> dict:
    raise NotImplementedError


# TODO 4: build the graph. START -> router; conditional from router to "model"
# (factual) or "action" (action); the model/tools cycle; action -> END. Compile with
# a checkpointer.
def build_app(checkpointer=None):
    raise NotImplementedError


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
    # TODO 5: run a factual question, then RESUME the same thread with an action
    # request; print both answers and the persisted watchlist.
    raise NotImplementedError


if __name__ == "__main__":
    main()
