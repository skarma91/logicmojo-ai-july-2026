"""Class 5.4 micro (reference solution): route by query type, and resume.

Extends the class LangGraph agent with a ROUTER: a conditional edge off the entry
that sends factual questions down the model-and-tools path, and action requests
("add X to my watchlist") down a separate action node. Then it shows the graph
RESUMING from a checkpoint on the same thread_id.

Two new LangGraph ideas beyond the class build:
  - a conditional edge that routes on query type (not just the tool loop), and
  - a second accumulating state field (watchlist) with its own append reducer,
    which the checkpointer persists across turns.

Run (native tool calling supports groq and ollama):
    pip install langgraph sentence-transformers numpy python-dotenv
    # plus your provider, e.g.:  pip install groq   (set PROVIDER=groq in .env)
    python solution.py
"""

from __future__ import annotations
import json
import logging
import operator
import pathlib
import re
from typing import Annotated, TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

import llm

DATA = pathlib.Path(__file__).parent.parent / "data" / "movies.jsonl"
log = logging.getLogger("course.micro")
MOVIES = [json.loads(line) for line in open(DATA)]

ACTION_WORDS = ("add", "save", "watchlist", "remember", "put")


# --- the factual tool ------------------------------------------------------
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


# --- state: messages AND a watchlist, both accumulating --------------------
class State(TypedDict):
    messages: Annotated[list, operator.add]
    watchlist: Annotated[list, operator.add]
    route: str            # overwrite: the latest routing decision


def classify(state: State) -> dict:
    """Router node: label the latest user turn 'action' or 'factual'."""
    text = [m for m in state["messages"] if m["role"] == "user"][-1]["content"].lower()
    route = "action" if any(w in text for w in ACTION_WORDS) else "factual"
    log.info("routed query to %s", route)
    return {"route": route}


def pick_route(state: State) -> str:
    return state["route"]


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


def action_node(state: State) -> dict:
    """Handle an action request: add the named title to the watchlist (persisted)."""
    text = [m for m in state["messages"] if m["role"] == "user"][-1]["content"]
    title = re.sub(r"(?i)\b(add|save|put|remember)\b", "", text)     # drop the action verb
    title = re.sub(r"(?i)\bto my watchlist\b", "", title).strip(" .\"'") or "(unknown)"
    return {"watchlist": [title],
            "messages": [{"role": "assistant", "content": f"Added {title!r} to your watchlist."}]}


def build_app(checkpointer=None):
    g = StateGraph(State)
    g.add_node("router", classify)
    g.add_node("model", model_node)
    g.add_node("tools", tool_node)
    g.add_node("action", action_node)
    g.add_edge(START, "router")
    g.add_conditional_edges("router", pick_route, {"factual": "model", "action": "action"})
    g.add_conditional_edges("model", should_continue, {"tools": "tools", END: END})
    g.add_edge("tools", "model")
    g.add_edge("action", END)
    return g.compile(checkpointer=checkpointer)


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
    app = build_app(checkpointer=MemorySaver())
    cfg = {"configurable": {"thread_id": "movie-1"}}

    out = app.invoke({"messages": [{"role": "system", "content": SYSTEM},
                                   {"role": "user", "content": "Which movie is about dreams within dreams?"}],
                      "watchlist": []}, config=cfg)
    print("factual ->", out["messages"][-1]["content"])

    # Resume the SAME thread with an action; watchlist and history persist.
    out = app.invoke({"messages": [{"role": "user", "content": "Add Inception to my watchlist."}]}, config=cfg)
    print("action  ->", out["messages"][-1]["content"])
    print("watchlist (persisted):", out["watchlist"])


if __name__ == "__main__":
    main()
