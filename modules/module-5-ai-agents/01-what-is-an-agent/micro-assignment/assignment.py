"""Class 5.1 micro (starter): a two-tool agent that chooses.

Take the think-act-observe loop from the class build and give the agent TWO tools
over the movie dataset, so it must choose which one fits each task, or use no tool
at all. The loop (`run_agent`) is given. Your job is the two tools, their schemas,
the system prompt, and showing the agent choosing.

Run:
    pip install python-dotenv
    # plus your provider, e.g.:  pip install ollama   (then `ollama serve`)
    python assignment.py

A worked reference is in solution/. Compare your output to it.
"""

from __future__ import annotations
import json
import logging
import pathlib
import re

import llm

DATA = pathlib.Path(__file__).parent / "data" / "movies.jsonl"
log = logging.getLogger("course.micro")

MOVIES = [json.loads(line) for line in open(DATA)]
BY_TITLE = {m["title"].lower(): m for m in MOVIES}


# TODO 1: a SEARCH tool. Return the movies whose plot best matches the query.
# A simple keyword match over each movie's "plot" is enough (no model needed).
def search_movies(query: str) -> str:
    raise NotImplementedError


# TODO 2: a LOOKUP tool. Return year, director, and genre for ONE movie by title
# (use BY_TITLE). Handle a title that is not in the catalog.
def movie_facts(title: str) -> str:
    raise NotImplementedError


# TODO 3: register both tools with a name, a clear description (the model reads the
# description to choose), and a typed parameter schema. See the class build for the
# shape of one entry.
TOOLS = {
    # "search_movies": {"fn": search_movies, "schema": {...}},
    # "movie_facts":   {"fn": movie_facts,   "schema": {...}},
}


def _tools_for_prompt() -> str:
    return json.dumps([t["schema"] for t in TOOLS.values()], indent=2)


# TODO 4: write the system prompt. State the tools, the one-JSON-object contract
# (a tool call or a final answer), and that the agent should answer directly when
# no tool is needed.
SYSTEM = ""


def _extract_json(text: str) -> str:
    m = re.search(r"\{.*\}", text, re.S)
    return m.group(0) if m else text


# Given: the same bounded think-act-observe loop as the class build.
def run_agent(question: str, max_steps: int = 5) -> str:
    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": question}]
    for step in range(1, max_steps + 1):
        raw = llm.chat(messages)
        try:
            action = json.loads(_extract_json(raw))
        except json.JSONDecodeError:
            messages.append({"role": "user", "content": "Reply with ONLY one JSON object."})
            continue
        if "final" in action:
            return action["final"]
        name, args = action.get("tool"), action.get("arguments", {})
        if name not in TOOLS:
            messages.append({"role": "assistant", "content": raw})
            messages.append({"role": "user", "content": f"No tool {name!r}. Available: {list(TOOLS)}."})
            continue
        result = TOOLS[name]["fn"](**args)
        messages.append({"role": "assistant", "content": json.dumps(action)})
        messages.append({"role": "user", "content": f"Tool result:\n{result}"})
    return "stopped: step cap reached"


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
    # TODO 5: three questions that exercise each path: one for search_movies, one
    # for movie_facts, and one that needs NO tool (answered directly).
    for q in []:
        print(f"\nQ: {q}")
        print("A:", run_agent(q))


if __name__ == "__main__":
    main()
