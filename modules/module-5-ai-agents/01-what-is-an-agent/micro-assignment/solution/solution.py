"""Class 5.1 micro (reference solution): a two-tool agent that chooses.

Same think-act-observe loop as the class build, but on a different dataset (movie
plots) and with TWO tools, so the agent must choose which one fits the task, or
choose to use no tool at all. That last case matters: choosing includes choosing
not to act.

The two tools are deliberately different in kind:
  search_movies(query): find movies by what happens in them (a search tool).
  movie_facts(title):    look up structured facts about one movie (a lookup tool).

search_movies here is a simple keyword match over plots (offline, no model needed),
which is enough to show tool CHOICE, the point of this micro. The dense retriever
lives in the class build.

Run:
    pip install python-dotenv
    # plus your provider, e.g.:  pip install ollama   (then `ollama serve`)
    python solution.py
"""

from __future__ import annotations
import json
import logging
import pathlib
import re

import llm

DATA = pathlib.Path(__file__).parent.parent / "data" / "movies.jsonl"
log = logging.getLogger("course.micro")

MOVIES = [json.loads(line) for line in open(DATA)]
BY_TITLE = {m["title"].lower(): m for m in MOVIES}


# --- Tool 1: search by plot (keyword match over the plots) -----------------
def search_movies(query: str) -> str:
    """Return movies whose plot best matches the query words."""
    words = [w for w in re.findall(r"[a-z]+", query.lower()) if len(w) > 3]
    scored = []
    for m in MOVIES:
        plot = m["plot"].lower()
        hits = sum(plot.count(w) for w in words)
        if hits:
            scored.append((hits, m))
    scored.sort(key=lambda x: -x[0])
    top = scored[:3] or [(0, m) for m in MOVIES[:3]]
    return "\n".join(f"[{m['id']}] {m['title']} ({m['year']}): {m['plot']}" for _, m in top)


# --- Tool 2: structured facts about one movie (a lookup) -------------------
def movie_facts(title: str) -> str:
    """Return year, director, and genre for one movie by title."""
    m = BY_TITLE.get(title.lower())
    if not m:
        return f"No movie titled {title!r} in the catalog."
    return f"{m['title']}: year={m['year']}, director={m['director']}, genre={m['genre']}"


TOOLS = {
    "search_movies": {
        "fn": search_movies,
        "schema": {"name": "search_movies",
                   "description": "Find movies by what happens in them (their plot). Use for "
                                  "'which movie is about ...' questions.",
                   "parameters": {"type": "object",
                                  "properties": {"query": {"type": "string"}},
                                  "required": ["query"]}},
    },
    "movie_facts": {
        "fn": movie_facts,
        "schema": {"name": "movie_facts",
                   "description": "Look up the year, director, and genre of ONE movie by its "
                                  "exact title. Use for 'who directed ...' or 'what year ...'.",
                   "parameters": {"type": "object",
                                  "properties": {"title": {"type": "string"}},
                                  "required": ["title"]}},
    },
}


def _tools_for_prompt() -> str:
    return json.dumps([t["schema"] for t in TOOLS.values()], indent=2)


SYSTEM = (
    "You are an agent that answers movie questions using tools when needed.\n"
    "Available tools:\n" + _tools_for_prompt() + "\n\n"
    "On each turn reply with EXACTLY ONE JSON object and nothing else, either:\n"
    '  a tool call:   {"tool": "<name>", "arguments": {...}}\n'
    '  or a final answer: {"final": "<your answer>"}\n'
    "Pick the tool that fits the task. If the question needs no lookup, answer directly."
)


def _extract_json(text: str) -> str:
    m = re.search(r"\{.*\}", text, re.S)
    return m.group(0) if m else text


def run_agent(question: str, max_steps: int = 5) -> str:
    """The think-act-observe loop, bounded by max_steps."""
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
            print(f"  step {step}: final")
            return action["final"]
        name, args = action.get("tool"), action.get("arguments", {})
        print(f"  step {step}: tool -> {name}({args})")
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
    for q in ["Which movie is about planting an idea in someone's dream?",  # -> search_movies
              "Who directed Inception?",                                     # -> movie_facts
              "What is 6 times 7?"]:                                         # -> no tool
        print(f"\nQ: {q}")
        print("A:", run_agent(q))
    # Expected: question 1 calls search_movies and names Inception; question 2 calls
    # movie_facts and returns the director; question 3 is answered directly with no tool.


if __name__ == "__main__":
    main()
