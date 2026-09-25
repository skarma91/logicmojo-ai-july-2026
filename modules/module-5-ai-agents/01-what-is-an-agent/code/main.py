"""Class 5.1 build: the simplest possible agent loop.

A plain LLM answers in one shot. An agent decides to use a tool, acts, observes
the result, and decides again, looping until it produces a final answer. This is
that loop, hand-rolled, with exactly one tool: the Module 4 retriever.

We use the provider-agnostic MANUAL-JSON protocol from class 4.3: the model replies
with ONE JSON object, either a tool request or a final answer, and our code parses
it. That is what "hand-roll the loop" means. Class 5.2 replaces this with the
provider's native tool-calling API and contrasts the two.

The loop is BOUNDED from day one: it stops when the model returns a final answer,
or when a step cap is reached (previewing the rails in class 5.6). An unbounded
agent does not stop on its own.

Run:
    pip install sentence-transformers numpy python-dotenv
    # plus your provider, e.g.:  pip install ollama   (and `ollama serve`)
    python main.py

Model calls are REAL (through llm.chat). Set a provider in a .env at the repo root.
"""

from __future__ import annotations
import json
import logging
import re

import llm            # the course wrapper: llm.chat(messages) -> reply text
import retriever      # retriever.search(query, k) -> list of passage dicts

log = logging.getLogger("course.agent")

# ---------------------------------------------------------------------------
# The one tool: the Module 4 retriever, wrapped with a schema the model reads.
# The description is what the model uses to decide WHEN to call the tool, so it
# is written for the model, not for us (class 5.2 makes this a first-class idea).
# ---------------------------------------------------------------------------
def search_tax_docs(query: str) -> str:
    """Search the tax-publication corpus and return the top passages with sources."""
    hits = retriever.search(query, k=3)
    return "\n".join(f"[{h['id']}] {h['text']} (source: {h['pub']} {h['tax_year']}, p.{h['page']})"
                     for h in hits)


TOOLS = {
    "search_tax_docs": {
        "fn": search_tax_docs,
        "schema": {
            "name": "search_tax_docs",
            "description": "Search official tax publications for a fact. Use this whenever "
                           "the question asks about tax rules, limits, deductions, or amounts.",
            "parameters": {"type": "object",
                           "properties": {"query": {"type": "string"}},
                           "required": ["query"]},
        },
    },
}


def _tools_for_prompt() -> str:
    return json.dumps([t["schema"] for t in TOOLS.values()], indent=2)


# The system prompt is what turns a chat model into an agent: it states the tools,
# the output contract (one JSON object), and the stopping rule (a final answer).
SYSTEM = (
    "You are an agent that answers questions using tools when needed.\n"
    "Available tools:\n" + _tools_for_prompt() + "\n\n"
    "On each turn reply with EXACTLY ONE JSON object and nothing else, either:\n"
    '  a tool call:   {"tool": "<name>", "arguments": {...}}\n'
    '  or a final answer: {"final": "<your answer, citing the [id] sources you used>"}\n'
    "Call a tool only when you need it. If you already know the answer, or the "
    "question needs no lookup, reply with a final answer directly."
)


def _extract_json(text: str) -> str:
    """Grab the first {...} block, since models sometimes add prose or code fences."""
    m = re.search(r"\{.*\}", text, re.S)
    return m.group(0) if m else text


# ---------------------------------------------------------------------------
# The think, act, observe loop.
# ---------------------------------------------------------------------------
def run_agent(question: str, max_steps: int = 5) -> str:
    """Loop: the model thinks and emits JSON (think), we run any tool (act), we feed
    the result back (observe), and repeat until a final answer or the step cap."""
    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": question}]

    for step in range(1, max_steps + 1):              # step cap: the loop is bounded
        raw = llm.chat(messages)                       # THINK: a real model call
        try:
            action = json.loads(_extract_json(raw))
        except json.JSONDecodeError:
            messages.append({"role": "user", "content": "Reply with ONLY one JSON object."})
            continue

        if "final" in action:                          # STOP: the model is done
            print(f"  step {step}: final answer")
            return action["final"]

        name = action.get("tool")
        args = action.get("arguments", {})
        print(f"  step {step}: tool -> {name}({args})")
        if name not in TOOLS:                           # guard: model asked for an unknown tool
            messages.append({"role": "assistant", "content": raw})
            messages.append({"role": "user", "content":
                             f"No tool named {name!r}. Available: {list(TOOLS)}."})
            continue

        result = TOOLS[name]["fn"](**args)             # ACT: run the tool (real call)
        messages.append({"role": "assistant", "content": json.dumps(action)})   # the request
        messages.append({"role": "user", "content": f"Tool result:\n{result}"}) # OBSERVE

    return "stopped: step cap reached before a final answer"


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    for q in ["How much can I contribute to my HSA?",         # needs the tool
              "What is 2 plus 2?"]:                            # needs no tool: answer directly
        print(f"\nQ: {q}")
        print("A:", run_agent(q))


if __name__ == "__main__":
    main()
