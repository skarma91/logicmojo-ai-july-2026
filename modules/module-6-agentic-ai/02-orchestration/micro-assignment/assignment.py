"""Class 6.2 micro-assignment: a writer-reviewer loop for movie blurbs.

A writer drafts a short blurb for a film from the real movie dataset
(data/movies.jsonl). A reviewer checks it and either approves it or sends it back
with specific feedback, until it passes or MAX_ROUNDS is reached.

    START -> writer -> reviewer --approve or cap--> END
               ^          |
               +-revise---+

A blurb passes only if (a) exact checks pass: under 60 words and it names the
director, and (b) the LLM reviewer approves (it checks the blurb is faithful to the
plot). Exact checks override the LLM. The writer's brief deliberately leaves out
the director rule, so the reviewer has something real to catch.

Fill in every TODO, then run and compare with the README's expected output.

Run:  python assignment.py      (needs PROVIDER=groq or ollama in .env)
"""

from __future__ import annotations
import json
import logging
import operator
import pathlib
from typing import Annotated, TypedDict

from langgraph.graph import StateGraph, START, END

import llm

log = logging.getLogger("course.blurb_loop")
DATA = pathlib.Path(__file__).parent / "data" / "movies.jsonl"
MOVIES = {m["id"]: m for m in (json.loads(l) for l in DATA.read_text().splitlines() if l.strip())}
MAX_WORDS = 60
MAX_ROUNDS = 3

WRITER_SYSTEM = ("You write punchy one-paragraph movie blurbs for a streaming catalog. "
                 "Use only the facts given. If you receive reviewer feedback, fix exactly that.")

REVIEWER_SYSTEM = ("You review a movie blurb against the film's record. Approve only if every "
                   "claim is supported by the record (no invented plot points, cast, or awards). "
                   "Otherwise request a revision with one specific fix. Call verdict exactly once.")

VERDICT_TOOL = {"type": "function", "function": {
    "name": "verdict",
    "description": "Approve the blurb, or request a revision with specific feedback.",
    "parameters": {"type": "object", "properties": {
        "approve": {"type": "boolean"},
        "feedback": {"type": "string", "description": "one specific fix; empty if approved"}},
        "required": ["approve", "feedback"]}}}


# Problem 1
def check_blurb(blurb: str, movie: dict) -> list[str]:
    """Exact checks. Returns the failed checks (empty list means all passed)."""
    # TODO(1): return the failed checks, in this order and with this exact wording:
    #   f"too long ({words} words; limit {MAX_WORDS})" if the blurb has more than MAX_WORDS words
    #   f"does not name the director ({movie['director']})" if the director's name (case-insensitive)
    #   does not appear in the blurb
    raise NotImplementedError


# Problem 2
def combine(failed: list[str], verdict: dict) -> tuple[bool, str]:
    """Merge exact failures with the LLM verdict. Exact failures override approve."""
    # TODO(2): approved only if the LLM approved AND no exact check failed.
    #          Approved -> (True, ""). Otherwise (False, the failed checks followed by the LLM's
    #          feedback if it gave any, joined with "; ").
    raise NotImplementedError


# Problem 3
def route_after_review(state: dict) -> str:
    """Back to the writer, or END on approval or at the round cap."""
    # TODO(3): return END if the blurb was approved or rounds has reached MAX_ROUNDS;
    #          otherwise return "writer".
    raise NotImplementedError


# Problem 4
def expected_rounds(q: float, k: int) -> float:
    """E[rounds] = (1 - (1-q)^k) / q for a loop capped at k rounds."""
    # TODO(4): implement the formula. Check what happens to it at q = 0 and handle that case.
    raise NotImplementedError


# Problem 5
def estimate_q(rounds: list[int], approved: list[bool]) -> float:
    """Per-round pass rate q estimated from observed runs."""
    # TODO(5): estimate the chance a single review round passes from observed runs.
    #          Think about what one trial is, and what a run that hit the cap contributes.
    raise NotImplementedError


# ---------------------------------------------------------------------------
# The loop (real model calls)
# ---------------------------------------------------------------------------
class BlurbState(TypedDict):
    movie: dict
    blurb: str
    approved: bool
    feedback: str
    rounds: Annotated[int, operator.add]
    reviews: Annotated[list, operator.add]


def record(movie: dict) -> str:
    return (f"Title: {movie['title']} ({movie['year']})\nGenre: {movie['genre']}\n"
            f"Director: {movie['director']}\nPlot: {movie['plot']}")


def build_loop():
    def writer(state: BlurbState) -> dict:
        ask = f"Write a blurb under {MAX_WORDS} words for:\n{record(state['movie'])}"
        if state["feedback"]:
            ask += f"\n\nYour previous blurb:\n{state['blurb']}\n\nReviewer feedback to fix:\n{state['feedback']}"
        blurb = llm.chat([{"role": "system", "content": WRITER_SYSTEM},
                          {"role": "user", "content": ask}], agent="writer")
        return {"blurb": blurb.strip()}

    def reviewer(state: BlurbState) -> dict:
        failed = check_blurb(state["blurb"], state["movie"])
        reply = llm.chat_tools(
            [{"role": "system", "content": REVIEWER_SYSTEM},
             {"role": "user", "content": f"Record:\n{record(state['movie'])}\n\nBlurb:\n{state['blurb']}"}],
            [VERDICT_TOOL], tool_choice="required", agent="reviewer")
        verdict = reply.tool_calls[0].args if reply.wants_tool else {"approve": False, "feedback": "no verdict returned"}
        approved, feedback = combine(failed, verdict)
        return {"approved": approved, "feedback": feedback, "rounds": 1,
                "reviews": [{"failed": failed, "llm_approve": bool(verdict.get("approve"))}]}

    g = StateGraph(BlurbState)
    g.add_node("writer", writer)
    g.add_node("reviewer", reviewer)
    g.add_edge(START, "writer")
    g.add_edge("writer", "reviewer")
    g.add_conditional_edges("reviewer", route_after_review, {"writer": "writer", END: END})
    return g.compile()


def main():
    logging.basicConfig(level=logging.WARNING)
    nolan = MOVIES["inception-2010"]
    good = "A thief who steals secrets inside dreams takes one last job. Directed by Christopher Nolan."
    print("[1] check_blurb(good)   ->", check_blurb(good, nolan))
    print("    check_blurb(no dir) ->", check_blurb("A thief who steals secrets inside dreams.", nolan))
    print("    check_blurb(long)   ->", check_blurb("word " * 61 + "Christopher Nolan", nolan))
    print("[2] combine(exact fail, LLM approves) ->",
          combine(["does not name the director (Christopher Nolan)"], {"approve": True, "feedback": ""}))
    print("    combine(no fails, LLM approves)   ->", combine([], {"approve": True, "feedback": ""}))
    print("[3] route(approved, round 1) ->", route_after_review({"approved": True, "rounds": 1}),
          "| route(rejected, round 1) ->", route_after_review({"approved": False, "rounds": 1}),
          "| route(rejected, round 3) ->", route_after_review({"approved": False, "rounds": 3}))
    print("[4] expected_rounds:", [round(expected_rounds(q, 3), 3) for q in (0.5, 0.8, 0.0, 1.0)])
    print("[5] estimate_q([1, 2, 3], [True, True, False]) ->", round(estimate_q([1, 2, 3], [True, True, False]), 3))

    loop = build_loop()
    llm.reset_usage()
    rounds, approved = [], []
    for mid in ["inception-2010", "parasite-2019", "coco-2017", "italian-job-1969"]:
        s = loop.invoke({"movie": MOVIES[mid], "blurb": "", "approved": False, "feedback": "",
                         "rounds": 0, "reviews": []}, config={"recursion_limit": 20})
        rounds.append(s["rounds"])
        approved.append(s["approved"])
        print(f"\n{MOVIES[mid]['title']} ({MOVIES[mid]['year']}): rounds={s['rounds']} approved={s['approved']}")
        for i, r in enumerate(s["reviews"], 1):
            print(f"   round {i}: exact failures={r['failed'] or 'none'}  llm_approve={r['llm_approve']}")
        print("   blurb:", s["blurb"])

    q_hat = estimate_q(rounds, approved)
    print(f"\nrounds per blurb={rounds}  max={max(rounds)} (never above {MAX_ROUNDS})")
    print(f"estimated q={q_hat:.2f}  predicted mean rounds={expected_rounds(q_hat, MAX_ROUNDS):.2f}  "
          f"observed mean={sum(rounds) / len(rounds):.2f}")
    print("calls per agent:", {a: u["calls"] for a, u in llm.usage_by_agent().items()})


if __name__ == "__main__":
    main()
