"""Class 6.3 micro-assignment (reference solution): trim the critic's context.

A two-agent movie team over the real movie dataset (data/movies.jsonl):

    scout  (deterministic)  finds the films in the requested genre
    critic (LLM)            picks one film for the viewer's request

The naive critic is handed everything: the whole catalog as verbose JSON plus the
scout's raw output. This solution trims its context to what the job needs (the
films in the genre, and only the fields that matter), then measures the critic's
real prompt tokens both ways and checks the pick passes the same test.

Run:  python assignment.py      (needs PROVIDER=groq, ollama, or gemini in .env)
"""

from __future__ import annotations
import json
import logging
import pathlib

import llm

log = logging.getLogger("course.trim_context")
DATA = pathlib.Path(__file__).parent / "data" / "movies.jsonl"
MOVIES = [json.loads(l) for l in DATA.read_text().splitlines() if l.strip()]
FIELDS = ["title", "year", "director", "plot"]      # what the critic actually uses

CRITIC_SYSTEM = ("You are a film critic on a streaming team. From the films provided, pick "
                 "exactly one for the viewer. Reply as: Title (year): one sentence on why.")

REQUESTS = [("a heartfelt animated film about family", "Animation"),
            ("a clever heist where the plan matters", "Crime")]


# --- The scout: deterministic, over the real dataset -----------------------
def scout_raw(genre: str) -> str:
    """What the scout hands over today: every field, pretty-printed."""
    return json.dumps([m for m in MOVIES if m["genre"] == genre], indent=2)


# Problem 1
def select_records(genre: str) -> list[dict]:
    """The films in this genre, newest first."""
    return sorted((m for m in MOVIES if m["genre"] == genre), key=lambda m: -m["year"])


# Problem 2
def project(record: dict, fields: list[str]) -> dict:
    """Keep only the named fields, in the order given."""
    return {f: record[f] for f in fields}


# Problem 3
def critic_context(genre: str) -> str:
    """The trimmed context: one line per film in the genre, only the needed fields."""
    lines = []
    for m in select_records(genre):
        p = project(m, FIELDS)
        lines.append(f"{p['title']} ({p['year']}), dir. {p['director']}: {p['plot']}")
    return "\n".join(lines)


def full_context(genre: str) -> str:
    """The naive context: the whole catalog plus the scout's raw output."""
    return (f"FULL CATALOG:\n{json.dumps(MOVIES, indent=2)}\n\n"
            f"SCOUT OUTPUT FOR {genre}:\n{scout_raw(genre)}")


# Problem 4
def shared_prefix_tokens(a: str, b: str) -> int:
    """How many leading whitespace-separated tokens two prompts share.

    A prompt cache can reuse only this shared prefix: matching stops at the first
    token that differs, however much identical text follows it.
    """
    n = 0
    for x, y in zip(a.split(), b.split()):
        if x != y:
            break
        n += 1
    return n


# Problem 5
def passes(answer: str, candidates: list[dict]) -> bool:
    """The pick names a candidate as 'Title (year)'.

    Matching the title alone is not enough: two films are called The Italian Job
    (1969 and 2003), so only the title and year together identify one film.
    """
    return any(f"{m['title']} ({m['year']})" in answer for m in candidates)


def ask_critic(request: str, context: str, agent: str) -> tuple[str, int]:
    llm.reset_usage()
    answer = llm.chat([{"role": "system", "content": CRITIC_SYSTEM},
                       {"role": "user", "content": f"Films:\n{context}\n\nViewer wants: {request}"}],
                      agent=agent)
    return answer.strip(), llm.usage_by_agent()[agent]["prompt"]


def main():
    logging.basicConfig(level=logging.WARNING)
    inception = next(m for m in MOVIES if m["title"] == "Inception")
    print("[1] select_records('Animation') ->", [m["title"] for m in select_records("Animation")])
    print("[2] project(Inception, title/year/director) ->", project(inception, ["title", "year", "director"]))
    ctx = critic_context("Animation")
    print("[3] critic_context('Animation'): lines =", len(ctx.splitlines()),
          "| first =", ctx.splitlines()[0][:40] + "...")
    print("[4] shared prefix, volatile first ->",
          shared_prefix_tokens("Request 1a2b You are a film critic. Pick one.",
                               "Request 9f8e You are a film critic. Pick one."),
          "| stable first ->",
          shared_prefix_tokens("You are a film critic. Pick one. Request 1a2b",
                               "You are a film critic. Pick one. Request 9f8e"))
    italian_1969 = [m for m in MOVIES if m["id"] == "italian-job-1969"]
    animation = select_records("Animation")
    print("[5] passes:", passes("I pick The Italian Job (2003).", italian_1969),
          passes("Watch Coco (2017): a moving story.", animation),
          passes("Watch Coco: a moving story.", animation))

    print()
    for request, genre in REQUESTS:
        candidates = select_records(genre)
        full_answer, full_tokens = ask_critic(request, full_context(genre), "critic_full")
        trim_answer, trim_tokens = ask_critic(request, critic_context(genre), "critic_trimmed")
        print(f"{request!r} ({genre})")
        print(f"   full    prompt_tokens={full_tokens:<6} passes={passes(full_answer, candidates)}  {full_answer}")
        print(f"   trimmed prompt_tokens={trim_tokens:<6} passes={passes(trim_answer, candidates)}  {trim_answer}")
        print(f"   tokens saved: {1 - trim_tokens / full_tokens:.0%}" if full_tokens else "")


if __name__ == "__main__":
    main()
