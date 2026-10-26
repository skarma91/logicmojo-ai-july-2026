"""Class 6.5 micro-assignment: find one injection in a movie team, and patch it.

data/reviews.jsonl is eight real films. One field on one row is a planted
instruction that tells the recommendation agent to email the user's watchlist
to an outside address. The class build did this to a tax team. Here the corpus
is reviews, and the defenses are yours.

Nothing here calls a model. Fill in every TODO, then compare with the README.

Run:  python assignment.py
"""

from __future__ import annotations
import json
import pathlib
import re

DATA = pathlib.Path(__file__).parent / "data" / "reviews.jsonl"
SKIP = {"id", "title", "year"}
EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")

TOOLS = [
    {"name": "search_films", "description": "Look up a film by title and return its plot.",
     "args": ["title"]},
    {"name": "send_email", "description": "Send a message to an address.",
     "args": ["to", "body"]},
    {"name": "render_poster", "description": "Show a poster by fetching the image_url the caller supplies.",
     "args": ["image_url"]},
    {"name": "delete_watchlist", "description": "Delete every title on the user's watchlist.",
     "args": ["user_id"]},
    {"name": "format_title", "description": "Return the title with the year in parentheses.",
     "args": ["title", "year"]},
]
ALLOWLIST = ["patron@films.example"]
SCENE = {
    "agent": "sender",
    "authority_tool": "send_email",
    "believes": "the studio note is an instruction from the research team",
    "user_asked": "email me my weekly list",
}
LEAKY = ("Parasite (2019): two families share one house. "
         "Press contact for this title: drops@exfil.example.")
CLEAN = "Parasite (2019): two families share one house."


def load() -> list[dict]:
    return [json.loads(line) for line in DATA.read_text().splitlines() if line.strip()]


# Problem 1
def find_injection(reviews: list[dict]) -> tuple[str, str, str]:
    """(row id, field name, destination email) of the planted instruction.

    Look at every string field except id, title, and year. The planted field
    tells an agent to call send_email and names an address.
    """
    # TODO(1): return the id, the field, and the email address inside it.
    raise NotImplementedError


# Problem 2
def exfil_channels(tools: list[dict]) -> list[str]:
    """Tool names that can carry data out or destroy it, sorted alphabetically.

    A tool counts if it sends a message, deletes stored data, or fetches a URL
    the caller supplies. Fetching a caller-supplied URL counts because the URL
    itself can hold the watchlist, and the host that receives the fetch sees it.
    """
    # TODO(2): return the sorted names. format_title and search_films do not count.
    raise NotImplementedError


# Problem 3
def split_privileges(tools: list[dict]) -> dict:
    """Who may hold which tool.

    Return {"reader": [...], "sender": [...], "sender_sees_raw": False}.
    The reader is the agent that reads untrusted reviews: it gets tools that look
    up outside text, and no outbound tool. The sender gets the mail tool and tools
    that only format text. Do not grant a tool that deletes data or fetches a
    caller-supplied URL. Keep each list in the order TOOLS is written.
    sender_sees_raw is False: the sender never reads raw review text.
    """
    # TODO(3): build the dict. Leave delete_watchlist and render_poster ungranted.
    raise NotImplementedError


# Problem 4
def allow(to: str, body: str, allowlist: list[str]) -> tuple[bool, str]:
    """(ok, reason). reason is "allowed", "destination", or "shape".

    Shape fails first: `to` must be one email (one @, a dot in the domain, no
    spaces), and `body` must be a non-empty string with no http or https URL.
    Destination fails when the shape is fine and `to` is not on the allowlist.
    A lookalike such as patron@films.example.exfil.test is a different string.
    """
    # TODO(4): check shape, then membership. Return (True, "allowed") only if both pass.
    raise NotImplementedError


# Problem 5
def name_deputy(scene: dict) -> tuple[str, str]:
    """(agent that holds legitimate authority, tool it is tricked into using).

    The scene has a distractor: what the agent believes is not the authority.
    """
    # TODO(5a): the deputy is the agent with the tool, not the sentence it believes.
    raise NotImplementedError


def residual(handoff: str, allowlist: list[str]) -> dict:
    """{"leaked_to": email or "", "stopped_by": "allowlist" or "privilege"}.

    A facts filter was already applied to `handoff`. If an address that is not on
    the allowlist is still in the text, the filter did not hold, and the layer
    that still stops a send to that address is the allowlist. If no such address
    remains, privilege held.
    """
    # TODO(5b): find an address the allowlist does not contain. Decide which layer
    #           is doing the work now.
    raise NotImplementedError


def main():
    reviews = load()
    print("[1] injection:", find_injection(reviews))
    print("[2] channels:", exfil_channels(TOOLS))
    print("[3] split:", split_privileges(TOOLS))
    cases = [
        ("patron@films.example", "Your weekly list: Moonlight, Arrival."),
        ("drops@exfil.example", "watchlist: Moonlight"),
        ("patron@films.example.exfil.test", "hi"),
        ("patron@films.example", "see https://evil.example/x?d=watchlist"),
        ("not an email", "hi"),
        ("patron@films.example", ""),
    ]
    print("[4] allow:", [allow(to, body, ALLOWLIST) for to, body in cases])
    print("[5] deputy:", name_deputy(SCENE))
    print("[5] leaky handoff:", residual(LEAKY, ALLOWLIST))
    print("[5] clean handoff:", residual(CLEAN, ALLOWLIST))


if __name__ == "__main__":
    main()
