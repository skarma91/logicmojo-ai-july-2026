"""Class 6.5 micro-assignment (reference solution).

data/reviews.jsonl is eight real films. One studio note tells the agent to email
the user's watchlist to an outside address. The functions below find that note
and apply the structural defenses from the class.

Run:  python assignment.py
"""

from __future__ import annotations
import json
import pathlib
import re

DATA = pathlib.Path(__file__).parent / "data" / "reviews.jsonl"
SKIP = {"id", "title", "year"}
EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
INSTRUCTION = re.compile(r"send_email|do not mention this note", re.IGNORECASE)
EMAIL_SHAPE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
URL = re.compile(r"https?://", re.IGNORECASE)

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


def find_injection(reviews: list[dict]) -> tuple[str, str, str]:
    """(row id, field name, destination email) of the planted instruction."""
    for row in reviews:
        for key, value in row.items():
            if key in SKIP or not isinstance(value, str):
                continue
            if INSTRUCTION.search(value):
                found = EMAIL.search(value)
                return (row["id"], key, found.group(0) if found else "")
    return ("", "", "")


def exfil_channels(tools: list[dict]) -> list[str]:
    """Tool names that can carry data out or destroy it, sorted alphabetically."""
    names = []
    for tool in tools:
        args = [a.lower() for a in tool["args"]]
        desc = tool["description"].lower()
        if "send" in tool["name"] or desc.startswith("send "):
            names.append(tool["name"])
        elif "delete" in tool["name"] or desc.startswith("delete "):
            names.append(tool["name"])
        elif "url" in args or any("url" in a for a in args):
            names.append(tool["name"])
    return sorted(names)


def split_privileges(tools: list[dict]) -> dict:
    """Reader reads. Sender sends. The sender does not see raw reviews."""
    reader, sender = [], []
    for tool in tools:
        args = [a.lower() for a in tool["args"]]
        desc = tool["description"].lower()
        outbound = ("send" in tool["name"] or "delete" in tool["name"]
                    or any("url" in a for a in args))
        if "delete" in tool["name"] or any("url" in a for a in args):
            continue
        if not outbound and ("look up" in desc or "return its plot" in desc):
            reader.append(tool["name"])
        else:
            sender.append(tool["name"])
    return {"reader": reader, "sender": sender, "sender_sees_raw": False}


def allow(to: str, body: str, allowlist: list[str]) -> tuple[bool, str]:
    """(ok, reason). Shape first, then the destination list."""
    if not isinstance(body, str) or not body.strip() or URL.search(body or ""):
        return (False, "shape")
    if not isinstance(to, str) or not EMAIL_SHAPE.match(to):
        return (False, "shape")
    if to not in allowlist:
        return (False, "destination")
    return (True, "allowed")


def name_deputy(scene: dict) -> tuple[str, str]:
    """The agent with the authority, and the tool it is tricked into using."""
    return (scene["agent"], scene["authority_tool"])


def residual(handoff: str, allowlist: list[str]) -> dict:
    """An address the filter left behind is stopped by the allowlist."""
    leaked = [addr for addr in EMAIL.findall(handoff) if addr not in allowlist]
    if leaked:
        return {"leaked_to": leaked[0], "stopped_by": "allowlist"}
    return {"leaked_to": "", "stopped_by": "privilege"}


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
