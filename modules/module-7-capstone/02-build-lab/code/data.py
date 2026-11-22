"""Load the help centre and the labeled tickets (pre-written).

Run `python fetch_data.py --sample` once if data/kb.jsonl or data/tickets.jsonl is missing.
"""

from __future__ import annotations
import json
import pathlib

DATA = pathlib.Path(__file__).parent / "data"


def _read(name: str) -> list[dict]:
    path = DATA / name
    if not path.exists():
        raise SystemExit(f"{path} not found: run  python fetch_data.py --sample  first")
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def load_kb() -> list[dict]:
    """27 help articles: {intent, category, title, body}."""
    return _read("kb.jsonl")


def load_tickets() -> list[dict]:
    """Labeled tickets: {id, text, intent, category, flags}. Agents see only `text`."""
    return _read("tickets.jsonl")


INTENTS = sorted(a["intent"] for a in load_kb())
CATEGORY = {a["intent"]: a["category"] for a in load_kb()}
