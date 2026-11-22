"""Load the Wikivoyage chunks, and the fixed vocabularies the agents share."""

from __future__ import annotations
import json
import pathlib

DATA = pathlib.Path(__file__).parent / "data" / "guides.jsonl"
PLACE_SECTIONS = ("See", "Do", "Eat", "Drink")   # sections that name visitable places
MAX_DAYS = 7
PACE_LIMIT = {"relaxed": 2, "moderate": 3, "packed": 4}   # activities per day, at most

# Each interest becomes a retrieval query: the words that signal it in guide text.
INTERESTS = {
    "food": "food restaurant eat dish cuisine cafe street food dining market",
    "history": "history historic old museum castle fortress palace monument ruins heritage",
    "art": "art museum gallery modern contemporary painting exhibition",
    "architecture": "architecture building cathedral church mosque temple tower bridge",
    "nature": "park garden beach hike mountain nature view lake river",
    "nightlife": "bar club nightlife live music pub cocktails",
    "shopping": "shopping market mall shops bazaar souk boutique",
    "culture": "culture festival theatre performance traditional neighbourhood district",
}


def load_chunks() -> list[dict]:
    if not DATA.exists():
        raise SystemExit(f"{DATA} not found: run  python fetch_data.py  first")
    return [json.loads(line) for line in DATA.read_text().splitlines() if line.strip()]


CHUNKS = load_chunks()
DESTINATIONS = sorted({c["destination"] for c in CHUNKS})
