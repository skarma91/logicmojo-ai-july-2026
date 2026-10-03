"""Download a real NASA corpus into data/corpus.jsonl (given).

The committed data/example_corpus.jsonl is a small, real sample from the NASA Image
and Video Library (images.nasa.gov, public domain). This script pulls a much larger
corpus from the same public API, so RAG has real retrieval work to do and the
RAG-versus-fine-tuning question becomes meaningful (a corpus of a few dozen passages
cannot settle it; a few hundred can).

The API needs no key and no login:
    https://images-api.nasa.gov/search?q=<term>&media_type=image

It writes one record per image in the schema the retriever expects:
    id, title, source, section (NASA center), url, text (the public-domain caption).

Network is required to run this; it is NOT needed to use the shipped sample.

Run:
    pip install requests
    python fetch_data.py                 # ~200 records across the default topics
    python fetch_data.py --per 30        # more per topic
"""

from __future__ import annotations
import argparse
import html
import json
import pathlib
import re
import sys

import requests

OUT = pathlib.Path(__file__).parent / "data" / "corpus.jsonl"
ENDPOINT = "https://images-api.nasa.gov/search"

# Broad space topics, chosen to give the retriever varied, non-overlapping material.
TOPICS = [
    "nebula", "galaxy", "black hole", "supernova", "Mars rover", "Moon landing",
    "Apollo", "International Space Station", "Saturn", "Jupiter", "solar flare",
    "Hubble telescope", "James Webb telescope", "asteroid", "comet", "Mercury planet",
    "Venus planet", "Pluto", "exoplanet", "spacewalk", "Voyager", "rocket launch",
]
_BOILERPLATE = re.compile(
    r"(NASA image use policy|Follow us on|Like us on|Find us on|Read more:|Credits?:).*",
    re.IGNORECASE | re.DOTALL,
)


def clean(text: str) -> str:
    """Strip HTML, entities, URLs, and trailing social/credit boilerplate."""
    text = html.unescape(text or "")
    text = re.sub(r"<[^>]+>", " ", text)
    text = _BOILERPLATE.sub("", text)
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    # Trim to the last complete sentence within 600 chars.
    text = text[:600]
    cut = text.rfind(". ")
    return (text[:cut + 1] if cut > 120 else text).strip()


def fetch_topic(term: str, per: int) -> list[dict]:
    resp = requests.get(ENDPOINT, params={"q": term, "media_type": "image",
                                          "page_size": per}, timeout=30)
    resp.raise_for_status()
    items = resp.json().get("collection", {}).get("items", [])
    records = []
    for it in items:
        data = (it.get("data") or [{}])[0]
        text = clean(data.get("description", ""))
        if len(text) < 120:                       # keep only informative passages
            continue
        nid = data.get("nasa_id")
        if not nid:
            continue
        records.append({
            "id": nid,
            "title": data.get("title", "").strip(),
            "source": "NASA Image and Video Library",
            "section": data.get("center", "NASA"),
            "url": f"https://images.nasa.gov/details/{nid}",
            "text": text,
        })
    return records


def main():
    ap = argparse.ArgumentParser(description="Download a NASA corpus for the milestone.")
    ap.add_argument("--per", type=int, default=12, help="images to pull per topic")
    args = ap.parse_args()

    seen, out = set(), []
    for term in TOPICS:
        try:
            for rec in fetch_topic(term, args.per):
                if rec["id"] not in seen:
                    seen.add(rec["id"])
                    out.append(rec)
        except Exception as e:                    # network or parse error
            print(f"skip {term!r}: {e}", file=sys.stderr)
    if not out:
        sys.exit("no records fetched (network required); the shipped sample still works")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(out)} records to {OUT}")


if __name__ == "__main__":
    main()
