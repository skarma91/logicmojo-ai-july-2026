"""Download Wikivoyage destination guides into data/guides.jsonl (given).

Wikivoyage (https://en.wikivoyage.org) is a free, worldwide travel guide. Its text is
licensed CC BY-SA 4.0: you may reuse it with attribution, and anything you share that
is built from it stays under the same license. Every chunk keeps its page URL, so the
planner can cite where each place came from.

This uses the public MediaWiki API with one request at a time and a pause between
requests, as Wikimedia asks of automated clients. Set USER_AGENT below (or the
WIKI_USER_AGENT environment variable) to name your project and a way to contact you.

    python fetch_data.py                      # the default destinations below
    python fetch_data.py Lisbon Kyoto Nairobi # any destinations you like

Output, one JSON object per line:
    {"id": "Lisbon#See#3", "destination": "Lisbon", "section": "See",
     "text": "...", "url": "https://en.wikivoyage.org/wiki/Lisbon"}
"""

from __future__ import annotations
import json
import os
import pathlib
import re
import sys
import time
import urllib.parse
import urllib.request

API = "https://en.wikivoyage.org/w/api.php"
OUT = pathlib.Path(__file__).parent / "data" / "guides.jsonl"
USER_AGENT = os.getenv("WIKI_USER_AGENT", "genai-course-travel-planner/1.0 (educational project)")
PAUSE_S = 1.0
DESTINATIONS = ["Lisbon", "Kyoto", "Mexico City", "Cape Town", "Istanbul", "Hanoi",
                "Edinburgh", "Buenos Aires", "Marrakech", "Vancouver", "Prague", "Singapore"]
SECTIONS = {"Understand", "Get around", "See", "Do", "Eat", "Drink", "Stay safe"}
MAX_WORDS = 120                       # chunk size: one listing or a short paragraph


def fetch_extract(title: str) -> str:
    """The page as plain text with == Section == headings."""
    params = urllib.parse.urlencode({"action": "query", "prop": "extracts", "explaintext": 1,
                                     "exsectionformat": "wiki", "redirects": 1,
                                     "titles": title, "format": "json", "formatversion": 2})
    req = urllib.request.Request(f"{API}?{params}", headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=60) as resp:
        pages = json.loads(resp.read())["query"]["pages"]
    return pages[0].get("extract", "") if pages else ""


def split_sections(extract: str) -> dict[str, str]:
    """Top-level sections only (== Name ==); subsections stay inside their parent."""
    parts = re.split(r"^==\s*([^=].*?)\s*==\s*$", extract, flags=re.M)
    return {parts[i].strip(): parts[i + 1] for i in range(1, len(parts) - 1, 2)}


def chunk(text: str) -> list[str]:
    """Paragraphs, merged or cut so each chunk holds at most MAX_WORDS words."""
    text = re.sub(r"^=+\s*(.*?)\s*=+\s*$", r"\1:", text, flags=re.M)   # subsection titles become labels
    chunks, current = [], []
    for para in (p.strip() for p in text.split("\n")):
        words = para.split()
        if not words:
            continue
        if len(current) + len(words) > MAX_WORDS and current:
            chunks.append(" ".join(current))
            current = []
        while len(words) > MAX_WORDS:
            chunks.append(" ".join(words[:MAX_WORDS]))
            words = words[MAX_WORDS:]
        current += words
    if current:
        chunks.append(" ".join(current))
    return [c for c in chunks if len(c.split()) >= 8]                  # drop stubs and lone labels


def guide_chunks(destination: str, extract: str) -> list[dict]:
    url = "https://en.wikivoyage.org/wiki/" + urllib.parse.quote(destination.replace(" ", "_"))
    out = []
    for section, body in split_sections(extract).items():
        if section in SECTIONS:
            for i, text in enumerate(chunk(body), 1):
                out.append({"id": f"{destination}#{section}#{i}", "destination": destination,
                            "section": section, "text": text, "url": url})
    return out


def main():
    destinations = sys.argv[1:] or DESTINATIONS
    rows = []
    for i, dest in enumerate(destinations):
        if i:
            time.sleep(PAUSE_S)
        try:
            new = guide_chunks(dest, fetch_extract(dest))
        except Exception as e:                   # one failed page should not lose the rest
            print(f"  skipped {dest}: {e}")
            continue
        rows += new
        print(f"  {dest:<14} {len(new):>4} chunks")
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    print(f"wrote {len(rows)} chunks for {len({r['destination'] for r in rows})} destinations to data/guides.jsonl")


if __name__ == "__main__":
    main()
