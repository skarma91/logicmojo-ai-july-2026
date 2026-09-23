"""Download a real MedlinePlus corpus into data/corpus.jsonl (given).

The committed data/example_corpus.jsonl is a small, real sample condensed from
MedlinePlus (U.S. National Library of Medicine, public domain). This script pulls a
much larger corpus from the official MedlinePlus Web Service, so RAG has real
retrieval work to do and the RAG-versus-fine-tuning question becomes meaningful (a
corpus of a dozen passages cannot settle it; a few dozen topics can). This is the
"real data plus a fetch script" workflow from classes 4.4 and 4.5.

It queries the Web Service (https://wsearch.nlm.nih.gov/ws/query), which returns
XML search results for the health-topics database, and writes one record per topic
in the same schema the assistant expects: id, title, source, section, url, text.

Network is required to run this; it is NOT needed to use the shipped sample.

Run:
    pip install requests
    python fetch_data.py              # writes data/corpus.jsonl
"""

from __future__ import annotations
import html
import json
import pathlib
import re
import sys
import xml.etree.ElementTree as ET

import requests

OUT = pathlib.Path(__file__).parent / "data" / "corpus.jsonl"
ENDPOINT = "https://wsearch.nlm.nih.gov/ws/query"

# The topics to build the corpus from. Add or change these to grow the corpus. A
# broad list gives the retriever varied, non-overlapping material (a few dozen topics
# yields a corpus large enough to make the RAG-versus-fine-tuning question meaningful).
TOPICS = [
    "high blood pressure", "cholesterol", "type 2 diabetes", "type 1 diabetes",
    "healthy sleep", "exercise and physical fitness", "benefits of exercise",
    "flu", "colds", "COVID-19", "asthma", "COPD", "heart disease prevention",
    "heart attack", "stroke", "nutrition", "dietary fats", "sodium in your diet",
    "obesity", "body weight", "smoking", "quitting smoking", "alcohol",
    "anxiety", "depression", "stress", "vaccines", "immunization", "hydration",
    "vitamin D", "calcium", "bone density", "arthritis", "migraine",
    "high blood pressure in adults", "dental health", "skin cancer", "sun exposure",
]


def _clean(text: str) -> str:
    """Strip HTML tags and collapse whitespace from a MedlinePlus summary."""
    text = re.sub(r"<[^>]+>", " ", html.unescape(text or ""))
    return re.sub(r"\s+", " ", text).strip()


def fetch_topic(term: str) -> dict | None:
    """Return the top health-topic result for a term, or None if nothing matched."""
    params = {"db": "healthTopics", "term": term, "retmax": "1"}
    resp = requests.get(ENDPOINT, params=params, timeout=30)
    resp.raise_for_status()
    root = ET.fromstring(resp.text)
    doc = root.find(".//document")
    if doc is None:
        return None
    fields = {c.get("name"): _clean(c.text or "") for c in doc.findall("content")}
    title = fields.get("title", term.title())
    summary = fields.get("FullSummary") or fields.get("snippet", "")
    url = doc.get("url", "")
    slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
    return {"id": f"{slug}-overview", "title": title, "source": "MedlinePlus",
            "section": "Overview", "url": url, "text": _clean(summary)[:600]}


def main():
    records = []
    for term in TOPICS:
        try:
            rec = fetch_topic(term)
        except Exception as e:                       # network or parse error
            print(f"skip {term!r}: {e}", file=sys.stderr)
            continue
        if rec and rec["text"]:
            records.append(rec)
            print(f"fetched: {rec['title']}")
    if not records:
        sys.exit("no records fetched (network required); the shipped sample still works")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(records)} records to {OUT}")


if __name__ == "__main__":
    main()
