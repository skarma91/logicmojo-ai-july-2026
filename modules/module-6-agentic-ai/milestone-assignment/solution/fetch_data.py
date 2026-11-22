"""Download a real arXiv corpus of abstracts into data/corpus.jsonl.

Uses the public arXiv API (https://info.arxiv.org/help/api/), which asks clients to
make at most one request every three seconds; this script waits three seconds
between pages. arXiv metadata, abstracts included, is released under CC0.

Each record keeps its arXiv id and abstract URL, so every answer the team writes
can be traced to its source:

    {"id": "2308.08155", "title": ..., "authors": ..., "year": 2023,
     "categories": "cs.AI cs.CL", "url": "https://arxiv.org/abs/2308.08155", "text": <abstract>}

    python fetch_data.py                 # about 300 papers on agents, RAG, and LLM evaluation
    python fetch_data.py --per-query 25  # a smaller corpus
"""

from __future__ import annotations
import argparse
import json
import pathlib
import re
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

API = "https://export.arxiv.org/api/query"
OUT = pathlib.Path(__file__).parent / "data" / "corpus.jsonl"
ATOM = {"a": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}
WAIT_S = 3.0                                   # arXiv's published request pacing

QUERIES = [                                    # topics the research team should know about
    'abs:"multi-agent" AND abs:"language model"',
    'abs:"tool use" AND abs:"language model"',
    'abs:"retrieval-augmented generation"',
    'abs:"LLM agent" AND abs:evaluation',
    'abs:"chain-of-thought" AND abs:reasoning',
    'abs:"prompt" AND abs:"self-reflection"',
    'abs:"LLM-as-a-judge"',
    'abs:"agent" AND abs:"planning" AND abs:"language model"',
    'abs:"hallucination" AND abs:"large language model"',
    'abs:"code generation" AND abs:"agent"',
    'abs:"long context" AND abs:"language model"',
    'abs:"instruction tuning"',
]


def clean(text: str) -> str:
    """Collapse whitespace, strip simple LaTeX, and avoid double hyphens in prose."""
    text = re.sub(r"\\(?:textit|textbf|emph|url)\{([^}]*)\}", r"\1", text)
    text = re.sub(r"\$([^$]*)\$", r"\1", text)
    text = re.sub(r"\s*-{2,3}\s*", ", ", text.replace("\\&", "&"))
    return " ".join(text.split())


def fetch_page(query: str, start: int, size: int) -> list[dict]:
    params = urllib.parse.urlencode({"search_query": query, "start": start, "max_results": size,
                                     "sortBy": "relevance", "sortOrder": "descending"})
    with urllib.request.urlopen(f"{API}?{params}", timeout=60) as resp:
        root = ET.fromstring(resp.read())
    rows = []
    for e in root.findall("a:entry", ATOM):
        arxiv_id = e.findtext("a:id", "", ATOM).rsplit("/abs/", 1)[-1]
        arxiv_id = re.sub(r"v\d+$", "", arxiv_id)
        authors = [a.findtext("a:name", "", ATOM) for a in e.findall("a:author", ATOM)]
        cats = " ".join(c.get("term") for c in e.findall("a:category", ATOM))
        rows.append({"id": arxiv_id, "title": clean(e.findtext("a:title", "", ATOM)),
                     "authors": ", ".join(authors[:3]) + (", et al." if len(authors) > 3 else ""),
                     "year": int(e.findtext("a:published", "0000", ATOM)[:4]),
                     "categories": cats, "url": f"https://arxiv.org/abs/{arxiv_id}",
                     "text": clean(e.findtext("a:summary", "", ATOM))})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-query", type=int, default=25, help="papers per topic query")
    args = ap.parse_args()
    seen, papers = set(), []
    for i, q in enumerate(QUERIES):
        if i:
            time.sleep(WAIT_S)
        try:
            rows = fetch_page(q, 0, args.per_query)
        except Exception as e:                  # one failed page should not lose the rest
            print(f"  skipped {q!r}: {e}")
            continue
        new = [r for r in rows if r["id"] not in seen and r["text"]]
        seen.update(r["id"] for r in new)
        papers += new
        print(f"  {q:<55} +{len(new)} (total {len(papers)})")
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text("\n".join(json.dumps(p, ensure_ascii=False) for p in papers) + "\n")
    print(f"wrote {len(papers)} papers to {OUT.relative_to(OUT.parent.parent)}")


if __name__ == "__main__":
    main()
