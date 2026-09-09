"""Build the FULL IRS corpus from the original public-domain PDFs.

The class build and the micro-assignment ship with a small hand-built
corpus.jsonl so they run out of the box. Run this script (where you have
internet) to replace it with the complete text of the source publications,
chunked with overlap and tagged with metadata.

    pip install requests
    pip install docling        # preferred parser (better tables and layout)
    pip install pypdf          # lightweight fallback parser
    python fetch_corpus.py     # downloads PDFs, writes corpus.jsonl

WHY DOCLING (with a pypdf fallback)
-----------------------------------
IRS publications are full of tables (deduction amounts, contribution limits).
pypdf extracts a flat character stream and tends to mangle those tables into
runs of loose numbers. Docling (an open document parser) understands page
layout and tables and exports clean Markdown, so the retrieved text is far more
faithful. Docling is a heavier install and downloads small layout models on
first run, so we fall back to pypdf if it is not available. The trade-off:
  - Docling path: best text quality; we export the whole document, so we do not
    track a per-chunk page number (page = 0).
  - pypdf path:   lighter; we read page by page, so each chunk gets its real page.

Everything here is a US government work and is in the public domain
(17 USC 105), so the downloaded files and the text are free to redistribute.

Schema per line: id, pub, title, tax_year, page, source_url, text
"""

from __future__ import annotations
import json
import pathlib
import re

# (pub label, title, tax_year, url). Current-year PDFs live under irs-pdf;
# prior-year PDFs under irs-prior with the p<NNN>--<YEAR>.pdf naming.
SOURCES = [
    ("Pub 501", "Dependents, Standard Deduction, and Filing Information", 2025, "https://www.irs.gov/pub/irs-pdf/p501.pdf"),
    ("Pub 501", "Dependents, Standard Deduction, and Filing Information", 2024, "https://www.irs.gov/pub/irs-prior/p501--2024.pdf"),
    ("Pub 502", "Medical and Dental Expenses", 2025, "https://www.irs.gov/pub/irs-pdf/p502.pdf"),
    ("Pub 969", "Health Savings Accounts and Other Tax-Favored Health Plans", 2025, "https://www.irs.gov/pub/irs-pdf/p969.pdf"),
    ("Pub 969", "Health Savings Accounts and Other Tax-Favored Health Plans", 2024, "https://www.irs.gov/pub/irs-prior/p969--2024.pdf"),
    ("Pub 526", "Charitable Contributions", 2025, "https://www.irs.gov/pub/irs-pdf/p526.pdf"),
    ("Pub 936", "Home Mortgage Interest Deduction", 2025, "https://www.irs.gov/pub/irs-pdf/p936.pdf"),
]

CHUNK_WORDS = 180      # target words per chunk
OVERLAP_WORDS = 40     # overlap so a fact split across a boundary is still findable
HERE = pathlib.Path(__file__).parent
PDF_DIR = HERE / "pdfs"


def download(url: str, dest: pathlib.Path) -> None:
    """Download the PDF once, caching it under pdfs/."""
    import requests
    if dest.exists():
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    r = requests.get(url, timeout=60, headers={"User-Agent": "course-corpus-builder"})
    r.raise_for_status()
    dest.write_bytes(r.content)


def clean(text: str) -> str:
    """Collapse stray whitespace so chunks are tidy."""
    text = text.replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{2,}", "\n", text)
    return text.strip()


def chunk_words(text: str, size: int, overlap: int) -> list[str]:
    """Split text into overlapping word windows (same idea as the class chunker)."""
    words = text.split()
    chunks, i = [], 0
    while i < len(words):
        chunks.append(" ".join(words[i:i + size]))
        if i + size >= len(words):
            break
        i += size - overlap
    return chunks


def extract_pages_docling(pdf_path: pathlib.Path):
    """Preferred parser. Return [(page_number, text)]; page 0 means 'whole document'.

    Docling's stable, documented API exports the whole document to Markdown, which
    preserves tables. We return it as a single (0, markdown) unit and let the
    caller chunk it. Raises ImportError if docling is not installed.
    """
    from docling.document_converter import DocumentConverter
    doc = DocumentConverter().convert(str(pdf_path)).document
    return [(0, clean(doc.export_to_markdown()))]


def extract_pages_pypdf(pdf_path: pathlib.Path):
    """Fallback parser. Return [(page_number, text)] with real per-page numbers."""
    from pypdf import PdfReader
    reader = PdfReader(str(pdf_path))
    out = []
    for page_num, page in enumerate(reader.pages, 1):
        text = clean(page.extract_text() or "")
        if len(text.split()) >= 25:          # skip covers, blank, and index pages
            out.append((page_num, text))
    return out


def extract_pages(pdf_path: pathlib.Path):
    """Use Docling if available, otherwise pypdf. Report which one ran."""
    try:
        pages = extract_pages_docling(pdf_path)
        print("  parsed with docling")
        return pages
    except ImportError:
        print("  docling not installed, falling back to pypdf")
        return extract_pages_pypdf(pdf_path)


def main():
    records = []
    for pub, title, year, url in SOURCES:
        slug = re.sub(r"[^a-z0-9]+", "-", f"{pub}-{year}".lower()).strip("-")
        pdf_path = PDF_DIR / f"{slug}.pdf"
        print(f"downloading {url}")
        download(url, pdf_path)
        for page_num, page_text in extract_pages(pdf_path):
            for j, ch in enumerate(chunk_words(page_text, CHUNK_WORDS, OVERLAP_WORDS)):
                records.append({
                    "id": f"{slug}-p{page_num}-{j}",
                    "pub": pub, "title": title, "tax_year": year,
                    "page": page_num, "source_url": url, "text": ch,
                })

    out = HERE / "corpus.jsonl"
    with out.open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(records)} chunks from {len(SOURCES)} publications to {out}")


if __name__ == "__main__":
    main()
