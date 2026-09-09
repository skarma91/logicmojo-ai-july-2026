# Class 4.4 code: embeddings and vector databases

Run `notebook.ipynb` to build a searchable store over a real corpus of IRS tax
publications (public domain), then query it by meaning, filter by metadata, and
blend in keyword search.

## What ships here

```
notebook.ipynb                  the class build (open this)
data/corpus.jsonl               the real corpus, ready to use (shipped)
data/make_corpus.py             regenerates the shipped sample corpus
data/fetch_corpus.py            builds the FULL corpus from the original PDFs
data/precompute_embeddings.py   caches dense embeddings for the corpus
```

`corpus.jsonl` already exists, so the notebook runs without any data step. Each
line is one passage with metadata: `id`, `pub`, `title`, `tax_year`, `page`,
`source_url`, `text`.

## Install

```
pip install sentence-transformers faiss-cpu rank-bm25 scikit-learn numpy
```

Chunking and BM25 run anywhere; the embedding cells download a small model
(`all-MiniLM-L6-v2`) on first use, so those need internet (Colab or a local
machine). Nothing here needs a GPU or any paid API.

## The data workflow

You only need this if you want to change or expand the corpus. All commands run
from inside the `data/` folder.

1. Regenerate the shipped sample (hand-built, genuine IRS content):

   ```
   cd data
   python make_corpus.py            # writes corpus.jsonl (19 passages)
   ```

2. Build the full corpus from the original public-domain PDFs (more chunks, real
   page numbers). This downloads the PDFs listed in `fetch_corpus.py`:

   ```
   pip install requests docling   # or: pip install pypdf (lightweight fallback)
   python fetch_corpus.py           # downloads PDFs -> data/pdfs/, writes corpus.jsonl
   ```

3. Cache dense embeddings once, so the notebook loads them instead of re-embedding:

   ```
   python precompute_embeddings.py  # writes embeddings.npy and ids.json
   ```

The notebook always reads `corpus.jsonl`, and its embedding cell loads
`embeddings.npy`/`ids.json` when they exist and match that corpus (otherwise it
embeds once and writes the cache itself). So the workflow is the same in both
cases and the only difference is whether you ran step 2: shipped `corpus.jsonl`
or PDF-rebuilt `corpus.jsonl`, then precompute, then run. To add your own
documents, edit the source list in `fetch_corpus.py` (or append records to
`corpus.jsonl` following the same fields) and rerun.

## What is committed vs generated

Committed to the repo: `notebook.ipynb`, `data/corpus.jsonl`, and the three
`data/*.py` scripts. Generated locally and git-ignored: `data/pdfs/` (downloaded
PDFs), `data/embeddings.npy`, and `data/ids.json`. Regenerate those any time
with the commands above.

## Note on the shipped corpus

`data/corpus.jsonl` is a small, hand-built sample condensed from public-domain
IRS publications; the figures were verified against the current publications, but
the wording is paraphrased and page numbers are representative. Run
`fetch_corpus.py` for the exact text and true page numbers.
