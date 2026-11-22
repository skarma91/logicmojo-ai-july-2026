# Capstone project: your starting point

These files are the plumbing you already built in earlier modules. Use them as they are;
your submission is judged on everything you build on top of them.

| File | What it gives you |
|---|---|
| `llm.py` | `chat(messages, agent=...)` over Groq, Ollama, or Gemini; `chat_json()` that never crashes on bad JSON; retries on HTTP 429; `usage_by_agent()` for cost per agent |
| `pii.py` | `redact(text)` returns `(text_with_placeholders, vault, spans)`; `restore(text, vault)` puts the values back for the final output only |
| `fetch_data.py` | downloads Wikivoyage destination guides into `data/guides.jsonl`, one chunk per line, each with its section and page URL |
| `data/guides.jsonl` | the committed sample, so everyone starts from the same real data |
| `requirements.txt`, `.env.example` | setup |

## Setup

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example ../../../../../.env                     # then fill in your key; or create .env at the repo root
python llm.py                                           # a one-line greeting proves the model works
```

To fetch more or different destinations (one request per second, as Wikimedia asks):

```bash
python fetch_data.py                       # the default 12 destinations
python fetch_data.py Lisbon Kyoto Nairobi  # your own list
```

## What you design yourself

Not given, on purpose: your data shapes, your agents, the supervisor and its routing,
retrieval, the constraint checks, tracing, and the evaluation. See `../README.md` for the
requirements and how the project is assessed.

## Data license

Wikivoyage text is CC BY-SA 4.0. Credit Wikivoyage, keep each chunk's page URL, and share
anything built from the text under the same license.
