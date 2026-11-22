# Class 7.2 (Build lab): the Customer Support Desk

A multi-agent customer-support desk in plain Python, designed in class 7.1 (Capstone
kickoff) and built live in this lab. A supervisor runs three agents on every ticket:

- **Triage** (model): the intent (27 possible), urgency, sentiment, and a confidence.
- **Knowledge** (no model): the matching help article, by TF-IDF retrieval, as an
  independent second opinion.
- **Reply writer** (model): a short reply grounded in that one article.

Rules in code decide when a person must take over (company policy, an angry or urgent
customer, an unsure triage, a weak article match, or a draft that fails the checks twice).
Personal data is redacted before any model sees the ticket and restored only in the final
reply. Every step is traced, and `eval.py` measures the desk on labeled tickets.

```
code/
  llm.py, pii.py, fetch_data.py, data.py     pre-written helpers
  schemas.py                                 the data shapes
  agents/triage.py, knowledge.py, writer.py  the three agents
  guards.py                                  escalation rules and reply checks
  tracing.py                                 one JSON line per step
  supervisor.py                              the routing loop
  main.py, eval.py                           run it, measure it
  service.py, Dockerfile, render.yaml        optional web API and free deploy
  data/                                      the Bitext sample, the help centre, the tickets
INSTRUCTOR_MANUAL.md                         the step-by-step guide for building it live
```

## 1. Install

From this folder, with Python 3.10 or newer:

```bash
cd code
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

`requirements.txt` installs the Groq client. For a fully local run, install Ollama, pull
`llama3.2`, and `pip install ollama` instead.

## 2. Configure a model

Create `.env` at the repository root (it is never committed):

```
PROVIDER=groq               # groq | ollama | gemini
GROQ_API_KEY=your-key       # only for groq; free at console.groq.com
MODEL_NAME=                 # optional; blank uses openai/gpt-oss-20b on groq, llama3.2 on ollama
```

Check it with `python llm.py`, which prints a one-line greeting and the tokens used.

## 3. Build the data

The committed sample is enough to run everything:

```bash
python fetch_data.py --sample     # 174 rows -> 27 articles (data/kb.jsonl), 147 tickets (data/tickets.jsonl)
```

For a larger ticket set, download the full Bitext CSV (about 19 MB) and keep 40 rows per
intent:

```bash
python fetch_data.py              # or: python fetch_data.py --per-intent 80
```

The data is the Bitext customer-support dataset (CDLA-Sharing-1.0). Bitext describes it
as hybrid synthetic: the messages are written in the style of real tickets, not taken
from real customers. See `data/SOURCE.md`.

## 4. Run the desk

```bash
python main.py                                  # 6 sample tickets, spread across intents
python main.py --n 12
python main.py --ticket "I'm Ana Ruiz, ana@example.com. How do I cancel order 4812?"
```

For each ticket you see the triage result, the steps the supervisor took, and either the
reply or the reason it went to a person, then the model calls and tokens per agent. The
trace is in `traces/desk.jsonl`: one line per step, with time, calls, and tokens, and no
ticket text.

## 5. Evaluate it

```bash
python eval.py --retrieval        # the knowledge agent alone: no model, a few seconds
python eval.py --limit 30         # the full desk on 30 tickets
python eval.py                    # all tickets
```

| Metric | What it means |
|---|---|
| intent accuracy | triage picked the labeled intent, with a 95% Wilson interval |
| escalation recall | of the tickets that need a person, the share escalated |
| escalation precision | of the tickets escalated, the share that needed a person |
| auto-resolved | the share answered without a person; each reply passed the checks |
| cost per ticket | model calls and tokens per ticket |

A ticket "needs a person" when its intent is a complaint, a payment problem, or a request
for a human, or when Bitext tagged it as offensive language. On the committed sample,
retrieval alone ranks the right article first for 102 of 147 tickets (69%) and in the top
three for 129 (88%). The full desk's numbers depend on the model, so run it and read
them; each full run makes about two model calls per ticket.

On Groq's free tier, rate limits apply (requests and tokens per minute); `llm.py` waits
and retries on HTTP 429 automatically. Use `--limit` while you iterate.

## 6. Optional: serve and deploy

```bash
uvicorn service:app --reload
curl -s localhost:8000/health
curl -s localhost:8000/ticket -H 'content-type: application/json' -d '{"text": "how do I reset my password"}'
```

To deploy on Render's free tier, push your fork, choose **New > Blueprint** in the Render
Dashboard, set the path to `modules/module-7-capstone/02-build-lab/code/render.yaml`, and
paste `GROQ_API_KEY` when asked. `plan: free` is written out on purpose: an omitted plan
defaults to a paid instance. A free service sleeps when idle and takes up to a minute to
wake.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `data/kb.jsonl not found` | run `python fetch_data.py --sample` |
| `GROQ_API_KEY not set` | put `.env` at the repository root |
| every ticket escalates with `intent unclear` | the model's reply is not JSON; try another model or provider |
| `rate limited (429)` warnings | free-tier limits; wait, or run fewer tickets |
| `ModuleNotFoundError: agents` | run the commands from inside `code/` |
