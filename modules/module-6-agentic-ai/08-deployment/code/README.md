# Class 6.8 code: ship the team as a live API

`service.py` wraps everything Module 6 built into one FastAPI service:

```
request -> rate limit -> cache -> concurrency limit
        -> gateway (class 6.4 (Guardrails, routing, and compliance): rules, PII redaction, semantic route, policy classifier)
        -> team (class 6.1 (Multi-agent architectures): supervisor, tax researcher, explainer)
        -> output guard -> response          with a trace per request (class 6.6)
```

| Endpoint | What it returns |
|---|---|
| `POST /chat` | one JSON answer: outcome, answer, route, PII kinds removed, model used, cache hit, trace id, token usage, latency |
| `POST /chat/stream` | Server-Sent Events: `accepted`, one `step` per graph step, then `final` with the same JSON |
| `GET /health` | `{"status": "ok", ...}` with no model call, for Render's health check |

Request and response bodies are Pydantic models, so a malformed request (an empty
question, or one over 2,000 characters) gets a `422` before any model is called.

## Built to fit 512 MB

Render's free web service has 0.1 CPU and 512 MB of memory. Three choices make the team fit:

1. **FastEmbed instead of sentence-transformers** (`retriever.py`): the same MiniLM
   model on the ONNX runtime, with no PyTorch. The image's requirements resolve without
   PyTorch at all.
2. **Vectors baked at build time** (`build_index.py`, run by the Dockerfile): the model
   and the corpus vectors are inside the image; the service only embeds queries and
   never downloads anything at startup.
3. **No MCP subprocess** (`local_tools.py`): the glossary server from class 5.5 (MCP)
   would load a second copy of the embedding model in a second process. `LocalBridge`
   offers the same `specs()` and `call()` interface in-process, so the worker code is
   unchanged.

Measured at authoring time: the service with every library loaded uses about 118 MB
before the model weights; the MiniLM ONNX model is about 90 MB on disk. Expect the
running service to stay well under 512 MB, and check it in the Render dashboard's
**Metrics** tab after deploying.

## Cost controls (`controls.py`)

| Control | Default | Why |
|---|---|---|
| Per-client token bucket | burst 5, then 6 requests per minute | one client cannot drain the shared Groq quota; refusals are `429` with `Retry-After` |
| Response cache | 6 hours, 256 entries | identical questions (after redaction and normalization) are answered once |
| Model routing | 20b by default, 120b for long, multi-part, or comparative questions | pay for the big model only when the question needs it |
| Per-request budget | 12 model calls, 20,000 prompt tokens | a looping request stops with `over_budget` instead of draining the quota |
| Concurrency limit | 2 requests at a time | about five model calls per request against Groq's free 30 requests per minute |

All of them are environment variables (`RATE_CAPACITY`, `RATE_PER_SEC`,
`MAX_CALLS_PER_REQUEST`, `MAX_PROMPT_TOKENS_PER_REQUEST`, `MAX_CONCURRENT`), so you
tune them on Render without a rebuild. The budget and the model choice are applied to
every model call through one wrapper, using context variables, so concurrent requests
never see each other's settings or token counts.

## Run it locally

Without Docker:

```
pip install -r requirements.txt
python build_index.py                 # once: downloads the model, bakes the vectors
uvicorn service:app --reload          # reads PROVIDER and GROQ_API_KEY from .env
```

With Docker (the same image Render will run):

```
docker build -t tax-team .
docker run --rm -p 8000:8000 --env-file ../../../../.env tax-team
```

Then try it:

```
curl localhost:8000/health
curl -X POST localhost:8000/chat -H "content-type: application/json" \
     -d '{"question": "How much can I contribute to an HSA for 2025?"}'
curl -N -X POST localhost:8000/chat/stream -H "content-type: application/json" \
     -d '{"question": "Explain in plain language who can open an HSA."}'
```

FastAPI also serves interactive docs at `localhost:8000/docs`.

## Deploy to Render (free, no credit card)

1. Push the course repo (or your fork) to GitHub.
2. Sign up at [render.com](https://render.com) with GitHub.
3. In the dashboard choose **New**, then **Blueprint**, pick the repo, and set the
   Blueprint path to `modules/module-6-agentic-ai/08-deployment/code/render.yaml`.
4. Render reads the file, shows one **free** web service, and asks for the values
   marked `sync: false`: paste your `GROQ_API_KEY` (and your `LANGSMITH_API_KEY` if you
   use LangSmith; then set `LANGSMITH_TRACING` to `true`).
5. Apply. The first build takes several minutes (it installs packages and bakes the
   model). When the log shows the service is live, open
   `https://<your-service>.onrender.com/health`.

`render.yaml` sets `plan: free` explicitly. Leave it there: if `plan` is omitted,
Render creates new web services on a paid instance type.

### What the free tier means in practice

- **Sleep and wake.** After 15 minutes without traffic the service spins down; the next
  request wakes it in about a minute. Fine for a course demo; not for production.
- **750 hours.** Each workspace gets 750 free instance hours a month, enough for one
  service running all month (31 days is 744 hours). A sleeping service uses none.
- **Ephemeral disk.** Local files (the audit log and the local trace file) are lost on
  every spin-down and redeploy. Use LangSmith for traces you want to keep.
- **Outbound mail ports are blocked** on free web services; this service sends none.

Hugging Face Docker Spaces now require a paid PRO plan; if you have one, the same image
runs there. On a company AWS account the same container runs on App Runner or ECS;
that is optional and not free.

## Before class (pre-bake)

Deploy once the day before, call `/health` and `/chat` to confirm a live answer, and
keep the URL. On the day, wake the service a minute before you need it.

## Files

```
service.py        FastAPI app: endpoints, Pydantic models, rate limit, cache, concurrency, streaming
controls.py       token bucket, rate limiter, response cache, model routing, per-request budget
retriever.py      FastEmbed retriever with baked vectors (same interface as before)
build_index.py    bakes the model and the corpus vectors into the image
local_tools.py    the glossary in-process, behind the same interface as the MCP bridge
Dockerfile, .dockerignore, requirements.txt, render.yaml
gateway.py, pii.py, policy.py, router.py, data/routes.json    the class 6.4 (Guardrails, routing, and compliance) gateway
team.py, tax_worker.py, guardrails.py, llm.py                 the class 6.1 (Multi-agent architectures) team
tracing.py        the class 6.6 (Observability and tracing) tracer (LangSmith when the variables are set)
data/corpus.jsonl
```
