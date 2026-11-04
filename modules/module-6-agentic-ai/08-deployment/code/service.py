"""Class 6.8 build: the team as a live API.

    POST /chat          one JSON answer
    POST /chat/stream   Server-Sent Events: progress per graph step, then the answer
    GET  /health        cheap liveness check for the host (no model call)

Every request goes through, in order:

  1. a per-client token-bucket rate limit       (429 with Retry-After when exceeded)
  2. the response cache                         (redacted, normalized question)
  3. a concurrency limit                        (an asyncio.Semaphore sized for Groq's free tier)
  4. the class 6.4 (Guardrails, routing, and compliance) gateway: rules, PII redaction,
     semantic routing, the policy classifier, then the class 6.1 (Multi-agent
     architectures) team, then the output guard
  5. a per-request model choice and budget      (controls.py)
  6. a trace per request                        (class 6.6, Observability and tracing)

Run locally:   uvicorn service:app --reload
Run in Docker: see the README (the same image deploys to Render's free tier).
"""

from __future__ import annotations
import asyncio
import contextvars
import json
import logging
import os
import pathlib
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.sse import EventSourceResponse, ServerSentEvent
from pydantic import BaseModel, Field

import controls
import llm
import pii
import retriever
from gateway import Gateway
from local_tools import LocalBridge
import router
from router import SemanticRouter
from tax_worker import build_tax_worker
from team import build_team
from tracing import Tracer

log = logging.getLogger("course.service")
MAX_CONCURRENT = int(os.getenv("MAX_CONCURRENT", "2"))       # about 5 model calls per request; 30 RPM free tier
RATE_CAPACITY = float(os.getenv("RATE_CAPACITY", "5"))       # burst per client
RATE_PER_SEC = float(os.getenv("RATE_PER_SEC", str(6 / 60)))  # 6 requests per minute per client, long run
MAX_CALLS = int(os.getenv("MAX_CALLS_PER_REQUEST", "12"))
MAX_PROMPT_TOKENS = int(os.getenv("MAX_PROMPT_TOKENS_PER_REQUEST", "20000"))
TRACE_DIR = pathlib.Path(os.getenv("TRACE_DIR", "/tmp/traces"))
_ON_EVENT = contextvars.ContextVar("on_event", default=None)


# ---------------------------------------------------------------------------
# Request and response shapes (Pydantic validates both; bad input gets a 422)
# ---------------------------------------------------------------------------
class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000, description="A US federal tax question")


class Usage(BaseModel):
    calls: int
    prompt_tokens: int
    completion_tokens: int


class ChatResponse(BaseModel):
    outcome: str = Field(description="answer, blocked, refused, canned, fallback, or over_budget")
    answer: str
    route: str
    pii_redacted: list[str] = Field(description="kinds of personal data removed before processing")
    model: str
    cached: bool
    trace_id: str
    usage: Usage
    latency_s: float


# ---------------------------------------------------------------------------
# Startup: load the baked index, build the router and the team, wire the controls
# ---------------------------------------------------------------------------
STATE: dict = {}


def run_team(team, question: str) -> dict:
    """Invoke the team. When a stream is listening, forward one event per graph step."""
    inputs = {"task": question, "findings": "", "explanation": "", "hops": 0, "route_log": []}
    on_event = _ON_EVENT.get()
    if on_event is None:
        return team.invoke(inputs, config={"recursion_limit": 25})
    final = None
    for mode, chunk in team.stream(inputs, config={"recursion_limit": 25}, stream_mode=["updates", "values"]):
        if mode == "values":
            final = chunk
        else:
            for node, update in chunk.items():
                decision = (update or {}).get("route_log", [{}])[-1] if isinstance(update, dict) else {}
                on_event({"event": "step", "node": node, "next": decision.get("next", "")})
    return final


@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
    n = retriever.load_index()
    TRACE_DIR.mkdir(parents=True, exist_ok=True)
    tracer = Tracer(TRACE_DIR / "service.jsonl")
    controls.instrument(tracer)
    team = build_team(build_tax_worker(LocalBridge()))
    STATE.update(
        passages=n, tracer=tracer,
        gateway=Gateway(team=team, run_team=run_team,
                        router=SemanticRouter(tau=float(os.getenv("ROUTER_TAU", router.TAU)))),
        semaphore=asyncio.Semaphore(MAX_CONCURRENT),
        limiter=controls.RateLimiter(RATE_CAPACITY, RATE_PER_SEC),
        cache=controls.ResponseCache(),
    )
    log.info("ready: %d passages, provider=%s, max concurrent=%d", n, llm.PROVIDER, MAX_CONCURRENT)
    yield


app = FastAPI(title="Tax research team", version="1.0", lifespan=lifespan)


# ---------------------------------------------------------------------------
# One request, end to end (runs in a worker thread)
# ---------------------------------------------------------------------------
def answer(question: str, on_event=None) -> dict:
    t0 = time.perf_counter()
    model = controls.choose_model(question)
    budget = controls.Budget(MAX_CALLS, MAX_PROMPT_TOKENS)
    controls.start_request(model, budget)
    _ON_EVENT.set(on_event)
    tracer: Tracer = STATE["tracer"]
    with tracer.span("request", "chain", inputs=question, metadata={"model": model}) as root:
        try:
            d = STATE["gateway"].handle(question)
            outcome, text, route, kinds = d.outcome, d.answer, d.route, sorted({k for k, _ in d.pii})
        except controls.BudgetExceeded as e:
            outcome, route = "over_budget", ""
            text = "This question needed more work than one request is allowed. Try a narrower question."
            kinds = sorted({s.kind for s in pii.detect(question)})
            root.metadata["budget"] = str(e)
        tracer.set_output(root, text)
    return {"outcome": outcome, "answer": text, "route": route, "pii_redacted": kinds, "model": model,
            "cached": False, "trace_id": root.trace_id,
            "usage": {"calls": budget.calls, "prompt_tokens": budget.prompt_tokens,
                      "completion_tokens": budget.completion_tokens},
            "latency_s": round(time.perf_counter() - t0, 3)}


def client_key(request: Request) -> str:
    return request.headers.get("x-api-key") or (request.client.host if request.client else "unknown")


def check_rate(request: Request):
    allowed, retry = STATE["limiter"].allow(client_key(request))
    if not allowed:
        raise HTTPException(429, "rate limit exceeded", headers={"Retry-After": str(max(1, round(retry)))})


def cached(question: str) -> tuple[str, dict | None]:
    key = controls.cache_key(pii.redact(question)[0])
    return key, STATE["cache"].get(key)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health")
def health() -> dict:
    return {"status": "ok", "passages": STATE.get("passages", 0), "provider": llm.PROVIDER,
            "max_concurrent": MAX_CONCURRENT}


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, request: Request) -> ChatResponse:
    check_rate(request)
    key, hit = cached(req.question)
    if hit:
        return ChatResponse(**{**hit, "cached": True, "latency_s": 0.0})
    async with STATE["semaphore"]:
        result = await asyncio.to_thread(answer, req.question)
    if result["outcome"] == "answer":
        STATE["cache"].put(key, result)
    return ChatResponse(**result)


@app.post("/chat/stream", response_class=EventSourceResponse)
async def chat_stream(req: ChatRequest, request: Request):
    check_rate(request)
    key, hit = cached(req.question)
    if hit:
        yield ServerSentEvent(event="final", data={**hit, "cached": True, "latency_s": 0.0})
        return
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()

    def on_event(ev: dict):                         # called from the worker thread
        loop.call_soon_threadsafe(queue.put_nowait, ev)

    async with STATE["semaphore"]:
        yield ServerSentEvent(event="accepted", data={"question_chars": len(req.question)})
        task = asyncio.create_task(asyncio.to_thread(answer, req.question, on_event))
        while not task.done() or not queue.empty():
            try:
                ev = await asyncio.wait_for(queue.get(), timeout=0.25)
                yield ServerSentEvent(event=ev.pop("event"), data=ev)
            except asyncio.TimeoutError:
                continue
        result = task.result()
    if result["outcome"] == "answer":
        STATE["cache"].put(key, result)
    yield ServerSentEvent(event="final", data=result)
