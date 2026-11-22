"""The research team as a live API (given; class 6.8, Deployment).

    POST /chat          one JSON answer with its cited papers
    POST /chat/stream   Server-Sent Events: one event per agent turn, then the answer
    GET  /health        liveness for the host (no model call)

Every request goes through, in order: a per-client token-bucket rate limit, the
response cache, a concurrency limit, a model choice and a per-request budget, the
gateway (rules, PII redaction, team, output guard), and one trace.

    uvicorn service:app --reload           # locally
    docker build -t research-team . && docker run --env-file ../../../../.env -p 8000:8000 research-team
"""

from __future__ import annotations
import asyncio
import contextvars
import functools
import logging
import os
import pathlib
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.sse import EventSourceResponse, ServerSentEvent
from pydantic import BaseModel, Field

import controls
import gateway
import llm
import pii
import retriever
from team import build_team, new_state
from tracing import Tracer

log = logging.getLogger("course.service")
MAX_CONCURRENT = int(os.getenv("MAX_CONCURRENT", "2"))        # about 6 model calls per request; 30 RPM free tier
RATE_CAPACITY = float(os.getenv("RATE_CAPACITY", "5"))
RATE_PER_SEC = float(os.getenv("RATE_PER_SEC", str(6 / 60)))
MAX_CALLS = int(os.getenv("MAX_CALLS_PER_REQUEST", "14"))
MAX_PROMPT_TOKENS = int(os.getenv("MAX_PROMPT_TOKENS_PER_REQUEST", "24000"))
TRACE_DIR = pathlib.Path(os.getenv("TRACE_DIR", "/tmp/traces"))
_ON_EVENT = contextvars.ContextVar("on_event", default=None)


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000, description="A question about LLM and agent research")


class Paper(BaseModel):
    id: str
    title: str


class Usage(BaseModel):
    calls: int
    prompt_tokens: int
    completion_tokens: int


class ChatResponse(BaseModel):
    outcome: str = Field(description="answer, no_results, blocked, or over_budget")
    answer: str
    papers: list[Paper]
    pii_redacted: list[str]
    model: str
    cached: bool
    trace_id: str
    usage: Usage
    latency_s: float


STATE: dict = {}


def run_team(team, question: str) -> dict:
    """Invoke the team; when a stream is listening, forward one event per agent turn."""
    on_event = _ON_EVENT.get()
    config = {"recursion_limit": 40}
    if on_event is None:
        return team.invoke(new_state(question), config=config)
    final = None
    for mode, chunk in team.stream(new_state(question), config=config, stream_mode=["updates", "values"]):
        if mode == "values":
            final = chunk
        else:
            for node in chunk:
                on_event({"event": "step", "node": node})
    return final


@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
    n = retriever.load_index()
    TRACE_DIR.mkdir(parents=True, exist_ok=True)
    tracer = Tracer(TRACE_DIR / "service.jsonl")
    controls.instrument(tracer)

    def wrap(name, fn):
        @functools.wraps(fn)
        def node(state):
            with tracer.span(name, "chain"):
                return fn(state)
        return node

    STATE.update(papers=n, tracer=tracer, team=build_team(wrap=wrap),
                 semaphore=asyncio.Semaphore(MAX_CONCURRENT),
                 limiter=controls.RateLimiter(RATE_CAPACITY, RATE_PER_SEC),
                 cache=controls.ResponseCache())
    log.info("ready: %d papers, provider=%s, max concurrent=%d", n, llm.PROVIDER, MAX_CONCURRENT)
    yield


app = FastAPI(title="arXiv research team", version="1.0", lifespan=lifespan)


def answer(question: str, on_event=None) -> dict:
    """One request end to end, in a worker thread."""
    t0 = time.perf_counter()
    model = controls.choose_model(question)
    budget = controls.Budget(MAX_CALLS, MAX_PROMPT_TOKENS)
    controls.start_request(model, budget)
    _ON_EVENT.set(on_event)
    tracer: Tracer = STATE["tracer"]
    with tracer.span("request", "chain", inputs=question, metadata={"model": model}) as root:
        try:
            r = gateway.handle(lambda q: run_team(STATE["team"], q), question)
        except controls.BudgetExceeded as e:
            r = {"outcome": "over_budget", "papers": [],
                 "answer": "This question needed more work than one request allows. Try a narrower question.",
                 "pii": sorted({s.kind for s in pii.detect(question)})}
            root.metadata["budget"] = str(e)
        tracer.set_output(root, r["answer"])
    return {"outcome": r["outcome"], "answer": r["answer"], "papers": r["papers"], "pii_redacted": r["pii"],
            "model": model, "cached": False, "trace_id": root.trace_id,
            "usage": {"calls": budget.calls, "prompt_tokens": budget.prompt_tokens,
                      "completion_tokens": budget.completion_tokens},
            "latency_s": round(time.perf_counter() - t0, 3)}


def check_rate(request: Request):
    key = request.headers.get("x-api-key") or (request.client.host if request.client else "unknown")
    allowed, retry = STATE["limiter"].allow(key)
    if not allowed:
        raise HTTPException(429, "rate limit exceeded", headers={"Retry-After": str(max(1, round(retry)))})


def cached(question: str) -> tuple[str, dict | None]:
    key = controls.cache_key(pii.redact(question)[0])
    return key, STATE["cache"].get(key)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "papers": STATE.get("papers", 0), "provider": llm.PROVIDER,
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

    def on_event(ev: dict):                          # called from the worker thread
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
