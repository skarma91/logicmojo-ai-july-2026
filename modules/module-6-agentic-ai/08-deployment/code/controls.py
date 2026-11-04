"""Cost controls for the deployed team.

Five controls, each small enough to test on its own:

  TokenBucket    per-client rate limit: bursts up to `capacity`, long-run `rate`/second
  ResponseCache  identical (redacted, normalized) questions are answered once per TTL
  choose_model   routing: easy requests to gpt-oss-20b, hard ones to gpt-oss-120b
  Budget         a per-request cap on model calls and prompt tokens (class 5.6,
                 Reliability and failure modes), enforced on every call
  instrument     wraps llm.chat / llm.chat_tools so routing and the budget apply to
                 every model call the team makes, without touching the team's code

The concurrency limit (an asyncio.Semaphore) lives in service.py, next to the requests
it holds back.
"""

from __future__ import annotations
import contextvars
import functools
import re
import threading
import time
from collections import OrderedDict

import llm

EASY_MODEL = "openai/gpt-oss-20b"
HARD_MODEL = "openai/gpt-oss-120b"
_MODEL = contextvars.ContextVar("request_model", default=None)
_BUDGET = contextvars.ContextVar("request_budget", default=None)


# ---------------------------------------------------------------------------
# Token bucket
# ---------------------------------------------------------------------------
class TokenBucket:
    """Holds at most `capacity` tokens and refills at `rate` tokens per second. A request
    is served only if a whole token is available. Bursts up to `capacity` pass at once;
    over a long window the served rate cannot exceed `rate`.

    `clock` is injectable so the behaviour can be tested without waiting.
    """

    def __init__(self, capacity: float, rate: float, clock=time.monotonic):
        self.capacity, self.rate, self.clock = capacity, rate, clock
        self.tokens, self.last = float(capacity), clock()
        self._lock = threading.Lock()

    def allow(self) -> tuple[bool, float]:
        """(allowed, seconds until the next token if refused)."""
        with self._lock:
            now = self.clock()
            self.tokens = min(self.capacity, self.tokens + (now - self.last) * self.rate)
            self.last = now
            if self.tokens >= 1:
                self.tokens -= 1
                return True, 0.0
            return False, (1 - self.tokens) / self.rate


class RateLimiter:
    """One bucket per client key (an API key header, or the caller's address)."""

    def __init__(self, capacity: float, rate: float, clock=time.monotonic):
        self.capacity, self.rate, self.clock = capacity, rate, clock
        self.buckets: dict[str, TokenBucket] = {}
        self._lock = threading.Lock()

    def allow(self, key: str) -> tuple[bool, float]:
        with self._lock:
            bucket = self.buckets.setdefault(key, TokenBucket(self.capacity, self.rate, self.clock))
        return bucket.allow()


# ---------------------------------------------------------------------------
# Response cache
# ---------------------------------------------------------------------------
def cache_key(redacted_question: str) -> str:
    """Normalize so trivial differences share one entry: case, spacing, end punctuation.
    The key is built from the REDACTED question, so no personal data sits in the cache."""
    return re.sub(r"[\s?!.]+$", "", " ".join(redacted_question.lower().split()))


class ResponseCache:
    """A small time-limited LRU cache. Tax rules change yearly, not hourly, so a TTL of
    hours is safe here; a cache in front of live data would need minutes or none."""

    def __init__(self, max_entries: int = 256, ttl_s: float = 6 * 3600, clock=time.monotonic):
        self.max_entries, self.ttl_s, self.clock = max_entries, ttl_s, clock
        self._data: OrderedDict[str, tuple[float, dict]] = OrderedDict()
        self._lock = threading.Lock()
        self.hits = self.misses = 0

    def get(self, key: str) -> dict | None:
        with self._lock:
            item = self._data.get(key)
            if item is None or self.clock() - item[0] > self.ttl_s:
                self._data.pop(key, None)
                self.misses += 1
                return None
            self._data.move_to_end(key)
            self.hits += 1
            return item[1]

    def put(self, key: str, value: dict):
        with self._lock:
            self._data[key] = (self.clock(), value)
            self._data.move_to_end(key)
            while len(self._data) > self.max_entries:
                self._data.popitem(last=False)


# ---------------------------------------------------------------------------
# Model routing
# ---------------------------------------------------------------------------
HARD_SIGNALS = ("compare", " versus ", " vs ", "difference between", "step by step", "which is better")


def choose_model(question: str) -> str:
    """A transparent rule, not a model: long, multi-part, or comparative questions go to
    the larger model; everything else to the smaller, cheaper, faster one. Only applies on
    Groq, where both models exist."""
    if llm.PROVIDER != "groq":
        return llm.MODEL_NAME
    q = f" {question.lower()} "
    hard = len(question.split()) > 35 or question.count("?") > 1 or any(s in q for s in HARD_SIGNALS)
    return HARD_MODEL if hard else EASY_MODEL


# ---------------------------------------------------------------------------
# Per-request budget
# ---------------------------------------------------------------------------
class BudgetExceeded(RuntimeError):
    pass


class Budget:
    def __init__(self, max_calls: int, max_prompt_tokens: int):
        self.max_calls, self.max_prompt_tokens = max_calls, max_prompt_tokens
        self.calls = self.prompt_tokens = self.completion_tokens = 0

    def before_call(self):
        if self.calls >= self.max_calls:
            raise BudgetExceeded(f"model-call budget of {self.max_calls} reached")
        if self.prompt_tokens >= self.max_prompt_tokens:
            raise BudgetExceeded(f"prompt-token budget of {self.max_prompt_tokens} reached")

    def after_call(self, usage_record: dict):
        self.calls += 1
        self.prompt_tokens += usage_record["prompt_tokens"]
        self.completion_tokens += usage_record["completion_tokens"]


def start_request(model: str, budget: Budget):
    """Set this request's model and budget. Context variables keep concurrent requests
    (each in its own thread) from seeing each other's settings."""
    _MODEL.set(model)
    _BUDGET.set(budget)


def instrument(tracer=None):
    """Wrap the model calls once, at startup.

    - llm.chat / llm.chat_tools: check the budget BEFORE each call, apply the request's
      model, and open an llm span.
    - llm._record (called inside every model call, in the calling thread): count the
      call's real tokens against THIS request's budget and attach them to the current
      span. Reading a shared usage list after the call would be wrong when two requests
      run at once; the hook sees exactly its own call.
    """
    import tracing
    original_record = llm._record

    def record(agent, provider, model, latency, usage):
        original_record(agent, provider, model, latency, usage)
        budget = _BUDGET.get()
        if budget is not None:
            budget.after_call({"prompt_tokens": usage.prompt_tokens, "completion_tokens": usage.completion_tokens})
        span = tracing.current()
        if span is not None and span.kind == "llm":
            span.tokens_in, span.tokens_out, span.cached = usage.prompt_tokens, usage.completion_tokens, usage.cached_tokens
    llm._record = record

    def wrap(fn):
        @functools.wraps(fn)
        def wrapper(messages, *args, model=None, agent=None, **kwargs):
            budget = _BUDGET.get()
            if budget is not None:
                budget.before_call()
            chosen = model or _MODEL.get()             # an explicit model (the classifier) wins
            if tracer is None:
                return fn(messages, *args, model=chosen, agent=agent, **kwargs)
            with tracer.span(f"llm:{agent or 'unattributed'}", "llm",
                             metadata={"model": chosen or llm.MODEL_NAME}):
                return fn(messages, *args, model=chosen, agent=agent, **kwargs)
        return wrapper
    llm.chat = wrap(llm.chat)
    llm.chat_tools = wrap(llm.chat_tools)
