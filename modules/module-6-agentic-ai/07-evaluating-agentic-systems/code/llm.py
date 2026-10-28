"""Unified LLM access for the course, extended for multi-agent teams (Module 6).

Same interface as Module 5 (AI Agents): chat() returns text, chat_tools() does native tool
calling. Module 6 adds three things a TEAM of agents needs:

  1. Retry with backoff on rate limits. A team multiplies model calls, so on a
     free tier you WILL see HTTP 429 (Too Many Requests). We wait and retry,
     honoring the provider's `retry-after` header when it sends one.
  2. Usage on every call. prompt_tokens, completion_tokens, and cached_tokens
     (Groq's automatic prompt cache) are read from the response, so cost and
     caching are measured, not guessed.
  3. Per-agent attribution. Pass agent="supervisor" (or use `with tagged(...)`)
     and every call is logged under that agent, so the team's cost can be summed
     per agent: the double sum from the class 6.1 (Multi-agent architectures) slides.

CONFIGURE (.env at the project root, never committed):
    PROVIDER=groq                # gemini | groq | ollama
    MODEL_NAME=                  # optional; blank uses the default below
    GROQ_API_KEY=...             # only for PROVIDER=groq
    GEMINI_API_KEY=...           # only for PROVIDER=gemini
    LLM_MAX_RETRIES=5            # optional; retries on 429

INSTALL only what you use:
    pip install python-dotenv groq        # or: ollama / google-genai

Free-tier note: at time of writing Groq's free plan for openai/gpt-oss-20b allows
about 30 requests/min, 1,000 requests/day, 8K tokens/min, 200K tokens/day. Check
your console; limits change. Native tool calling (chat_tools) supports groq and
ollama.
"""

from __future__ import annotations
import contextlib
import contextvars
import json
import logging
import os
import random
import time
from dataclasses import dataclass, field

log = logging.getLogger("course.llm")
log.addHandler(logging.NullHandler())

try:
    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv())
except Exception:
    pass

DEFAULT_MODEL = {
    "gemini": "gemini-3.5-flash-lite",
    "groq": "openai/gpt-oss-20b",
    "ollama": "llama3.2",
}
PROVIDER = os.getenv("PROVIDER", "ollama").lower()
MODEL_NAME = os.getenv("MODEL_NAME") or DEFAULT_MODEL.get(PROVIDER, "llama3.2")
MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "5"))


# ===========================================================================
# Usage tracking and per-agent attribution
# ===========================================================================
@dataclass
class Usage:
    """Tokens for one model call. cached_tokens is the part of the prompt served
    from the provider's prompt cache (cheaper, and on Groq it does not count
    against rate limits)."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0


# Every call appends one record here: {agent, provider, model, latency, tokens...}.
USAGE: list[dict] = []
_CURRENT_AGENT = contextvars.ContextVar("current_agent", default="unattributed")


@contextlib.contextmanager
def tagged(agent: str):
    """Attribute every model call inside the block to `agent`."""
    token = _CURRENT_AGENT.set(agent)
    try:
        yield
    finally:
        _CURRENT_AGENT.reset(token)


def reset_usage():
    USAGE.clear()


def usage_by_agent() -> dict:
    """Totals per agent: {agent: {"calls", "prompt", "completion", "cached"}}."""
    out: dict = {}
    for r in USAGE:
        a = out.setdefault(r["agent"], {"calls": 0, "prompt": 0, "completion": 0, "cached": 0})
        a["calls"] += 1
        a["prompt"] += r["prompt_tokens"]
        a["completion"] += r["completion_tokens"]
        a["cached"] += r["cached_tokens"]
    return out


def _record(agent, provider, model, latency, usage: Usage):
    USAGE.append({"agent": agent or _CURRENT_AGENT.get(), "provider": provider,
                  "model": model, "latency": latency,
                  "prompt_tokens": usage.prompt_tokens,
                  "completion_tokens": usage.completion_tokens,
                  "cached_tokens": usage.cached_tokens})


def _usage_from_openai_style(resp) -> Usage:
    """Read usage from a Groq (OpenAI-compatible) response, tolerating gaps."""
    u = getattr(resp, "usage", None)
    if u is None:
        return Usage()
    details = getattr(u, "prompt_tokens_details", None)
    cached = getattr(details, "cached_tokens", 0) if details is not None else 0
    return Usage(getattr(u, "prompt_tokens", 0) or 0,
                 getattr(u, "completion_tokens", 0) or 0,
                 cached or 0)


def _usage_from_ollama(resp) -> Usage:
    get = resp.get if isinstance(resp, dict) else (lambda k, d=None: getattr(resp, k, d))
    return Usage(get("prompt_eval_count", 0) or 0, get("eval_count", 0) or 0, 0)


# ===========================================================================
# Retry with exponential backoff on rate limits (HTTP 429)
# ===========================================================================
def _is_rate_limit(err) -> bool:
    status = getattr(err, "status_code", None) or getattr(getattr(err, "response", None), "status_code", None)
    return status == 429 or "RateLimit" in type(err).__name__ or " 429" in str(err)


def _retry_after(err) -> float | None:
    """Seconds the provider asked us to wait, from the `retry-after` header."""
    try:
        value = err.response.headers.get("retry-after")
        return float(value) if value is not None else None
    except Exception:
        return None


def _with_retries(call):
    """Run `call()`; on a 429, wait (provider hint, or 2^attempt seconds plus a
    little random jitter so parallel agents do not retry in lockstep) and retry."""
    for attempt in range(MAX_RETRIES + 1):
        try:
            return call()
        except Exception as e:
            if not _is_rate_limit(e) or attempt == MAX_RETRIES:
                raise
            hint = _retry_after(e)                      # honor the provider's hint, even 0
            wait = hint if hint is not None else min(30.0, 2 ** attempt + random.uniform(0, 1))
            log.warning("rate limited (429); retry %d/%d in %.1fs", attempt + 1, MAX_RETRIES, wait)
            time.sleep(wait)


def _resolve(provider, model):
    provider = (provider or PROVIDER).lower()
    if model is None:
        model = MODEL_NAME if provider == PROVIDER else DEFAULT_MODEL[provider]
    return provider, model


# ===========================================================================
# chat(): plain text in, text out
# ===========================================================================
def chat(messages, provider=None, model=None, temperature=0.2, agent=None) -> str:
    """Send chat messages and return the reply text. Logs usage under `agent`."""
    provider, model = _resolve(provider, model)
    t0 = time.perf_counter()
    try:
        if provider == "groq":
            text, usage = _with_retries(lambda: _groq(messages, model, temperature))
        elif provider == "ollama":
            text, usage = _with_retries(lambda: _ollama(messages, model, temperature))
        elif provider == "gemini":
            text, usage = _with_retries(lambda: _gemini(messages, model, temperature))
        else:
            raise ValueError(f"unknown PROVIDER {provider!r}; use gemini, groq, or ollama")
    except Exception as e:
        log.warning("llm call failed provider=%s model=%s error=%s", provider, model, e)
        raise
    latency = time.perf_counter() - t0
    _record(agent, provider, model, latency, usage)
    log.info("llm call agent=%s provider=%s model=%s latency=%.3fs tokens=%d+%d cached=%d",
             agent or _CURRENT_AGENT.get(), provider, model, latency,
             usage.prompt_tokens, usage.completion_tokens, usage.cached_tokens)
    return text


def _groq(messages, model, temperature):
    from groq import Groq
    key = os.getenv("GROQ_API_KEY")
    if not key:
        raise RuntimeError("GROQ_API_KEY not set (add it to .env)")
    resp = Groq(api_key=key).chat.completions.create(
        model=model, messages=messages, temperature=temperature)
    return (resp.choices[0].message.content or "").strip(), _usage_from_openai_style(resp)


def _ollama(messages, model, temperature):
    import ollama
    resp = ollama.chat(model=model, messages=messages, options={"temperature": temperature})
    return (resp["message"]["content"] or "").strip(), _usage_from_ollama(resp)


def _gemini(messages, model, temperature):
    from google import genai
    from google.genai import types
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("GEMINI_API_KEY not set (add it to .env)")
    system = "\n".join(m["content"] for m in messages if m["role"] == "system")
    convo = "\n\n".join(f"{m['role']}: {m['content']}" for m in messages if m["role"] != "system") or " "
    resp = genai.Client(api_key=key).models.generate_content(
        model=model, contents=convo,
        config=types.GenerateContentConfig(system_instruction=system or None, temperature=temperature))
    meta = getattr(resp, "usage_metadata", None)
    usage = Usage(getattr(meta, "prompt_token_count", 0) or 0,
                  getattr(meta, "candidates_token_count", 0) or 0,
                  getattr(meta, "cached_content_token_count", 0) or 0) if meta else Usage()
    return (resp.text or "").strip(), usage


# ===========================================================================
# chat_tools(): native tool calling (from class 5.2, Tool calling and ReAct), now with usage
# ===========================================================================
@dataclass
class ToolCall:
    """One tool the model asked to run."""
    id: str
    name: str
    args: dict


@dataclass
class ToolReply:
    """A normalized reply: text, or tool calls to run (or both), plus usage."""
    text: "str | None"
    tool_calls: list = field(default_factory=list)
    raw: object = None
    usage: Usage = field(default_factory=Usage)

    @property
    def wants_tool(self) -> bool:
        return bool(self.tool_calls)


def chat_tools(messages, tools, provider=None, model=None, temperature=0.2,
               tool_choice="auto", agent=None) -> ToolReply:
    """One model call with native tool calling; returns a ToolReply.

    tool_choice: "auto" (model decides), "required" (must call a tool), or "none".
    Honored by groq; ollama ignores it. Logs usage under `agent`.
    """
    provider, model = _resolve(provider, model)
    t0 = time.perf_counter()
    try:
        if provider == "groq":
            reply = _with_retries(lambda: _groq_tools(messages, tools, model, temperature, tool_choice))
        elif provider == "ollama":
            reply = _with_retries(lambda: _ollama_tools(messages, tools, model, temperature))
        elif provider == "gemini":
            raise RuntimeError("chat_tools (native tool calling) supports groq and ollama; "
                               "set PROVIDER=groq or PROVIDER=ollama for Module 6.")
        else:
            raise ValueError(f"unknown PROVIDER {provider!r}; use groq or ollama")
    except Exception as e:
        log.warning("chat_tools failed provider=%s model=%s error=%s", provider, model, e)
        raise
    latency = time.perf_counter() - t0
    _record(agent, provider, model, latency, reply.usage)
    log.info("chat_tools agent=%s provider=%s model=%s calls=%d latency=%.3fs tokens=%d+%d cached=%d",
             agent or _CURRENT_AGENT.get(), provider, model, len(reply.tool_calls), latency,
             reply.usage.prompt_tokens, reply.usage.completion_tokens, reply.usage.cached_tokens)
    return reply


def _groq_tools(messages, tools, model, temperature, tool_choice) -> ToolReply:
    from groq import Groq
    key = os.getenv("GROQ_API_KEY")
    if not key:
        raise RuntimeError("GROQ_API_KEY not set (add it to .env)")
    resp = Groq(api_key=key).chat.completions.create(
        model=model, messages=messages, tools=tools,
        tool_choice=tool_choice, temperature=temperature)
    msg = resp.choices[0].message
    calls = []
    for tc in (msg.tool_calls or []):
        args = tc.function.arguments
        if isinstance(args, str):                  # groq returns arguments as a JSON string
            args = json.loads(args or "{}")
        calls.append(ToolCall(id=tc.id, name=tc.function.name, args=args))
    return ToolReply(text=msg.content, tool_calls=calls, raw=resp, usage=_usage_from_openai_style(resp))


def _ollama_tools(messages, tools, model, temperature) -> ToolReply:
    import ollama
    resp = ollama.chat(model=model, messages=messages, tools=tools,
                       options={"temperature": temperature})
    msg = resp["message"]
    calls = [ToolCall(id=f"call_{i}", name=tc["function"]["name"], args=dict(tc["function"]["arguments"]))
             for i, tc in enumerate(msg.get("tool_calls") or [])]
    return ToolReply(text=msg.get("content") or None, tool_calls=calls, raw=resp, usage=_usage_from_ollama(resp))


def assistant_tool_call_message(reply: ToolReply) -> dict:
    """The assistant turn that requested the tools, appended BEFORE the results."""
    return {"role": "assistant", "content": reply.text or "",
            "tool_calls": [{"id": c.id, "type": "function",
                            "function": {"name": c.name, "arguments": json.dumps(c.args)}}
                           for c in reply.tool_calls]}


def tool_result_message(call: ToolCall, result: str) -> dict:
    """The tool result appended AFTER running one tool call (the observation)."""
    return {"role": "tool", "tool_call_id": call.id, "name": call.name, "content": result}


if __name__ == "__main__":
    print(f"PROVIDER={PROVIDER}  MODEL_NAME={MODEL_NAME}")
    print(chat([{"role": "user", "content": "Say hello in one short sentence."}], agent="smoke-test"))
    print(usage_by_agent())
