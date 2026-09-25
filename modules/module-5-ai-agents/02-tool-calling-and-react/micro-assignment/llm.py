"""Unified LLM access for the course: one chat() call, three providers.

WHY THIS FILE EXISTS
--------------------
Every provider (Google Gemini, Groq, Ollama) has a slightly different Python
library and a different way to send a chat message. Rewriting that in every
class would be repetitive and error prone. So we hide all of it behind a single
function, chat(messages), and pick the provider from configuration. The rest of
the course just calls llm.chat(...) and does not care which provider is behind it.

HOW YOU CONFIGURE IT
--------------------
Create a file named .env at the project root (copy .env.example) and put your
settings there. .env is gitignored, so your keys never get committed:

    PROVIDER=ollama            # which provider to use: gemini | groq | ollama
    MODEL_NAME=                # optional; leave blank to use the default below
    GEMINI_API_KEY=...         # only needed if PROVIDER=gemini
    GROQ_API_KEY=...           # only needed if PROVIDER=groq

ollama runs a model on your own machine, so it is free and needs no key. gemini
and groq are cloud services with free tiers that need an API key.

WHAT TO INSTALL
---------------
Only the library for the provider you actually use:

    pip install python-dotenv        # reads the .env file (always useful)
    pip install google-genai         # only for PROVIDER=gemini
    pip install groq                 # only for PROVIDER=groq
    pip install ollama               # only for PROVIDER=ollama (+ run `ollama serve`)
"""

from __future__ import annotations
import logging
import os
import time

# --- Logging -----------------------------------------------------------------
# We log every model call (provider, model, latency) so cost and behavior are
# observable, a habit that pays off when debugging agents later. As a library,
# llm.py only gets a logger and attaches a NullHandler; the program that imports
# it decides how logs are shown by calling logging.basicConfig(level=logging.INFO).
log = logging.getLogger("course.llm")
log.addHandler(logging.NullHandler())

# --- Step 1: load the .env file so its values appear in os.environ ---------
# python-dotenv reads .env and copies each line into the environment. We wrap it
# in try/except so the file still works even if python-dotenv is not installed
# (for example, when the values are already set some other way).
try:
    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv())     # find_dotenv walks up the folders to locate .env
except Exception:
    pass

# --- Step 2: the default model for each provider ---------------------------
# If MODEL_NAME is not set in .env, we fall back to one sensible model per provider.
DEFAULT_MODEL = {
    # Two gemini options (set MODEL_NAME in .env to switch):
    #   "gemini-3.5-flash-lite" (default): low cost, high speed; best for
    #       high-volume routine work such as extraction, classification, search.
    #   "gemini-3.5-flash": higher quality but several times pricier; better for
    #       coding and multi-step agentic tasks.
    #   (gemini-2.5-flash is retiring on 2026-10-20, so we do not use it.)
    "gemini": "gemini-3.5-flash-lite",
    "groq": "openai/gpt-oss-20b",
    "ollama": "llama3.2",
}

# --- Step 3: read the choice of provider and model from the environment ----
# os.getenv("PROVIDER", "ollama") means "use PROVIDER if set, otherwise ollama".
PROVIDER = os.getenv("PROVIDER", "ollama").lower()
# Use MODEL_NAME if the user set one; otherwise use the default for this provider.
MODEL_NAME = os.getenv("MODEL_NAME") or DEFAULT_MODEL.get(PROVIDER, "llama3.2")


def chat(messages, provider=None, model=None, temperature=0.2) -> str:
    """Send chat messages to the model and return the reply text.

    messages is a list of dicts, each shaped like
        {"role": "system", "content": "..."}   # sets the assistant's job
        {"role": "user",   "content": "..."}    # the user's request
        {"role": "assistant", "content": "..."} # a past model reply (optional)

    provider/model default to the PROVIDER/MODEL_NAME resolved above, but you can
    override them per call. temperature controls randomness: near 0 is focused
    and repeatable, higher is more varied.
    """
    # Fall back to the module defaults if the caller did not specify.
    provider = (provider or PROVIDER).lower()
    if model is None:
        # If the caller asked for a different provider, use that provider's default.
        model = MODEL_NAME if provider == PROVIDER else DEFAULT_MODEL[provider]

    # Time the call and log it. On failure, log a warning and re-raise so the
    # caller still sees the error.
    t0 = time.perf_counter()
    try:
        if provider == "gemini":
            reply = _gemini(messages, model, temperature)
        elif provider == "groq":
            reply = _groq(messages, model, temperature)
        elif provider == "ollama":
            reply = _ollama(messages, model, temperature)
        else:
            raise ValueError(f"unknown PROVIDER {provider!r}; use gemini, groq, or ollama")
    except Exception as e:
        log.warning("llm call failed provider=%s model=%s error=%s", provider, model, e)
        raise
    log.info("llm call provider=%s model=%s latency=%.3fs", provider, model, time.perf_counter() - t0)
    return reply


def _split_system(messages):
    """Separate the system message(s) from the rest of the conversation.

    Gemini takes the system instruction as a separate argument, not as a message
    in the list, so we pull those out here.
    """
    system = "\n".join(m["content"] for m in messages if m["role"] == "system")
    convo = [m for m in messages if m["role"] != "system"]
    return system, convo


def _gemini(messages, model, temperature) -> str:
    # The library is imported inside the function so you only need it installed
    # when you actually use Gemini.
    from google import genai
    from google.genai import types

    key = os.getenv("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("GEMINI_API_KEY not set (add it to .env)")
    client = genai.Client(api_key=key)

    # Gemini wants the system prompt separately and the rest as one text blob.
    system, convo = _split_system(messages)
    contents = "\n\n".join(f"{m['role']}: {m['content']}" for m in convo) or " "

    resp = client.models.generate_content(
        model=model,
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=system or None, temperature=temperature),
    )
    return (resp.text or "").strip()


def _groq(messages, model, temperature) -> str:
    from groq import Groq

    key = os.getenv("GROQ_API_KEY")
    if not key:
        raise RuntimeError("GROQ_API_KEY not set (add it to .env)")
    client = Groq(api_key=key)

    # Groq uses the OpenAI-style API: pass the messages list as-is.
    resp = client.chat.completions.create(
        model=model, messages=messages, temperature=temperature)
    return resp.choices[0].message.content.strip()


def _ollama(messages, model, temperature) -> str:
    import ollama

    # Ollama also takes the messages list directly and talks to a local server.
    resp = ollama.chat(model=model, messages=messages,
                       options={"temperature": temperature})
    return resp["message"]["content"].strip()


# ===========================================================================
# Native tool calling (class 5.2)
# ===========================================================================
# chat() returns text. For agents we also want the model to REQUEST a tool in a
# structured way. chat_tools() exposes the provider's NATIVE tool-calling API and
# returns a normalized reply: either text, or a list of tool calls to run.
#
# Messages are the OpenAI-style list used across the course. A tool spec is the
# OpenAI / JSON-schema shape:
#     {"type": "function", "function": {"name": ..., "description": ...,
#                                       "parameters": {a JSON schema}}}
# Groq and Ollama accept this shape directly. Gemini's SDK uses a different tool
# object, so native tool calling through chat_tools() supports groq and ollama;
# for gemini, use the provider-agnostic manual-JSON loop from class 5.1
# (llm.chat), which works with every provider. This is the "same loop, different
# transport" point from the slides: the think-act-observe loop is identical;
# native calling is just a cleaner transport than hand-parsed JSON.

from dataclasses import dataclass, field


@dataclass
class ToolCall:
    """One tool the model asked to run."""
    id: str
    name: str
    args: dict


@dataclass
class ToolReply:
    """A normalized reply: text, or tool calls to run (or both)."""
    text: "str | None"
    tool_calls: list = field(default_factory=list)   # list[ToolCall]
    raw: object = None

    @property
    def wants_tool(self) -> bool:
        return bool(self.tool_calls)


def chat_tools(messages, tools, provider=None, model=None, temperature=0.2,
               tool_choice="auto") -> "ToolReply":
    """One model call with native tool calling. Returns a ToolReply.

    messages: OpenAI-style list (may include prior assistant tool-call turns and
              {"role": "tool", ...} results).
    tools:    list of OpenAI-style tool specs.
    tool_choice: "auto" (model decides), "required" (must call a tool), or "none"
              (no tools). Honored by groq; ollama ignores it.
    """
    provider = (provider or PROVIDER).lower()
    if model is None:
        model = MODEL_NAME if provider == PROVIDER else DEFAULT_MODEL[provider]
    t0 = time.perf_counter()
    try:
        if provider == "groq":
            reply = _groq_tools(messages, tools, model, temperature, tool_choice)
        elif provider == "ollama":
            reply = _ollama_tools(messages, tools, model, temperature)
        elif provider == "gemini":
            raise RuntimeError(
                "chat_tools (native tool calling) supports groq and ollama; for "
                "gemini use the manual-JSON loop from class 5.1 (llm.chat).")
        else:
            raise ValueError(f"unknown PROVIDER {provider!r}; use groq or ollama")
    except Exception as e:
        log.warning("chat_tools failed provider=%s model=%s error=%s", provider, model, e)
        raise
    log.info("chat_tools provider=%s model=%s tools=%d calls=%d latency=%.3fs",
             provider, model, len(tools), len(reply.tool_calls), time.perf_counter() - t0)
    return reply


def _groq_tools(messages, tools, model, temperature, tool_choice) -> "ToolReply":
    import json as _json
    from groq import Groq
    key = os.getenv("GROQ_API_KEY")
    if not key:
        raise RuntimeError("GROQ_API_KEY not set (add it to .env)")
    client = Groq(api_key=key)
    resp = client.chat.completions.create(
        model=model, messages=messages, tools=tools,
        tool_choice=tool_choice, temperature=temperature)
    msg = resp.choices[0].message
    calls = []
    for tc in (msg.tool_calls or []):
        args = tc.function.arguments
        if isinstance(args, str):                 # groq returns arguments as a JSON string
            args = _json.loads(args or "{}")
        calls.append(ToolCall(id=tc.id, name=tc.function.name, args=args))
    return ToolReply(text=msg.content, tool_calls=calls, raw=resp)


def _ollama_tools(messages, tools, model, temperature) -> "ToolReply":
    import ollama
    resp = ollama.chat(model=model, messages=messages, tools=tools,
                       options={"temperature": temperature})
    msg = resp["message"]
    calls = []
    for i, tc in enumerate(msg.get("tool_calls") or []):
        fn = tc["function"]
        args = fn["arguments"]                    # ollama returns arguments as a dict already
        calls.append(ToolCall(id=f"call_{i}", name=fn["name"], args=dict(args)))
    return ToolReply(text=msg.get("content") or None, tool_calls=calls, raw=resp)


def assistant_tool_call_message(reply: "ToolReply") -> dict:
    """The assistant turn that requested the tools, to append BEFORE the results."""
    import json as _json
    return {
        "role": "assistant",
        "content": reply.text or "",
        "tool_calls": [
            {"id": c.id, "type": "function",
             "function": {"name": c.name, "arguments": _json.dumps(c.args)}}
            for c in reply.tool_calls
        ],
    }


def tool_result_message(call: "ToolCall", result: str) -> dict:
    """The tool result to append AFTER running one tool call (the observation)."""
    return {"role": "tool", "tool_call_id": call.id, "name": call.name, "content": result}


if __name__ == "__main__":
    # Running `python llm.py` directly prints the resolved config and makes one
    # call, a quick way to check your .env is set up correctly.
    print(f"PROVIDER={PROVIDER}  MODEL_NAME={MODEL_NAME}")
    print(chat([{"role": "user", "content": "Say hello in one short sentence."}]))
