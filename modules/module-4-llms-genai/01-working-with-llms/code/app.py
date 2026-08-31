"""Class 4.1 build: one LLM wrapper behind a small Streamlit web page.

WHAT THIS DOES
--------------
It shows a text box and a provider dropdown. You type a question, pick a backend
(Ollama, Groq, or Gemini), and it prints the reply plus what the call cost and
how long it took. It is the class notebook's wrapper turned into a tiny app.

Run it with:

    pip install streamlit ollama groq google-genai python-dotenv
    streamlit run app.py

Every call is a REAL model call, so set up a provider first: run Ollama locally
(free, the default) or put a Groq or Gemini API key in a .env file at the project
root (see .env.example). Every call is logged with provider, token counts, cost,
and latency.
"""

import logging
import os
import time

import streamlit as st

# Load the .env file so GROQ_API_KEY / GEMINI_API_KEY become available via
# os.environ. Wrapped in try/except so the app still starts if python-dotenv is
# not installed (Ollama, the default, needs no key).
try:
    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv())
except Exception:
    pass

# ----------------------------------------------------------------------------
# Logging: configure it once. Every model call writes one INFO line with a
# timestamp, so you can see what happened (a habit that pays off in Module 5).
# ----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger("llm")

# ----------------------------------------------------------------------------
# Cost model. A call is priced per token, and input and output tokens cost
# different amounts. PRICES maps a "tier" to (price_per_input, price_per_output)
# in dollars per 1,000,000 tokens. These numbers are ILLUSTRATIVE and change
# often; the point is the ratio between tiers, not the exact figure.
# TIER_OF maps each provider to the tier it belongs to.
# ----------------------------------------------------------------------------
PRICES = {"local": (0.0, 0.0), "hosted": (0.05, 0.08), "frontier": (2.50, 10.0)}
TIER_OF = {"ollama": "local", "groq": "hosted", "gemini": "hosted"}


def call_cost(n_in, n_out, tier):
    """Dollar cost of one call: input tokens times input price, plus output."""
    p_in, p_out = PRICES[tier]
    return n_in / 1e6 * p_in + n_out / 1e6 * p_out


# ----------------------------------------------------------------------------
# Providers. Each function takes the messages list and returns the same shape:
# {"text": reply, "in_tokens": N, "out_tokens": N}. Keeping the return shape
# identical is what lets one chat() function treat them interchangeably. Each
# imports its library inside the function, so you only need the one you use. Every
# provider here makes a REAL model call.
# ----------------------------------------------------------------------------
def ollama_provider(messages, temperature=0.7, model="llama3.2"):
    """Call a model running locally via Ollama (free, private, needs a server)."""
    import ollama

    r = ollama.chat(model=model, messages=messages, options={"temperature": temperature})
    return {
        "text": r["message"]["content"],
        # Ollama reports how many tokens it read and wrote; default to 0 if absent.
        "in_tokens": r.get("prompt_eval_count", 0),
        "out_tokens": r.get("eval_count", 0),
    }


def groq_provider(messages, temperature=0.7, model="openai/gpt-oss-20b"):
    """Call a hosted open model on Groq (OpenAI-style API, needs GROQ_API_KEY)."""
    from groq import Groq

    client = Groq(api_key=os.environ["GROQ_API_KEY"])
    # Groq accepts the messages list directly.
    r = client.chat.completions.create(model=model, messages=messages, temperature=temperature)
    u = r.usage                       # exact token counts reported by the API
    return {
        "text": r.choices[0].message.content,
        "in_tokens": u.prompt_tokens,
        "out_tokens": u.completion_tokens,
    }


def gemini_provider(messages, temperature=0.7, model="gemini-3.5-flash-lite"):
    """Call Google Gemini (needs GEMINI_API_KEY)."""
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    # Gemini takes the system prompt separately, and the rest as one text blob,
    # so we split the messages into a system string and a joined conversation.
    system = "\n".join(m["content"] for m in messages if m["role"] == "system")
    convo = "\n\n".join(f"{m['role']}: {m['content']}"
                        for m in messages if m["role"] != "system") or " "
    r = client.models.generate_content(
        model=model, contents=convo,
        config=types.GenerateContentConfig(
            system_instruction=system or None, temperature=temperature))
    u = r.usage_metadata              # Gemini's token usage lives here
    return {
        "text": r.text or "",
        "in_tokens": u.prompt_token_count,
        "out_tokens": u.candidates_token_count,
    }


# A name -> function lookup, so we can pick a provider by the dropdown's string.
PROVIDERS = {
    "ollama": ollama_provider,
    "groq": groq_provider,
    "gemini": gemini_provider,
}


def chat(messages, provider="ollama", temperature=0.7):
    """One entry point for all three backends, timed, priced, and logged.

    This is the whole point of the wrapper: the caller says which provider to use
    and everything else (timing, cost, logging) happens the same way regardless.
    """
    if provider not in PROVIDERS:
        raise ValueError(f"unknown provider: {provider}")
    t0 = time.perf_counter()                              # start the stopwatch
    out = PROVIDERS[provider](messages, temperature=temperature)   # the actual call
    out["latency"] = time.perf_counter() - t0            # seconds the call took
    # Look up this provider's price tier and compute the dollar cost.
    out["cost"] = call_cost(out["in_tokens"], out["out_tokens"], TIER_OF[provider])
    # One structured log line per call: provider, tokens, cost, latency.
    log.info(
        "provider=%s in=%d out=%d cost=$%.6f latency=%.3fs",
        provider, out["in_tokens"], out["out_tokens"], out["cost"], out["latency"],
    )
    return out


# ----------------------------------------------------------------------------
# The Streamlit page. Streamlit re-runs this file top to bottom on every
# interaction, and each st.* call draws one widget on the page.
# ----------------------------------------------------------------------------
st.title("Ask the model")
st.caption("One wrapper, three real backends: Ollama (local), Groq and Gemini (hosted).")

# Two side-by-side controls: a provider dropdown and a temperature slider.
col1, col2 = st.columns(2)
provider = col1.selectbox("Provider", list(PROVIDERS))
temperature = col2.slider("Temperature", 0.0, 1.5, 0.7, 0.1)

# Two text boxes: the system prompt (the model's standing instructions) and the
# user's question for this turn.
system = st.text_input("System prompt", "You are a concise assistant.")
question = st.text_input("Your question", "How do I export invoices to CSV?")

# st.button returns True only on the run where it was just clicked.
if st.button("Send") and question:
    # Build the message list the same way the notebook did.
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": question},
    ]
    out = chat(messages, provider=provider, temperature=temperature)
    st.write(out["text"])                                # show the reply
    st.caption(                                          # show the metrics under it
        f"provider={provider} | in={out['in_tokens']} out={out['out_tokens']} "
        f"| cost=${out['cost']:.6f} | {out['latency']:.3f}s"
    )
