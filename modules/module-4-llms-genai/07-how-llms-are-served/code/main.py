"""Class 4.7 build: benchmark the same prompt two ways, with real streaming.

The slides explained WHY serving speed and cost vary (KV cache, quantization,
hardware). This script MEASURES it. It streams one prompt from a local 4-bit model
(Ollama) and from Groq, and reports the two serving-latency numbers that matter:

  * time to first token (TTFT): how long until the first token appears. This is
    the prefill wait, and it is what makes a chat feel responsive.
  * throughput: tokens per second once generation starts (the decode speed). This
    is what sets the total time for a long answer.

It also runs the prompt twice so the repeated prefix shows the prompt-cache cost
saving, and prices tokens per tier.

Run:
    pip install python-dotenv
    pip install ollama    # local, free; then: ollama serve && ollama pull llama3.2
    pip install groq      # hosted; set GROQ_API_KEY in .env at the project root
    python main.py

Streaming is done here in the benchmark (not in the shared llm.py) because TTFT
only exists when you read tokens as they arrive. The chat model config still comes
from llm.py (same .env, same per-provider defaults).

NOTE: the exact numbers depend on your machine, the model, and the provider. The
point is the SHAPE: Groq's hardware usually wins on TTFT and throughput; local is
free per token but slower on modest hardware; a repeated prefix is cheaper.
"""

from __future__ import annotations
import logging
import os
import time

import llm   # for the .env config and per-provider default model names

log = logging.getLogger("course.serving")

# A deliberately long, FIXED system prompt. Being the same on every call, it is
# exactly the shared prefix that prompt caching can reuse.
SYSTEM = (
    "You are a careful assistant for a small business accounting tool. Answer in "
    "plain language, at most three sentences. Do not invent numbers. If you are "
    "unsure, say so briefly. Always stay on the topic of the user's question and "
    "avoid filler. " * 3
)
QUESTION = "In one short paragraph, what is the difference between a debit and a credit?"

# Illustrative prices per 1,000,000 tokens, by tier. Real prices change often, so
# treat these as rough teaching numbers. "cached_in" is the discounted rate for
# input tokens served from a cached prefix (frontier APIs charge roughly 10-25%
# of the normal input rate).
PRICE_PER_1M = {
    "local":    {"in": 0.0,  "cached_in": 0.0,  "out": 0.0},
    "hosted":   {"in": 0.10, "cached_in": 0.02, "out": 0.50},
    "frontier": {"in": 3.0,  "cached_in": 0.30, "out": 15.0},
}
PROVIDER_TIER = {"ollama": "local", "groq": "hosted", "gemini": "frontier"}


def estimate_tokens(text: str) -> int:
    """A rough token count: about 4 characters per token in English."""
    return max(1, len(text) // 4)


def cost_usd(in_tokens: int, out_tokens: int, tier: str) -> float:
    """Price a call at a tier, using the per-1,000,000-token rates above."""
    p = PRICE_PER_1M[tier]
    return in_tokens / 1_000_000 * p["in"] + out_tokens / 1_000_000 * p["out"]


def cost_usd_cached(prefix_tokens: int, new_tokens: int, out_tokens: int, tier: str) -> float:
    """Price a call whose PREFIX is served from cache (the discounted rate) while
    the NEW part and the output are charged normally. This is the second call in
    the benchmark, where the shared system prompt is reused."""
    p = PRICE_PER_1M[tier]
    return (prefix_tokens / 1_000_000 * p["cached_in"]
            + new_tokens / 1_000_000 * p["in"]
            + out_tokens / 1_000_000 * p["out"])


def _resolve_model(provider: str) -> str:
    """Use the .env MODEL_NAME if it belongs to this provider, else the default."""
    return llm.MODEL_NAME if provider == llm.PROVIDER else llm.DEFAULT_MODEL[provider]


# ---------------------------------------------------------------------------
# Streaming timers. Each yields tokens as they arrive, so we can stopwatch the
# FIRST one (TTFT) and the total, then derive throughput. One per provider,
# because each SDK streams a little differently.
# ---------------------------------------------------------------------------
def stream_ollama(model, messages):
    import ollama
    t0 = time.perf_counter()
    ttft, pieces = None, []
    for chunk in ollama.chat(model=model, messages=messages, stream=True,
                             options={"temperature": 0.2}):
        piece = chunk.get("message", {}).get("content", "")
        if piece and ttft is None:
            ttft = time.perf_counter() - t0        # first token has arrived
        pieces.append(piece)
    return "".join(pieces), ttft, time.perf_counter() - t0


def stream_groq(model, messages):
    from groq import Groq
    key = os.getenv("GROQ_API_KEY")
    if not key:
        raise RuntimeError("GROQ_API_KEY not set (add it to .env)")
    client = Groq(api_key=key)
    t0 = time.perf_counter()
    ttft, pieces = None, []
    for chunk in client.chat.completions.create(model=model, messages=messages,
                                                temperature=0.2, stream=True):
        piece = chunk.choices[0].delta.content or ""
        if piece and ttft is None:
            ttft = time.perf_counter() - t0
        pieces.append(piece)
    return "".join(pieces), ttft, time.perf_counter() - t0


STREAMERS = {"ollama": stream_ollama, "groq": stream_groq}


def measure(provider: str) -> dict:
    """Stream one prompt from a provider and return its timing."""
    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": QUESTION}]
    reply, ttft, total = STREAMERS[provider](_resolve_model(provider), messages)
    out_tokens = estimate_tokens(reply)
    tps = out_tokens / total if total > 0 else 0.0     # throughput: tokens/second
    log.info("stream provider=%s ttft=%.3fs total=%.3fs tok/s=%.1f",
             provider, ttft or 0.0, total, tps)
    return {"ttft": ttft, "total": total, "out_tokens": out_tokens, "tps": tps}


def benchmark(provider: str) -> dict | None:
    """Run the same prompt twice. Returns None if the provider is not set up."""
    try:
        return {"first": measure(provider), "second": measure(provider)}
    except Exception as e:
        print(f"  [skipped {provider}: {e}]")
        return None


def main():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    prefix, new = estimate_tokens(SYSTEM), estimate_tokens(QUESTION)
    print(f"Prompt: system {prefix} tokens + question {new} tokens\n")

    results = {}
    for provider in ("ollama", "groq"):
        print(f"=== {provider} ({PROVIDER_TIER[provider]} tier) ===")
        r = benchmark(provider)
        if r is None:
            continue
        results[provider] = r
        f, s = r["first"], r["second"]
        full = cost_usd(prefix + new, f["out_tokens"], PROVIDER_TIER[provider])
        cached = cost_usd_cached(prefix, new, s["out_tokens"], PROVIDER_TIER[provider])
        ttft = f"{f['ttft']:.2f}s" if f["ttft"] is not None else "n/a"
        print(f"  TTFT: {ttft}   total: {f['total']:.2f}s   throughput: {f['tps']:.1f} tok/s")
        print(f"  cost run 1 (full): ${full:.6f}   run 2 (cached prefix): ${cached:.6f}")
        print()

    # Compare TTFT, the number Groq's hardware targets.
    if "ollama" in results and "groq" in results:
        a, b = results["ollama"]["first"]["ttft"], results["groq"]["first"]["ttft"]
        if a and b:
            faster = "Groq" if b < a else "local"
            print(f"{faster} had the lower time-to-first-token. Groq's inference "
                  f"hardware usually wins on TTFT and throughput; local is free per token.")
    elif results:
        print("Only one provider was available. Set up the other to compare.")
    else:
        print("No provider was reachable. Install and configure Ollama or Groq "
              "(see the header), then run again.")


if __name__ == "__main__":
    main()
