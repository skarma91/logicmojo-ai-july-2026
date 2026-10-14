"""Micro-assignment 4.7 reference solution: measure latency and cost.

Real calls through llm.chat(..., provider=...). Set up Ollama (free, local) and a
Groq key in .env to compare both tiers.

    pip install python-dotenv ollama groq
"""

import time
import llm

SYSTEM = "You are a terse assistant. Answer in one short sentence."
QUESTION = "Name one benefit of running a model locally."

PRICE_PER_1M = {"local": {"in": 0.0, "out": 0.0},
                "hosted": {"in": 0.10, "out": 0.50}}
TIER = {"ollama": "local", "groq": "hosted"}


# ---- 1. one timed, real call to a specific provider ----
def time_call(provider, messages):
    t0 = time.perf_counter()
    reply = llm.chat(messages, provider=provider)      # REAL call
    return reply, time.perf_counter() - t0


# ---- 2. token estimate and cost ----
def estimate_tokens(text):
    return max(1, len(text) // 4)                      # ~4 chars per token


def cost_usd(in_tokens, out_tokens, tier):
    p = PRICE_PER_1M[tier]
    return in_tokens / 1_000_000 * p["in"] + out_tokens / 1_000_000 * p["out"]
    # cost_usd(900, 300, "hosted") = 0.00009 + 0.00015 = 0.000240


# ---- 3. benchmark one provider (skip cleanly if not set up) ----
def benchmark(provider):
    msgs = [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": QUESTION}]
    try:
        reply, s1 = time_call(provider, msgs)
        _, s2 = time_call(provider, msgs)              # same prompt again
    except Exception as e:
        print(f"  [skipped {provider}: {e}]")
        return None
    it, ot = estimate_tokens(SYSTEM + QUESTION), estimate_tokens(reply)
    return {"s1": s1, "s2": s2, "cost": cost_usd(it, ot, TIER[provider])}


if __name__ == "__main__":
    for provider in ("ollama", "groq"):
        r = benchmark(provider)
        if r:
            print(f"{provider}: run1 {r['s1']:.2f}s  run2 {r['s2']:.2f}s  "
                  f"cost ${r['cost']:.6f}")

    # 4. Why run 2 can be faster: the system prompt is the same on both calls, so
    #    the provider can reuse the processed prefix (KV / prompt cache) instead of
    #    recomputing it. The model is also "warm" (already loaded) on the second call.
    #
    # 5. Which tier for which job: ship a HIGH-VOLUME background feature (batch
    #    summaries, overnight extraction) on the cheaper or local tier, where a bit
    #    of latency does not matter and cost adds up. Ship an INTERACTIVE chat on the
    #    fast tier (Groq), where users feel every extra second even if it costs more.
