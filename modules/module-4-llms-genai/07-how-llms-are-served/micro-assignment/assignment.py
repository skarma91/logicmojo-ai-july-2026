"""Micro-assignment 4.7: measure latency and cost across providers.

Real calls through llm.chat(..., provider=...) (shipped as llm.py here). Set up at
least one provider first (Ollama local, and a Groq key in .env to compare). See
README.md for the five problems.

    pip install python-dotenv ollama groq
"""

import time
import llm

# Given: a fixed prompt (the system message is the shared prefix).
SYSTEM = "You are a terse assistant. Answer in one short sentence."
QUESTION = "Name one benefit of running a model locally."

# Given: rough per-1,000,000-token prices by tier, and the provider -> tier map.
PRICE_PER_1M = {"local": {"in": 0.0, "out": 0.0},
                "hosted": {"in": 0.10, "out": 0.50}}
TIER = {"ollama": "local", "groq": "hosted"}

# ---- 1. time_call(provider, messages) -> (reply, seconds) ----
# your code here

# ---- 2. estimate_tokens(text) and cost_usd(in_tokens, out_tokens, tier) ----
# your code here

# ---- 3. benchmark(provider): run the prompt, print latency + cost; skip if unavailable ----
# your code here

# ---- 4 and 5. Run twice for the prefix effect, then reason about tiers ----
if __name__ == "__main__":
    # your code here
    pass
