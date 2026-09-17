# Micro-assignment 4.7: measure latency and cost across providers

Turn the serving ideas into numbers. Send the same prompt through a local model (Ollama) and through Groq, time each, price the tokens, and reason about the trade-off. Work in `assignment.py`. These are **real** calls through `llm.chat(..., provider=...)` (shipped as `llm.py` here), so set up at least one provider first: Ollama local (free), and a Groq key in `.env` at the project root to compare.

## Setup

```
pip install python-dotenv
pip install ollama        # local; then: ollama serve && ollama pull llama3.2
pip install groq          # hosted; set GROQ_API_KEY in .env
```

## Problems

1. **Timed call.** Write `time_call(provider, messages)` that makes a real `llm.chat(..., provider=provider)` call and returns `(reply, seconds)`. **Expected:** a reply string and an elapsed time greater than 0.

2. **Tokens and cost.** Write `estimate_tokens(text)` (about 4 characters per token) and `cost_usd(in_tokens, out_tokens, tier)` using the given per-1,000,000-token rates. **Expected:** `cost_usd(900, 300, "hosted")` is about `$0.000240`.

3. **Benchmark two providers.** Run the same prompt through `ollama` and `groq`, printing latency and estimated cost for each, and skip a provider cleanly if it is not set up. **Expected:** one row per available provider.

4. **Repeated prefix (reasoning).** Run the same prompt twice on one provider, print both timings, and write one or two sentences on why the second call can be faster. **Expected:** two timings plus your reasoning (the shared prefix can be cached, so it is not reprocessed).

5. **Pick a tier (reasoning).** From your measured latency and cost, say which setup you would ship a high-volume background feature on, and which for an interactive chat, and why. **Expected:** a justified choice for each.

## How this is checked

A reference solution is in the `solution/` folder. Because the calls are real, your exact timings and costs will vary; compare your structure and reasoning, and the cost math in problem 2, to the expected outputs.
