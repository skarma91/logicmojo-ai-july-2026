# Class 4.7 code: how LLMs are served (a benchmark)

`main.py` measures what the slides explained. It **streams** one fixed prompt
through a local 4-bit model (Ollama) and through Groq, so it can time the first
token (TTFT) and the throughput (tokens per second), not just the total. It prices
the tokens at a rough tier rate, and runs the prompt twice so the repeated prefix
shows the prompt-cache cost saving. Streaming lives in `main.py` (not the shared
`llm.py`) because TTFT only exists when you read tokens as they arrive.

## What ships here

```
main.py     the latency/cost benchmark (run this)
llm.py      unified LLM access: gemini | groq | ollama (same file as 4.1-4.6)
```

## Run

```
pip install python-dotenv
pip install ollama      # local, free; then: ollama serve && ollama pull llama3.2
pip install groq        # hosted; set GROQ_API_KEY in .env at the project root
python main.py
```

The benchmark calls each provider explicitly with `llm.chat(..., provider=...)`.
If a provider is not installed or configured, it is skipped with a note, so the
script still runs with whichever you have. Per-provider default model: groq ->
`openai/gpt-oss-20b`, ollama -> `llama3.2` (override with `MODEL_NAME` in `.env`).

## What you should see

Per provider: TTFT, total time, and throughput (tokens per second), plus two costs.
Run 1 is priced at the full input rate, run 2 with the shared prefix at the
discounted `cached_in` rate
(frontier APIs bill cached-prefix tokens at roughly 10-25% of the normal input
rate). Typically the local 4-bit model is free but slower on modest hardware,
while Groq is metered but very fast, and run 2 is a little faster and cheaper than
run 1 when the shared prefix is cached.

## Honest note

The numbers depend on your machine, the chosen model, and the provider, and the
tier prices in `main.py` are rough teaching values, not a quote. The lesson is
the shape of the result, not the exact figures. Token counts are estimated at
about four characters per token, not measured from the provider's tokenizer.
