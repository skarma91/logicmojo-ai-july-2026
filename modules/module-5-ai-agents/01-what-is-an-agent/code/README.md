# Class 5.1 code: the simplest agent loop

`main.py` hand-rolls the think, act, observe loop with exactly one tool, the
Module 4 retriever (`retriever.py`). The model decides whether to search, reads the
result, and answers, looping until it emits a final answer or hits the step cap.

## Why manual JSON here (and native tool calling in 5.2)

`main.py` uses the provider-agnostic manual-JSON protocol from class 4.3: the model
replies with one JSON object, a tool call or a final answer, and our code parses it.
That is what "hand-roll the loop" means, and it works with any provider through
`llm.chat`. Class 5.2 replaces this with the provider's native tool-calling API and
contrasts the two.

## The loop is bounded

It stops on a final answer, or when `max_steps` is reached. An agent does not stop
on its own; the step cap is the first of the reliability rails built out in 5.6.

## What ships here

```
main.py             the agent loop (run this)
retriever.py        the Module 4 dense retriever, wrapped as one tool
llm.py              the shared model wrapper (llm.chat over gemini/groq/ollama)
data/corpus.jsonl   the tax-publication corpus the retriever searches
```

## Run

```
pip install sentence-transformers numpy python-dotenv
# plus your provider, e.g.:  pip install ollama   (then `ollama serve`)
python main.py
```

Set `PROVIDER` / `MODEL_NAME` / keys in a `.env` at the repo root (copy
`.env.example`). The retriever caches its embeddings to `data/embeddings.npy` plus
`data/ids.json` on first run and reuses them after (git-ignored).

## The two demo questions

The first ("how much can I contribute to my HSA?") needs the tool, so the agent
searches, then answers with a cited `[id]`. The second ("what is 2 plus 2?") needs
no tool, so the agent answers directly: choosing includes choosing not to act.
