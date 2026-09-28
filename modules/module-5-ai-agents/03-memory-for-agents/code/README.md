# Class 5.3 code: an agent that remembers

This is the class 5.2 agent given **memory**. Two kinds work together (`memory.py`):

- **Working memory** (`Conversation`): recent turns kept verbatim, older turns
  **compacted** into a running summary (a real `llm.chat` call) so the context stays
  bounded.
- **Long-term memory** (`MemoryStore`): durable facts and preferences in a vector
  store (the class 4.4 embedding idea), retrieved by similarity each turn.

## The write policy is the hard part

`MemoryStore.add` does not blindly append. Given a stable `key` (e.g.
`filing_status`), a changed value **updates** the existing item; without a key, a
near-duplicate is **merged** into the existing one. So a preference that changes
overwrites the old value instead of leaving two contradictory copies.

## Each turn

`run_turn` retrieves relevant long-term memories, assembles the model input
(system + memories + running summary + recent turns), runs the tool loop
(`search_tax_docs`, `save_memory`), records the answer, and compacts if the buffer
grew. The `save_memory` tool writes through the dedupe/update policy.

## What ships here

```
main.py             the memory-aware agent (run this)
memory.py           MemoryStore (long-term) + Conversation (working memory)
llm.py              chat + chat_tools (from class 5.2)
retriever.py        the Module 4 retriever, as a tool
data/corpus.jsonl   the corpus the retriever searches
```

`main.py` runs two sessions that share one `MemoryStore`: session 1 saves the user's
preferences, session 2 (a fresh conversation) **recalls** them, showing memory
persist across sessions. `data/memory.jsonl` is generated and git-ignored.

## Run

```
pip install sentence-transformers numpy python-dotenv
# plus your provider:  pip install groq   (set PROVIDER=groq in .env), or ollama
python main.py
```

Set `PROVIDER` / keys in a `.env` at the repo root. Native tool calling targets
groq and ollama; for gemini, use the class 5.1 manual-JSON loop (the memory design
is unchanged).
