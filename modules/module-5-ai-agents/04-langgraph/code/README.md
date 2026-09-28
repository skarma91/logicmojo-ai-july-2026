# Class 5.4 code: the agent on LangGraph

`main.py` refactors the same agent from classes 5.1 to 5.3 onto a **LangGraph state
machine**. The Python for-loop becomes an explicit graph: a `model` node and a
`tools` node, a conditional edge that loops until the model stops, and a
checkpointer for persistence.

## The three LangGraph ideas

- **Typed State with a reducer.** `State.messages` is annotated with `operator.add`,
  so each node **returns new messages that get appended**, not a whole replacement.
  The append reducer is the thing beginners get wrong first.
- **Cycles.** `model -> (tools -> model)* -> END`. The conditional edge
  `should_continue` routes back to the model while it keeps asking for tools, then to
  `END`. The loop is now a visible cycle, not a hidden `while`.
- **Checkpointer.** `compile(checkpointer=MemorySaver())` snapshots the state after
  each node. A `thread_id` names one conversation, so a follow-up on the same thread
  resumes with the earlier turns automatically. This one feature gives memory,
  resumability, and human-in-the-loop.

## What ships here

```
main.py             the LangGraph agent (run this)
llm.py              chat + chat_tools (from class 5.2)
retriever.py        the Module 4 retriever, as a tool
data/corpus.jsonl   the corpus the retriever searches
```

`main.py` runs a query, then a follow-up on the **same `thread_id`**: the second
call sends only the new turn and the checkpointer supplies the rest, so the message
count grows across turns without you resending history.

## Run

```
pip install langgraph sentence-transformers numpy python-dotenv
# plus your provider:  pip install groq   (set PROVIDER=groq in .env), or ollama
python main.py
```

Set `PROVIDER` / keys in a `.env` at the repo root. Native tool calling targets groq
and ollama; for gemini, swap the model node for the class 5.1 manual-JSON call (the
graph is unchanged).
