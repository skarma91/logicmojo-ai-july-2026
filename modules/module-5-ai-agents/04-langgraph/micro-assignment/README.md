# Class 5.4 micro-assignment: route by query type, and resume

Extend the class LangGraph agent with a **router**: a conditional edge off the entry
that sends factual questions down the model-and-tools path and action requests down a
separate action node. Then show the graph **resuming** from a checkpoint on the same
thread, with a watchlist that persists.

Work in `assignment.py`. The model and tools nodes and the movie search tool are
given. Native tool calling supports groq and ollama.

## What to build

1. **State**: add a `watchlist` field with an **append** reducer (like `messages`),
   and a `route` field (plain string, overwrite).
2. **Router node** (`classify`): read the latest user message; if it contains an
   action word (`ACTION_WORDS`), set `route="action"`, else `route="factual"`.
3. **Action node** (`action_node`): pull the title from the user message, append it
   to the watchlist, and add an assistant confirmation.
4. **The graph**: `START -> router`; a conditional edge from `router` to `model`
   (factual) or `action` (action); the `model`/`tools` cycle; `action -> END`.
   Compile with a `MemorySaver` checkpointer.
5. **Run and resume**: ask a factual question, then **resume the same thread** with
   an action request. Print both answers and the persisted watchlist.

## Expected behavior

The factual question routes to `model`, runs `search_movies`, and answers. The action
request, on the **same thread**, routes to `action` and adds the title. Because the
checkpointer persisted the thread, the watchlist survives across the two calls and
prints with the added title.

## How this is checked

A reference solution is in the `solution/` folder. Compare your routing, your
watchlist output, and the resume behavior to it.
