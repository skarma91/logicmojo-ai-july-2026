# Class 5.2 micro-assignment: a real-API tool and its failures

The class build handled a real API's failures (timeout, bad response). Here you do
it yourself on the movie dataset. A tool that hits the network is where reliability
is won or lost, so the point of this micro is the failure handling, not the happy
path.

Work in `assignment.py`. The native-tool-calling loop (`run_agent`) and one offline
tool (`search_movies`) are given. Native tool calling supports groq and ollama.

## What to build

1. **A real-API tool**, `lookup_show(title)`: call a keyless public API (for example
   TVmaze, `https://api.tvmaze.com/singlesearch/shows?q=<title>`) with `requests`
   and a **timeout**, and return a short factual summary (name, premiere, genres).
   Let its errors propagate; you handle them in step 3.
   **Expected:** for "Inception" it returns a one-line factual summary.
2. **Register both tools** with a name, a clear description, and a typed schema. The
   description is what drives tool choice, so make the two clearly distinct.
3. **Complete `dispatch`** so every failure becomes an observation fed back to the
   model, never a crash. Handle at least: an unknown tool, a missing required
   argument, a **timeout** (`requests.Timeout`), a **bad response**
   (`requests.HTTPError` from `raise_for_status`), and an **unexpected payload**
   (`ValueError` or `KeyError`).
   **Expected:** each failure returns a clear `error: ...` string; the agent reads
   it and recovers or answers without the tool.
4. **Show it working** on three questions: one that routes to `search_movies`, one
   that routes to `lookup_show`, and one that needs **no tool**.

## Expected behavior

The plot question uses `search_movies`; the "when did it premiere" question uses
`lookup_show` and returns real details; the no-lookup question is answered directly.
When the API is unreachable or the title is unknown, the agent reports the failure
gracefully instead of crashing. The loop stays bounded by `max_steps`.

## How this is checked

A reference solution is in the `solution/` folder. Compare your output to the
expected behavior above.
