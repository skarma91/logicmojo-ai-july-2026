# Capstone: the Travel Planner

Build a multi-agent system that turns a short trip request into a day-by-day itinerary
that respects the traveller's constraints, uses only places found in real Wikivoyage
guides, and cites a source for each one. The problem, the approach, and the high-level
design are in the class 7.3 (Interview prep and capstone) deck. The low-level design is
yours.

> "Four relaxed days in Lisbon. I love food and old churches."
> becomes a 4-day plan, at most a few activities a day, every place taken from the
> Lisbon guide, with its page URL, and every interest covered.

## What you are given

The `assignment/` folder is where you build. It starts with plumbing you built in earlier modules. Use it as it is:

| File | What it gives you |
|---|---|
| `llm.py` | `chat()` over Groq, Ollama, or Gemini; `chat_json()`; retries on HTTP 429; cost per agent |
| `pii.py` | `redact()` personal data to placeholders; `restore()` them in the final output |
| `fetch_data.py` | downloads Wikivoyage guides into `data/guides.jsonl` |
| `data/guides.jsonl` | the committed sample: guide chunks for 12 cities, each with section and page URL |
| `requirements.txt`, `.env.example` | setup |

See `assignment/README.md` for setup.

## Requirements

Your system must:

1. **Use a team of agents** with at least a researcher (retrieval over the guides), a
   planner (writes the itinerary), and a critic (checks it), coordinated by a supervisor.
   Plain Python or LangGraph, your choice.
2. **Redact personal data** with `pii.py` before any model call, log, or trace.
3. **Read the constraints** from each request: destination, number of days, pace, and
   interests.
4. **Ground every place.** The planner may schedule only places from the pool your
   researcher retrieved, and the output cites each place's page URL.
5. **Check the hard constraints in code**, not in a prompt: exact number of days, no
   day above the pace limit you define, every place in the pool, every interest covered,
   no repeats.
6. **Bound the review loop:** a fixed maximum number of revisions, plus a hard step
   limit. When the cap is reached, return the best plan that passes the code checks,
   clearly marked, or refuse.
7. **Refuse clearly** when the destination has no guide or the constraints cannot be met,
   instead of inventing a plan.
8. **Validate every structured model reply**, with a safe fallback for each field.
9. **Trace every step:** agent, duration, model calls, tokens, errors. No request text or
   personal data in traces.
10. **Evaluate** on at least 10 fixed requests that you write, across several cities,
    paces, and trip lengths, including at least one destination with no guide and one
    request with more interests than days.

## A tip about the data

Wikivoyage keeps most individual listings for big cities in district pages. The main city
pages you are given often name places inside prose ("Alfama", "the 25th of April Bridge")
rather than as a list. Decide how your researcher turns that prose into named places, and
how you check that each name really appears in the guide. Some interests have little or no
content for some cities; decide what your system does then.

## What to submit

A repository (or folder) containing:

- **`DESIGN.md`**, your low-level design: the modules and what each owns, your data
  shapes, each agent's inputs and outputs, the supervisor's routing rules and revision cap,
  how retrieval works and how you turn guide text into named places, and the reason
  behind each major choice. One architecture diagram.
- **The code**, runnable from your `README.md` on a fresh machine.
- **`EVAL_REPORT.md`**: your evaluation requests, every metric below defined in plain
  words and reported, with a 95% Wilson interval where you report a rate, and at least
  two failures explained from your traces.
- **A short demo:** the command and output for one successful plan and one refusal.
- Optional: a live URL, deployed as in class 6.8 (Deployment).

Metrics to report: constraint pass rate, grounding rate (the share of scheduled places
found in the retrieved text), interest coverage, revisions per request, correct refusals,
and model calls, tokens, and seconds per request.

## Rubric

| Part | What good looks like | Weight |
|---|---|---|
| Low-level design document | Clear modules, data shapes, agent contracts, routing rules, and reasons for each choice | 25% |
| Working system | Runs from the README on a fresh machine; produces plans that pass the code checks | 25% |
| Grounding and constraints | Code checks for every hard constraint; no invented places in the evaluation | 20% |
| Evaluation report | At least 10 requests, every metric defined and reported, failures explained | 20% |
| Tracing and privacy | Useful traces with no personal data; redaction before any model call | 10% |

## Data license

Wikivoyage text is CC BY-SA 4.0. Credit Wikivoyage, keep the page URL with every chunk
you use, and share anything built from the text under the same license.

## How this is checked

A reference solution, with its own design document and evaluation report, is in the
`solution/` folder. Compare your outputs to it.
