# Travel Planner: reference solution

A team of agents that turns "Three relaxed days in Lisbon, I love food and old churches"
into a day-by-day plan built only from places in the Wikivoyage guide, with sources. Read
`DESIGN.md` for the low-level design and `EVAL_REPORT.md` for how it is measured.

```
code/
  llm.py, pii.py, fetch_data.py     given helpers (same as assignment/)
  guides.py                         the data and shared vocabularies
  schemas.py                        the data shapes
  agents/intake.py                  request -> destination, days, pace, interests
  agents/researcher.py              retrieval, place extraction, verbatim grounding
  agents/planner.py                 a plan that refers to places by pool id
  agents/critic.py                  exact checks in code, then a judgment
  supervisor.py                     the bounded plan-and-critique loop
  tracing.py                        one JSON line per step
  main.py, eval.py                  run it, measure it
  data/guides.jsonl                 1,278 Wikivoyage chunks, 12 cities (CC BY-SA 4.0)
  data/eval_requests.jsonl          13 evaluation requests
```

## Run

```bash
cd code
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python llm.py                                           # checks your .env at the repository root
python main.py                                          # the sample request
python main.py "Two days in Hanoi, street food and culture, packed pace"
python eval.py --retrieval                              # no model: what each guide supports
python eval.py                                          # the full evaluation
```

Each run prints the constraints read from the request, the status (`plan`, `best_effort`,
or `refused`), the steps taken, the plan with each place's section, the source pages, and
the model calls and tokens per agent. Traces go to `traces/`, with no request text.

A request costs four model calls when the first plan is approved (intake, extraction,
plan, judgment) and two more per revision, at most eight.
