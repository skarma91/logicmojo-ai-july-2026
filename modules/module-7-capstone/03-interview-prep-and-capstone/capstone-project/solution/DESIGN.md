# Travel Planner: reference low-level design

One valid design for the capstone, not the only one. Compare the decisions and their
reasons with yours; a different choice with a good reason is as strong as this one.

## 1. Architecture

```
request text
  -> pii.redact                 personal data becomes <EMAIL_1>, <PERSON_1>, ...
  -> IntakeAgent      (model)   destination, days, pace, interests; validated in code
  -> ResearcherAgent  (model)   TF-IDF retrieval, then place extraction, then a verbatim check
  -> PlannerAgent     (model)   a day-by-day plan that refers to places by pool id
  -> CriticAgent      (code, then model)   exact checks; a judgment only on valid plans
       ^                   |
       +---- feedback -----+    at most MAX_REVISIONS = 2 revisions (3 plans)
  -> Result: "plan" | "best_effort" | "refused"
```

The `Supervisor` is a plain Python loop. `next_step(state)` reads the state and returns
a step name; it never calls a model. Every step runs inside a `Tracer.span`.

## 2. Modules

| Module | Owns |
|---|---|
| `llm.py`, `pii.py`, `fetch_data.py` | given helpers, unchanged from `assignment/` |
| `guides.py` | loads `data/guides.jsonl`; the shared vocabularies: `DESTINATIONS`, `INTERESTS` (interest to query words), `PLACE_SECTIONS`, `PACE_LIMIT`, `MAX_DAYS` |
| `schemas.py` | the data shapes below |
| `agents/intake.py` | `IntakeAgent.run(text) -> (TripRequest or None, reason)` |
| `agents/researcher.py` | `ResearcherAgent.retrieve(req)`, `ResearcherAgent.run(req) -> Pool` |
| `agents/planner.py` | `PlannerAgent.run(req, pool, feedback=None) -> Plan` |
| `agents/critic.py` | `exact_checks(plan, req, pool) -> list[str]`, `CriticAgent.run(plan, req, pool) -> Critique` |
| `supervisor.py` | `Supervisor.next_step(state)`, `handle(text, request_id)`, `finish(state)` |
| `tracing.py` | `Tracer.span(name, **attrs)`: one JSON line per step |
| `main.py`, `eval.py` | the command line; the evaluation |

Imports point one way: `schemas` and `guides` depend on nothing of ours; agents depend on
`schemas`, `guides`, and `llm`; the supervisor depends on everything. No agent imports
another agent.

## 3. Data shapes

| Dataclass | Fields | Produced by |
|---|---|---|
| `TripRequest` | destination, days, pace, interests | IntakeAgent |
| `Place` | id (`P1`...), name, section, interests, chunk_id, url | ResearcherAgent |
| `Pool` | destination, places; `by_id()` | ResearcherAgent |
| `Activity` | place_id, note | PlannerAgent |
| `Plan` | days: a list of lists of `Activity` | PlannerAgent |
| `Critique` | problems, approved, feedback | CriticAgent |
| `TripState` | text, request, pool, plan, critique, best, drafts, notes, rounds, trail | Supervisor |
| `Result` | status, reason, request, plan, pool, rounds, trail, drafts, notes | Supervisor |

## 4. Routing: `next_step`

| # | Condition, in order | Step |
|---|---|---|
| 1 | no request yet | `intake` (refuse if it returns a reason) |
| 2 | no pool yet | `research` (drop interests the guide cannot support, with a note; refuse if none remain) |
| 3 | fewer places than days | `refuse` |
| 4 | no critique yet | `plan` if there is no plan, else `critique` |
| 5 | the critique approved | `finish` |
| 6 | `rounds > MAX_REVISIONS` | `finish` |
| 7 | otherwise | `plan` again, with the critique as feedback |

`finish` returns `plan` if approved; otherwise `best_effort` with the last plan that
passed every exact check; otherwise `refused` with the failing checks. `MAX_STEPS = 10`
backs up the loop.

## 5. Key decisions and why

1. **Places by id, not by name.** The planner returns `P3`, never a free-text name. An
   invented place becomes an id that is not in the pool, which `exact_checks` catches
   with a dictionary lookup.
2. **Extraction by a model, grounding by code.** Big-city Wikivoyage pages name places
   inside prose ("Alfama", "the 25th of April Bridge") and keep individual listings in
   district pages. A model is good at spotting names; code keeps a name only if it
   appears word for word in the chunk it came from.
3. **Interests as retrieval queries.** Each interest maps to a short list of signal words
   (`guides.INTERESTS`). A place inherits the interests whose query retrieved its chunk,
   so "interest covered" is a deterministic check, not a model's opinion.
4. **Unsupported interests are dropped with a note.** The retrieval table in
   `EVAL_REPORT.md` shows, for example, no art content in the Kyoto guide. The planner
   plans what the guide supports and tells the traveller what was dropped.
5. **Word TF-IDF here, character n-grams in the support desk.** Guide text is edited and
   spelled correctly, and interest queries are clean words, so whole words work. The
   support desk's tickets were full of typos.
6. **The judge runs only on valid plans.** No tokens are spent on opinions about a plan
   that fails a hard rule. A valid plan the judge never approves is still returned, as
   `best_effort`, because it meets every constraint the traveller stated.
7. **Plain Python supervisor.** One request, one bounded loop, no persistence or
   streaming needed. LangGraph would be equally valid.

## 6. Known limitations

- **Single first names are not redacted.** `pii.py` detects names of two or three words
  after a cue ("my name is Daniel Okafor"); "I'm Priya" passes through. A named-entity
  model (for example Presidio with spaCy) would close the gap.
- **Grounding proves a name is in the text, not that it is a place.** A model could
  extract "Every Tuesday" from a listing. The judge and the evaluation catch some of
  this; a stricter version would keep only names from See, Do, Eat, and Drink listings
  that start a sentence or line.
- **Main city pages only.** District pages hold richer listings; adding them to
  `fetch_data.py` would enlarge every pool.
- **Facts age.** Wikivoyage text can be out of date; notes must not state opening hours
  or prices, and the planner prompt says so.
