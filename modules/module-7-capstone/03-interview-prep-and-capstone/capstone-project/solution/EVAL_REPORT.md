# Travel Planner: evaluation report (reference)

## What is measured

Thirteen fixed requests in `code/data/eval_requests.jsonl`: eleven that should get a
plan, across all twelve cities, one to four days, every pace, and two that must be refused.

| Case | Request (short) | Tests |
|---|---|---|
| e01 | 3 relaxed days, Lisbon, food and old churches | the standard path |
| e02 | 2 days, Kyoto, temples and culture | a thin guide |
| e03 | 4 days, Mexico City, food, art, history | the longest trip |
| e04 | 3 packed days, Cape Town, nature, food | the packed limit |
| e05 to e08 | Istanbul, Hanoi, Edinburgh, Buenos Aires | mixed interests and paces |
| e09 | 1 packed day, Marrakech, five interests | more interests than slots |
| e10 | 1 relaxed day, Vancouver, nature | the shortest trip |
| e11 | 2 days, Singapore, with a name and a phone number | redaction |
| e12 | Paris | no guide: must refuse |
| e13 | 10 days, Prague | too long: must refuse |

Metrics, defined in `code/eval.py`: constraint pass rate (95% Wilson interval), grounding
rate over every drafted plan, interest coverage, revisions per plan, correct refusals, and
calls, tokens, and seconds per request.

## Measured without a model: what the guides can support

`python eval.py --retrieval` counts the chunks each interest retrieves per city (at most 4).
These numbers are real, from the committed `guides.jsonl`:

| City | food | history | art | architecture | nature | nightlife | shopping | culture |
|---|---|---|---|---|---|---|---|---|
| Buenos Aires | 4 | 4 | 1 | 0 | 4 | 4 | 4 | 4 |
| Cape Town | 4 | 4 | 4 | 4 | 4 | 4 | 4 | 4 |
| Edinburgh | 4 | 4 | 4 | 4 | 4 | 4 | 1 | 4 |
| Hanoi | 4 | 4 | 4 | 4 | 4 | 4 | 4 | 4 |
| Istanbul | 4 | 4 | 4 | 4 | 4 | 4 | 4 | 4 |
| Kyoto | 4 | 4 | 0 | 4 | 4 | 1 | 2 | 4 |
| Lisbon | 4 | 4 | 4 | 2 | 4 | 4 | 1 | 4 |
| Marrakech | 4 | 4 | 4 | 4 | 4 | 4 | 4 | 4 |
| Mexico City | 4 | 4 | 4 | 4 | 4 | 4 | 2 | 4 |
| Prague | 4 | 4 | 4 | 4 | 4 | 4 | 2 | 4 |
| Singapore | 4 | 4 | 4 | 4 | 4 | 4 | 4 | 4 |
| Vancouver | 4 | 4 | 4 | 1 | 4 | 4 | 4 | 4 |

Reading it: Kyoto has no art content and Buenos Aires no architecture content on their main
pages, which is why the supervisor drops an unsupported interest with a note instead of
failing the coverage check forever.

## Verified offline: the mechanics

With scripted model replies (the sandbox that built this has no model access), every path
behaves as designed:

- a valid plan on the first draft; a first plan with an invented id and an over-packed day,
  rejected by the exact checks and fixed on revision; a plan the judge rejects once, then
  approves; a judge that never approves, giving `best_effort` after three plans; a planner
  that never produces a valid plan, giving `refused` after three plans;
- Paris and a ten-day trip refused at intake; an interest with no guide content refused or
  dropped with a note;
- an invented place name from extraction removed by the verbatim check;
- no email, phone number, or two-word name in any model prompt or trace line.

These runs prove the routing and checks work. They say nothing about plan quality,
because the scripted replies are not a real model.

## Live results: fill in from your run

Run `python eval.py` with a provider configured, then record the output here. Until then,
the table holds no numbers on purpose: any figure written before a live run would be
invented.

| Metric | Result | Notes |
|---|---|---|
| Constraint pass rate | | 95% interval: |
| Grounding rate (drafted plans) | | |
| Interest coverage | | |
| Revisions per plan | | |
| Correct refusals | | |
| Cost per request | | calls, tokens, seconds |

What to look for in a live run, and how to read it:

- **Grounding below 100% in drafts** means the planner wrote ids outside the pool; the
  critic caught them, but each one cost a revision.
- **Many revisions** with few failing checks means the judge is too strict; tighten its
  prompt or accept `best_effort` more often.
- **Low interest coverage** on thin guides (Kyoto, Edinburgh) points at retrieval or at
  the main-page-only data, not at the planner.
- Explain at least two failures from `traces/eval.jsonl`: which step, which check, and
  what you changed.
