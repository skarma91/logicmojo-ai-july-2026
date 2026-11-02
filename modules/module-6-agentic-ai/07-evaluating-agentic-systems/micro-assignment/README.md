# Class 6.7 micro-assignment: an eval that catches a regression

A movie team (a supervisor with `plot_search`, `facts`, and `recommender` workers) was
run on eight tasks, twice:

- `data/baseline_runs.jsonl`: a version everyone trusted.
- `data/candidate_runs.jsonl`: a new version someone wants to ship.

Each run records the worker steps it took, its tool calls with arguments, and its
answer. `data/tasks.jsonl` gives every task its reference steps and a success rule
(phrases the answer must contain).

The candidate looks fine if you only read answers. Your job is an evaluation that
judges the path as well, with a **pinned rule** (a director question must call the
`facts` tool before answering), and a gate that decides whether the candidate ships.
Everything is recorded, so no model is called and every number is exact.

## Problems

1. **`in_order(actual, reference)`.** True if every reference step appears in `actual`
   in the same order; extra steps are allowed.
2. **`step_pr(actual, reference)`.** Step precision and recall with multisets, rounded
   to two places. Two identical calls against one required call count one as useful and
   one as waste. Decide what an empty `actual` should score.
3. **`director_rule(task, run)`.** The pinned rule: a question that asks who directed a
   film must call `facts`. Other questions pass. Read the recorded runs first: should
   you check the steps, the tool calls, or both?
4. **`wilson(k, n)`.** The Wilson 95% interval for a pass rate, rounded to two places.
5. **`gate(baseline, candidate)`.** Task ids that passed in the baseline and fail in the
   candidate, sorted. A task passes only if its answer succeeds, its path is in order,
   and it keeps the director rule.

## Expected output

```
[1] in_order(['plot_search', 'facts'], ['facts']) -> True | in_order(['facts', 'plot_search'], ['plot_search', 'facts']) -> False
[2] step_pr(['recommender', 'recommender'], ['recommender']) -> (0.5, 1.0) | step_pr([], ['facts']) -> (0.0, 0.0)
[3] director rule broken in candidate: ['m1', 'm2', 'm4', 'm7', 'm8']
[4] wilson(8, 8) -> (0.68, 1.0) | wilson(3, 8) -> (0.14, 0.69)
    baseline  pass 8/8  wilson (0.68, 1.0)  |  answer-only 8/8
    candidate pass 3/8  wilson (0.14, 0.69)  |  answer-only 7/8
    right answer, wrong path in candidate: ['m1', 'm2', 'm4', 'm8']
[5] gate -> FAIL ['m1', 'm2', 'm4', 'm7', 'm8']
```

Read the last four lines together. An answer-only eval scores the candidate 7 out of 8
and would ship it. It answered four director questions from the model's memory and got
them right, but the one ambiguous title (*The Italian Job*, two films) went wrong,
because the step that would have caught the ambiguity was skipped. Notice also that
the candidate's m3 still passes, with a precision of 0.5: a wasted step is a cost
problem, not a regression.

## Run

```
python assignment.py
```

No packages beyond the Python standard library, and no API key.

## How this is checked

A reference solution is in the `solution/` folder. Compare your output to the expected
output above.
