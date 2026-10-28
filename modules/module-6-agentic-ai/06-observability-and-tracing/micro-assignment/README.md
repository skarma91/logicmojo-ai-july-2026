# Class 6.6 micro-assignment: diagnose a failed run from its trace

A movie team was asked: *"Recommend films like the heist movie The Italian Job, and
tell me who directed it."* It ran for nine seconds, made four model calls, and ended
with no answer. You have only its trace: `data/failed_trace.jsonl`, one span per
line, in the format the class tracer writes (span_id, parent_id, name, kind, start,
end, tokens, cost, error, metadata).

The lines are in the order the spans **finished**, the way a real tracer writes them,
so a child appears before its parent. Nothing here calls a model: diagnosing from a
trace is reading data, which is why it works on runs you did not see happen.

## Problems

1. **`build_tree(spans)`.** Return the root span and a map from each `parent_id` to
   its children sorted by start time.
2. **`failed_spans(spans)`.** Return `(span_id, error)` for every span with an error,
   in start order.
3. **`self_time(span, kids)`.** A span's duration minus the time covered by its
   children. Two of the root's children ran at the same time; make sure overlapping
   time is not subtracted twice.
4. **`cost_share(spans)`.** For each span with a cost, its share of the run's cost,
   $s_i = c_i / \sum_j c_j$, rounded to three places. Look at how the shares change
   from hop to hop, and work out why.
5. **`critical_path(span, kids)`.** The span ids of the chain that set the wall time:
   start from the child that finished last, step back to the child that finished last
   before it started, repeat, and recurse into each chosen child.
6. **`diagnose(spans)` and `DIAGNOSIS`.** Return the failure that *caused* the outcome
   (not the root's error, which is only the symptom), how many times that exact
   failure repeated, the cost of the model calls that could not have helped, and the
   time wasted after the first failure. Then write `DIAGNOSIS`: which span failed, why,
   why the run never recovered, and what you would change in the tool and in the
   supervisor.

## Expected output

The six lines are exact:

```
[1] root: run | children of root: ['s1', 'f1', 'c1', 's2', 'f2', 's3', 'f3', 's4']
[2] failed spans: [('r0', 'RuntimeError: hop cap (4) reached without an answer'), ('f1', "ValueError: 2 films titled 'The Italian Job' (1969, 2003); pass a year"), ('f2', "ValueError: 2 films titled 'The Italian Job' (1969, 2003); pass a year"), ('f3', "ValueError: 2 films titled 'The Italian Job' (1969, 2003); pass a year")]
[3] self time: root 0.47 | supervisor hop 1 0.04
[4] cost share: [('l1', 0.183), ('l2', 0.231), ('l3', 0.272), ('l4', 0.314)]
[5] critical path: ['s1', 'l1', 'c1', 's2', 'l2', 'f2', 's3', 'l3', 'f3', 's4', 'l4']
[6] diagnosis: {'first_failure': 'f1', 'name': 'facts', 'error': "ValueError: 2 films titled 'The Italian Job' (1969, 2003); pass a year", 'repeats': 3, 'wasted_cost': 0.0002205, 'wasted_time': 7.65}
```

Your `DIAGNOSIS` is free text, but it should cover:

- the `facts` tool failed because the title matches two films;
- the supervisor retried with the same input until the hop cap, so the run could only
  fail the same way;
- each retry cost more than the last, because the supervisor's context grows every hop;
- a fix in the tool (accept a year, or return every match) and in the supervisor
  (do not repeat a call that already failed with the same arguments).

You can cross-check your numbers with the class analyzer:
`python ../code/analyze.py data/failed_trace.jsonl`.

## Run

```
python assignment.py
```

No packages beyond the Python standard library, and no API key.

## How this is checked

A reference solution is in the `solution/` folder. Compare your output to the expected
output above.
