# Class 6.2 micro-assignment: a writer-reviewer loop for movie blurbs

The class build looped a tax answer between a writer and a reviewer. Here you build
the same bounded loop on a different dataset: short catalog blurbs for films in
`data/movies.jsonl` (18 real films).

```
START -> writer -> reviewer --approve, or 3 rounds--> END
           ^          |
           +-revise---+
```

A blurb passes only if:

- **exact checks pass**: at most 60 words, and it names the film's director; and
- **the LLM reviewer approves**: every claim is supported by the film's record.

Exact checks override the LLM. The writer's brief deliberately leaves out the
director rule, so the reviewer has something real to catch on the first round.

## Problems

1. **`check_blurb(blurb, movie)`.** Return the failed exact checks, in order:
   `too long (N words; limit 60)` and `does not name the director (<director>)`
   (case-insensitive match). An empty list means the blurb passed.
2. **`combine(failed, verdict)`.** Merge the exact failures with the LLM's verdict
   (`{"approve": bool, "feedback": str}`) into `(approved, feedback)`. Approved only
   if the LLM approved *and* nothing failed; otherwise the feedback is the failed
   checks followed by the LLM's feedback, joined with `"; "`.
3. **`route_after_review(state)`.** Return `END` on approval or once `rounds`
   reaches `MAX_ROUNDS`; otherwise `"writer"`. This is the loop's guaranteed exit.
4. **`expected_rounds(q, k)`.** Implement the formula from the deck,
   $\mathbb{E}[\text{rounds}] = \frac{1-(1-q)^k}{q}$, where $q$ is the chance one
   draft passes and $k$ the round cap. It must also return the right answer for
   every $q$ between 0 and 1, inclusive.
5. **`estimate_q(rounds, approved)`.** From observed runs (rounds each blurb took,
   and whether it was approved), estimate $q$, the chance a single review round
   passes. Decide carefully what counts as one trial, and what a blurb that hit the
   cap without approval contributes.

## Expected output

The deterministic lines are exact:

```
[1] check_blurb(good)   -> []
    check_blurb(no dir) -> ['does not name the director (Christopher Nolan)']
    check_blurb(long)   -> ['too long (63 words; limit 60)']
[2] combine(exact fail, LLM approves) -> (False, 'does not name the director (Christopher Nolan)')
    combine(no fails, LLM approves)   -> (True, '')
[3] route(approved, round 1) -> __end__ | route(rejected, round 1) -> writer | route(rejected, round 3) -> __end__
[4] expected_rounds: [1.75, 1.24, 3.0, 1.0]
[5] estimate_q([1, 2, 3], [True, True, False]) -> 0.333
```

The loop then runs on four films with a real model, so the blurbs vary, but these
properties must hold:

- Every film shows its rounds, the exact failures per round, and the final blurb.
- No film takes more than 3 rounds (`max` is at most 3).
- An approved blurb has at most 60 words and names its director.
- The last lines print the estimated $q$, the mean rounds the formula predicts from
  it, and the observed mean; with only four films they need not match exactly.

## Run

```
pip install langgraph python-dotenv groq      # or ollama
python assignment.py                          # PROVIDER=groq (or ollama) in .env
```

At most 24 calls (4 films, 3 rounds, a writer and a reviewer call each), well
inside Groq's free tier.

## How this is checked

A reference solution is in the `solution/` folder. Compare your output to the expected
output above.
