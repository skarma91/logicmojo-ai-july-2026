# Class 6.3 micro-assignment: trim the critic's context

The class build cut a tax team's tokens by scoping what each agent sees. Here you do
the same for one agent on a movie team, using `data/movies.jsonl` (18 real films).

- **scout** (deterministic) finds the films in the requested genre.
- **critic** (LLM) picks one film for the viewer and replies as `Title (year): why`.

Today the critic is handed everything: the whole catalog as pretty-printed JSON plus
the scout's raw output. You trim its context to the films in the genre and the four
fields it actually uses (title, year, director, plot), then measure its real prompt
tokens both ways. A trim only counts if the pick still passes the same check.

## Problems

1. **`select_records(genre)`.** The films in that genre, newest first.
2. **`project(record, fields)`.** A new dict with only the named fields, in order.
3. **`critic_context(genre)`.** The trimmed context: one line per film from
   `select_records`, formatted `Title (year), dir. Director: plot`.
4. **`shared_prefix_tokens(a, b)`.** How many leading whitespace-separated tokens two
   prompts share. This is the part a prompt cache could reuse. The expected output
   compares a prompt that starts with a request id against one that ends with it.
5. **`passes(answer, candidates)`.** True if the critic's pick names one of the
   candidate films. Read the dataset's titles before you decide what to match on: a
   check that accepts the wrong film is worse than no check.

## Expected output

The deterministic lines are exact:

```
[1] select_records('Animation') -> ['Coco', 'Finding Nemo', 'Spirited Away', 'Toy Story']
[2] project(Inception, title/year/director) -> {'title': 'Inception', 'year': 2010, 'director': 'Christopher Nolan'}
[3] critic_context('Animation'): lines = 4 | first = Coco (2017), dir. Lee Unkrich: A boy who...
[4] shared prefix, volatile first -> 1 | stable first -> 8
[5] passes: False True False
```

Then the critic runs twice per request (full context, then trimmed) with a real
model. Picks and exact counts vary, but these properties must hold:

- `prompt_tokens` are the provider's own numbers, and the trimmed call uses far fewer
  than the full one (roughly 80 to 90 percent fewer).
- The trimmed pick passes the check for both requests.

## Run

```
pip install python-dotenv groq      # or ollama
python assignment.py                # PROVIDER=groq (or ollama) in .env
```

Four critic calls in total. The full-context calls are about 2,000 tokens each, well
inside Groq's free tier.

## How this is checked

A reference solution is in the `solution/` folder. Compare your output to the expected
output above.
