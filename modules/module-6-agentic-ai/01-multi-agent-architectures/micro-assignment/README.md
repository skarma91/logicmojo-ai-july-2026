# Class 6.1 micro-assignment: add a third specialist

The class build turned one agent into a team. Here you grow a **movie team** by one
member and get the routing right, on a different dataset (movies, not tax documents).

The team in `assignment.py`: a **supervisor** (the only LLM) routes each request to
worker tools over `data/movies.jsonl` (18 real films). It calls a `route` tool that
names the next worker, the input to give it, and, on FINISH, the final answer. Two
workers exist:

- **plot_search**: finds films whose plot matches a description.
- **facts**: year, genre, and director for an exact title (note: two films share the
  title *The Italian Job*, so it lists every match).

You add **recommender**, and the rules that keep the supervisor using it correctly.

## Problems

1. **`recommend(title)`.** Return up to 3 *other* titles: same director first, then
   same genre, each group newest first, no duplicates; an unknown title returns `[]`.
2. **Register the recommender.** Add it to `WORKERS` as
   `("<role description>", recommender)`, so the supervisor's `route` tool offers it.
3. **Write the role description.** The supervisor chooses workers by reading their
   descriptions (class 5.2, Tool calling and ReAct: descriptions drive tool choice). Make it use the
   recommender **only** when the user asks for recommendations or similar films, and
   pass it a title.
4. **`workers_used(route_log)`.** Return the workers the supervisor sent work to, in
   order, excluding FINISH. You will use it to check that routing includes *not*
   routing.
5. **Guard the recommender.** It needs an exact catalog title. If the supervisor hands
   it a description instead ("the movie about thieves who enter dreams"), reroute to
   `plot_search` first with the reason `recommender needs a catalog title; resolving it
   first`. A cheap rule beats a wasted call.

## Expected output

The deterministic lines are exact:

```
[1] recommend('Inception') -> ['Interstellar', 'The Dark Knight', 'The Matrix']
    recommend('Coco')      -> ['Finding Nemo', 'Spirited Away', 'Toy Story']
    recommend('Avatar')    -> []
[2] route tool workers     -> ['plot_search', 'facts', 'recommender', 'FINISH']
[5] guard on a description -> plot_search
```

The routed requests use a real model, so exact wording varies, but these properties
must hold:

| Request | `recommender_used` | Workers include |
|---|---|---|
| Who directed Inception? | `False` | `facts` |
| Suggest movies like Inception. | `True` | `recommender` |
| Which movie is about thieves who enter dreams? Recommend similar ones. | `True` | `plot_search` **before** `recommender` |

The answers name Christopher Nolan for the first request and recommend Interstellar,
The Dark Knight, and The Matrix for the other two.

## Run

```
pip install langgraph python-dotenv groq      # or ollama
python assignment.py                          # PROVIDER=groq (or ollama) in .env
```

Each request costs only two or three supervisor calls, comfortably inside Groq's free
tier.

## How this is checked

A reference solution is in the `solution/` folder. Compare your output to the expected
output above.
