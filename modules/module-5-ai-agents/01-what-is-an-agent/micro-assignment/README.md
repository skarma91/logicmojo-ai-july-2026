# Class 5.1 micro-assignment: a two-tool agent that chooses

The class build had one tool. Here you give the agent **two** tools over a different
dataset (movie plots), so it must choose which one fits each task, and sometimes
choose to use no tool at all. Choosing not to act is part of being an agent.

Work in `assignment.py`. The think-act-observe loop (`run_agent`) is given; you write
the tools and the prompt.

## The dataset

`data/movies.jsonl`: 18 movies, each with a title, year, genre, director, and plot.

## What to build

1. **A search tool**, `search_movies(query)`: return the movies whose plot best
   matches the query. A simple keyword match over each `plot` is enough (no model).
   **Expected:** for "planting an idea in a dream", Inception ranks first.
2. **A lookup tool**, `movie_facts(title)`: return the year, director, and genre for
   one movie by title, and handle a title not in the catalog.
   **Expected:** `movie_facts("Inception")` returns its year, director, and genre.
3. **Register both tools** with a name, a clear description, and a typed parameter
   schema. The description is what the model reads to choose, so make the two
   clearly distinct.
4. **Write the system prompt**: the one-JSON-object contract (a tool call or a final
   answer), and answer directly when no tool is needed.
5. **Show it choosing** on three questions, one that should call `search_movies`,
   one that should call `movie_facts`, and one that needs **no tool** (for example a
   simple arithmetic question), answered directly.

## Expected behavior

The plot question routes to `search_movies` and names the right film; the "who
directed" or "what year" question routes to `movie_facts`; the no-lookup question is
answered in one step with no tool call. Keep the step cap so the loop stays bounded.

## How this is checked

A reference solution is in the `solution/` folder. Compare your output to the
expected behavior above.
