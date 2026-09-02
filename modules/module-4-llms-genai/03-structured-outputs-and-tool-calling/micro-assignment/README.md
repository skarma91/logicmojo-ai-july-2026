# Micro-assignment 4.3: validated extraction and a tool call (movies)

Same pattern as class (a Pydantic contract, validate-and-retry, a tool round trip), on a **new** theme: movies. Work in `assignment.ipynb`. Pydantic and the tool registry are plain Python, but the extraction and tool-call steps make **real** calls through `llm.chat` (shipped as `llm.py` here), so set up a provider first (Ollama local, or a Groq/Gemini key in `.env`).

## Problems

1. **Schema.** Define a Pydantic model `Movie` with `title: str`, `year: int`, and `genre: Literal["action","animation","crime","drama","romance","science fiction"]`. **Expected:** all three fields required.

2. **Valid case.** Validate `'{"title":"Inception","year":2010,"genre":"science fiction"}'` and print the typed fields. **Expected:** `title` is `"Inception"`, `year` is a real `int`.

3. **Malformed case.** Validate a bad genre (e.g. `"thriller"`) and catch the error. **Expected:** a `ValidationError` on `genre` (a `literal_error`).

4. **Validate and retry.** Write `extract_movie(text, max_tries=3)` that calls the real model, validates the JSON against `Movie`, and on failure feeds the error back and retries. **Expected:** a validated `Movie`, usually in 1 try.

5. **Tool call (reasoning).** Write the schema for `lookup_movie_rating(title)` and a real `run(question)` round trip (the model emits a JSON action, your code runs the tool and feeds the result back, the model answers). Run it on a rating question, then say why a step cap matters. **Expected:** an answer containing the rating, plus your reasoning.

## How this is checked

A reference solution is in the `solution/` folder. Problems 1 to 3 are exact; the extraction and tool-call steps use a real model, so exact text varies while the validation, retry, and tool round trip are what matter.
