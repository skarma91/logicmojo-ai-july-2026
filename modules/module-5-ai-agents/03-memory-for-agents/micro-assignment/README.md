# Class 5.3 micro-assignment: remember a preference across sessions

Build a movie assistant that remembers the user's genre preference in **long-term
memory**, recalls it in a **later session**, and, when the preference **changes**,
updates the one stored fact instead of piling up a second, contradictory copy.

Work in `assignment.py`. The `MemoryStore` and `Conversation` are given (`memory.py`),
and the memory-aware turn loop is given. Your job is the tools and the write policy.
Native tool calling supports groq and ollama.

## What to build

1. **The tool registry** (`make_tools`), with two tools:
   - `save_preference(text)`: call `store.add(text, kind="semantic", key=PREF_KEY)`.
     Using the same `key` is what makes a changed preference **overwrite** the old
     one rather than duplicate it.
   - `recommend_movie(genre)`: given (returns a movie of that genre).
   Give each a clear description and a typed schema.
2. **Two sessions sharing one `store`** (in `main`):
   - Session 1: the user states a preference; the agent saves it and recommends.
   - Session 2: a **new `Conversation`** (fresh working memory), same store: confirm
     the preference is **recalled** (retrieved and used).
   - Then the user **changes** the preference. Confirm exactly **one**
     genre-preference item remains in the store, with the new value.

## Expected behavior

Session 1 saves "science fiction" and recommends a sci-fi film. Session 2, with no
working memory of session 1, still recommends sci-fi because the preference is in
long-term memory. After "I prefer crime now", the store holds a single
genre-preference item reading "crime", not two conflicting ones.

## How this is checked

A reference solution is in the `solution/` folder. Compare your output, and the
final count of genre-preference items (it should be 1), to it.
