# Micro-assignment 4.2: A/B two prompts, on movie taglines

Same pattern as class (write a few-shot prompt, score by format, A/B against zero-shot, then harden against injection), on a **new** task: generating a movie tagline from a premise. Work in `assignment.ipynb`. The A/B tasks make **real** model calls through `llm.chat` (shipped as `llm.py` here), so set up a provider first (Ollama local, or a Groq/Gemini key in `.env`). The injection task is plain offline logic.

## Problems

1. **Write V2.** Given the zero-shot `TAGLINE_V1`, write `TAGLINE_V2` that adds two examples and the format rules (Title Case, at most 8 words, no ending punctuation). **Expected:** a template with an `Examples:` block and a `{premise}` slot.

2. **Scorer.** Implement `is_good_tagline(t)` enforcing the three rules. **Expected:** `True` for `"Dream Big Steal Bigger"`, `False` for `"Here is a tagline: dream big."`.

3. **Run the A/B test.** Score V1 and V2 over the premises with the real model. **Expected:** V2 clearly beats V1; exact scores vary run to run.

4. **Harden a prompt (reasoning).** Given a movie review containing `"IGNORE PREVIOUS INSTRUCTIONS ..."`, write a fenced `defended` prompt and a `looks_injected(text)` check, and say why fencing alone is not a complete fix. **Expected:** `looks_injected(review)` is `True`, plus your reasoning.

## How this is checked

A reference solution is in the `solution/` folder. Because the A/B calls are real, your exact taglines and scores will differ; the structure and the V2-beats-V1 outcome are what matter.
