# Micro-assignment 4.1: a logged, multi-provider LLM wrapper (movie theme)

Build the small wrapper the class sketched, make it observable, and wrap it in a Streamlit page. Work in `assignment.ipynb`. These tasks make **real** model calls through `llm.chat` (shipped as `llm.py` in this folder), so set up a provider first: run Ollama locally (free, the default) or put a Groq or Gemini key in a `.env` file at the project root.

## Problems

1. **Message builder.** Write `build_messages(system, user)` that returns the two-message list in the correct role format. Expected: a list of two dicts with roles `system` then `user`.

2. **Token and cost estimate.** Reuse `estimate_tokens` and `call_cost`. For a prompt of 900 input and 300 output tokens, print the cost at all three tiers. Expected: local `$0.000000`, hosted `~$0.000069`, frontier `~$0.005250`.

3. **Provider switch.** Write `chat(messages, provider)` that makes a real call through `llm.chat` and returns a dict with `text`, `in_tokens`, `out_tokens` (estimate the token counts from the text). Expected: `chat(msgs)` returns a real reply dict from the default provider; `chat(msgs, provider="nope")` raises a `ValueError` from `llm.chat` you can explain.

4. **Cost logging.** Wrap the call so each one logs `provider`, `in_tokens`, `out_tokens`, `cost`, and `latency` at INFO. Expected: one INFO line per call in the shown format.

5. **A/B two tiers (reasoning).** Make one real call, then price its token counts at two tiers, print the comparison, and write one or two sentences on which tier you would ship this feature on and why. Expected: a short comparison plus a justified choice (exact numbers vary with the real reply; the reasoning is graded).

## Streamlit app

Copy the wrapper into `app.py` (a complete reference build is in `../code/app.py`) and run `streamlit run app.py`. It should show a text box, a provider dropdown, and print the reply plus the logged cost.

## How this is checked

A reference solution is in the `solution/` folder. Compare your outputs and reasoning to it. Because the calls are real, your exact text, token counts, and latency will differ from the reference; the structure and the cost/logging behavior are what matter.
