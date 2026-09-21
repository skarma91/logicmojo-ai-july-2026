# Micro-assignment 4.9: a house-style dataset for movies

Prepare a small supervised fine-tuning dataset that teaches a **house style** on a **new** track, movies, not the IRS assistant: answer a plot description with the film as `Title (Year)` plus one spoiler-free line, and a fixed refusal when the film is unknown. This teaches a behavior and format, not facts. Work in `assignment.ipynb`. No training is run; this is dataset preparation, like the class build.

## Problems

1. **House-style rule.** Write the `SYSTEM` rule: reply with `Title (Year)` then one spoiler-free sentence, and a fixed refusal (`"I do not recognize that film."`) when unknown. **Expected:** a single system string stating the format.

2. **Build chat rows.** Write `to_rows(pairs)` that turns `(description, target)` pairs into chat-format rows with `system`, `user`, and `assistant` messages, and build a handful of examples (include one refusal). **Expected:** a list of dicts, each with a three-message `messages` list.

3. **Inspect.** Print the number of examples and one formatted example (its three messages). **Expected:** the count and a readable system/user/assistant example.

4. **Split.** Split the rows into train and validation. **Expected:** two disjoint lists whose sizes sum to the total.

5. **Justify and name a risk (reasoning).** In two to four sentences, argue why fine-tuning (not RAG or prompting) fits this behavior, and name one risk. **Expected:** a justification plus a concrete risk (for example, train and eval overlap, or facts leaking into the targets).

## How this is checked

A reference solution is in the `solution/` folder. Compare your dataset shape, the formatted example, and your reasoning to the expected output.
