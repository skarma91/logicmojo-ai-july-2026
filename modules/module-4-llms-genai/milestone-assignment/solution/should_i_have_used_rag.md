# Should I have used RAG instead? (reference answer)

**Short answer: I needed both, for different jobs. RAG for the facts, fine-tuning
for the style. Neither one alone would have produced the assistant.**

## What each lever actually changed

The assistant has two separable requirements:

- **Know the right facts and cite them.** The normal blood pressure ranges, the
  recommended hours of sleep, the weekly physical-activity target, all live in the
  corpus and are revised as health guidance is updated.
- **Answer in a consistent house style.** Concise, always cite the source topic and
  section, decline politely when the answer is not in the documents, and stay general
  health information rather than personal medical advice.

RAG solves the first. The facts are retrieved at query time from the Chroma store,
so they are current, grounded, and citable, and I can update them by re-indexing a
new document, no retraining. Fine-tuning cannot do this reliably: a model tuned on
older guidance would state it confidently and be wrong, with no citation and no way
to refresh short of another training run. This is the "behavior, not facts" rule
from class 4.9.

Fine-tuning solves the second. The house style is a *behavior*: it is the same
regardless of which passage was retrieved, so it does not belong in the context. I
could push it into the prompt instead ("be concise, cite the source and section,
decline if unknown"), and for a first version I would. I reach for fine-tuning only
once that prompt is long, repeated on every call, and still applied inconsistently.
Then baking the style into the weights makes it the default and shortens every prompt.

## Why not fine-tune on the documents and skip RAG?

This is the tempting shortcut and it fails on all three counts that matter here:

- **Staleness.** Health guidance is revised over time; the weights would not be.
- **No citations.** A tuned model cannot point to the passage it used, and this
  assistant must cite the source topic and section.
- **Hallucination.** Tuning teaches phrasing, not reliable recall of specific
  numbers; it would produce fluent, confident, sometimes wrong figures, which is
  especially unacceptable for health information.

## Why not RAG-only and skip fine-tuning?

RAG-only is a perfectly good v1, and honestly where I would start. The case for
adding the tune is narrow: the style was not holding from the prompt alone across
many questions, and the repeated style instructions were adding tokens to every
call (a real cost at scale, class 4.7). The tune removed that instruction from the
prompt and made the style consistent. If the prompt had been enough, I would not
have fine-tuned, exactly the "reach for fine-tuning last" order from class 4.9.

## Evidence from the build

Running the assistant with the base model versus the tuned model (`--model
dka-assistant`) on the class 4.8 eval questions: both cite the same retrieved
facts (RAG did its job in both), but the tuned model holds the concise, always-cited
format without being told, while the base model needs the full style prompt and
still drifts. The house-style adherence metric from class 4.10 is higher for the
tuned model; faithfulness is essentially the same, because faithfulness comes from
the retrieved context, not the tune.

## Verdict

Use RAG for what the model must *know* and keep current; use fine-tuning for how it
must *behave*. For this assistant that means RAG is mandatory and fine-tuning is a
justified, but secondary, optimization once a prompt alone stopped carrying the
style. Start with prompt plus RAG; add the tune only when the style will not hold.
