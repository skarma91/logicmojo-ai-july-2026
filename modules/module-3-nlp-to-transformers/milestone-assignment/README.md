# Module 3 milestone: the text-intelligence layer for a support inbox

## Goal

You own the text-understanding layer for a SaaS accounting product (think invoices, subscriptions, bank reconciliation). Support messages arrive all day. Build the pipeline that reads each one and turns it into something the business can act on: how it tokenizes, whether the customer is unhappy, which team it belongs to, which help article answers it, and a drafted reply. Every stage uses a pretrained transformer, and every stage connects back to a concept from the module.

This is a build, not an essay. The inputs are fixed and given below. Work through `assignment.ipynb` top to bottom. It needs an internet connection for the first model download.

## The dataset (already in the notebook)

Eight real-looking inbox messages:

1. "I was charged twice for my subscription this month and I want a refund."
2. "My card was declined at checkout but I was still billed for the upgrade."
3. "The app crashes every time I open the bank reconciliation screen."
4. "After the latest update my reports show the wrong totals, this is broken."
5. "How do I export my invoices to a CSV file?"
6. "Where is the setting to add a second user to my account?"
7. "Please cancel my subscription, I do not need the service any more."
8. "I want to close my account and stop all future payments."

Four teams the inbox routes to: **Billing**, **Bug report**, **How-to**, **Cancellation** (each with a one-line description in the notebook).

A six-entry help centre (FAQ) covering export, adding users, payment methods, cancelling, reconciliation, and report totals.

## What to build

### Part 1. Tokenization audit (class 3.1, Text as data)

Load a WordPiece tokenizer (`bert-base-uncased`) and run it over all eight messages. For each message print the word count, the token count, and the token pieces. Then answer, in the notebook: which domain words split into several subword pieces, and why the token count matters for cost.

**Expected output:** a per-message table where the token count is at or above the word count; domain terms such as "reconciliation", "invoices", and "CSV" break into multiple `##` pieces, while common words stay whole. A one-line takeaway that a model is billed and budgeted in tokens, not words.

### Part 2. Sentiment triage (class 3.4, Using pretrained transformers)

Run the sentiment pipeline over all eight messages and flag the high-priority ones (a confident negative). Print each message with its label and score, sorted so the angriest customers surface first.

**Expected output:** messages 1 to 4 come back `NEGATIVE` with high confidence (~0.99) and are flagged high-priority; the cancellations (7, 8) usually read `NEGATIVE` too. The two how-to questions (5, 6) expose a real limitation: an SST-2 model has only `POSITIVE`/`NEGATIVE`, no neutral, so it forces a polarity on a neutral question, often at lower confidence. Name that limitation in a comment.

### Part 3. Routing and FAQ retrieval with embeddings (classes 3.2 Attention, 3.3 The Transformer, 3.4)

First, the mechanic: build a sentence vector by mean-pooling a DistilBERT `last_hidden_state`, and show that a paraphrase pair scores a high cosine similarity while an unrelated sentence scores low. Then the task: with a sentence-embedding model, embed the eight messages and the four team descriptions, and route each message to the nearest team by cosine similarity. Finally, for the two how-to questions, retrieve the single best-matching FAQ entry.

**Expected output:** the paraphrase pair scores clearly higher than the unrelated pair. The routing table sends messages 1 and 2 to Billing, 3 and 4 to Bug report, 5 and 6 to How-to, and 7 and 8 to Cancellation. Message 5 retrieves the "export invoices to CSV" FAQ and message 6 retrieves the "add a second user" FAQ.

### Part 4. Draft a reply and control the decoding (class 3.5, From models to LLMs)

Take message 5 (the CSV export question) and use a small generative model (`distilgpt2`) to draft a reply. Generate three times from the same prompt: greedy, then sampling at a high temperature, then nucleus (top-p) sampling, seeding for reproducibility. Print the token length of the prompt and say what fraction of the model's context window it uses.

**Expected output:** greedy is deterministic and tends to repeat itself; the high-temperature sample is more varied and more likely to wander off-topic; top-p sits in between, fluent but controlled. The three outputs are visibly different from one another, and the prompt token count prints as a small fraction of the context window. (`distilgpt2` is tiny, so the wording will be rough. That is the point: you are studying the decoding knobs, not the quality of a small model.)

### Write-up

Four or five sentences tying it together: how tokenization shaped the token counts and cost in Part 1, and how attention (each token pulling in context from the rest) explains both the sentiment flips in Part 2 (for example "not" attaching to "recommend") and the routing similarities in Part 3.

## Deliverable

The completed `assignment.ipynb` with its outputs and your write-up.

## How this is checked

A reference solution is in the `solution/` folder. Compare your outputs and reasoning to it.
