# Interview Preparation Guide, Module 4: LLMs and GenAI

Questions and answers for the applied-LLM stack taught in Module 4: calling models, prompting, structured outputs, embeddings and RAG, serving and efficiency, evaluation, and fine-tuning with alignment. This is a study companion, not part of the published course site.

**Coverage map.** The questions follow the ten classes: 4.1 (Working with LLMs), 4.2 (Prompt engineering), 4.3 (Structured outputs and tool calling), 4.4 (Embeddings and vector databases), 4.5 (RAG from scratch), 4.6 (RAG with frameworks), 4.7 (How LLMs are served), 4.8 (Evaluating LLM and RAG), 4.9 (Fine-tuning I: the decision and the data), and 4.10 (Fine-tuning II: QLoRA and alignment). RAG and the "prompt vs RAG vs fine-tune" decision are the most heavily interviewed topics here, so spend the most time on Parts 5, 6, and 9.

## How to use this guide

Read the question, answer it out loud or on paper first, then expand the answer to check yourself. Each question is tagged by difficulty:

- **[Warm-up]** a definition or one-liner an interviewer opens with.
- **[Core]** the standard question you are expected to answer cleanly.
- **[Deep]** mechanism, trade-off, or a "why" that separates strong candidates.
- **[Numerical]** a small calculation to do by hand.
- **[Applied]** a scenario or short design or debugging prompt.

The RAG loop, the "behavior not facts" decision, and `W' = W + BA` are worth being able to draw and decode from memory.

---

## Part 1: Working with LLMs (class 4.1)

### Q1. [Warm-up] What is a token, and why are you billed in tokens rather than words?

<details><summary>Answer</summary>

A token is a subword piece the model reads and generates; common words are one token, rare words split into several. You are billed in tokens because that is the unit the model processes: cost and the context limit are both counted in input tokens plus output tokens, not words. A rough rule is about 4 characters or 0.75 words per token in English.
</details>

### Q2. [Core] What is the context window, and what has to fit inside it?

<details><summary>Answer</summary>

The context window is the maximum number of tokens a model can attend to in one call. Everything must fit: the system prompt, any conversation history you resend, retrieved context (RAG), the current user message, and the space reserved for the output. The model is stateless between calls, so "memory" of earlier turns exists only because you resend those tokens each time.
</details>

### Q3. [Core] Explain temperature and top-p.

<details><summary>Answer</summary>

Both control sampling from the next-token distribution. Temperature rescales the logits before softmax: below 1 sharpens toward the top token (more deterministic), above 1 flattens (more random). Top-p (nucleus) keeps the smallest set of tokens whose probabilities sum to p and samples from just those, cutting the long tail. Use low temperature for extraction and code, higher for brainstorming. Temperature 0 is greedy and reproducible.
</details>

### Q4. [Core] What is prompt caching and when does it save money?

<details><summary>Answer</summary>

Prompt caching stores the model's computed state for a long, unchanging prefix (a big system prompt, a fixed instruction block, few-shot examples) so repeated calls that share that prefix skip recomputing it. Cached input tokens are billed at a large discount versus normal input tokens. It pays off when many requests share the same large prefix and only the tail differs; it does nothing if every prompt is unique.
</details>

### Q5. [Applied] Your app resends the full chat history every turn and cost grows each message. Why, and what do you do?

<details><summary>Answer</summary>

Because the model is stateless, so each turn re-bills all resent history as input tokens; a long conversation means quadratic-ish token spend. Levers: cap or summarize old turns (a rolling summary), drop turns irrelevant to the current question, and put the stable instructions in a cached prefix so they are billed at the cached rate. Retrieve only what the current turn needs rather than carrying everything.
</details>

---

## Part 2: Prompt engineering (class 4.2)

### Q6. [Warm-up] What is few-shot prompting?

<details><summary>Answer</summary>

Including a handful of input-output examples in the prompt so the model infers the pattern and format you want, without any training. Zero-shot is instructions only; few-shot adds examples. It is the cheapest lever to try before RAG or fine-tuning.
</details>

### Q7. [Core] What is chain-of-thought prompting and when does it help or hurt?

<details><summary>Answer</summary>

Asking the model to reason step by step before answering. It helps on multi-step reasoning, math, and logic by giving the model tokens to "work" in. It hurts when you need a terse answer (it adds latency and cost), and the visible reasoning is a plausible narrative, not a guaranteed faithful trace of the computation. For extraction or classification it is usually unnecessary.
</details>

### Q8. [Core] What belongs in a system prompt versus a user prompt?

<details><summary>Answer</summary>

The system prompt sets stable, cross-turn behavior: role, tone, format rules, guardrails, and what to do when unsure. The user prompt carries the specific request for this turn. Keeping durable instructions in the system prompt makes behavior consistent and makes the prefix cacheable; stuffing per-turn detail into the system prompt defeats both.
</details>

### Q9. [Deep] What is prompt injection and how do you reduce it?

<details><summary>Answer</summary>

Prompt injection is when untrusted content (a retrieved document, a web page, user input) contains instructions the model then follows, overriding your intent. Reductions: keep a clear boundary that only the developer or user chat is a source of instructions and retrieved or tool content is data; never concatenate untrusted text into the instruction region; constrain outputs and validate them; least-privilege on any tools; and do not let model output trigger side effects without a check. It cannot be fully eliminated by prompting alone.
</details>

### Q10. [Applied] The model mostly follows your format but occasionally drifts. Prompt-only fixes?

<details><summary>Answer</summary>

Give an explicit output template and one or two few-shot examples of it; state the format rules in the system prompt; lower temperature; and for machine-readable output ask for JSON and validate (class 4.3). If it still drifts across many calls and the style prompt is long and repeated, that is the signal to consider fine-tuning for the behavior (class 4.9), but only after prompting is exhausted.
</details>

---

## Part 3: Structured outputs and tool calling (class 4.3)

### Q11. [Core] How do you get reliable JSON out of an LLM?

<details><summary>Answer</summary>

Ask for JSON and, where the provider supports it, use a structured-output or JSON mode that constrains decoding to a schema, so the output parses by construction. Define the schema (for example with Pydantic), validate the response against it, and on a validation failure retry with the error fed back. Do not parse free text with regex when a schema mode is available.
</details>

### Q12. [Core] What is tool calling (function calling), mechanically?

<details><summary>Answer</summary>

You describe tools to the model as named functions with typed parameter schemas. The model does not run anything; it emits a structured request naming a tool and its arguments. Your code executes the tool, then feeds the result back into the conversation, and the model uses it to produce the final answer. It is a loop: model proposes, you execute, you return the result.
</details>

### Q13. [Deep] Whose job is it to actually run the tool, and why does that matter for safety?

<details><summary>Answer</summary>

Your application runs it, not the model. That boundary is the safety control: the model only proposes a call, so you validate the arguments, enforce permissions, and decide whether to execute, especially for anything with side effects (writes, payments, deletes). Treating the model's proposed call as an intent to be checked, not a command to be obeyed, is what contains prompt injection and mistakes.
</details>

### Q14. [Applied] A tool-calling agent sometimes invents an argument the user never gave. How do you catch it?

<details><summary>Answer</summary>

Validate arguments against the schema and against what is actually known (do not send a fabricated id to a real API); require missing required fields to trigger a clarifying question instead of a guess; and for risky tools, confirm before executing. Log every call. The schema catches type or shape errors; the "is this grounded in the input" check catches invented but well-typed values.
</details>

---

## Part 4: Embeddings and vector databases (class 4.4)

### Q15. [Warm-up] What is an embedding?

<details><summary>Answer</summary>

A dense vector that represents the meaning of a piece of text, so that texts with similar meaning have nearby vectors. It is produced by an embedding model (for example all-MiniLM-L6-v2). Similar meaning, nearby vectors is the whole basis of semantic search.
</details>

### Q16. [Core] Why cosine similarity rather than raw dot product or Euclidean distance?

<details><summary>Answer</summary>

Cosine measures the angle between vectors, ignoring magnitude, so it compares direction (meaning) rather than length, which for text embeddings often reflects incidental things like document length. With L2-normalized vectors, cosine similarity and dot product are equivalent, which is why many indexes normalize and use inner product.
</details>

### Q17. [Core] What is the difference between a vector index and a vector database?

<details><summary>Answer</summary>

An index (like FAISS) stores only the vectors and returns nearest neighbors; you keep the text and metadata elsewhere and join by hand. A vector database (like Chroma, Qdrant, pgvector) stores vectors, the source text, and metadata together, filters on metadata inside the query, and persists to disk. For an application you usually want the database; for a fast in-memory index the bare index is fine.
</details>

### Q18. [Deep] What is HNSW and what trade-off does it make?

<details><summary>Answer</summary>

HNSW (Hierarchical Navigable Small World) is the common approximate-nearest-neighbor index: a multi-layer graph where search starts coarse at the top and refines downward, giving roughly logarithmic search instead of scanning every vector. The trade-off is approximate: you gain large speedups on big corpora at the cost of occasionally missing a true neighbor, tuned by parameters like the number of links and the search breadth (ef).
</details>

### Q19. [Applied] Retrieval is slow at a million documents with a flat index. What changes?

<details><summary>Answer</summary>

A flat index compares the query to every vector (exact but linear). Switch to an approximate index like HNSW (or IVF) to get sublinear search; add metadata filtering to shrink the candidate set; and consider quantizing the stored vectors to cut memory. Accept the small recall loss and measure it, rather than assuming exact search you cannot afford.
</details>

---

## Part 5: RAG from scratch (class 4.5)

### Q20. [Core] Walk through the RAG loop end to end.

<details><summary>Answer</summary>

Ingest: load documents, chunk them, embed each chunk, store the vectors with text and metadata. Query time: embed the question, retrieve the top-k nearest chunks, augment a prompt with those chunks as context, generate an answer that uses only the context and cites the chunks. Retrieve, augment, generate on top of an offline ingest step.
</details>

### Q21. [Core] Why RAG instead of putting the knowledge in the model?

<details><summary>Answer</summary>

RAG keeps facts external, so they are current (re-index to update), citable (you can point to the source), and controllable (you choose the sources). Baking facts into weights makes them stale, uncitable, and prone to confident hallucination, and updating means retraining. RAG adds knowledge; fine-tuning changes behavior.
</details>

### Q22. [Deep] How does chunking affect retrieval quality?

<details><summary>Answer</summary>

Chunks that are too large dilute the embedding (many topics in one vector) and waste context tokens; too small and a chunk loses the context needed to answer, and the answer spans several chunks. You tune chunk size and overlap so each chunk is a coherent, self-contained unit. Overlap avoids splitting a fact across a boundary. There is no universal size; it depends on the documents and the questions.
</details>

### Q23. [Deep] Name the common ways RAG fails and how you would diagnose each.

<details><summary>Answer</summary>

Retrieval miss (the right chunk was never fetched): low context recall, so fix chunking, the embedding model, or k. Right chunk fetched but answer ignores it or contradicts it: low faithfulness, so fix the prompt (answer only from context) or the model. Retrieved but irrelevant chunks crowd the context: low context precision, so rerank or filter. Stale or duplicate documents: fix the corpus. Diagnose by separating retrieval metrics from generation metrics (class 4.8), because they point to different fixes.
</details>

### Q24. [Applied] The assistant cites a number that is a year out of date. Where is the bug?

<details><summary>Answer</summary>

Almost always the corpus or retrieval, not the model: the store still holds last year's document, or retrieval fetched the old chunk over the new one (no recency filter, or both years embed similarly). Fixes: remove or supersede stale documents, add a metadata filter on year, and make sure ingest re-embeds updated files. If the number is not in any retrieved chunk at all, the model hallucinated it, which is a faithfulness or prompting problem.
</details>

---

## Part 6: RAG with frameworks (class 4.6)

### Q25. [Core] What does a framework like LangChain plus Chroma give you over hand-built RAG?

<details><summary>Answer</summary>

The plumbing: a persisted store that holds vectors, text, and metadata together with native metadata filtering; a retriever that drops into a chain; and composable steps (retrieve, prompt, model, parse) as an LCEL pipeline. You stop hand-writing the parallel text list, the filter loop, and the persistence. The concepts are identical to the from-scratch loop; the framework removes boilerplate.
</details>

### Q26. [Core] What is a cross-encoder reranker and why add it after retrieval?

<details><summary>Answer</summary>

The initial vector search uses a bi-encoder: query and passage are embedded separately, so it is fast but approximate. A cross-encoder reads the (query, passage) pair together and scores relevance far more accurately, but it is too slow to run over the whole corpus. So you retrieve a wide pool cheaply with the bi-encoder, then rerank that small pool with the cross-encoder and keep the top few. Two stages: fast recall, then accurate precision.
</details>

### Q27. [Deep] Explain HyDE, and contrast it with HyPE.

<details><summary>Answer</summary>

HyDE (Hypothetical Document Embeddings) is query-time: ask the model to draft a hypothetical answer, then retrieve real passages nearest to that draft. It helps because a terse question and the passage that answers it often use different words, while the draft reads like the target documents. HyPE is index-time: for each chunk, generate hypothetical questions it could answer and index those, so real questions match stored questions. HyDE costs an extra model call per query; HyPE moves that cost to ingest.
</details>

### Q28. [Applied] Retrieval quality is mediocre and queries are vague one-liners. Two upgrades?

<details><summary>Answer</summary>

Query rewriting: use the model to turn the vague question into a specific search query before retrieving. And a cross-encoder rerank over a wider pool to fix ordering. HyDE is a third option for the vocabulary-mismatch case. Measure each with hit@k on a small labeled set rather than assuming it helped, since each adds cost.
</details>

---

## Part 7: How LLMs are served (class 4.7)

### Q29. [Core] What is the KV cache and what problem does it solve?

<details><summary>Answer</summary>

During generation, each new token attends to all previous tokens' keys and values. Without caching you would recompute them every step (quadratic work). The KV cache stores the keys and values already computed, so each new token only computes its own, making generation roughly linear. The cost is memory: the cache grows with sequence length and batch size, which is what serving systems fight to manage.
</details>

### Q30. [Core] What are MHA, MQA, and GQA?

<details><summary>Answer</summary>

Multi-Head Attention (MHA): every query head has its own key and value heads. Multi-Query Attention (MQA): all query heads share a single key or value head, which shrinks the KV cache a lot at some quality cost. Grouped-Query Attention (GQA): a middle ground where groups of query heads share key or value heads, most of the memory saving with little quality loss. GQA is the common modern default.
</details>

### Q31. [Deep] Explain FlashAttention at a high level. What does it actually speed up?

<details><summary>Answer</summary>

FlashAttention is IO-aware exact attention. Standard attention is bottlenecked not by math but by moving the large attention matrix between the GPU's big-but-slow HBM and its small-but-fast on-chip SRAM. FlashAttention tiles the computation and fuses the softmax so it never materializes the full matrix in HBM, using an online softmax to combine tiles. Same result, far fewer memory reads and writes, so it is faster and uses less memory, especially on long sequences.
</details>

### Q32. [Deep] What is speculative decoding, and what happens when the draft is always accepted?

<details><summary>Answer</summary>

A small fast draft model proposes several tokens ahead; the large model verifies them in one parallel pass and keeps the longest correct prefix. It speeds up generation when the draft is usually right, because one big-model pass yields several tokens instead of one, with identical output distribution. If acceptance rate alpha is 1 (draft always accepted), you get the maximum speedup, k+1 tokens per verification step for a k-token draft.
</details>

### Q33. [Core] What is PagedAttention (vLLM) and what does it fix?

<details><summary>Answer</summary>

PagedAttention manages the KV cache like operating-system virtual memory: it stores the cache in fixed-size blocks (pages) instead of one contiguous chunk per request. That removes the fragmentation and over-reservation that waste memory when sequences grow and shrink unpredictably, so you can batch far more requests. It is the core idea behind vLLM's throughput gains.
</details>

### Q34. [Deep] Distinguish TTFT from throughput. Why measure both?

<details><summary>Answer</summary>

Time to first token (TTFT) is latency until the first output token appears, dominated by prefill (processing the prompt); it is what a user feels as responsiveness. Throughput is tokens per second across the whole generation (and across concurrent requests), what determines cost and capacity. They trade off: large batches raise throughput but can raise TTFT. You measure both because a chat UI cares about TTFT while a batch job cares about throughput.
</details>

### Q35. [Deep] What does quantization cost you, and where is it risky?

<details><summary>Answer</summary>

Quantization stores weights (and sometimes activations) in fewer bits (int8, 4-bit NF4), cutting memory and often speeding inference, with the arithmetic still done in higher precision after dequantizing. The cost is some accuracy loss, which is usually small at 8-bit and 4-bit for inference but grows at very low bit-widths, and it can hit outlier-sensitive layers or long-tail behaviors harder than aggregate benchmarks show. Measure on your task, not just perplexity.
</details>

---

## Part 8: Evaluating LLM and RAG (class 4.8)

### Q36. [Core] Define faithfulness, answer relevance, and context relevance and recall.

<details><summary>Answer</summary>

Faithfulness: is the answer supported by the retrieved context (no claims beyond it)? Answer relevance: does the answer actually address the question? Context relevance: are the retrieved passages on-topic for the question (precision of retrieval)? Context recall: did retrieval fetch the passages needed to answer (a deterministic check against gold ids). The first three are judged; recall is measured against ground truth.
</details>

### Q37. [Core] What is LLM-as-judge and what biases does it have?

<details><summary>Answer</summary>

Using an LLM to score outputs against a rubric (faithful? relevant?). Known biases: verbosity (prefers longer answers), position (favors the first or a particular slot when comparing), self-preference (favors text from the same model family), and leniency (grades generously). It is cheap and scalable but noisy, so it needs mitigation.
</details>

### Q38. [Deep] How do you mitigate LLM-as-judge biases?

<details><summary>Answer</summary>

Do not let a model grade only itself (use a different judge model or a panel); randomize and swap positions in pairwise comparisons and average; give a strict rubric with few-shot anchored examples rather than "rate 1 to 10"; ask for a short justification before the score; control for length; and calibrate the judge against a small set of human labels. Keep deterministic metrics (like context recall) where you can, and treat the judge as one signal, not truth.
</details>

### Q39. [Deep] One overall score or several? How do you combine metrics?

<details><summary>Answer</summary>

Keep the metrics separate to diagnose, then combine deliberately for a decision. Use gates for non-negotiables (for example faithfulness must exceed a threshold, or the answer fails regardless of other scores), and a weighted average for the rest, with weights set by the use case (a medical assistant weights faithfulness heavily; a brainstorming tool weights relevance and helpfulness). A single blended number hides which stage failed, so never report only that.
</details>

### Q40. [Core] What is RAGAS and when would you not lean on it?

<details><summary>Answer</summary>

RAGAS is the standard library for RAG evaluation: faithfulness, answer relevance, context precision and recall, mostly via LLM-as-judge. Use it as a reference and a fast start. Reasons to hand-build instead: dependency conflicts with your framework versions, the need for a metric it does not have, or wanting full transparency into how each score is computed. Name it, understand it, and reach for a small hand-built harness when it gets in the way.
</details>

### Q41. [Applied] Faithfulness is high but users say answers miss the point. Which metric, which fix?

<details><summary>Answer</summary>

High faithfulness means the answer sticks to the context; missing the point is low answer relevance, and often low context relevance upstream (the retrieved passages were on-corpus but not what the question needed). Fix retrieval first (rewrite the query, rerank, tune k and chunking), then the prompt so the answer targets the question. Faithful but irrelevant is a retrieval-and-framing problem, not a hallucination problem.
</details>

---

## Part 9: Fine-tuning I, the decision and the data (class 4.9)

### Q42. [Core] Prompt, RAG, or fine-tune: how do you choose?

<details><summary>Answer</summary>

Cheapest lever that closes the gap, in order. Prompt for instructions, format, and tone. RAG when the gap is missing or changing knowledge (your documents, current facts). Fine-tune only when you need a behavior, style, or narrow skill the model will not follow from a prompt, and prompting has been exhausted. They combine: a fine-tuned model for behavior plus RAG for facts plus a good prompt is a common production shape.
</details>

### Q43. [Core] "Fine-tune it on my documents so it knows them." Why is that wrong?

<details><summary>Answer</summary>

Fine-tuning changes behavior, not reliable factual recall. A model tuned on documents will not dependably remember specific facts, cannot cite them, and cannot be updated without retraining, and it will state stale facts confidently. Knowledge that must be current and citable belongs in RAG; fine-tuning is for how the model responds, not what it knows.
</details>

### Q44. [Deep] Distinguish SFT, full fine-tuning, and PEFT. Are SFT and full fine-tuning the same thing?

<details><summary>Answer</summary>

No, they answer different questions. SFT versus preference optimization is the objective axis: learn from labeled input-output pairs, or from comparisons of better versus worse answers. Full fine-tuning versus PEFT is the parameter axis: update every weight, or update a small added subset with the base frozen. They are independent: our build is SFT plus QLoRA (supervised pairs, parameters updated the PEFT way). Conflating SFT with full fine-tuning is a common mistake.
</details>

### Q45. [Deep] Explain LoRA and decode W' = W + BA.

<details><summary>Answer</summary>

LoRA freezes the pretrained weight W and learns a small low-rank correction. W is the original d-by-d matrix (frozen). B is d-by-r and A is r-by-d, with rank r tiny (8 or 16). Their product BA is d-by-d, a compact approximation of the change the task needs. W' = W + BA is the effective weight at inference. You train only B and A, roughly 2dr numbers instead of d-squared, so a tiny fraction of the parameters. It works because the task-specific update is intrinsically low-rank, and freezing W preserves the base ability (avoiding catastrophic forgetting).
</details>

### Q46. [Core] Which weights get a LoRA adapter, and how does that affect the trainable count?

<details><summary>Answer</summary>

You choose the target modules; the usual choice is the four attention projections (q, k, v, o), a strong cheap baseline. Adding the MLP or feed-forward projections reaches more of the network at more cost and overfit risk; embeddings, the output head, and layer-norms are normally left alone. Two dials set the size: the rank r and how many modules you target. Even a 7 to 8B base at r=16 on q,k,v,o trains only about 10 to 14M parameters, well under 1 percent.
</details>

### Q47. [Numerical] For a hidden size d = 4096 and rank r = 8, roughly how many parameters does one attention projection's LoRA adapter add?

<details><summary>Answer</summary>

An adapter on a d-by-d matrix adds B (d-by-r) plus A (r-by-d) = 2dr = 2 x 4096 x 8 = 65,536 parameters, versus d-squared = about 16.8 million for a full update of that matrix, a few hundred times fewer. Across the four attention projections and all layers it still stays a small fraction of the base.
</details>

### Q48. [Applied] Is a few-hundred-example dataset enough to fine-tune?

<details><summary>Answer</summary>

For a behavior or style tune, a few hundred to about a thousand clean, consistent examples is a realistic range; quality and consistency matter more than volume, since one sloppy target teaches the wrong habit. It is not enough to teach broad new knowledge (that is RAG). Keep training, validation, and the eval set disjoint, and watch for overfitting on a small set.
</details>

---

## Part 10: Fine-tuning II, QLoRA and alignment (class 4.10)

### Q49. [Core] What is QLoRA, and how do 4-bit and 16-bit work together?

<details><summary>Answer</summary>

QLoRA is LoRA on a 4-bit base. The frozen base is stored in 4-bit (NF4) so it barely uses memory; the LoRA adapter stays in 16-bit. On the forward pass each 4-bit weight is dequantized to 16-bit just for that matmul and then discarded, so activations are full precision; on backprop only the adapter receives gradients and the base stays 4-bit. So 4-bit is storage, 16-bit is the arithmetic and the part that learns. This is what lets a large base fine-tune on a free Colab T4 (16 GB).
</details>

### Q50. [Core] After training, what precision is the served model, merged or not?

<details><summary>Answer</summary>

The adapter is always 16-bit. If you keep it separate, the base stays 4-bit and does the same dequantize-then-matmul at inference with the adapter added on top. If you merge (W' = W + BA), the base must be dequantized to add BA, so the merged model is a full 16-bit model, not 4-bit. To ship something small, re-quantize the merged model afterward (for example to 4-bit GGUF for Ollama).
</details>

### Q51. [Core] What is catastrophic forgetting, and how does LoRA help?

<details><summary>Answer</summary>

Catastrophic forgetting is when fine-tuning on a narrow dataset overwrites unrelated skills the model learned in pretraining, so it gets worse at everything else. Full fine-tuning risks it because every weight moves. LoRA largely sidesteps it by freezing the base and training only a small add-on, so general ability is preserved by construction. You still watch for it: a small or repetitive set can make even a LoRA tune overfit and drift.
</details>

### Q52. [Core] Explain DPO and where the preference pairs come from.

<details><summary>Answer</summary>

DPO (Direct Preference Optimization) trains directly on (prompt, chosen, rejected) triples: it raises the probability of the chosen answer and lowers the rejected one, measured against a frozen reference copy, with no reward model and no RL loop, which is why it is the modern default. The pairs are collected offline from human raters or from an AI judge (RLAIF, as in Constitutional AI). The chat app's "which answer do you prefer?" picker is exactly this collection, pooled and used in a later training run, not applied live.
</details>

### Q53. [Deep] Contrast PPO-RLHF, DPO, and RLVR/GRPO.

<details><summary>Answer</summary>

PPO-RLHF: train a reward model from human comparisons, then use RL (PPO) to push the model toward high reward; powerful but complex (reward model plus an RL loop). DPO: skip the reward model and RL, optimize the preference objective directly; simpler, the common default. RLVR (reinforcement learning from verifiable rewards): the reward is a deterministic verifier (math answer key, code tests), not a learned model, used for reasoning. GRPO (Group Relative Policy Optimization) is the algorithm often paired with RLVR: a PPO variant with no value network that samples a group of answers per prompt and pushes toward those above the group average. RLVR is the reward source; GRPO is the optimizer; they are different axes.
</details>

### Q54. [Deep] "Correctness can be checked automatically." How, and why does it matter for RLVR?

<details><summary>Answer</summary>

A program scores the output and that score is the reward: compare a math answer to the known solution, run unit tests on generated code, validate format with a regex or JSON schema, or use a rules engine for a puzzle. It matters because it removes the human labeler and the learned reward model from the loop, giving a cheap, exact, unhackable-in-the-obvious-way signal, which is what drives the current wave of reasoning models. It only applies where ground truth exists.
</details>

### Q55. [Applied] Your tuned model nails the house style but is now worse at general questions. What happened and what do you do?

<details><summary>Answer</summary>

Overfitting toward the narrow tuning set, a mild catastrophic-forgetting effect. Diagnose by evaluating general ability on a held-out set, not just the target behavior. Fixes: fewer epochs, a lower learning rate, a smaller rank, more diverse training data, or mixing in some general examples; and confirm the tune was even necessary versus a prompt. Always check that general answers did not degrade, not just that the new behavior improved.
</details>

---

## Rapid-fire (mixed tiers)

<details><summary>Open</summary>

- **Why is the model stateless between API calls?** It only sees the tokens you send; "memory" is you resending history.
- **One reason RAG beats fine-tuning for facts.** Facts stay current and citable; update by re-indexing.
- **Bi-encoder vs cross-encoder in one line.** Bi-encoder embeds separately (fast, recall); cross-encoder reads the pair together (accurate, rerank).
- **What does temperature 0 give you?** Greedy, deterministic decoding.
- **Cosine vs dot product on normalized vectors?** Equivalent.
- **What does the KV cache trade?** Memory for speed (avoids recomputing past keys and values).
- **NF4 in one line.** A 4-bit float format tuned to the bell-curve spread of weights.
- **alpha in LoRA?** A scaling factor; the update is applied as (alpha/r) times BA.
- **What is hit@k?** 1 if the gold passage is in the top-k retrieved, else 0.
- **RLAIF expands to?** Reinforcement learning from AI feedback.
- **GRPO expands to?** Group Relative Policy Optimization.
- **Why divide serving into TTFT and throughput?** One is user-felt latency, the other is cost and capacity.
- **When is prompt caching useless?** When every prompt has a unique prefix.
- **Faithful but irrelevant answer points to which stage?** Retrieval and framing, not hallucination.

</details>

## Whiteboard drills (do these on paper)

<details><summary>Open</summary>

1. Draw the RAG loop: ingest (load, chunk, embed, store) and query time (embed, retrieve, augment, generate, cite).
2. Write `W' = W + BA` and label the shape of each matrix, then compute the trainable-parameter count for one d-by-d projection at rank r.
3. Draw the two-stage retrieve-then-rerank pipeline and mark which stage is the bi-encoder and which is the cross-encoder.
4. Sketch the alignment pipeline: pretraining, SFT, preference optimization (DPO / PPO-RLHF / RLVR-GRPO).
5. Draw the QLoRA forward pass: 4-bit base, dequantize-to-16-bit for the matmul, 16-bit LoRA add-on, gradients only to the adapter.
6. Lay out an evaluation scorecard: faithfulness, answer relevance, context relevance, context recall, with which are judged and which are deterministic.

</details>

## Senior / stretch questions (beyond core coverage)

### S1. [Senior] How would you design an evaluation harness for a RAG system you are about to ship?

<details><summary>Answer</summary>

Build a labeled eval set with questions, gold answers, and gold passage ids, including unanswerable questions to test refusal. Separate retrieval metrics (context recall and precision, deterministic where possible) from generation metrics (faithfulness, answer relevance via a well-controlled LLM-as-judge). Gate on faithfulness and refusal correctness, weight the rest by use case. Track per-stage so a regression tells you whether retrieval or generation broke. Keep the set versioned and disjoint from any tuning data, and calibrate the judge against a small human-labeled sample.
</details>

### S2. [Senior] A production RAG assistant is too slow and too expensive. Levers?

<details><summary>Answer</summary>

Retrieval: smaller or quantized embeddings, an ANN index (HNSW), fewer and better chunks, rerank only a small pool. Generation: cache the fixed prompt prefix (prompt caching), shrink the context to what is needed, use a smaller or quantized model where quality allows, stream to cut felt latency. Serving: batch on a system like vLLM (PagedAttention) for throughput, use GQA models, consider speculative decoding. Measure TTFT, throughput, and cost per query, and attack the dominant term.
</details>

### S3. [Senior] When is fine-tuning genuinely the right call over prompt-plus-RAG, at scale?

<details><summary>Answer</summary>

When a behavior must hold consistently and the prompt that enforces it is long and paid on every call (so baking it into weights removes recurring token cost, class 4.7), when latency budget cannot afford few-shot exemplars in-context, when you need a narrow skill or format the base will not follow reliably, or when you must run a small open-weight model that cannot follow the instruction zero-shot. Facts still go in RAG; you fine-tune the style and skill, and you justify the ongoing training and serving cost.
</details>

### S4. [Senior] Why is fine-tuning mostly an open-weight activity, and what does it cost on closed models?

<details><summary>Answer</summary>

Fine-tuning needs access to the weights, so LoRA and QLoRA live on open-weight models (Llama, Qwen, Mistral, Gemma) that you download, train, and host yourself. Closed frontier models do not release weights, so you cannot LoRA them; a subset offer a managed fine-tune (your data on their platform, SFT only), and it is constrained and pricey: tuned endpoints are billed at a premium per token, and some require dedicated or provisioned capacity billed by the hour regardless of use. Availability also shifts. Default to open-weight; for closed models prefer prompt plus RAG.
</details>

### S5. [Senior] Chat apps collect preference clicks. What are the failure modes of that signal?

<details><summary>Answer</summary>

Human preference is noisy and biased: people click the longer, more confident, better-formatted answer, not the more correct one, the same verbosity and style biases that afflict LLM-as-judge. Selection effects (who bothers to click), position effects, and prompt-injection or adversarial voting also creep in. Providers filter and de-bias heavily, aggregate over many votes, and never apply a single click live. Preference optimizes for what people prefer, which is not identical to what is true or safe, so it is paired with other safeguards.
</details>
