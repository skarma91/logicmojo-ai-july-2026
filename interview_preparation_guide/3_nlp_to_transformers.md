# Interview Preparation Guide, Module 3: NLP to Transformers

Questions and answers for the NLP and transformer foundations taught in Module 3.

**Coverage map.** The questions follow the five classes of the module: class 3.1 (Text as data), class 3.2 (Attention, the core idea), class 3.3 (The Transformer), class 3.4 (Using pretrained transformers), and class 3.5 (From models to LLMs). Attention and the transformer block are the most heavily interviewed topics in this module, so spend the most time there.

## How to use this guide

Read the question, answer it out loud or on paper first, then expand the answer to check yourself. Each question is tagged by difficulty:

- **[Warm-up]** a definition or one-liner an interviewer opens with.
- **[Core]** the standard question you are expected to answer cleanly.
- **[Deep]** mechanism, trade-off, or a "why" that separates strong candidates.
- **[Applied]** a scenario or short design or debugging prompt.

The attention formula and the transformer block are worth being able to draw and decode from memory. Work the numerical questions by hand before expanding.

---

## Part 1: Text as data (class 3.1)

### Q1. [Warm-up] What is tokenization, and what is a token?

<details><summary>Answer</summary>

Tokenization is the step that turns raw text into the discrete pieces a model actually reads, called tokens. A token is a piece of a word (a subword), not necessarily a whole word: frequent words are often a single token, rare words split into several pieces.
</details>

### Q2. [Core] Explain byte-pair encoding (BPE) intuitively.

<details><summary>Answer</summary>

BPE starts from individual characters and repeatedly merges the most frequent adjacent pair into a new token, building up a vocabulary. Frequent words end up as one token; rare words stay split into smaller pieces.

**Analogy:** Lego pieces. Common shapes are single bricks, rare shapes are built from smaller bricks. For example "unbelievable" might split into "un", "believ", "able", while "the" stays whole.
</details>

### Q3. [Core] "One word equals one token." Why is that wrong, and why does it matter?

<details><summary>Answer</summary>

The number of tokens per word varies: short common words are one token, long or rare words are several. It matters because cost, latency, and the context window are all counted in tokens, not words, so token count, not word count, is the unit of budgeting for any LLM (a point that returns in class 3.5 and Module 4).
</details>

### Q4. [Warm-up] Define corpus and vocabulary.

<details><summary>Answer</summary>

A corpus is the body of text a model or tokenizer learns from. A vocabulary is the fixed set of tokens the tokenizer knows, each with an integer id, built from the corpus. The vocabulary size drives everything downstream: a one-hot vector is as long as the vocabulary, and the embedding table has one row per vocabulary token.
</details>

### Q5. [Deep] Why is one-hot encoding a poor representation of tokens? What replaces it?

<details><summary>Answer</summary>

A one-hot vector is as long as the vocabulary (tens of thousands of dimensions), almost all zeros, and it carries no notion of similarity: every token is equidistant from every other, so "cat" is no closer to "dog" than to "invoice." A dense embedding replaces it: a short learned vector per token where similar tokens land nearby. Word order is a separate issue that neither representation fixes on its own.
</details>

### Q6. [Core] What is an embedding, mechanically?

<details><summary>Answer</summary>

An embedding is a lookup table with one row per vocabulary token and a fixed number of columns (the embedding dimension). A token's id selects its row. Those rows are parameters, learned like any other weight (nn.Embedding in PyTorch), either by a dedicated algorithm or jointly as the model's first layer.
</details>

### Q7. [Deep] How does word2vec learn its vectors, and where does the embedding table actually come from?

<details><summary>Answer</summary>

Word2vec trains a shallow network on a fake task: CBOW predicts a word from its surrounding context (fill in the blank), and skip-gram predicts the context from the word. Because the input is one-hot, the hidden vector is just one row of the input weight matrix, and that input weight matrix is the embedding table. After training you keep that matrix and discard the output layer. The guiding idea is "a word is known by the company it keeps."

**Common wrong answer:** "word2vec is what modern LLMs use." It is the ancestor; LLMs learn far richer, context-aware vectors.
</details>

### Q8. [Deep] Distinguish an embedding from a sequence model like an RNN or a transformer.

<details><summary>Answer</summary>

An embedding is a static table that maps token ids to vectors. A sequence model (RNN, LSTM, transformer) consumes those embedding vectors and models relationships across a sequence. The embedding is the input representation; the sequence model is what runs on top of it. They are not the same kind of object.
</details>

### Q9. [Core] Why did RNNs and LSTMs fall short, motivating attention?

<details><summary>Answer</summary>

An RNN carries a hidden state forward step by step; an LSTM adds gates to hold context longer. Two limits remain. First, the memory of early words fades over distance, so long-range dependencies are lost. Second, the read is inherently sequential (each step needs the previous hidden state), so it cannot be parallelized across the sequence, which caps training speed. Attention answers both by letting every token look directly at every other token at once.
</details>

---

## Part 2: Attention, the core idea (class 3.2)

### Q10. [Core] What problem does attention solve? Give the classic example.

<details><summary>Answer</summary>

To understand a word you need the right other words. In "the bank of the river," "bank" means riverbank only because "river" is nearby. Attention lets each word pull in information from the words that matter to it, however far away, rather than relying on a hidden state that decays with distance.
</details>

### Q11. [Core] Explain query, key, and value.

<details><summary>Answer</summary>

Each token produces three vectors. The query is what this token is looking for, the key is what a token offers, the value is the information a token carries. A token's query is compared against every token's key to decide how much attention to pay, then it collects a weighted blend of the matching tokens' values.

**Analogy:** a soft dictionary lookup. The query "love" is compared to every key (genre labels) and returns a weighted blend of the values (movies), more from the keys that match best.
</details>

### Q12. [Deep] Where do Q, K, and V come from?

<details><summary>Answer</summary>

They are not given. Each token's embedding x is multiplied by three learned matrices: q = x W_Q, k = x W_K, v = x W_V. Same embedding, three different learned views: what to look for, what to advertise, what to share. These projection matrices are trained along with the rest of the network.
</details>

### Q13. [Core] Write the scaled dot-product attention formula and decode every part.

<details><summary>Answer</summary>

Attention(Q, K, V) = softmax( (Q K^T) / sqrt(d_k) ) V

- Q, K, V are the stacks of query, key, value vectors for all tokens.
- Q K^T is the dot product of each query with every key, giving a relevance score of every token against every other token.
- d_k is the length of a key vector; dividing by sqrt(d_k) keeps the scores from growing too large when vectors are long. It is a stabilizer.
- softmax turns those raw scores into weights that are positive and sum to 1: what fraction of attention goes to each token.
- multiplying the weights by V and summing gives each token a context vector, a weighted blend of every token's value.

Read aloud: "score every token against every token, turn scores into percentages, then blend everyone's information by those percentages."
</details>

### Q14. [Deep] Why divide by sqrt(d_k)? What breaks without it?

<details><summary>Answer</summary>

The dot product of two vectors of length d_k grows in magnitude with d_k. Large scores push softmax into a very peaky region where one weight is near 1 and the rest near 0, and there the softmax gradient is tiny, so learning stalls. Dividing by sqrt(d_k) keeps the score variance roughly constant regardless of dimension, so softmax stays in a well-behaved range and gradients flow.
</details>

### Q15. [Deep] "Attention is just a dictionary lookup." Why is that not quite right?

<details><summary>Answer</summary>

A dictionary lookup returns one exact match. Attention is soft and weighted: every token contributes a fraction of its value, set by the softmax weights, so the output is a blend of all tokens, not a single retrieved entry. It is a weighted average over the whole sequence, dominated by the most relevant tokens but never exclusive.
</details>

### Q16. [Core] What is self-attention, and why does it enable parallel training?

<details><summary>Answer</summary>

Self-attention is attention where Q, K, and V all come from the same sequence, so tokens attend to each other within one sentence. Unlike an RNN, there is no step-by-step dependency: every token's context vector is computed at the same time from the same matrices, so the whole sequence is processed in parallel, which is the main reason transformers train fast on modern hardware.
</details>

### Q17. [Deep] What is causal masking, why is it needed, and how is it implemented?

<details><summary>Answer</summary>

A left-to-right generator must not let a token attend to tokens that come after it, or it would cheat by peeking at the answer during next-token training. Causal masking blocks the future positions by setting their scores to negative infinity before the softmax, so their weights become zero and the attention matrix is lower-triangular. Encoders like BERT skip the mask because they are allowed to see the whole sentence.
</details>

### Q18. [Numerical] A query has raw scores [2, 1, 0] against three keys (already scaled). What are the attention weights, and which value dominates the blend?

<details><summary>Answer</summary>

Softmax of [2, 1, 0]: e^2 = 7.39, e^1 = 2.72, e^0 = 1.00, sum = 11.11, weights about [0.665, 0.245, 0.090]. The first token's value dominates the context vector at about 67 percent, but the output still mixes in about 25 percent and 9 percent from the other two.
</details>

---

## Part 3: The Transformer (class 3.3)

### Q19. [Core] What is multi-head attention and why use several heads?

<details><summary>Answer</summary>

Multi-head attention runs attention several times in parallel, each head with its own learned Q, K, V projections, then concatenates and mixes the results. Different heads can specialize: one on syntax, one on who-refers-to-what, one on nearby context. It lets the model attend to several kinds of relationship at once.

**Analogy:** several readers each highlighting a different aspect of the same sentence, then pooling their notes.
</details>

### Q20. [Numerical] A model dimension is 512 with 8 heads. What is each head's dimension, and do heads add width?

<details><summary>Answer</summary>

Each head works in 512 / 8 = 64 dimensions. Heads do not add width: the 512-dim vector is split across the 8 heads of size 64, each attends in its own 64-dim slice, and concatenating the 8 outputs recovers 512, after which one linear layer mixes them. Total compute is comparable to single-head attention at the same model dimension.
</details>

### Q21. [Deep] Why does a transformer need positional encoding?

<details><summary>Answer</summary>

Self-attention is order-blind: it treats the input as a set, so "dog bites man" and "man bites dog" would look identical to it. Positional encoding injects information about each token's position so the model knows word order. Without it, the model has no inherent sense of sequence.

**Common wrong answer:** "the model inherently knows word order." It does not; order is added in.
</details>

### Q22. [Core] Contrast sinusoidal positional encoding with RoPE.

<details><summary>Answer</summary>

Sinusoidal encoding adds fixed sine and cosine signals of different frequencies to the token embeddings, giving each position a unique pattern. RoPE (rotary position embedding) instead rotates the query and key vectors by an angle that grows with position, so the attention score between two tokens depends on their relative distance. RoPE is the modern default and extends more gracefully to longer contexts than the original fixed encodings.
</details>

### Q23. [Core] What is the feed-forward network in a transformer block, and what does it do that attention does not?

<details><summary>Answer</summary>

After attention mixes information across tokens, the feed-forward network is a small per-token neural network (a couple of linear layers with a nonlinearity) applied to each token's vector independently. Attention mixes across tokens; the feed-forward layer transforms each token on its own. The two alternate, and the feed-forward layers hold a large share of the model's parameters.
</details>

### Q24. [Deep] What do residual connections and LayerNorm do, and why are they essential for depth?

<details><summary>Answer</summary>

A residual connection adds a sublayer's input back to its output (output = x + layer(x)), so information and gradients can flow straight through a deep stack instead of vanishing. LayerNorm rescales each token's vector to a stable range so activations do not drift as they pass through many layers. Together they are the plumbing that makes it possible to stack many blocks without training falling apart.
</details>

### Q25. [Deep] Why LayerNorm rather than BatchNorm in transformers?

<details><summary>Answer</summary>

LayerNorm normalizes across the features of a single token, independently of other tokens or the batch. BatchNorm normalizes across the batch dimension, so it depends on batch statistics. LayerNorm's batch independence matters for sequences of varying length and especially for generating one token at a time, where there is effectively no batch of comparable positions to normalize over.
</details>

### Q26. [Core] Draw the structure of one transformer (decoder) block.

<details><summary>Answer</summary>

Input -> masked multi-head self-attention -> add and norm (residual + LayerNorm) -> feed-forward network -> add and norm -> output. This block is repeated N times to form the stack. A decoder block uses the causal mask; an encoder block is the same without the mask (and an encoder-decoder adds a cross-attention sublayer).
</details>

### Q27. [Core] Compare encoder-only, decoder-only, and encoder-decoder architectures.

<details><summary>Answer</summary>

Encoder-only (BERT) sees the whole sentence at once, good for understanding tasks like classification and named-entity recognition. Decoder-only (GPT) uses the causal mask and predicts the next token, which is what most LLMs are. Encoder-decoder (T5) encodes an input then decodes an output, natural for translation and summarization. The course uses decoder-only stacks. Any of these can be pushed at many tasks; the families describe where each design shines.
</details>

### Q28. [Core] What is the output head of a decoder-only transformer?

<details><summary>Answer</summary>

After the last block, a single linear layer maps each token's vector to a score for every word in the vocabulary (the logits), and softmax turns those into a probability distribution over the next token. This is where the stack connects to decoding in class 3.5.
</details>

---

## Part 4: Using pretrained transformers (class 3.4)

### Q29. [Warm-up] What is the Hugging Face hub?

<details><summary>Answer</summary>

A public library of pretrained models and datasets. Model families differ by size and purpose; you pick one that fits your task and hardware and load it with a few lines rather than training from scratch. It is also where the open models used with Ollama and Groq originate.
</details>

### Q30. [Core] Why must the tokenizer and model be loaded as a matched pair?

<details><summary>Answer</summary>

A model was trained with one specific tokenizer, so its token ids and embedding rows only mean anything under that tokenizer's vocabulary. Pairing a model with a different tokenizer feeds it ids that map to the wrong embeddings, producing meaningless output. Always load both with from_pretrained on the same model name.
</details>

### Q31. [Deep] What do the Auto classes give you, and how does that relate to the output head?

<details><summary>Answer</summary>

AutoModel loads the bare transformer stack. The task-specific classes load the stack plus a head that fits the task: AutoModelForSequenceClassification (classification head), AutoModelForCausalLM (next-token head), AutoModelForTokenClassification (per-token head). The head is the class 3.3 output layer, sized for the task. Reading a classification output by hand means taking the logits, applying softmax, taking argmax, and mapping the index through id2label, which is exactly what a pipeline wraps.
</details>

### Q32. [Core] What is a pipeline, and when would you not use one?

<details><summary>Answer</summary>

A pipeline wraps tokenize, run, and decode into one call for common tasks (sentiment, NER, feature extraction, generation, summarization, question answering). It is the fast path to a result. You drop below it when you need control the pipeline hides: custom batching, access to raw logits or hidden states, or non-standard pre or post-processing.
</details>

### Q33. [Applied] You are running inference on many inputs and it is slow. Two levers from this class?

<details><summary>Answer</summary>

Batch several inputs together so the model processes them in one forward pass (callback to batching in class 2.5, Training in practice), and run in inference mode: model.eval() plus no gradient tracking (torch.no_grad()), which avoids building the autograd graph and saves time and memory. Padding to a common length is what makes a batch a single tensor.
</details>

---

## Part 5: From models to LLMs (class 3.5)

### Q34. [Core] What is the pretraining objective of an LLM, and how does it end up "knowing things"?

<details><summary>Answer</summary>

The objective is next-token prediction: given the previous tokens, predict the next one. Trained over a huge amount of text, the model absorbs grammar, facts, and patterns as a side effect of getting good at that one task. That is the honest answer to "how does it know things": there is no explicit knowledge store, just a model that became very good at predicting the next token.

**Common wrong answer:** "it is trained to answer questions." A base model is trained to continue text; faced with a question it may just write more questions.
</details>

### Q35. [Core] Describe the three training stages that produce a chat model.

<details><summary>Answer</summary>

Pretraining: next-token prediction on trillions of tokens, producing a base model. Instruction tuning (supervised fine-tuning): training on instruction-response pairs, producing a model that follows instructions. RLHF or preference tuning: humans rank responses and the model is tuned toward the preferred ones, producing an aligned chat model with better tone, helpfulness, and safety. Each stage builds on the last.
</details>

### Q36. [Deep] What is emergent behavior with scale?

<details><summary>Answer</summary>

As data, parameters, and compute grow, performance does not only improve smoothly; past certain thresholds, qualitatively new abilities appear, such as following instructions, few-shot learning, and basic reasoning. Below the threshold the ability is near chance; above it, it appears. The exact framing is debated, but the practical point is that scale unlocks capabilities that small models do not show.
</details>

### Q37. [Core] What is the context window, and what must fit inside it?

<details><summary>Answer</summary>

The context window is the maximum number of tokens the model can consider at once, its working memory. Everything must fit: the system prompt, the user prompt, any retrieved documents, and the conversation so far. It is measured in tokens, not words (callback to class 3.1, Text as data).
</details>

### Q38. [Deep] "The model remembers our past chats by default." Why is that wrong?

<details><summary>Answer</summary>

Each call is stateless: the model only sees what is in the current context window. Any apparent memory across turns is because the application resends prior messages as part of the new prompt. Persistent memory beyond the window is something you engineer (a theme in Module 5, AI Agents), not a built-in property of the model.
</details>

### Q39. [Core] How does decoding work, and what do temperature and top-p control?

<details><summary>Answer</summary>

At each step the model outputs a probability for every possible next token; decoding is how you pick one, then you append it and repeat (the autoregressive loop). Temperature divides the logits by a value T before softmax: T below 1 sharpens the distribution (more focused, more deterministic), T above 1 flattens it (more random), and near 0 it always takes the top token. Top-p (nucleus) sampling restricts the choice to the smallest set of tokens whose probabilities add up to p, ignoring the unlikely long tail.
</details>

### Q40. [Deep] Is temperature 0 always best?

<details><summary>Answer</summary>

No. Temperature 0 (greedy) is best for exact, deterministic tasks such as extraction, classification, or code where you want the single most likely token and reproducibility. It is a poor choice for creative or open-ended generation, where it produces flat, repetitive text. Temperature is a dial to match the task, not a quality setting.
</details>

### Q41. [Numerical] Logits (2, 1, 0). How does the top token's probability move as temperature falls from 2 to 1 to 0.5?

<details><summary>Answer</summary>

Dividing the logits by T before softmax sharpens the distribution as T falls. The top token's probability rises from about 0.51 at T = 2, to about 0.67 at T = 1, to about 0.87 at T = 0.5. Lower temperature concentrates probability on the leading token; higher temperature spreads it out.
</details>

### Q42. [Numerical] Next-token probabilities are mat 0.50, rug 0.25, sofa 0.15, floor 0.07, cliff 0.03. Which tokens are in the nucleus at top-p = 0.9?

<details><summary>Answer</summary>

Accumulate from the top: mat 0.50, +rug = 0.75, +sofa = 0.90. The first three (mat, rug, sofa) reach 0.90, so the nucleus is those three; floor and cliff are cut. Sampling then happens only among the nucleus, renormalized.
</details>

---

## Rapid-fire (mixed tiers)

<details><summary>Reveal all rapid-fire answers</summary>

1. **Subword tokenization in one line?** Splitting text into word pieces so frequent words are one token and rare words are several.
2. **Why do LLMs count cost in tokens?** Because tokens, not words, are the unit the model processes and providers bill.
3. **Self-attention versus cross-attention?** Self-attention: Q, K, V from the same sequence. Cross-attention: queries from one sequence, keys and values from another (as in encoder-decoder).
4. **Softmax in attention does what?** Turns raw relevance scores into weights that are positive and sum to 1.
5. **Causal mask shape?** Lower-triangular attention weights; future positions are zeroed.
6. **Why is a decoder-only model called autoregressive?** It generates one token at a time, feeding each output back in as input.
7. **Logits versus probabilities?** Logits are raw scores; softmax turns them into probabilities.
8. **What does the feed-forward layer operate on?** Each token's post-attention vector, independently.
9. **BERT versus GPT in one line?** BERT is encoder-only for understanding; GPT is decoder-only for generation.
10. **Base model versus instruct model?** Base continues text; instruct follows instructions after supervised fine-tuning.
11. **Top-k versus top-p?** Top-k keeps a fixed number of tokens; top-p keeps a variable number that reaches cumulative probability p.
12. **Why does attention enable parallelism but an RNN does not?** Attention computes all positions at once; an RNN's step depends on the previous hidden state.
</details>

---

## Whiteboard drills (do these on paper)

<details><summary>Reveal drills and solutions</summary>

1. **Draw the attention formula and label the four steps.** Score (Q K^T), scale (/ sqrt(d_k)), softmax (weights), blend (times V). Be able to say what each produces.
2. **Head split.** Model dim 768, 12 heads. Head dim = 64. Concatenating 12 heads of 64 recovers 768.
3. **Attention weights.** Softmax of scaled scores (3, 3, 0): e^3 = 20.09 twice and e^0 = 1, sum 41.17, weights about (0.488, 0.488, 0.024). Two tokens share attention almost equally.
4. **Causal mask.** Write the 4x4 attention weight pattern for a decoder: row i can attend to columns 0..i only, so the matrix is lower-triangular.
5. **Temperature intuition.** Explain in one sentence why dividing logits by a large T flattens the distribution: it shrinks the gaps between logits, so after softmax the probabilities are closer to uniform.
6. **Top-p selection.** Given cumulative probabilities 0.4, 0.7, 0.85, 0.95, 1.0 and p = 0.8, the nucleus is the first three tokens (0.85 is the first cumulative value at or above 0.8).
7. **Param intuition.** For a feed-forward block of dim d with hidden 4d, count weights: d*4d + 4d*d = 8d^2 (ignoring biases). This is why feed-forward layers hold much of a transformer's parameters.
</details>

---

## Senior / stretch questions (beyond course coverage)

These go past what Module 3 teaches directly, into the territory a senior or staff candidate is expected to reason about: inference cost, scaling, long context, alignment, and modern architecture choices. Each has a working answer and a "Go deeper" pointer into the module's suggested reading list.

### S1. [Senior] What is the computational cost of self-attention, and what is the KV cache?

<details><summary>Answer</summary>

Self-attention compares every token with every other token, so time and memory scale as the square of the sequence length (order n squared times the model dimension). That quadratic term is why long context is expensive. At inference, a decoder generates one token at a time, and recomputing attention over all previous tokens each step would be wasteful, so the keys and values of past tokens are stored and reused: the KV cache. It turns per-step work from quadratic to linear in the sequence length but costs memory that grows with context length and batch size, which is often the real limit on how long a context you can serve.

**Go deeper:** Attention? Attention! (Lilian Weng) for the mechanics; The Annotated Transformer (Harvard NLP) for the implementation.
</details>

### S2. [Senior] Efficient-attention variants exist. Name the problem they attack and a couple of approaches.

<details><summary>Answer</summary>

They attack the quadratic cost of full attention. FlashAttention keeps the exact computation but reorders it to be memory-aware, avoiding materializing the full attention matrix in slow memory, which speeds up training and inference without changing the result. Sparse attention restricts each token to a subset of positions (local windows plus a few global tokens), and linear or kernel-based attention approximates the softmax so cost grows linearly. The trade-off is that approximate methods can lose some quality or long-range fidelity, while FlashAttention is exact and now standard.

**Go deeper:** Attention? Attention! (Lilian Weng), the survey of attention variants.
</details>

### S3. [Senior] Why have decoder-only models become the default for general-purpose LLMs?

<details><summary>Answer</summary>

A decoder-only stack trained on next-token prediction is simple, scales cleanly, and a single model handles both understanding and generation by framing every task as text continuation. It uses the full parameter budget for one objective rather than splitting into an encoder and a decoder, benefits directly from in-context and few-shot learning, and reuses the KV cache efficiently at inference. Encoder-decoder models remain strong for fixed input-to-output mappings such as translation and summarization, and encoder-only models are excellent for pure understanding tasks, but decoder-only won the general-assistant race on simplicity and scaling.

**Go deeper:** The Illustrated Transformer (Jay Alammar); Hugging Face LLM Course for the architecture families in practice.
</details>

### S4. [Senior] What do scaling laws and the compute-optimal (Chinchilla) result tell us?

<details><summary>Answer</summary>

Scaling laws show that model loss falls predictably as a power law in parameters, data, and compute, so performance is forecastable before training. The compute-optimal finding (Chinchilla) is that, for a fixed compute budget, many early large models were undertrained: parameters and training tokens should scale together, roughly in proportion, rather than growing the model while starving it of data. The practical consequences are smaller models trained on more tokens, and a sharper focus on dataset size and quality, not just parameter count.

**Go deeper:** Hugging Face LLM Course, the sections on how large models are trained.
</details>

### S5. [Senior] Compare tokenization algorithms and their failure modes.

<details><summary>Answer</summary>

BPE merges the most frequent character or token pairs bottom-up; WordPiece merges by what most improves the training likelihood and marks continuations (the ## convention); Unigram (used by SentencePiece) starts from a large vocabulary and prunes tokens to maximize likelihood, and SentencePiece can operate directly on raw bytes so it is language-agnostic. Failure modes matter in interviews: numbers and dates fragment inconsistently, which hurts arithmetic; code and whitespace tokenize awkwardly unless the tokenizer was trained for it; and languages underrepresented in the training corpus split into many tokens, raising cost and lowering quality. Byte-level fallbacks avoid unknown-token failures at the price of longer sequences.

**Go deeper:** Hugging Face LLM Course, the tokenizer chapters; The Illustrated Word2vec (Jay Alammar) for representation background.
</details>

### S6. [Senior] How do models handle context longer than they were trained on, and why can it fail?

<details><summary>Answer</summary>

Positional schemes decide how well a model extrapolates. Fixed sinusoidal encodings and naive learned positions degrade past the trained length because the model never saw those position signals. RoPE rotates queries and keys by a position-dependent angle, so it encodes relative distance and extrapolates better, and it can be stretched further with position interpolation or NTK-aware scaling that rescales the rotation frequencies. Failures show up as attention that cannot reliably reach distant tokens (the "lost in the middle" effect) and rising KV-cache memory. Genuinely long context usually needs training or fine-tuning at that length, not just a scaling trick.

**Go deeper:** Attention Is All You Need (Vaswani et al.) for the original positional encoding; Attention? Attention! (Lilian Weng) for later developments.
</details>

### S7. [Senior] Distinguish RLHF from DPO, and explain why alignment is separate from capability.

<details><summary>Answer</summary>

Both align a model to human preferences using ranked responses. RLHF first trains a separate reward model from human comparisons, then optimizes the language model against it with reinforcement learning (typically PPO), which is powerful but complex and can be unstable. DPO (direct preference optimization) skips the explicit reward model and reinforcement loop, optimizing the model directly on preference pairs with a simple classification-style loss, which is more stable and cheaper and has become popular. Alignment is separate from capability because pretraining and instruction tuning build what the model can do, while preference tuning shapes how it behaves (tone, helpfulness, refusal), and a highly capable base model can still be poorly aligned, and vice versa.

**Go deeper:** Hugging Face LLM Course, the sections on instruction tuning and preference alignment.
</details>

### S8. [Senior] Why do LLMs hallucinate, and what reduces it?

<details><summary>Answer</summary>

The training objective is next-token likelihood, not truth. A model produces the most plausible continuation given its parameters, and when it lacks the fact it still generates fluent, confident text because nothing in the objective rewards saying "I do not know." There is also no grounding to an external source at generation time. Mitigations attack different parts: retrieval-augmented generation supplies real passages and citations so answers are grounded and checkable (the whole thrust of Module 4, LLMs and GenAI), decoding and prompting can encourage abstention, tool use offloads facts and computation, and calibration or verification steps catch low-confidence claims. None fully eliminates it, so production systems design for verification.

**Go deeper:** Neural Networks: Zero to Hero (Andrej Karpathy) to see next-token training from the ground up; The Illustrated Transformer (Jay Alammar) for how generation works.
</details>
