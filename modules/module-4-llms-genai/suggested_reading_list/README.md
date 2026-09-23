# Suggested reading: Module 4 (LLMs and GenAI)

Module 4 is the widest module: prompting, RAG, serving, evaluation, and fine-tuning.
These are the free resources that go deepest on each, plus the original papers behind
the ideas we build. Start with the guides; reach for the papers when you want the
detail under a concept.

## Prompting and building with LLMs

- [Anthropic: Prompt engineering overview](https://docs.claude.com/en/docs/build-with-claude/prompt-engineering/overview) - practical, current guidance on getting reliable behavior from a model.
- [DeepLearning.AI short courses](https://www.deeplearning.ai/short-courses/) - free, hands-on mini-courses on prompting, RAG, and evaluation with real APIs.

## Embeddings, vector databases, and RAG

- [Pinecone: Retrieval-Augmented Generation](https://www.pinecone.io/learn/retrieval-augmented-generation/) - the four-step ingest, retrieve, augment, generate loop, drawn out clearly.
- [Pinecone: What is a vector database?](https://www.pinecone.io/learn/vector-database/) - vectors, similarity, and approximate nearest-neighbor search, the engine under retrieval.
- [Retrieval-Augmented Generation, Lewis et al. (2020)](https://arxiv.org/abs/2005.11401) - the paper that named RAG.

## How LLMs are served (efficiency)

- [Making LLMs more accessible: bitsandbytes, 4-bit, and QLoRA, Hugging Face](https://huggingface.co/blog/4bit-transformers-bitsandbytes) - what 4-bit storage and NF4 actually do, with the exact `BitsAndBytesConfig` we use.
- [FlashAttention, Dao et al. (2022)](https://arxiv.org/abs/2205.14135) - the IO-aware attention that trades HBM reads for on-chip SRAM work (the class 4.7 story).
- [PagedAttention / vLLM, Kwon et al. (2023)](https://arxiv.org/abs/2309.06180) - managing the KV cache like OS virtual memory, the idea behind vLLM.

## Evaluating LLMs and RAG

- [Ragas documentation](https://docs.ragas.io/en/stable/) - the standard RAG-evaluation library: faithfulness, answer relevance, context precision and recall.

## Fine-tuning and alignment

- [LoRA: Low-Rank Adaptation, Hu et al. (2021)](https://arxiv.org/abs/2106.09685) - the low-rank add-on that makes fine-tuning cheap; the `W' = W + BA` we decode in class 4.9.
- [QLoRA, Dettmers et al. (2023)](https://arxiv.org/abs/2305.14314) - LoRA on a 4-bit base; how a large model fine-tunes on one small GPU.
- [Hugging Face PEFT documentation](https://huggingface.co/docs/peft) - the library behind LoRA and friends, with runnable recipes.
- [RLHF: Reinforcement Learning from Human Feedback, Chip Huyen](https://huyenchip.com/2023/05/02/rlhf.html) - the three stages (pretrain, SFT, RLHF) with the reward model and PPO explained plainly.
- [Training language models to follow instructions (InstructGPT), Ouyang et al. (2022)](https://arxiv.org/abs/2203.02155) - the paper that made SFT-plus-preferences the standard alignment recipe.
