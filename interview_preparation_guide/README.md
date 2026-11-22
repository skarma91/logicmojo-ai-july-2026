# Interview preparation guides

One guide per module, written to be practised out loud. Every guide follows the classes of
its module, and every answer is hidden until you have tried the question yourself.

| Module | Guide | What it covers | Questions |
|---|---|---|---|
| 1. Python for AI | [Python for AI](1_python_for_ai.md) | core Python and its gotchas, OOP, errors, files and JSON, NumPy, Pandas, APIs | 30 |
| 2. ML and DL essentials | [ML and DL essentials](2_ml_dl_essentials.md) | what ML is, how models learn, neural networks, PyTorch, training in practice | 55 |
| 3. NLP to Transformers | [NLP to Transformers](3_nlp_to_transformers.md) | text as data, attention, the Transformer, pretrained models, from models to LLMs | 50 |
| 4. LLMs and GenAI | [LLMs and GenAI](4_llms_and_genai.md) | prompting, structured outputs, embeddings, RAG, serving, evaluation, fine-tuning | 60 |
| 5. AI Agents | [AI Agents](5_ai_agents.md) | the agent loop, tool calling and ReAct, memory, LangGraph, MCP, reliability, cost | 45 |
| 6. Agentic AI | [Agentic AI](6_agentic_ai.md) | multi-agent teams, orchestration, context, guardrails, security, tracing, evaluation, deployment | 52 |

Module 7 (Capstone) has no separate guide: its interview preparation is the system-design
discussion in class 7.3 (Interview prep and capstone), built on the project you deliver.

## How to use the guides

1. **Answer first, then open the answer.** Say it out loud or write it down before you
   expand the answer. Recognising an answer is much easier than producing one.
2. **Use the difficulty tags.** Each question is marked:
   - **[Warm-up]** a definition an interviewer opens with;
   - **[Core]** the standard question you must answer cleanly;
   - **[Deep]** a mechanism, trade-off, or "why" that separates strong candidates;
   - **[Numerical]** a small calculation to do by hand;
   - **[Applied]** a scenario, design, or debugging prompt.
3. **Finish with the senior questions.** Each guide ends with stretch questions on
   system design and judgment, the part of an interview that decides the level.

## Which guides to start with

| Your target role | Start with | Then |
|---|---|---|
| Data or ML engineer | ML and DL essentials | NLP to Transformers, then the Python for AI rapid-fire |
| LLM application engineer | LLMs and GenAI | AI Agents, then the transformer questions in NLP to Transformers |
| AI agent or platform engineer | AI Agents | Agentic AI, then RAG and evaluation in LLMs and GenAI |
| Any role, the week before | the senior questions in LLMs and GenAI, AI Agents, and Agentic AI | your capstone story, from class 7.3 (Interview prep and capstone) |

## Before an interview

- Draw the core diagrams from memory: the training loop, attention, a RAG pipeline, the
  agent loop, a supervisor with workers.
- Redo the numerical questions on paper; interviewers check the reasoning, not the answer.
- Prepare a two-minute walkthrough of your capstone: the problem, the design, a decision
  you defend, what broke, and the measured result.
