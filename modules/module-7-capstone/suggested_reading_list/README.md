# Suggested reading: Module 7 (Capstone)

Module 7 turns everything into two products: the Customer Support Desk you watch being
designed and built, and the Travel Planner you design yourself. These free resources help
with the three things that decide how good a capstone is: the design, the evaluation, and
how you explain it. All of them are free to read.

## Designing a team of agents

- [Anthropic: Building effective agents](https://www.anthropic.com/engineering/building-effective-agents) - workflows versus agents, and the orchestrator-workers and evaluator-optimizer patterns behind both capstones.
- [Anthropic: How we built our multi-agent research system](https://www.anthropic.com/engineering/multi-agent-research-system) - a production team of agents, with its costs and failure modes stated plainly.
- [Cognition: Don't build multi-agents](https://cognition.com/blog/dont-build-multi-agents) - the case against splitting an agent; read it before you defend your own team in an interview.

## The two projects

- [TravelPlanner: A Benchmark for Real-World Planning with Language Agents, Xie et al. (2024)](https://arxiv.org/abs/2402.01622) - language agents fail at multi-day travel plans mostly by losing track of constraints, the problem your critic's code checks exist to solve.
- [Self-Refine, Madaan et al. (2023)](https://arxiv.org/abs/2303.17651) - a model that critiques and revises its own output: the idea under the Travel Planner's plan, critique, and revise loop.
- [tau-bench, Yao et al. (2024)](https://arxiv.org/abs/2406.12045) - agents in customer-service settings, measured on whether they succeed reliably, not just once; useful context for the support desk.
- [Bitext customer-support dataset](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset) - the support desk's data: 27 intents, with the language tags the evaluation uses.
- [MediaWiki TextExtracts API](https://www.mediawiki.org/wiki/Extension:TextExtracts) - the API behind the Travel Planner's `fetch_data.py`, if you want to fetch more pages or district guides.

## From prototype to product

- [Eugene Yan: Patterns for building LLM-based systems and products](https://eugeneyan.com/writing/llm-patterns) - evals, RAG, guardrails, caching, and defensive design, each as a reusable pattern.
- [Chip Huyen: Building a generative AI platform](https://huyenchip.com/2024/07/25/genai-platform.html) - the components a production system grows around a model, a good map for the "how would it scale" interview question.
- [Hamel Husain: Your AI product needs evals](https://hamel.dev/blog/posts/evals/) - why small, specific evaluations built from your own traces beat generic benchmarks; the habit behind your evaluation report.

## For interviews

- [The course's interview preparation guides](../../../interview_preparation_guide/) - one guide per module, with questions from warm-up to senior and answers to check yourself against.
