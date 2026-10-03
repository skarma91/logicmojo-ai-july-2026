# Suggested reading: Module 5 (AI agents)

Module 5 is where models become agents: a loop that thinks, calls tools, and acts.
These free resources go deepest on each piece we build, plus the original papers behind
the ideas. Start with the guides; reach for the papers when you want the detail under a
concept.

## What an agent is (and when to build one)

- [Anthropic: Building effective agents](https://www.anthropic.com/research/building-effective-agents) - the clearest short read on the agent-versus-workflow distinction and when each is the right choice (the class 5.1 framing).
- [Chip Huyen: Agents](https://huyenchip.com/2025/01/07/agents.html) - a thorough tour of tools, planning, memory, and failure modes, from first principles.

## Tool calling and the reasoning loop

- [ReAct: Synergizing Reasoning and Acting, Yao et al. (2022)](https://arxiv.org/abs/2210.03629) - the think-act-observe loop we implement in class 5.2.
- [Toolformer, Schick et al. (2023)](https://arxiv.org/abs/2302.04761) - how a model learns when and how to call tools.
- [Reflexion, Shinn et al. (2023)](https://arxiv.org/abs/2303.11366) - agents that improve by reflecting on their own failed attempts.

## Orchestration with LangGraph

- [LangGraph documentation](https://langchain-ai.github.io/langgraph/) - the state graph, reducers, cycles, and the checkpointer we build on in class 5.4.
- [LangGraph: persistence and human-in-the-loop](https://langchain-ai.github.io/langgraph/concepts/persistence/) - how the checkpointer persists a thread and enables the approval pause we use in class 5.6.

## Memory

- [MemGPT, Packer et al. (2023)](https://arxiv.org/abs/2310.08560) - treating the context window like memory tiers, the intuition behind working versus long-term memory in class 5.3.

## MCP (the Model Context Protocol)

- [Model Context Protocol: introduction](https://modelcontextprotocol.io/) - the official docs for the protocol we implement, client and server, in class 5.5.
- [Anthropic: Introducing the Model Context Protocol](https://www.anthropic.com/news/model-context-protocol) - why a common protocol turns the N times M integration problem into N plus M.

## Reliability, injection, and evaluation

- [Simon Willison: prompt injection series](https://simonwillison.net/series/prompt-injection/) - the definitive running write-up on why tool and retrieval output must be treated as data, not instructions (the class 5.6 defense).
- [OWASP Top 10 for LLM Applications](https://owasp.org/www-project-top-10-for-large-language-model-applications/) - the security checklist for LLM systems, injection first among them.
- [LangSmith documentation](https://docs.smith.langchain.com/) - tracing and evaluating agents on trajectory and outcome, the observability from class 5.6.
