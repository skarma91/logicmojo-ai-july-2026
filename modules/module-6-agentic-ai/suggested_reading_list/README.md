# Suggested reading: Module 6 (Agentic AI)

Module 6 turns one agent into a team and then into a service: a supervisor and workers,
loops with a cap, context that stays lean, a gateway in front, traces underneath, an
evaluation that checks the path as well as the answer, and a free-tier deploy. These free
resources go deepest on each of those pieces. Start with the engineering write-ups; reach
for the papers when you want the detail under a concept.

## Multi-agent architectures

- [Anthropic: How we built our multi-agent research system](https://www.anthropic.com/engineering/multi-agent-research-system) - an orchestrator with parallel subagents in production, with the token cost and the failure modes stated plainly: the trade-off from class 6.1 (Multi-agent architectures), in practice.
- [Cognition: Don't build multi-agents](https://cognition.com/blog/dont-build-multi-agents) - the strongest case against splitting an agent, built on shared context; read it next to the Anthropic post and decide where each argument holds.
- [Anthropic: Building effective agents](https://www.anthropic.com/engineering/building-effective-agents) - workflows versus agents, and the orchestrator-workers and evaluator-optimizer patterns behind class 6.1 and class 6.2 (Orchestration).
- [LangChain docs: Multi-agent](https://docs.langchain.com/oss/python/langchain/multi-agent) - subagents, handoffs, and routers as the framework names them.
- [AutoGen, Wu et al. (2023)](https://arxiv.org/abs/2308.08155) - agents that converse to solve a task; the paper that popularized the conversation framing.
- [MetaGPT, Hong et al. (2023)](https://arxiv.org/abs/2308.00352) - roles and standard operating procedures as the structure of a team.

## Orchestration: parallel work and review loops

- [LangGraph docs: Use the graph API](https://docs.langchain.com/oss/python/langgraph/use-graph-api) - reducers, `Send` for map-reduce fan-out, conditional edges, and recursion limits, the mechanics of class 6.2 (Orchestration).
- [Self-Refine, Madaan et al. (2023)](https://arxiv.org/abs/2303.17651) - a model that critiques and revises its own output, the idea under the writer-reviewer loop.
- [Improving Factuality and Reasoning through Multiagent Debate, Du et al. (2023)](https://arxiv.org/abs/2305.14325) - several agents arguing toward a better answer, and what it costs.

## Context engineering

- [Anthropic: Effective context engineering for AI agents](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents) - context as a finite budget: compaction, structured notes, and subagents that return summaries, as in class 6.3 (Context engineering).
- [LangChain: Context engineering for agents](https://www.langchain.com/blog/context-engineering-for-agents) - the write, select, compress, and isolate framing, with examples of each.
- [Lost in the Middle, Liu et al. (2023)](https://arxiv.org/abs/2307.03172) - models use the start and end of a long context better than the middle, the evidence for keeping context short.
- [Groq docs: Prompt caching](https://console.groq.com/docs/prompt-caching) - exact-prefix caching, the discount, and why a stable prefix comes first.

## Guardrails, routing, and compliance

- [Presidio](https://presidio.dataprivacystack.org/) - the open-source PII detection and anonymization toolkit; compare its recognizers with the regex-plus-validator approach in class 6.4 (Guardrails, routing, and compliance).
- [semantic-router](https://github.com/aurelio-labs/semantic-router) - embedding-based routing with thresholds, the library version of the centroid router we build.
- [Groq docs: gpt-oss-safeguard-20b](https://console.groq.com/docs/model/openai/gpt-oss-safeguard-20b) - a policy-following safety classifier you write the policy for (a preview model, with a tight free-tier rate limit).
- [NIST AI Risk Management Framework](https://www.nist.gov/itl/ai-risk-management-framework) - the reference vocabulary for governing AI risk, useful when compliance asks how a system is controlled.

## AI security

- [OWASP Top 10 for LLM and Gen AI Applications](https://genai.owasp.org/llm-top-10/) - the standard checklist of risks and mitigations for LLM systems, the map for class 6.5 (AI security).
- [Simon Willison: The lethal trifecta for AI agents](https://simonwillison.net/2025/Jun/16/the-lethal-trifecta/) - why private data, untrusted content, and an outbound channel together are the combination to design out.
- [Greshake et al. (2023): Indirect prompt injection](https://arxiv.org/abs/2302.12173) - the paper that showed retrieved content can steer an LLM application, and why tool output is data, not instructions.

## Observability and tracing

- [LangSmith docs: Observability](https://docs.langchain.com/langsmith/observability) - traces, runs, metadata, and feedback, the hosted side of class 6.6 (Observability and tracing).
- [OpenTelemetry: GenAI semantic conventions](https://github.com/open-telemetry/semantic-conventions-genai) - the emerging standard names for model, token, and agent spans, so traces stay portable across vendors.
- [Google SRE book: Monitoring distributed systems](https://sre.google/sre-book/monitoring-distributed-systems/) - the four golden signals and symptom-based alerting, older than LLMs and still the right frame.

## Evaluating agentic systems

- [Hamel Husain: Your AI product needs evals](https://hamel.dev/blog/posts/evals/) - a practitioner's case for looking at traces and building small, specific evals first, the habit class 6.7 (Evaluating agentic systems) builds.
- [agentevals](https://github.com/langchain-ai/agentevals) - ready-made trajectory evaluators (strict, unordered, subset, superset) to compare with the matchers we write by hand.
- [LangSmith docs: Evaluation concepts](https://docs.langchain.com/langsmith/evaluation-concepts) - datasets, evaluators, experiments, and regression testing.
- [Evan Miller: How not to sort by average rating](https://www.evanmiller.org/how-not-to-sort-by-average-rating.html) - the clearest explanation of why the Wilson interval beats a raw rate on small samples.
- [AgentBench, Liu et al. (2023)](https://arxiv.org/abs/2308.03688) and [tau-bench, Yao et al. (2024)](https://arxiv.org/abs/2406.12045) - two agent benchmarks; tau-bench's pass^k measures whether an agent succeeds reliably, not just once.

## Deployment

- [FastAPI: Server-Sent Events](https://fastapi.tiangolo.com/tutorial/server-sent-events/) - the streaming endpoint pattern used in class 6.8 (Deployment).
- [Docker: Get started](https://docs.docker.com/get-started/) - images, containers, and layers, from zero.
- [Render docs: Deploy for free](https://render.com/docs/free) and [Blueprint YAML reference](https://render.com/docs/blueprint-spec) - the free tier's limits (sleep, monthly hours, memory) and every field of `render.yaml`.
- [FastEmbed](https://github.com/qdrant/fastembed) - ONNX embeddings without PyTorch, how the image fits in 512 MB.
- [Groq docs: Rate limits](https://console.groq.com/docs/rate-limits) - requests and tokens per minute and per day for each model, the numbers the token bucket and concurrency limit are sized against.

## Courses (video)

- [Hugging Face: AI Agents Course](https://huggingface.co/learn/agents-course) - a free, hands-on course on agents, tools, and multi-agent frameworks.
- [DeepLearning.AI: AI Agents in LangGraph](https://www.deeplearning.ai/courses/ai-agents-in-langgraph) and [Evaluating AI Agents](https://www.deeplearning.ai/courses/evaluating-ai-agents) - short video courses on building and tracing agents and on evaluating them (free to enroll during the platform's beta; certificates need a paid plan).
