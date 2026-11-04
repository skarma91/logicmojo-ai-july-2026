# Interview Preparation Guide, Module 6: Agentic AI

Questions and answers for the multi-agent and production stack taught in Module 6: team architectures, orchestration mechanics, context engineering, guardrails and compliance, AI security, observability, evaluation of agentic systems, and deployment with cost controls. This is a study companion, not part of the published course site.

**Coverage map.** The questions follow the eight classes: 6.1 (Multi-agent architectures), 6.2 (Orchestration), 6.3 (Context engineering), 6.4 (Guardrails, routing, and compliance), 6.5 (AI security), 6.6 (Observability and tracing), 6.7 (Evaluating agentic systems), and 6.8 (Deployment). The single-agent-versus-team decision, trajectory evaluation, and cost control in production are the most heavily interviewed topics here, so spend the most time on Parts 1, 7, and 8.

## How to use this guide

Read the question, answer it out loud or on paper first, then expand the answer to check yourself. Each question is tagged by difficulty:

- **[Warm-up]** a definition or one-liner an interviewer opens with.
- **[Core]** the standard question you are expected to answer cleanly.
- **[Deep]** mechanism, trade-off, or a "why" that separates strong candidates.
- **[Numerical]** a small calculation to do by hand.
- **[Applied]** a scenario or short design or debugging prompt.

The supervisor graph, the expected-rounds formula for a bounded loop, the critical path of a trace, and the Wilson interval are worth being able to draw and write from memory.

---

## Part 1: Multi-agent architectures (class 6.1)

### Q1. [Warm-up] What is a multi-agent system, in one sentence?

<details><summary>Answer</summary>

Several LLM-driven agents, each with its own prompt, tools, and slice of context, that cooperate on one task, usually with a coordinator deciding who works next. Each agent is still the class 5.1 (What is an agent) loop; the new part is how they divide work and pass results.
</details>

### Q2. [Core] When does splitting one agent into a team pay off, and when does it not?

<details><summary>Answer</summary>

It pays off when there is a specific bottleneck a split removes: independent subtasks that can run in parallel, a context window overloaded by unrelated tools and documents, or a step that needs a different model, prompt, or permission set. It does not pay off for a short, sequential task where every step needs the full context, because a team adds model calls, latency, handoff loss, and more ways to fail. The senior default is one agent first, then split on a named bottleneck you have measured.
</details>

### Q3. [Core] Name three team shapes and the task each fits.

<details><summary>Answer</summary>

- **Supervisor**: a coordinator routes to workers and combines results. Fits tasks where the next step depends on what earlier steps found.
- **Hierarchical**: supervisors of supervisors. Fits large tasks with clear sub-teams (research, writing, review), at the cost of more hops.
- **Pipeline**: a fixed sequence of agents. Fits repeatable processes where the order never changes; it is a workflow, and that is a feature, not a weakness.
</details>

### Q4. [Deep] Agent-as-tool versus handoff: what is the difference?

<details><summary>Answer</summary>

With agent-as-tool, the supervisor calls a worker like a function and gets a result back; the supervisor stays in control and sees only the return value. With a handoff, control transfers to the worker (in LangGraph, a `Command(goto=..., update=...)`), which then decides where to go next. Agent-as-tool keeps routing centralized and easier to trace; handoffs give workers autonomy and can cut supervisor calls, but make the control flow harder to predict and to bound.
</details>

### Q5. [Numerical] A request passes through five agent steps, each succeeding independently 95% of the time. What is the end-to-end success rate, and what does it imply?

<details><summary>Answer</summary>

$0.95^5 \approx 0.774$, so about 77%. Failure compounds multiplicatively: every agent you add multiplies in another chance to fail. Splitting must buy more reliability per step (narrower prompts, exact checks, a review loop) than it costs in added steps, or the team is worse than the single agent.
</details>

### Q6. [Applied] Your team uses 3x the tokens of the single agent and scores the same. What do you do?

<details><summary>Answer</summary>

Measure where the tokens go per agent (supervisor routing calls are a common surprise), then check whether the split removed any real bottleneck. If quality is equal and nothing runs in parallel, fold the workers back into one agent or a fixed pipeline. If one worker truly needs isolation (a large tool set, different permissions), keep only that split and make the supervisor cheaper: a smaller model, or deterministic routing where the next step is obvious.
</details>

---

## Part 2: Orchestration (class 6.2)

### Q7. [Warm-up] What is a subgraph in LangGraph?

<details><summary>Answer</summary>

A compiled graph used as a node inside a parent graph. Each worker can be its own graph with private state, tested on its own, and wired in either directly (shared state keys) or through a wrapper function that maps parent state to child input and back.
</details>

### Q8. [Core] How do reducers decide what happens when parallel branches write the same key?

<details><summary>Answer</summary>

A key with no reducer is overwritten, and two parallel writes to it in one step raise an error because the merge is ambiguous. A key annotated with a reducer, for example `Annotated[list, operator.add]`, combines the writes: `operator.add` appends both lists. Choose the reducer by meaning: append for findings, overwrite for a single current draft, a custom function for dedup or max.
</details>

### Q9. [Core] What does `Send` do, and what limits its speedup in practice?

<details><summary>Answer</summary>

`Send(node, payload)` launches one invocation of a node per payload, in parallel, so a supervisor can fan out the same worker over several subtopics; a reducer fans the results back in. In theory latency drops from the sum of the branches to the slowest one. In practice provider rate limits cap real parallelism: on a free tier with about 30 requests per minute, five parallel branches each making several calls can hit 429s and fall back to retries, so the measured speedup is smaller than the ideal.
</details>

### Q10. [Deep] Why should a reviewer return a structured verdict and run exact checks first?

<details><summary>Answer</summary>

A structured verdict (`approve: bool`, `feedback: str` through a tool schema) is machine-readable, so the loop's routing is deterministic. Exact checks (word limit, required citations present, citations actually retrieved) are free, never wrong, and should override the model: a model reviewer can approve a draft that cites a source nobody retrieved. The model then judges only what code cannot, such as whether each claim is supported.
</details>

### Q11. [Numerical] A writer-reviewer loop is capped at 3 rounds, and each draft passes review with probability 0.6. How many rounds run on average?

<details><summary>Answer</summary>

For a cap of $k$ rounds and a per-round pass probability $q$, the expected number of rounds is

$$E[\text{rounds}] = \frac{1 - (1-q)^k}{q}$$

where $(1-q)^k$ is the chance that every one of the $k$ drafts fails. With $q = 0.6$ and $k = 3$: $(1 - 0.4^3)/0.6 = 0.936/0.6 = 1.56$ rounds. When $q = 0$ the formula divides by zero; the loop simply always runs all $k$ rounds. Use this to budget calls: 1.56 rounds is 3.12 model calls for a writer plus a reviewer.
</details>

### Q12. [Applied] A review loop always hits its cap. Diagnose.

<details><summary>Answer</summary>

Read the feedback across rounds in the trace. Common causes: the writer never sees the feedback (it is not in its context), the reviewer's criterion cannot be met with the given sources (a loop that cannot win), the feedback is vague so revisions change nothing, or an exact check is buggy and fails every draft. Fix the cause; raising the cap only multiplies the cost of a loop that cannot converge.
</details>

---

## Part 3: Context engineering (class 6.3)

### Q13. [Warm-up] What is context engineering?

<details><summary>Answer</summary>

Deciding exactly what each model call sees: which instructions, tools, documents, history, and handoff notes, in what order. In a team, it means each agent gets a scoped view of shared state rather than everything, because tokens cost money, latency, and accuracy.
</details>

### Q14. [Core] Why hand off a note rather than the worker's transcript?

<details><summary>Answer</summary>

The transcript carries tool schemas, raw search results, and intermediate reasoning the next agent does not need. A compact note (the facts found, with their source ids) shrinks the next agent's input, keeps citations traceable, and stops raw tool output, which may be wrong or untrusted, from travelling between agents. Measure it with a compression ratio: transcript tokens divided by note tokens.
</details>

### Q15. [Core] How does provider prompt caching work, and what layout does it reward?

<details><summary>Answer</summary>

The provider reuses computation for an exact prefix it has seen recently. On Groq it is automatic, applies above a minimum prefix length, discounts cached input tokens by 50%, and expires after a period of inactivity. It rewards stable content first (system prompt, tool schemas, fixed instructions) and variable content last (the request, a timestamp, a request id). One changing byte early in the prompt invalidates everything after it.
</details>

### Q16. [Numerical] A call has 6,000 input tokens, of which a 4,000-token prefix is cached at a 50% discount. What is the input saving?

<details><summary>Answer</summary>

Billed-equivalent input is $4000 \times 0.5 + 2000 = 4000$ token-equivalents instead of 6,000, a saving of $2000/6000 \approx 33\%$ on input. In general the saving is the cached fraction times the discount: $(4000/6000) \times 0.5 = 1/3$.
</details>

### Q17. [Deep] "Give every agent everything, to be safe." What is wrong with that?

<details><summary>Answer</summary>

It is not safe. Irrelevant context lowers quality (models use the middle of long inputs poorly, and distractors pull answers off course), raises cost per call, slows responses, and spreads untrusted tool output into agents that act on it. It also leaks personal data into places that do not need it. Scoped context is both cheaper and more accurate.
</details>

### Q18. [Applied] Your engineered-context run shows a higher cache-hit rate than the baseline, and you want to claim caching as a win. What do you check first?

<details><summary>Answer</summary>

Whether the comparison is fair. If the baseline ran first, it warmed the cache that the second run reused, so the second run's hits are partly an artifact of order. Run them in alternating order, or give each run a volatile header that prevents cross-run reuse, and compare tokens and quality over several questions before claiming the saving.
</details>

---

## Part 4: Guardrails, routing, and compliance (class 6.4)

### Q19. [Warm-up] What is a guardrail gateway?

<details><summary>Answer</summary>

One layer that wraps the whole team: input rules, PII redaction, routing, and a policy classifier before the team runs, and an output guard after. Putting it outside the agents means one place to audit, test, and change policy, instead of rules scattered across prompts.
</details>

### Q20. [Core] Why order the stages rules, then route, then classify?

<details><summary>Answer</summary>

Cost funnel: run the cheapest, most certain checks first. Local rules (length, patterns) are free and block the obvious; semantic routing is one embedding and diverts out-of-scope requests to a canned reply; only what remains reaches the model classifier, which costs a call and has its own rate limit. Every stage that stops a request saves all the stages after it.
</details>

### Q21. [Core] How does centroid routing work, and how do you pick the threshold?

<details><summary>Answer</summary>

Embed example utterances for each route and average them into a centroid. A new request goes to the nearest centroid by cosine similarity, but only if that similarity clears a threshold $\tau$; otherwise it goes to a fallback. Choose $\tau$ from labeled data: sweep values and pick the one that balances misroutes against fallbacks for your use, rather than guessing.
</details>

### Q22. [Numerical] A PII detector finds 20 spans on a labeled set: 18 are real PII, 2 are not. The set has 22 real PII spans. Give precision and recall, and say which matters more here.

<details><summary>Answer</summary>

Precision $= 18/20 = 0.90$ (of what was flagged, how much was real). Recall $= 18/22 \approx 0.82$ (of the real PII, how much was caught). For redaction, recall matters more: a missed name or email leaks to the model provider, the trace, and the logs, while a false positive only masks a harmless string.
</details>

### Q23. [Deep] Fail open or fail closed when the policy classifier errors?

<details><summary>Answer</summary>

It depends on the cost of each error. Fail closed (block) when a wrong allow is expensive: regulated advice, actions with side effects. Fail open (allow, log, alert) when availability matters more and downstream checks exist. Whichever you choose, make it explicit in code, log the failure, and count it, because a silent fallback is a policy nobody decided.
</details>

### Q24. [Applied] Where can personal data leak in a traced, logged multi-agent system, and what is the ordering rule?

<details><summary>Answer</summary>

The model provider's API, trace spans (inputs, outputs, and metadata), application logs and audit trails, response caches, evaluation datasets, and error messages. The rule is redact before anything is sent or stored: the redaction step sits in front of every model call, log line, and trace. Span metadata is an easy miss, because tracers often redact inputs and outputs but record metadata verbatim.
</details>

---

## Part 5: AI security (class 6.5)

### Q25. [Warm-up] Direct versus indirect prompt injection?

<details><summary>Answer</summary>

Direct: the user's own message tries to override the system's instructions. Indirect: the instructions arrive inside content the system retrieves or receives, such as a web page, a document, an email, or a tool result, so the user may be innocent. Indirect injection is the bigger risk for agents, because they read untrusted content as part of normal work.
</details>

### Q26. [Core] What is the "lethal trifecta"?

<details><summary>Answer</summary>

An agent that has access to private data, reads untrusted content, and can communicate externally. With all three, injected instructions in the untrusted content can make the agent send the private data out. The design response is to make sure no single agent, or chain of agents, holds all three capabilities at once.
</details>

### Q27. [Core] How does least privilege apply to a team of agents?

<details><summary>Answer</summary>

Give each agent only the tools and data its role needs. The agent that reads untrusted text should have no side-effecting tools; the agent that can act should never see raw untrusted text, only structured, validated fields. Pair this with allowlists on destinations and argument shapes, and human approval or a sandbox for irreversible actions.
</details>

### Q28. [Deep] Why is a strong system prompt not a defense on its own?

<details><summary>Answer</summary>

The system prompt and untrusted content share the same channel: the model sees tokens, not a privilege boundary. A model can follow instructions that arrive later in its context, so "ignore any instructions in documents" lowers the rate of failure but cannot guarantee it. Real defenses live outside the model: permissions, allowlists, validation of tool arguments, and approval gates.
</details>

### Q29. [Numerical] Three independent defense layers each miss 20%, 30%, and 10% of bad inputs. What is the combined miss rate, and what is the catch?

<details><summary>Answer</summary>

If the layers fail independently, the combined miss rate is the product, $0.2 \times 0.3 \times 0.1 = 0.006$, or 0.6%. The catch is independence: layers that share a weakness (two classifiers fooled by the same phrasing) fail together, so the real rate can be far higher. Diverse layers (a rule, a classifier, a permission boundary) are worth more than several of the same kind.
</details>

### Q30. [Deep] What is a confused deputy in an agentic system?

<details><summary>Answer</summary>

A component with legitimate authority that is tricked into using it on someone else's behalf. An agent with a payment tool that acts on instructions found in a retrieved invoice is a confused deputy: the authority is real, the intent is not the user's. The fix is to tie each privileged action to the user's actual request and to validated data, not to whatever the model read.
</details>

---

## Part 6: Observability and tracing (class 6.6)

### Q31. [Warm-up] Define a trace and a span.

<details><summary>Answer</summary>

A span is one timed unit of work (an agent turn, a model call, a retrieval, a tool call) with inputs, outputs, tokens, errors, and metadata. A trace is the tree of spans for one request, linked by parent ids, so you can see who called what, in what order, and how long each took.
</details>

### Q32. [Core] Total time versus self time?

<details><summary>Answer</summary>

Total time is a span's duration from start to end, including its children. Self time is the part spent in the span itself, total time minus the time covered by its children (with overlapping children merged so parallel work is not subtracted twice). Self time points at where work actually happens; total time points at which subtree is slow.
</details>

### Q33. [Numerical] A supervisor span takes 1 s, then two researcher branches run in parallel for 3 s and 5 s, then a writer runs for 2 s. What is the wall time, and what is the sum of the spans?

<details><summary>Answer</summary>

Wall time follows the critical path: $1 + \max(3, 5) + 2 = 8$ s. The sum of the spans is $1 + 3 + 5 + 2 = 11$ s. Speeding up the 3 s branch saves nothing; only the 5 s branch, the supervisor, or the writer are on the critical path.
</details>

### Q34. [Deep] Why is the slowest span rarely the costliest one?

<details><summary>Answer</summary>

Latency and cost have different drivers. A retrieval or a slow tool can dominate time while costing nothing in tokens, and a supervisor that makes many short routing calls with long prompts can dominate cost while each call is quick. Rank spans twice, by self time and by token cost share, and fix the bottleneck that matches the complaint.
</details>

### Q35. [Applied] Traces must go to a hosted tool. What do you do about privacy and volume?

<details><summary>Answer</summary>

Redact PII before recording, including metadata; truncate long inputs and outputs; never put secrets in spans. For volume, sample in production (keep every error and a fraction of successes), set retention to what debugging and compliance need, and keep a local JSONL fallback so tracing never blocks the request path.
</details>

---

## Part 7: Evaluating agentic systems (class 6.7)

### Q36. [Warm-up] What is a trajectory, and why evaluate it?

<details><summary>Answer</summary>

The ordered list of work steps a run took: agent turns and tool calls. An agent can produce a right answer by a wrong path, for example answering from memory without retrieving, or after calling a forbidden tool. Answer-only scoring passes those runs; trajectory scoring catches them before they turn into wrong answers on the next input.
</details>

### Q37. [Core] Compare exact, in-order, and set trajectory matching.

<details><summary>Answer</summary>

Exact: same steps, same order, nothing extra; precise but brittle, since one harmless extra search fails it. In-order: every reference step appears in the same order, with extras allowed; the usual choice for agents. Set: every reference step appears somewhere, in any order; use it when order does not matter. Pair any of them with a forbidden-steps list.
</details>

### Q38. [Core] What do step precision and step recall measure?

<details><summary>Answer</summary>

Precision is the share of steps taken that were required; low precision means wasted steps (cost). Recall is the share of required steps that were taken; low recall means skipped steps (risk). Count with multisets, so two searches against one required search counts one as useful and one as waste.
</details>

### Q39. [Numerical] An agent passes 9 of 10 tasks. Give the Wald and Wilson 95% intervals and say which to report.

<details><summary>Answer</summary>

Wald: $\hat p \pm z\sqrt{\hat p(1-\hat p)/n} = 0.9 \pm 1.96\sqrt{0.009}$, about 0.71 to 1.0 (clipped from 1.09). Wilson, which centers the interval at $(\hat p + z^2/2n)/(1 + z^2/n)$ and widens it correctly for small $n$: about 0.60 to 0.98. Report Wilson. Wald is too narrow for small samples and claims zero uncertainty at 10 of 10, where Wilson still gives about 0.72 to 1.0. The honest reading of 9 of 10 is "somewhere between 60% and 98%".
</details>

### Q40. [Numerical] How many tasks do you need to estimate a pass rate near 80% within plus or minus 10 points at 95% confidence?

<details><summary>Answer</summary>

$n \approx z^2 p(1-p)/E^2 = 1.96^2 \times 0.8 \times 0.2 / 0.1^2 \approx 61.5$, so about 62 tasks. Halving the margin to 5 points quadruples the count to about 246. This is why a 10-task set detects only large changes.
</details>

### Q41. [Deep] What is a regression gate, and why gate on tasks rather than on the average?

<details><summary>Answer</summary>

A check in the release path that fails if any task that passed in the pinned baseline fails now. Gating on the average lets one fixed task hide one newly broken task; gating per task pins behavior, so anything that used to work must keep working. Keep the baseline under version control and refresh it deliberately, not on every run.
</details>

### Q42. [Applied] Answer-only success is 7 of 8, but path-aware pass rate is 3 of 8. What does that tell you?

<details><summary>Answer</summary>

Most correct answers are arriving by wrong paths: skipped retrieval, wrong tools, or forbidden steps. The system is right for now by luck or memorization, and will fail as soon as the questions move off what the model already knows. Read the failing trajectories, fix routing or tool selection, and gate on the path-aware rate.
</details>

---

## Part 8: Deployment and cost control (class 6.8)

### Q43. [Warm-up] Image, container, layer: define each.

<details><summary>Answer</summary>

An image is a read-only package of an app and its dependencies, built from a Dockerfile. A container is a running instance of an image. A layer is one cached filesystem step of the image; ordering the Dockerfile so dependencies install before code is copied means code edits rebuild only the last layers.
</details>

### Q44. [Core] How do you fit an embedding-based service into a 512 MB instance?

<details><summary>Answer</summary>

Drop PyTorch and use ONNX runtime embeddings (FastEmbed) with the same small model; bake the model weights and the corpus vectors into the image at build time so the service only embeds queries; run one worker; keep heavy work out of the request path. The general rule is to do slow work at build time, not at request time.
</details>

### Q45. [Core] Why stream with Server-Sent Events, and what goes in a health check?

<details><summary>Answer</summary>

A team can take tens of seconds, and SSE sends progress events (one per agent step) over a plain HTTP response so the client sees work happening and proxies do not time out. A health check should be cheap and dependency-light, with no model call, so the host can probe it often without spending quota.
</details>

### Q46. [Numerical] A token bucket has capacity 5 and refills at 0.1 tokens per second. A client sends 60 requests evenly over one minute. Roughly how many are allowed?

<details><summary>Answer</summary>

The burst uses the 5 stored tokens, then refills add $60 \times 0.1 = 6$ more during the minute, so about 11 are allowed and the rest get 429 with a `Retry-After`. Capacity sets the burst; rate sets the long-run throughput.
</details>

### Q47. [Numerical] The provider allows 30 requests per minute, each user request makes about 6 model calls and takes about 20 s. What concurrency limit fits?

<details><summary>Answer</summary>

Sustainable throughput is $30/6 = 5$ user requests per minute. By Little's law, requests in flight equal arrival rate times time in system: $5 \times (20/60) \approx 1.7$, so a limit of 2. Higher concurrency does not raise throughput; it only produces 429s and retries.
</details>

### Q48. [Deep] Name five controls that bound cost in a deployed agent service.

<details><summary>Answer</summary>

A per-client rate limit (token bucket), a response cache keyed on the redacted, normalized question, model routing (the larger model only for questions that need it), a per-request budget on model calls and tokens that stops the run cleanly when exceeded, and a concurrency limit sized to the provider quota. Cache only successful answers, and never key the cache on raw PII.
</details>

### Q49. [Applied] What does "free tier" really mean on a host such as Render, and how do you design around it?

<details><summary>Answer</summary>

The instance sleeps after a period of inactivity and takes time to wake, there is a monthly cap on instance hours, memory is small, and an omitted plan in a blueprint can default to a paid instance, so write `plan: free` explicitly. Design around it: a cheap health check, a client that tolerates a cold start, baked artifacts so the wake is fast, and secrets set as environment variables in the dashboard, never in the repo or the image.
</details>

---

## Senior stretch: system-design and judgment

### S1. [Deep] Design a research assistant that answers from a paper corpus with citations, served as an API. Walk through it.

<details><summary>Answer</summary>

Start with one agent and a retriever; move to a supervisor with a paper finder, a summarizer, and a critic only if traces show a bottleneck. Put a gateway in front (rules, PII redaction) and an output guard after (every cited id must be one the team retrieved). Use a relevance threshold chosen from data so off-topic questions get a clear refusal. Trace every agent turn and model call with redaction. Evaluate on a holdout task set with trajectory checks and a regression gate. Serve with FastAPI and SSE, with a rate limit, cache, model routing, per-request budget, and a concurrency limit sized to the provider quota.
</details>

### S2. [Deep] When is a multi-agent design the wrong answer even though the task "sounds" like a team?

<details><summary>Answer</summary>

When the steps are sequential and each needs the full context, when there is no parallelism to exploit, or when the free-tier quota makes extra calls the binding constraint. A pipeline or one agent with well-designed tools is cheaper, faster, easier to evaluate, and has fewer failure points. Choose the team only after measuring the single agent and naming what the split fixes.
</details>

### S3. [Applied] A deployed team's cost doubled overnight with no code change. How do you investigate?

<details><summary>Answer</summary>

Compare traces before and after: tokens per request by agent, calls per request, cache-hit rate, and the request mix. Likely causes: a prompt-prefix change that broke provider caching, a change in traffic toward questions routed to the larger model, a review loop now hitting its cap, retries from 429s, or a response cache that stopped hitting because its key changed. Fix the cause, then add an alert on tokens per request so the next jump is caught in hours, not on the invoice.
</details>
