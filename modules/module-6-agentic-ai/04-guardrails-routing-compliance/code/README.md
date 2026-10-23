# Class 6.4 code: a guardrail gateway in front of the team

The class 6.1 (Multi-agent architectures) team is unchanged (`team.py`). Every rail
lives outside it, in `gateway.py`, so one safety and compliance layer wraps the whole
team. Each request passes these stages, cheapest first, and stops at the first one
that decides it:

| # | Stage | Where it runs | What it does |
|---|---|---|---|
| 1 | Rules | local, free | length cap; obvious prompt-attack patterns |
| 2 | Redact | local, free | PII becomes typed placeholders (`<EMAIL_1>`); nothing downstream sees a raw value |
| 3 | Route | local, cheap | semantic router: `team`, `refuse`, `canned`, or `fallback` |
| 4 | Classify | one model call | `gpt-oss-safeguard-20b` against a written policy, **only** for team-bound requests |
| 5 | Team | several calls | the class 6.1 (Multi-agent architectures) team, given the redacted request |
| 6 | Output guard | local | re-scan the answer for PII; withhold an answer with no citation |

Routing comes before classification on purpose. The refuse, canned, and fallback paths
never reach a model, so a classifier call there buys nothing; the paid check guards the
only path that can act.

## Files

```
main.py           runs data/requests.jsonl through the gateway and scores it
gateway.py        the pipeline and the audit log (redacted text only)
pii.py            detection with validators (Luhn, SSN rules), typed placeholders, precision and recall
router.py         semantic router: route centroids, cosine, threshold TAU, fallback
policy.py         the written policy and the safeguard classifier, with a fallback model
team.py           the class 6.1 (Multi-agent architectures) team, unchanged
data/requests.jsonl   16 labeled requests: expected outcome, expected route, exact PII spans
data/routes.json      example utterances per route, and each route's action
tax_worker.py, llm.py, retriever.py, guardrails.py, mcp_bridge.py, server.py, data/corpus.jsonl
```

## What the report shows

- **Per request:** outcome, the stage that decided it, route and similarity score, PII kinds.
- **Decisions:** how many match the label; **block rate** (unsafe requests stopped,
  blocked or refused, divided by all unsafe requests) and **false-block rate** (safe
  requests wrongly stopped, divided by all safe requests).
- **Routing accuracy** over the requests that reached the router.
- **PII precision** (flagged spans that really are PII, divided by all flagged spans) and
  **recall** (real PII spans that were flagged, divided by all real PII spans).
- **Compliance:** the audit log is searched for every labeled raw PII value.
- **Model calls per agent:** the classifier against the team.

## Two results to expect, and why

1. **One name leaks.** Request r05 ends "Thanks, Daniel Okafor". The PERSON rule only
   finds a capitalized name after a context cue ("my neighbor", "patient", "my name
   is"), so this one is missed: recall drops below 1 and the compliance check reports
   the leak. That is the lesson. Rules plus validators handle structured identifiers
   well and names badly; production adds a named-entity model, for example
   [Microsoft Presidio](https://microsoft.github.io/presidio/) with spaCy.
2. **Some routing mistakes, until you tune TAU.** The router threshold `TAU` (in
   `router.py`) is a starting value, not a measured one. Run `python main.py --tune`:
   it prints every request's best route score and suggests a TAU midway between the
   weakest in-scope score and the strongest out-of-scope one. If those overlap, no
   threshold separates them; add example utterances to `data/routes.json` instead.

## The classifier model

`gpt-oss-safeguard-20b` is a Groq **preview** model at time of writing, which Groq can
withdraw at short notice. If the call fails, or the provider is not Groq, `policy.py`
falls back to your default chat model with the same policy and the report names the
model used. An unreadable verdict counts as a violation (fail closed). The policy
follows Groq's guidance: Instructions, Definitions, Criteria, Examples, about 570
tokens, static policy first and the request last, so the policy prefix can be cached.

## Free-tier note

`--no-team` stops at the team's door: one classifier call per team-bound request,
about nine on this set. The full run adds about four team calls (supervisor and
researcher) for each of the roughly seven answered requests. Both fit Groq's free plan;
`llm.py` backs off on HTTP 429.

## Run

```
pip install langgraph "mcp<2" sentence-transformers numpy python-dotenv groq
# set PROVIDER=groq and GROQ_API_KEY in .env (or PROVIDER=ollama)
python main.py --tune       # router scores and a suggested TAU, no model calls
python main.py --no-team    # gateway decisions only
python main.py              # full run, team included
python main.py --tau 0.42   # try another threshold
```

Not legal advice: the compliance checks here illustrate data minimization and
redaction-before-logging. Real GDPR or HIPAA obligations depend on your situation.
