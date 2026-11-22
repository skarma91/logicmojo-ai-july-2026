# Instructor manual: building the Customer Support Desk live

This is the script for class 7.2 (Build lab). You build the whole Customer Support Desk
designed in class 7.1 (Capstone kickoff), by hand, in front of the class, one building
block at a time. Each step gives you:

- **Goal**: what the block does and where it sits in the design.
- **Type**: the file to create, with its full code. Comments and docstrings can be pasted
  or shortened while you type; the code lines are the point.
- **Say**: the talking points that connect the code to the design.
- **Checkpoint**: a command that proves the block works before you move on, and what
  to expect.

The order follows the low-level design: data shapes first, then each agent on its own,
then the rules, then the supervisor that wires them, then evaluation. Every checkpoint
runs on what exists so far, so a typo surfaces at the step that caused it.

## Timing (one 2-hour session)

| Block | Step | Time |
|---|---|---|
| Setup | 0. Before class | (done in advance) |
| Tour | 1. The pre-written files | 10 min |
| Contract | 2. `schemas.py` | 8 min |
| Agents | 3. Knowledge agent | 12 min |
| | 4. Triage agent | 12 min |
| | 5. Reply writer agent | 8 min |
| Rules | 6. `guards.py` | 15 min |
| Observability | 7. `tracing.py` | 6 min |
| Orchestration | 8. `supervisor.py` | 20 min |
| Entry point | 9. `main.py` | 6 min |
| Quality | 10. `eval.py` | 13 min |
| Buffer | questions, debugging | 10 min |
| Optional | 11. The web API and deploy | 10 min, or homework |

If you fall behind, paste the docstrings, then trim step 10 to the `--retrieval` mode and
leave the full eval as homework. Step 11 is optional by design.

## The finished layout

```
code/
  llm.py          pre-written  one chat() over gemini, groq, ollama; chat_json(); usage per agent
  pii.py          pre-written  redact() personal data to placeholders; restore() at the very end
  fetch_data.py   pre-written  builds data/kb.jsonl and data/tickets.jsonl from Bitext
  data.py         pre-written  load_kb(), load_tickets(), INTENTS, CATEGORY
  data/           committed    raw_sample.jsonl (174 real rows), kb.jsonl, tickets.jsonl, SOURCE.md
  schemas.py      typed        Ticket, Triage, Article, Draft, Handoff, CaseState, Outcome
  agents/
    knowledge.py  typed        KnowledgeAgent: TF-IDF retrieval, no model
    triage.py     typed        TriageAgent: intent, urgency, sentiment, confidence
    writer.py     typed        ReplyWriterAgent: a reply grounded in one article
  guards.py       typed        escalation_reason(), check_reply()
  tracing.py      typed        Tracer.span(): one JSON line per step
  supervisor.py   typed        Supervisor.next_step(), handle(), escalate()
  main.py         typed        run sample tickets or your own message
  eval.py         typed        intent accuracy, escalation recall and precision, cost
  service.py      optional     FastAPI: POST /ticket, GET /health
  Dockerfile, render.yaml, requirements.txt   optional deploy
```

---

## Step 0. Before class

Do this on the machine you will project, the day before.

```bash
cd modules/module-7-capstone/02-build-lab/code
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Create `.env` at the repository root (never committed):

```
PROVIDER=groq
GROQ_API_KEY=your-key
```

Then confirm the model and the data work:

```bash
python llm.py                       # prints a one-line greeting and the usage per agent
python fetch_data.py --sample       # 174 rows -> 27 articles in data/kb.jsonl, 147 tickets in data/tickets.jsonl
```

Keep a second copy of the finished `code/` folder open in another window. If a step
breaks and the fix is not obvious within a minute, copy that one file across and move on.

For the live session, start from an empty folder that contains only the pre-written
files and `data/`:

```bash
mkdir live && cp -r llm.py pii.py fetch_data.py data.py data requirements.txt live/ && cd live
mkdir agents
```

---

## Step 1. Tour the pre-written files (10 min)

**Goal.** Show the four files nobody types today, and why they are pre-written: they are
the same building blocks the class already wrote in Modules 4 to 6.

**Show, do not type.**

- `llm.py`: `chat(messages, agent=...)` returns text and logs tokens under the agent's
  name; `chat_json()` asks for JSON and returns `{}` when the reply does not parse;
  `usage_by_agent()` sums cost per agent. Retries on HTTP 429 are built in.
- `pii.py`: `redact(text)` returns the text with `<EMAIL_1>`, `<PHONE_1>`, `<PERSON_1>`
  placeholders plus a vault; `restore(text, vault)` puts the values back. Run it live:

  ```bash
  python -c "import pii; print(pii.redact(\"I'm Ana Ruiz, ana@example.com, cancel order 4812\")[0])"
  ```

  Expect: `I'm <PERSON_1>, <EMAIL_1>, cancel order 4812`. The order number stays: it is
  not personal data, and the agents need it.
- `fetch_data.py` and `data.py`: one help article per intent (27), and 147 labeled
  tickets that never overlap with the articles. Open one line of each file.

**Say.** "The model never sees a raw email or name. That one line in the supervisor is
the most important privacy decision in the whole system." "The tickets are synthetic,
written in the style of real ones: Bitext says so, and so do we."

---

## Step 2. `schemas.py`: the contract (8 min)

**Goal.** Agree on the data shapes before writing any agent, so each agent's input and
output is a named type, not a loose dict.

**Type** `schemas.py`:

`schemas.py`

```python
"""The data shapes every module agrees on.

Agents never pass loose dicts to each other: each one returns one of these, so the
supervisor (and a reader) always knows exactly what a step produced.
"""

from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class Ticket:
    id: str
    text: str                    # exactly what the customer wrote; never sent to a model


@dataclass
class Triage:
    intent: str                  # one of the 27 intents, or "unknown"
    urgency: str                 # low | normal | high
    sentiment: str               # calm | frustrated | angry
    confidence: float            # the model's own estimate, 0 to 1


@dataclass
class Article:
    intent: str
    title: str
    body: str
    score: float                 # how similar the ticket is to this article (cosine, 0 to 1)
    agrees: bool                 # was this also retrieval's best match on its own?


@dataclass
class Draft:
    text: str
    problems: list[str] = field(default_factory=list)   # filled by guards.check_reply


@dataclass
class Handoff:
    reason: str                  # why a human must take it
    summary: str                 # what the human needs to know, redacted


@dataclass
class CaseState:
    """Everything known about one ticket while the supervisor works on it."""
    ticket: Ticket
    redacted: str                # the ticket with personal data replaced by placeholders
    vault: dict                  # placeholder -> original value; stays in memory
    triage: Triage | None = None
    article: Article | None = None
    draft: Draft | None = None
    attempts: int = 0            # how many drafts the writer has produced
    trail: list[str] = field(default_factory=list)       # the steps taken, in order


@dataclass
class Outcome:
    ticket_id: str
    action: str                  # "reply" (sent to the customer) or "escalate" (to a human)
    reply: str = ""
    handoff: Handoff | None = None
    triage: Triage | None = None
    trail: list[str] = field(default_factory=list)
```

**Say.**

- "`CaseState` is the whiteboard the supervisor reads. Agents write one field each."
- "`Ticket.text` is raw and never leaves the process. Agents get `CaseState.redacted`."
- "`trail` is the cheapest observability there is: the list of steps, in order."

**Checkpoint.**

```bash
python -c "from schemas import CaseState, Ticket; print(CaseState(Ticket('t1', 'hi'), 'hi', {}))"
```

Expect a `CaseState(...)` line with `triage=None, article=None, draft=None, attempts=0, trail=[]`.

---

## Step 3. The knowledge agent (12 min)

**Goal.** Find the help article for a ticket, with no model call. It is the desk's second
opinion: retrieval, independent of the model.

**Type** `agents/__init__.py` (one line) and `agents/knowledge.py`:

`agents/__init__.py`

```python
"""The three agents of the support desk: triage, knowledge, and reply writer."""
```

`agents/knowledge.py`

```python
"""Knowledge agent: find the help article that answers the ticket. No model call.

It gives the supervisor a second, independent opinion. Triage (a model) says what
the customer wants; retrieval (TF-IDF over the 27 articles) says which article the
words of the ticket are closest to. Tickets are short and full of typos ("cancel
oorder", "refudn"), so we compare character n-grams (3 to 5 letters inside each
word) instead of whole words: a typo still shares most of its n-grams.

When the two opinions agree, we trust the article. When they disagree, someone is
wrong, and the supervisor may hand the ticket to a human.
"""

from __future__ import annotations
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from data import load_kb
from schemas import Article


class KnowledgeAgent:
    name = "knowledge"

    def __init__(self):
        self.kb = load_kb()
        # the title and intent name are repeated so they outweigh the long, chatty body
        docs = [f"{a['title']}. {a['intent'].replace('_', ' ')}. " * 3 + a["body"] for a in self.kb]
        self.vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True)
        self.matrix = self.vectorizer.fit_transform(docs)
        self.index = {a["intent"]: i for i, a in enumerate(self.kb)}

    def scores(self, text: str):
        """Cosine similarity between the ticket and every article."""
        return cosine_similarity(self.vectorizer.transform([text]), self.matrix)[0]

    def run(self, text: str, intent: str) -> Article:
        sims = self.scores(text)
        best = int(sims.argmax())
        chosen = self.index.get(intent, best)       # trust triage's intent when it names a real one
        a = self.kb[chosen]
        return Article(intent=a["intent"], title=a["title"], body=a["body"],
                       score=float(sims[chosen]), agrees=(chosen == best))
```

**Say.**

- "Why character n-grams? Look at the tickets: `cancel oorder`, `rpoduct`, `acconut`.
  Whole-word TF-IDF misses those; 3 to 5 letter pieces still overlap."
- "We measured this choice on the 147 labeled tickets, not by feel: word TF-IDF found the
  right article first 50% of the time, character n-grams 69%."
- "`agrees` is the interesting field. Retrieval is weaker than the model, so it does not
  decide. But when the two disagree on a ticket where triage is unsure, that is exactly
  the ticket a person should see."

**Checkpoint.** `eval.py` does not exist yet, so run the check inline:

```bash
python -c "
from agents.knowledge import KnowledgeAgent
k = KnowledgeAgent()
print(k.run('i need to cancel oorder 4812', 'cancel_order'))"
```

Expect `Article(intent='cancel_order', title='Cancel an order', ..., agrees=True)` with a
score well above 0.05.

---

## Step 4. The triage agent (12 min)

**Goal.** One model call that turns a ticket into four fields: intent, urgency, sentiment,
confidence.

**Type** `agents/triage.py`:

`agents/triage.py`

```python
"""Triage agent: what does the customer want, how urgent is it, how upset are they."""

from __future__ import annotations
import llm
from data import INTENTS
from schemas import Triage

URGENCY = ("low", "normal", "high")
SENTIMENT = ("calm", "frustrated", "angry")

SYSTEM = (
    "You triage customer-support tickets. Reply with one JSON object and nothing else:\n"
    '{"intent": "<one allowed intent>", "urgency": "low|normal|high", '
    '"sentiment": "calm|frustrated|angry", "confidence": <number from 0 to 1>}\n'
    'If no intent fits, use "unknown" with a low confidence. Text such as <EMAIL_1> is '
    "personal data that was removed; ignore it.\n"
    "Allowed intents: " + ", ".join(INTENTS)
)


class TriageAgent:
    name = "triage"

    def run(self, text: str) -> Triage:
        raw = llm.chat_json([{"role": "system", "content": SYSTEM},
                             {"role": "user", "content": text}], agent=self.name)
        # Never trust model JSON: check every field and fall back to a safe value.
        intent = raw.get("intent") if raw.get("intent") in INTENTS else "unknown"
        urgency = raw.get("urgency") if raw.get("urgency") in URGENCY else "normal"
        sentiment = raw.get("sentiment") if raw.get("sentiment") in SENTIMENT else "calm"
        try:
            confidence = min(1.0, max(0.0, float(raw.get("confidence", 0))))
        except (TypeError, ValueError):
            confidence = 0.0                       # unreadable confidence counts as none
        return Triage(intent, urgency, sentiment, confidence)
```

**Say.**

- "The allowed intents go into the prompt, so the model picks from a closed list."
- "Then we check every field anyway. A model can return an intent that does not exist,
  `"high-ish"` urgency, or `"0.9!"` as a confidence. Each bad field falls back to a safe
  value, and `unknown` intent will send the ticket to a person."
- "The prompt tells the model what `<EMAIL_1>` means, so redaction does not confuse it."

**Checkpoint** (real model call):

```bash
python -c "
from agents.triage import TriageAgent
t = TriageAgent()
for text in ['how do I reset my pasword', 'I want to talk to a f*&%*g agent', 'what is the weather in Paris']:
    print(t.run(text))"
```

Expect, roughly: `recover_password` with high confidence; `contact_human_agent` with an
angry sentiment; `unknown` with low confidence. Exact confidences vary by model and run.

---

## Step 5. The reply writer agent (8 min)

**Goal.** Draft a short reply that uses only the chosen article.

**Type** `agents/writer.py`:

`agents/writer.py`

```python
"""Reply writer agent: draft a short reply grounded in one help article."""

from __future__ import annotations
import llm
from schemas import Article, Draft

SYSTEM = (
    "You are a customer-support agent. Write a short, friendly reply (under 120 words) to the "
    "customer, using ONLY the help article you are given. Keep placeholders such as "
    "{{Order Number}} and <PERSON_1> exactly as written. Never invent order numbers, dates, "
    "prices, or policies, and never promise anything the article does not describe."
)


class ReplyWriterAgent:
    name = "writer"

    def run(self, text: str, article: Article, feedback: list[str] | None = None) -> Draft:
        prompt = f"Customer message:\n{text}\n\nHelp article: {article.title}\n{article.body}"
        if feedback:                                 # a second attempt: say exactly what to fix
            prompt += "\n\nYour previous draft was rejected. Fix these problems:\n- " + "\n- ".join(feedback)
        reply = llm.chat([{"role": "system", "content": SYSTEM},
                          {"role": "user", "content": prompt}], agent=self.name)
        return Draft(text=reply.strip())
```

**Say.**

- "The writer sees the redacted ticket and one article, nothing else. Less context means
  fewer chances to invent."
- "`feedback` is how the retry works: the second draft is told exactly which check failed."
- "The writer does not check its own work. That is the next file's job, in plain code."

**Checkpoint** (real model call):

```bash
python -c "
from agents.knowledge import KnowledgeAgent
from agents.writer import ReplyWriterAgent
a = KnowledgeAgent().run('how do I reset my password', 'recover_password')
print(ReplyWriterAgent().run('how do I reset my password', a).text)"
```

Expect a short reply describing the reset steps, keeping placeholders such as
`{{Login Page URL}}` as written.

---

## Step 6. `guards.py`: the rules (15 min)

**Goal.** Two functions: when a person must take over, and whether a draft is safe to
send. Rules in code, not in prompts.

**Type** `guards.py`:

`guards.py`

```python
"""The desk's rules, in plain code: when a human must take over, and what a reply may say.

Rules live here, not in prompts, so they are cheap, testable, and never ignored.
"""

from __future__ import annotations
import re

import pii
from schemas import Article, CaseState

ALWAYS_HUMAN = {"complaint", "payment_issue", "contact_human_agent"}   # company policy
MIN_CONFIDENCE = 0.6        # below this, triage is a guess
TRUST_CONFIDENCE = 0.8      # at or above this, triage outweighs a retrieval disagreement
MIN_SCORE = 0.05            # the chosen article barely matches the ticket's words
MAX_ATTEMPTS = 2            # drafts before a human takes over
MAX_WORDS = 150

NUMBER = re.compile(r"\d{3,}")                    # order numbers, amounts, phone fragments
PLACEHOLDER = re.compile(r"\{\{[^}]+\}\}")         # {{Order Number}} and friends


def escalation_reason(state: CaseState) -> str | None:
    """Why this ticket must go to a human now, or None to keep going."""
    t, a, d = state.triage, state.article, state.draft
    if t is not None:
        if t.intent == "unknown":
            return "intent unclear"
        if t.intent in ALWAYS_HUMAN:
            return f"policy: {t.intent} always goes to a person"
        if t.sentiment == "angry":
            return "customer is angry"
        if t.urgency == "high":
            return "high urgency"
        if t.confidence < MIN_CONFIDENCE:
            return f"low triage confidence ({t.confidence:.2f})"
    if a is not None:
        if a.score < MIN_SCORE:
            return f"help article barely matches the ticket (score {a.score:.2f})"
        if not a.agrees and t.confidence < TRUST_CONFIDENCE:
            return "triage and knowledge disagree"
    if d is not None and d.problems and state.attempts >= MAX_ATTEMPTS:
        return "reply failed the checks twice: " + "; ".join(d.problems)
    return None


def check_reply(reply: str, article: Article, ticket_text: str) -> list[str]:
    """Problems that make a draft unsafe to send. An empty list means send it."""
    if not reply:
        return ["empty reply"]
    problems = []
    if len(reply.split()) > MAX_WORDS:
        problems.append(f"too long ({len(reply.split())} words; limit {MAX_WORDS})")
    source = article.body + " " + ticket_text
    invented = set(NUMBER.findall(reply)) - set(NUMBER.findall(source))
    if invented:
        problems.append(f"numbers that are not in the article or ticket: {sorted(invented)}")
    made_up = set(PLACEHOLDER.findall(reply)) - set(PLACEHOLDER.findall(source))
    if made_up:
        problems.append(f"placeholders that are not in the article: {sorted(made_up)}")
    if pii.detect(reply):                          # the writer never saw real personal data
        problems.append("reply contains personal data the writer could not have known")
    return problems
```

**Say.**

- "Read `escalation_reason` top to bottom: it is the escalation policy, in priority order.
  A product manager can review this file without reading any prompt."
- "`ALWAYS_HUMAN` is a business decision: complaints, payment problems, and anyone asking
  for a person. The eval uses the same set as its definition of 'needs a person'."
- "`check_reply` catches the two classic failures: invented numbers (an order number or a
  refund amount the article never mentioned) and invented placeholders."
- "Every threshold is a named constant at the top. `MIN_SCORE` came from the data: about
  5% of tickets score below 0.05 against their own article."

**Checkpoint** (no model):

```bash
python -c "
import guards
from schemas import Article, CaseState, Ticket, Triage
s = CaseState(Ticket('t', 'x'), 'x', {})
s.triage = Triage('complaint', 'normal', 'calm', 0.9); print(guards.escalation_reason(s))
s.triage = Triage('track_order', 'normal', 'calm', 0.9); print(guards.escalation_reason(s))
a = Article('track_order', 'Track an order', 'Use {{Order Number}} on the tracking page.', 0.3, True)
print(guards.check_reply('Your order 99999 ships today.', a, 'where is my order'))"
```

Expect: `policy: complaint always goes to a person`, then `None`, then a list naming the
invented number `99999`.

---

## Step 7. `tracing.py`: one line per step (6 min)

**Goal.** Record what each step cost, in time, model calls, and tokens, without recording
any ticket text.

**Type** `tracing.py`:

`tracing.py`

```python
"""A small tracer: one JSON line per step, with its time, model calls, and tokens.

Spans record step names and numbers only, never ticket text, so a trace file holds
no personal data by construction. Class 6.6 (Observability and tracing) shows the
full version with nested trees and LangSmith.
"""

from __future__ import annotations
import contextlib
import json
import pathlib
import time

import llm


class Tracer:
    def __init__(self, path: str | None = "traces/desk.jsonl"):
        self.path = pathlib.Path(path) if path else None
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    @contextlib.contextmanager
    def span(self, name: str, **attrs):
        start, calls_before = time.perf_counter(), len(llm.USAGE)
        record = {"span": name, **attrs, "error": None}
        try:
            yield record
        except Exception as e:
            record["error"] = f"{type(e).__name__}: {e}"
            raise
        finally:
            new = llm.USAGE[calls_before:]            # the model calls made inside this span
            record["ms"] = round((time.perf_counter() - start) * 1000, 1)
            record["calls"] = len(new)
            record["tokens"] = sum(u["prompt_tokens"] + u["completion_tokens"] for u in new)
            if self.path:
                with open(self.path, "a") as f:
                    f.write(json.dumps(record) + "\n")
```

**Say.**

- "It counts model calls by looking at how many usage records `llm.py` added while the
  span was open. No changes to the agents."
- "There is no ticket text in a span, on purpose. Traces get shared; personal data must not."

**Checkpoint.**

```bash
python -c "
from tracing import Tracer
t = Tracer('traces/check.jsonl')
with t.span('demo', ticket_id='t1'): pass
print(open('traces/check.jsonl').read())"
```

Expect one JSON line with `"span": "demo"`, `"calls": 0`, and a small `"ms"`.

---

## Step 8. `supervisor.py`: wiring the team (20 min)

**Goal.** The loop that runs the agents: redact, then ask `next_step`, run that step, and
repeat until the ticket is answered or handed to a person.

**Type** `supervisor.py`:

`supervisor.py`

```python
"""The supervisor: a plain Python loop that decides which agent works next.

    redact -> triage -> (escalate?) -> knowledge -> (escalate?) -> write -> check
                                                                   ^        |
                                                                   +-retry--+   at most MAX_ATTEMPTS
           -> send the reply, or hand the ticket to a human

Routing is rules, not a model: every decision is cheap, testable, and explainable.
The agents do the language work; the supervisor only reads the state and picks a step.
"""

from __future__ import annotations
import guards
import pii
from agents.knowledge import KnowledgeAgent
from agents.triage import TriageAgent
from agents.writer import ReplyWriterAgent
from data import CATEGORY
from schemas import CaseState, Handoff, Outcome, Ticket
from tracing import Tracer

MAX_STEPS = 8               # a hard stop, whatever the rules say


class Supervisor:
    def __init__(self, tracer: Tracer | None = None):
        self.triage = TriageAgent()
        self.knowledge = KnowledgeAgent()
        self.writer = ReplyWriterAgent()
        self.tracer = tracer or Tracer(None)

    def next_step(self, s: CaseState) -> str:
        """Read the state, return the next step's name."""
        if s.triage is None:
            return "triage"
        if guards.escalation_reason(s):
            return "escalate"
        if s.article is None:
            return "knowledge"
        if s.draft is None or s.draft.problems:
            return "write"
        return "send"

    def handle(self, ticket: Ticket) -> Outcome:
        redacted, vault, _ = pii.redact(ticket.text)        # before any model sees the text
        s = CaseState(ticket=ticket, redacted=redacted, vault=vault)
        with self.tracer.span("ticket", ticket_id=ticket.id) as root:
            for _ in range(MAX_STEPS):
                step = self.next_step(s)
                s.trail.append(step)
                with self.tracer.span(step, ticket_id=ticket.id):
                    if step == "triage":
                        s.triage = self.triage.run(s.redacted)
                    elif step == "knowledge":
                        s.article = self.knowledge.run(s.redacted, s.triage.intent)
                    elif step == "write":
                        feedback = s.draft.problems if s.draft else None
                        s.draft = self.writer.run(s.redacted, s.article, feedback)
                        s.draft.problems = guards.check_reply(s.draft.text, s.article, s.redacted)
                        s.attempts += 1
                    elif step == "escalate":
                        outcome = self.escalate(s, guards.escalation_reason(s))
                        break
                    elif step == "send":
                        outcome = Outcome(ticket.id, "reply", reply=pii.restore(s.draft.text, s.vault),
                                          triage=s.triage, trail=s.trail)
                        break
            else:
                outcome = self.escalate(s, f"step limit of {MAX_STEPS} reached")
            root["action"] = outcome.action
        return outcome

    def escalate(self, s: CaseState, reason: str) -> Outcome:
        """Hand the ticket to a person with a summary they can act on at once."""
        t = s.triage
        about = f"{t.intent} ({CATEGORY.get(t.intent, '?')}), urgency {t.urgency}, " \
                f"sentiment {t.sentiment}" if t else "not triaged"
        summary = f"{about}. Customer wrote: {s.redacted[:300]}"
        return Outcome(s.ticket.id, "escalate", handoff=Handoff(reason, summary), triage=t, trail=s.trail)
```

**Say.**

- "`next_step` reads the state and returns a name. It never calls a model. That is why we
  can test routing without spending a token."
- "Redaction is the first line of `handle`. Everything after it sees placeholders."
- "`pii.restore` runs only on the final reply to the customer who wrote the ticket."
- "`MAX_STEPS` is the seatbelt: even if a bug in the rules made the loop spin, it stops
  and escalates. The `for ... else` runs the `else` only when the loop never hit `break`."
- "The handoff summary is built in code from triage, so the person gets the intent,
  urgency, and the redacted message without another model call."

**Checkpoint** (real model calls):

```bash
python -c "
from schemas import Ticket
from supervisor import Supervisor
d = Supervisor()
for text in ['I need to cancel purchase 4812', 'I am furious, cancel my order right now']:
    o = d.handle(Ticket('t', text)); print(o.action, o.trail, o.handoff.reason if o.handoff else '')"
```

Expect the first to end with `reply ['triage', 'knowledge', 'write', 'send']` and the
second with `escalate ['triage', 'escalate'] customer is angry`.

---

## Step 9. `main.py`: the entry point (6 min)

**Goal.** Run a spread of sample tickets, or one message typed at the command line.

**Type** `main.py`:

`main.py`

```python
"""Run the support desk on sample tickets, or on a message you type.

    python main.py                       # 6 tickets spread across the sample
    python main.py --n 12
    python main.py --ticket "I'm Ana Ruiz, ana@example.com. How do I cancel order 4812?"
"""

from __future__ import annotations
import argparse
import logging

import llm
from data import load_tickets
from schemas import Outcome, Ticket
from supervisor import Supervisor
from tracing import Tracer


def show(text: str, o: Outcome):
    print("=" * 78)
    print(f"[{o.ticket_id}] {text}")
    if o.triage:
        t = o.triage
        print(f"  triage   : {t.intent} | urgency {t.urgency} | {t.sentiment} | confidence {t.confidence:.2f}")
    print(f"  steps    : {' > '.join(o.trail)}")
    if o.action == "reply":
        print(f"  REPLY    : {o.reply}")
    else:
        print(f"  ESCALATE : {o.handoff.reason}")
        print(f"  handoff  : {o.handoff.summary}")


def main():
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s | %(name)s | %(message)s")
    ap = argparse.ArgumentParser(description="Customer Support Desk")
    ap.add_argument("--ticket", help="your own customer message")
    ap.add_argument("--n", type=int, default=6, help="how many sample tickets to run")
    args = ap.parse_args()

    if args.ticket:
        tickets = [Ticket("you", args.ticket)]
    else:
        sample = load_tickets()
        step = max(1, len(sample) // args.n)
        tickets = [Ticket(t["id"], t["text"]) for t in sample[::step][:args.n]]

    desk = Supervisor(Tracer("traces/desk.jsonl"))
    for ticket in tickets:
        show(ticket.text, desk.handle(ticket))

    print("=" * 78)
    print("model calls per agent")
    for agent, u in llm.usage_by_agent().items():
        print(f"  {agent:<10} calls={u['calls']:<3} tokens in={u['prompt']:<6} out={u['completion']}")
    print("trace: traces/desk.jsonl")


if __name__ == "__main__":
    main()
```

**Checkpoint.**

```bash
python main.py --n 6
python main.py --ticket "I'm Ana Ruiz, ana@example.com. How do I cancel order 4812?"
```

Expect six tickets, each with its triage line, its steps, and either a REPLY or an
ESCALATE with a reason; then the calls and tokens per agent. For the second command, the
reply greets Ana by name (restored at the end) while `traces/desk.jsonl` contains no name
or email. Open the trace file to show it.

**Say.** "Look at the per-agent cost: triage runs on every ticket, the writer only on the
ones we answer. Escalating early is also the cheapest path."

---

## Step 10. `eval.py`: is it any good? (13 min)

**Goal.** Measure the desk on the 147 labeled tickets: did triage get the intent, did the
right tickets reach a person, and what did it cost.

**Type** `eval.py`:

`eval.py`

```python
"""Evaluate the desk on the labeled tickets.

    python eval.py --retrieval        # knowledge agent alone: no model, runs in seconds
    python eval.py --limit 30         # the full desk on 30 tickets (real model calls)
    python eval.py                    # every ticket

What each number means:
  intent accuracy      triage picked the labeled intent, with a 95% Wilson interval
  escalation recall    of the tickets policy says need a person, the share escalated
                       (a miss means an upset or sensitive customer got a bot reply)
  escalation precision of the tickets escalated, the share that really needed a person
                       (low precision wastes human time)
  auto-resolved        share of tickets answered without a person; every one passed the checks
  cost                 model calls and tokens per ticket

Policy (the gold label): a ticket needs a person when its intent is in
guards.ALWAYS_HUMAN, or when Bitext tagged it as offensive language (flag "W").
"""

from __future__ import annotations
import argparse
import math
from collections import Counter

import guards
import llm
from agents.knowledge import KnowledgeAgent
from data import load_tickets
from schemas import Ticket
from supervisor import Supervisor
from tracing import Tracer


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% interval for a rate measured on a small sample."""
    if n == 0:
        return 0.0, 0.0
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, centre - half), min(1.0, centre + half)


def needs_person(t: dict) -> bool:
    return t["intent"] in guards.ALWAYS_HUMAN or "W" in t["flags"]


def retrieval_only(tickets: list[dict]):
    k = KnowledgeAgent()
    top1 = top3 = 0
    for t in tickets:
        order = k.scores(t["text"]).argsort()[::-1]
        gold = k.index[t["intent"]]
        top1 += order[0] == gold
        top3 += gold in order[:3]
    n = len(tickets)
    print(f"retrieval alone on {n} tickets: top-1 {top1}/{n} = {top1 / n:.0%}, top-3 {top3}/{n} = {top3 / n:.0%}")
    print("so retrieval is a second opinion, not the decider: triage names the intent")


def full_desk(tickets: list[dict]):
    desk = Supervisor(Tracer("traces/eval.jsonl"))
    llm.reset_usage()
    rows = [(t, desk.handle(Ticket(t["id"], t["text"]))) for t in tickets]
    n = len(rows)

    correct = sum(o.triage is not None and o.triage.intent == t["intent"] for t, o in rows)
    lo, hi = wilson(correct, n)
    print(f"intent accuracy      {correct}/{n} = {correct / n:.0%}  (95% interval {lo:.0%} to {hi:.0%})")

    tp = sum(needs_person(t) and o.action == "escalate" for t, o in rows)
    fn = sum(needs_person(t) and o.action == "reply" for t, o in rows)
    fp = sum(not needs_person(t) and o.action == "escalate" for t, o in rows)
    print(f"escalation recall    {tp}/{tp + fn}" + (f" = {tp / (tp + fn):.0%}" if tp + fn else ""))
    print(f"escalation precision {tp}/{tp + fp}" + (f" = {tp / (tp + fp):.0%}" if tp + fp else ""))

    replied = sum(o.action == "reply" for _, o in rows)
    print(f"auto-resolved        {replied}/{n} = {replied / n:.0%}")
    reasons = Counter(o.handoff.reason.split(":")[0].split(" (")[0] for _, o in rows if o.handoff)
    for reason, count in reasons.most_common():
        print(f"  escalated: {reason:<45} {count}")

    misses = Counter((t["intent"], o.triage.intent) for t, o in rows if o.triage and o.triage.intent != t["intent"])
    for (gold, got), count in misses.most_common(5):
        print(f"  confused: {gold} -> {got}  x{count}")

    calls = len(llm.USAGE)
    tokens = sum(u["prompt_tokens"] + u["completion_tokens"] for u in llm.USAGE)
    print(f"cost per ticket      {calls / n:.1f} model calls, {tokens / n:.0f} tokens")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--retrieval", action="store_true", help="score the knowledge agent alone (no model)")
    ap.add_argument("--limit", type=int, default=0, help="only the first N tickets")
    args = ap.parse_args()
    tickets = load_tickets()
    if args.limit:
        tickets = tickets[:args.limit]
    retrieval_only(tickets) if args.retrieval else full_desk(tickets)


if __name__ == "__main__":
    main()
```

**Say.**

- "Escalation recall is the number that protects customers: a miss means an angry or
  sensitive ticket got a bot reply. Precision protects the support team's time."
- "The Wilson interval is there because 147 tickets is a small sample; class 6.7
  (Evaluating agentic systems) covered why the simple interval misleads here."
- "Run retrieval alone first. It costs nothing and tells you how much to trust the
  knowledge agent's second opinion."

**Checkpoint.**

```bash
python eval.py --retrieval          # no model: top-1 102/147 = 69%, top-3 129/147 = 88% on the committed sample
python eval.py --limit 30           # real model calls; about 2 calls per ticket
```

The retrieval numbers are measured and deterministic on the committed sample. The full
desk's numbers depend on the model and the run, so read them live with the class, and
discuss one confused intent pair and one escalation reason.

---

## Step 11 (optional). The web API and a free deploy (10 min, or homework)

**Goal.** The same desk behind an HTTP endpoint, deployable to Render's free tier exactly
as in class 6.8 (Deployment).

**Show** `service.py`, `Dockerfile`, and `render.yaml` (pre-written; do not type them).

`service.py`

```python
"""Optional: the support desk as a small web API (class 6.8, Deployment).

    POST /ticket   {"text": "..."}  ->  the outcome: a reply, or a handoff to a person
    GET  /health                    ->  liveness for the host; no model call

    uvicorn service:app --reload
"""

from __future__ import annotations
import uuid

from fastapi import FastAPI
from pydantic import BaseModel, Field

from schemas import Ticket
from supervisor import Supervisor
from tracing import Tracer

app = FastAPI(title="Customer Support Desk")
desk = Supervisor(Tracer("/tmp/traces/service.jsonl"))


class TicketIn(BaseModel):
    text: str = Field(min_length=1, max_length=2000)


class OutcomeOut(BaseModel):
    ticket_id: str
    action: str                       # reply | escalate
    reply: str
    escalation_reason: str
    intent: str
    steps: list[str]


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/ticket", response_model=OutcomeOut)
def ticket(body: TicketIn) -> OutcomeOut:
    o = desk.handle(Ticket(uuid.uuid4().hex[:8], body.text))
    return OutcomeOut(ticket_id=o.ticket_id, action=o.action, reply=o.reply,
                      escalation_reason=o.handoff.reason if o.handoff else "",
                      intent=o.triage.intent if o.triage else "", steps=o.trail)
```

**Checkpoint.**

```bash
uvicorn service:app --reload
curl -s localhost:8000/health
curl -s localhost:8000/ticket -H 'content-type: application/json' -d '{"text": "how do I reset my password"}'
```

Expect `{"status":"ok"}`, then a JSON outcome with `"action": "reply"` and the steps.
Deploying follows the 6.8 steps: push, New > Blueprint in Render, paste `GROQ_API_KEY`.

---

## When something breaks

| Symptom | Likely cause | Fix |
|---|---|---|
| `data/kb.jsonl not found` | data not built | `python fetch_data.py --sample` |
| `GROQ_API_KEY not set` | `.env` missing or in the wrong folder | put `.env` at the repository root |
| Every ticket escalates with `intent unclear` | triage reply is not JSON | print the raw reply in `chat_json`; try another model |
| `rate limited (429); retry ...` warnings | free-tier requests per minute | wait, or use `--n 3`; retries are automatic |
| `ModuleNotFoundError: agents` | running from the wrong folder | run from `code/`, where `agents/` lives |
| Replies rejected twice | writer adds numbers or placeholders | read the reason in the handoff; tighten the writer prompt |
