"""A small tracer: one span per agent turn, model call, and tool call, in a tree.

Every span records its name, kind, parent, start and end time, redacted inputs and
outputs, token usage and cost (model spans), and any error. Spans are written to a
local JSONL file (one span per line) ALWAYS, so a run can be analyzed offline with
analyze.py. When LangSmith tracing is switched on (LANGSMITH_TRACING=true plus an API
key), every span is ALSO sent to LangSmith as a run, nested the same way, and the
LangGraph runs that LangSmith records on its own appear alongside them.

Privacy (class 6.4, Guardrails, routing, and compliance): a trace is a copy of your
data sent to a third party. Inputs and outputs pass through pii.redact() BEFORE they
are recorded anywhere, locally or remotely, and long values are truncated.

Kinds follow LangSmith's run types: chain (a whole run or an agent turn), llm, tool,
retriever.
"""

from __future__ import annotations
import contextlib
import contextvars
import json
import os
import pathlib
import threading
import time
import uuid
from dataclasses import dataclass, field, asdict

import pii

MAX_CHARS = 600                                   # truncate long inputs and outputs
_CURRENT = contextvars.ContextVar("current_span", default=None)


def langsmith_enabled() -> bool:
    return (os.getenv("LANGSMITH_TRACING", "").lower() == "true"
            and bool(os.getenv("LANGSMITH_API_KEY")))


def _clean(value) -> str:
    """Redact PII, then truncate. Applied to everything a span records."""
    text = value if isinstance(value, str) else json.dumps(value, default=str)
    redacted = pii.redact(text)[0]
    return redacted if len(redacted) <= MAX_CHARS else redacted[:MAX_CHARS] + " ...[truncated]"


@dataclass
class Span:
    span_id: str
    trace_id: str
    parent_id: str | None
    name: str
    kind: str
    start: float
    end: float = 0.0
    inputs: str = ""
    outputs: str = ""
    tokens_in: int = 0
    tokens_out: int = 0
    cached: int = 0
    cost: float = 0.0
    error: str = ""
    metadata: dict = field(default_factory=dict)
    langsmith_run_id: str = ""


class Tracer:
    def __init__(self, path: pathlib.Path, project: str | None = None):
        self.path = pathlib.Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.project = project or os.getenv("LANGSMITH_PROJECT")
        self.remote = langsmith_enabled()
        self._lock = threading.Lock()
        self.roots: list[Span] = []

    @contextlib.contextmanager
    def span(self, name: str, kind: str = "chain", inputs=None, metadata: dict | None = None):
        """Open a span as a child of the current one. Yields the Span, so the caller
        can set outputs, tokens, or metadata before it closes."""
        parent = _CURRENT.get()
        s = Span(span_id=uuid.uuid4().hex[:12], trace_id=parent.trace_id if parent else uuid.uuid4().hex[:12],
                 parent_id=parent.span_id if parent else None, name=name, kind=kind,
                 start=time.time(), inputs=_clean(inputs) if inputs is not None else "",
                 metadata=dict(metadata or {}))
        if parent is None:
            self.roots.append(s)
        token = _CURRENT.set(s)
        remote_cm = None
        if self.remote:
            from langsmith import run_helpers           # imported only when tracing remotely
            remote_cm = run_helpers.trace(name, run_type=kind, inputs={"input": s.inputs},
                                          metadata=s.metadata, project_name=self.project)
            rt = remote_cm.__enter__()
            s.langsmith_run_id = str(rt.id)
        try:
            yield s
        except Exception as e:
            s.error = f"{type(e).__name__}: {e}"
            raise
        finally:
            s.end = time.time()
            if remote_cm is not None:
                rt.end(outputs={"output": s.outputs, "tokens_in": s.tokens_in,
                                "tokens_out": s.tokens_out, "cost": s.cost},
                       error=s.error or None)
                remote_cm.__exit__(None, None, None)
            _CURRENT.reset(token)
            self._write(s)

    def set_output(self, s: Span, value):
        s.outputs = _clean(value)

    def _write(self, s: Span):
        with self._lock, open(self.path, "a") as f:
            f.write(json.dumps(asdict(s)) + "\n")


def current() -> Span | None:
    return _CURRENT.get()
