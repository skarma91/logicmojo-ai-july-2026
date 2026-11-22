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
