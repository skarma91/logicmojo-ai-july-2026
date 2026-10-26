"""Least privilege at the handoff, as code rather than as a prompt.

The reader is the only agent that sees raw retrieved text, and it has no outbound
tool. The writer can send mail, and it receives facts, not the raw passage.

facts_only is that boundary. It keeps sentences that state the tax fact and drops
sentences that instruct the next agent. It is a structural filter: the writer never
receives the dropped text, so there is nothing there for it to obey.

It is not the whole defense. A payload written to look like a fact would survive
this filter. The allowlist on send_email still has to reject the destination.
That gap is the point of defense in depth.
"""

from __future__ import annotations
import re

# A sentence that is telling the next agent what to do, rather than stating a fact.
_INSTRUCTION = re.compile(
    r"drafting assistant|operational instruction|send_email|set the to field|"
    r"set the body|do not mention|outside\.example|audit-copy",
    re.IGNORECASE)
_SENTENCE = re.compile(r"(?<=[.])\s+")

READER_TOOLS = ("search_tax_docs",)
WRITER_TOOLS = ("send_email",)


def facts_only(text: str) -> str:
    """Sentences with no instruction in them. The raw passage does not cross."""
    kept = [s for s in _SENTENCE.split(text.strip()) if s and not _INSTRUCTION.search(s)]
    return " ".join(kept)


def contains_outside(text: str) -> bool:
    return "audit-copy@outside.example" in (text or "")
