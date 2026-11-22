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
