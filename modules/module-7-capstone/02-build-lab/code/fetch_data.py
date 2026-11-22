"""Build the support desk's data files from the Bitext customer-support dataset (pre-written).

Source: "Bitext - Customer Service Tagged Training Dataset for LLM-based Virtual
Assistants" on Hugging Face (bitext/Bitext-customer-support-llm-chatbot-training-dataset),
licensed CDLA-Sharing-1.0. Bitext describes it as HYBRID SYNTHETIC: the customer
messages are generated in the style of real tickets, not copied from real customers.
Each row has: flags (language-variation tags), instruction (the customer message),
category, intent (27 intents), and response (an example agent reply).

What this script writes into data/:

  kb.jsonl       27 help articles, one per intent. Each is the most detailed example
                 response for that intent, with placeholders such as {{Order Number}}
                 kept as written. This is the desk's help centre.
  tickets.jsonl  the remaining rows as labeled tickets: id, text, intent, category,
                 flags. The desk only ever sees `text`; eval.py uses the labels.

A row used as an article is never also a ticket, so the eval never grades the desk
on a message whose answer it was given.

    python fetch_data.py               # download the full CSV, keep --per-intent rows per intent
    python fetch_data.py --sample      # rebuild from the committed data/raw_sample.jsonl (offline)
"""

from __future__ import annotations
import argparse
import csv
import io
import json
import pathlib
import random
import urllib.request

DATA = pathlib.Path(__file__).parent / "data"
CSV_URL = ("https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset/"
           "resolve/main/Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv")

TITLES = {
    "cancel_order": "Cancel an order", "change_order": "Change items in an order",
    "change_shipping_address": "Change a shipping address", "check_cancellation_fee": "Cancellation fees",
    "check_invoice": "View an invoice", "check_payment_methods": "Accepted payment methods",
    "check_refund_policy": "Refund policy", "complaint": "Make a complaint",
    "contact_customer_service": "Contact customer service", "contact_human_agent": "Talk to a person",
    "create_account": "Create an account", "delete_account": "Close an account",
    "delivery_options": "Delivery options", "delivery_period": "Delivery times",
    "edit_account": "Edit account details", "get_invoice": "Download an invoice",
    "get_refund": "Request a refund", "newsletter_subscription": "Newsletter subscription",
    "payment_issue": "Report a payment problem", "place_order": "Place an order",
    "recover_password": "Reset a password", "registration_problems": "Sign-up problems",
    "review": "Leave a review", "set_up_shipping_address": "Add a shipping address",
    "switch_account": "Switch account type", "track_order": "Track an order",
    "track_refund": "Track a refund",
}


def clean(text: str) -> str:
    return " ".join(text.split())


def build(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    """Split rows into 27 articles (one per intent) and labeled tickets."""
    by_intent: dict[str, list[dict]] = {}
    for r in rows:
        by_intent.setdefault(r["intent"], []).append(r)
    articles, tickets = [], []
    for intent in sorted(by_intent):
        group = by_intent[intent]
        source = max(group, key=lambda r: len(r["response"]))   # the most detailed answer
        articles.append({"intent": intent, "category": source["category"],
                         "title": TITLES.get(intent, intent.replace("_", " ").capitalize()),
                         "body": clean(source["response"])})
        for r in group:
            if r is not source:
                tickets.append({"text": clean(r["instruction"]), "intent": intent,
                                "category": r["category"], "flags": r["flags"]})
    random.Random(7).shuffle(tickets)                          # mix intents; reproducible
    for i, t in enumerate(tickets, 1):
        t["id"] = f"t{i:03d}"
    return articles, tickets


def download(per_intent: int) -> list[dict]:
    print("downloading the Bitext CSV (about 19 MB) ...")
    with urllib.request.urlopen(CSV_URL, timeout=120) as resp:
        text = resp.read().decode("utf-8")
    rows = list(csv.DictReader(io.StringIO(text)))
    rng = random.Random(13)
    rng.shuffle(rows)
    kept, count = [], {}
    for r in rows:
        if count.get(r["intent"], 0) < per_intent:
            kept.append(r)
            count[r["intent"]] = count.get(r["intent"], 0) + 1
    return kept


def write_jsonl(path: pathlib.Path, items: list[dict]):
    path.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in items))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", action="store_true", help="use the committed data/raw_sample.jsonl")
    ap.add_argument("--per-intent", type=int, default=40, help="rows kept per intent when downloading")
    args = ap.parse_args()
    if args.sample:
        rows = [json.loads(l) for l in (DATA / "raw_sample.jsonl").read_text().splitlines() if l.strip()]
    else:
        rows = download(args.per_intent)
    articles, tickets = build(rows)
    DATA.mkdir(exist_ok=True)
    write_jsonl(DATA / "kb.jsonl", articles)
    write_jsonl(DATA / "tickets.jsonl", tickets)
    print(f"{len(rows)} rows -> {len(articles)} articles in data/kb.jsonl, {len(tickets)} tickets in data/tickets.jsonl")


if __name__ == "__main__":
    main()
