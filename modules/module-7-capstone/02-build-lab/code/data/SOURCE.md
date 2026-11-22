# Data source and license

`raw_sample.jsonl` holds 174 real rows (all 27 intents) from the Bitext customer-support
dataset on Hugging Face: `bitext/Bitext-customer-support-llm-chatbot-training-dataset`,
pulled through the Hugging Face datasets-server rows API. `kb.jsonl` and `tickets.jsonl`
are built from it by `python fetch_data.py --sample`; `python fetch_data.py` rebuilds them
from the full CSV.

License: Community Data License Agreement, Sharing, version 1.0 (CDLA-Sharing-1.0),
(c) Bitext Innovations. Data shared from this folder stays under the same license.

Bitext describes the dataset as hybrid synthetic: the customer messages are generated in
the style of real support tickets, not taken from real customers, so they contain no real
personal data.
