# Class 6.4 micro-assignment: block piracy requests at the door

The class build put a gateway in front of a tax team. Here you build the input guard
for a movie team: it must stop requests for pirated copies of films before any agent
sees them, without refusing innocent questions that happen to share the words.

`data/requests.jsonl` holds 16 labeled requests: 8 unsafe (torrents, magnet links,
cam rips, illegal streams, DRM removal, redistribution) and 8 safe, several of them
deliberate traps (*Pirates of the Caribbean*, a "torrential" rain scene, how
BitTorrent works, a legal free download).

The guard has two stages, cheapest first:

1. **Rules:** word-boundary patterns over normalized text. Free and local.
2. **Classifier:** a written piracy policy on `gpt-oss-safeguard-20b` (Groq), called
   **only** for requests the rules did not block that still contain an ambiguous word.
   It falls back to your default model if the safeguard model is unavailable, and an
   unreadable verdict counts as a block (fail closed).

## Problems

1. **`normalize(text)`.** Lowercase, undo simple digit and symbol swaps with the `LEET`
   table (`t0rr3nt` becomes `torrent`), and collapse runs of whitespace.
2. **Rules.** (a) Write the three missing `RULES` patterns described in the TODO.
   (b) Write `rule_block(text)`: normalize, then return `(True, rule name)` for the
   first pattern that matches, or `(False, "")`.
3. **`rates(preds, labels)`.** Return `(block rate, false-block rate)`. Block rate is
   blocked unsafe requests divided by all unsafe requests; false-block rate is blocked
   safe requests divided by all safe requests. It must not crash on an empty group.
4. **`disagreements(rows)`.** `naive_block` is the substring blocker people write
   first. Return `(id, naive verdict, rule verdict, label)` for every request where it
   and your rules disagree. Then read the list: which one is right each time, and why?
5. **`needs_classifier(text)`.** Decide which requests are worth a model call. A
   request the rules already settled needs none; neither does one with no ambiguous
   word (see `AMBIGUOUS`).

## Expected output

The deterministic lines are exact:

```
[1] normalize('  T0rr3nt   the  M@trix ') -> 'torrent the matrix'
[2] rule_block('Where can I t0rrent Oppenheimer?') -> (True, 'torrent')
[2] rule_block('Who directed Pirates of the Caribbean?') -> (False, '')
[2] rule_block('Is there a torrential rain scene in Blade Runner?') -> (False, '')
[3] rules only: block rate 0.88, false-block rate 0.00
    naive only: block rate 0.75, false-block rate 0.62
    rates([], []) -> (0.0, 0.0)
[4] disagreements (id, naive, rules, label): [('m03', True, False, True), ('m06', False, True, True), ('m07', False, True, True), ('m09', True, False, False), ('m10', True, False, False), ('m14', True, False, False), ('m15', True, False, False), ('m16', True, False, False)]
[5] sent to the classifier: ['m03', 'm11', 'm14']  (3 of 16 requests; 13 model calls saved)
```

The two-stage run then calls a real model on those three requests, so its verdicts
can vary, but these properties must hold:

- Every request shows `BLOCK` or `pass` and the stage that decided it.
- Exactly three classifier calls.
- The two-stage block rate is at least the rules-only 0.88, and the false-block rate
  stays at or near 0.00. The model should block m03 (a "free download" that avoids
  paying) and pass m11 and m14 (legal streaming, an official free download).

## Run

```
pip install python-dotenv groq      # or ollama
python assignment.py                # PROVIDER=groq (or ollama) in .env
```

## How this is checked

A reference solution is in the `solution/` folder. Compare your output to the expected
output above.
