# Class 6.5 micro-assignment: find one injection, then patch it

The class build planted a passage in a tax corpus. Here the corpus is film
reviews. `data/reviews.jsonl` holds eight films. One field on one row tells the
recommendation agent to email the user's watchlist to an outside address. The
other fields are ordinary notes.

No model is called. Finding the vector, and showing a structural patch, is
reading and deciding.

## Problems

1. **`find_injection(reviews)`.** Return `(id, field, destination)` for the planted
   instruction. Check every string field, not only `review`.
2. **`exfil_channels(tools)`.** Which tools can carry data out or destroy it?
   Include a tool that never says "send": it fetches a URL the caller supplies,
   and that URL can hold the watchlist. Return the names sorted alphabetically.
3. **`split_privileges(tools)`.** The reader gets tools that look up outside text
   and gets no outbound tool. The sender gets mail and pure formatting, and
   `sender_sees_raw` is false. Leave the destructive tools and the URL-fetching
   tool ungranted.
4. **`allow(to, body, allowlist)`.** Return `(True, "allowed")`, or
   `(False, "shape")`, or `(False, "destination")`. Shape is checked first. A
   lookalike domain is a different string, so it is not on the list.
5. **`name_deputy(scene)` and `residual(handoff, allowlist)`.** Name the agent
   that holds real authority and the tool it is tricked into using. Then read a
   handoff a facts filter already produced. If an outside address is still in
   it, say which layer stops a send there anyway.

## Expected output

```
[1] injection: ('rv-05', 'studio_note', 'drops@exfil.example')
[2] channels: ['delete_watchlist', 'render_poster', 'send_email']
[3] split: {'reader': ['search_films'], 'sender': ['send_email', 'format_title'], 'sender_sees_raw': False}
[4] allow: [(True, 'allowed'), (False, 'destination'), (False, 'destination'), (False, 'shape'), (False, 'shape'), (False, 'shape')]
[5] deputy: ('sender', 'send_email')
[5] leaky handoff: {'leaked_to': 'drops@exfil.example', 'stopped_by': 'allowlist'}
[5] clean handoff: {'leaked_to': '', 'stopped_by': 'privilege'}
```

The leaky handoff is the thinking step. The facts filter kept a sentence that
still names `drops@exfil.example`. Privilege did not hold on that sentence. The
allowlist is the layer that still stops the send. The clean handoff has no
address left, so privilege held.

## Run

```
python assignment.py
```

No packages beyond the Python standard library, and no API key.

## How this is checked

A reference solution is in the `solution/` folder. Compare your output to the expected
output above.
