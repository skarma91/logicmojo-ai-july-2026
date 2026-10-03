"""Memory for agents: working memory (short-term) and long-term memory.

Two kinds, as taught in the slides:

- Conversation (WORKING memory): the running dialogue and intermediate results in
  the current context. It stays bounded by keeping only the last few turns verbatim
  and COMPACTING older turns into a running summary (a real llm.chat call).

- MemoryStore (LONG-TERM memory): durable facts and preferences kept across
  sessions in a vector store (the class 4.4 idea), retrieved by similarity. The
  hard part is the WRITE policy: dedupe near-duplicates and UPDATE a fact that
  changed, rather than appending a second, contradictory copy.

The embedding model is injectable (embed_fn) so this module is testable offline;
by default it uses sentence-transformers, the same all-MiniLM-L6-v2 as class 4.4.

Install:
    pip install sentence-transformers numpy
"""

from __future__ import annotations
import json
import logging
import pathlib

import numpy as np

log = logging.getLogger("course.memory")
MODEL_NAME = "all-MiniLM-L6-v2"
DEDUP_THRESHOLD = 0.88   # cosine above this counts as "the same fact" on write


def _default_embedder():
    """Lazily load sentence-transformers and return an encode function."""
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(MODEL_NAME)
    return lambda texts: np.asarray(model.encode(list(texts)), dtype=float)


def _normalize(m: np.ndarray) -> np.ndarray:
    return m / np.clip(np.linalg.norm(m, axis=1, keepdims=True), 1e-12, None)


# ===========================================================================
# Long-term memory: a small vector store with a real write policy
# ===========================================================================
class MemoryStore:
    def __init__(self, path, embed_fn=None):
        self.path = pathlib.Path(path)
        self._embed_fn = embed_fn
        self.items: list[dict] = []          # each: {id, kind, key, text}
        self._matrix = np.zeros((0, 0))      # normalized embeddings, one row per item
        if self.path.exists():
            self.items = [json.loads(l) for l in open(self.path)]
            self._reembed()

    def _embed(self, texts) -> np.ndarray:
        if self._embed_fn is None:
            self._embed_fn = _default_embedder()
        return _normalize(self._embed_fn(texts))

    def _reembed(self):
        self._matrix = self._embed([it["text"] for it in self.items]) if self.items else np.zeros((0, 0))

    def add(self, text: str, kind: str = "semantic", key: str = None) -> str:
        """Write policy. Returns 'added', 'updated', or 'skipped'.

        1. If a `key` is given and already exists, UPDATE that item (a preference
           that changed overwrites the old value, it does not pile up a duplicate).
        2. Else if the text is a near-duplicate of an existing item, UPDATE that
           one (keep the latest wording) instead of adding a second copy.
        3. Otherwise APPEND a new item.
        """
        vec = self._embed([text])[0]

        if key is not None:
            for it in self.items:
                if it.get("key") == key:
                    it["text"] = text
                    self._reembed(); self.save()
                    log.info("memory updated key=%s", key)
                    return "updated"

        if self.items:
            sims = self._matrix @ vec
            j = int(np.argmax(sims))
            if float(sims[j]) >= DEDUP_THRESHOLD:
                self.items[j]["text"] = text          # fold into the near-duplicate
                self._reembed(); self.save()
                log.info("memory merged into near-duplicate (sim=%.2f)", float(sims[j]))
                return "updated"

        item = {"id": f"m{len(self.items)}", "kind": kind, "key": key, "text": text}
        self.items.append(item)
        self._reembed(); self.save()
        log.info("memory added id=%s kind=%s", item["id"], kind)
        return "added"

    def search(self, query: str, k: int = 3) -> list[dict]:
        if not self.items:
            return []
        q = self._embed([query])[0]
        sims = self._matrix @ q
        order = np.argsort(-sims)[:k]
        return [dict(self.items[i], score=float(sims[i])) for i in order]

    def save(self):
        with open(self.path, "w") as f:
            for it in self.items:
                f.write(json.dumps(it) + "\n")


# ===========================================================================
# Working memory: a bounded conversation with running-summary compaction
# ===========================================================================
class Conversation:
    def __init__(self, keep_last: int = 6):
        self.turns: list[dict] = []   # {role, content}, recent turns kept verbatim
        self.summary: str = ""        # older turns, compacted
        self.keep_last = keep_last

    def add(self, role: str, content: str):
        self.turns.append({"role": role, "content": content})

    def compact(self, chat_fn):
        """When the buffer grows past keep_last, fold the OLDEST turns into the
        running summary with one real model call, then drop them. This keeps the
        token count bounded while the essentials survive (a real llm.chat call)."""
        if len(self.turns) <= self.keep_last:
            return
        old = self.turns[:-self.keep_last]
        self.turns = self.turns[-self.keep_last:]
        transcript = "\n".join(f"{t['role']}: {t['content']}" for t in old)
        prompt = (
            "Update the running summary of a conversation. Keep durable facts, "
            "decisions, and user preferences; drop small talk. Be concise.\n\n"
            f"Current summary:\n{self.summary or '(none)'}\n\n"
            f"New turns to fold in:\n{transcript}\n\nUpdated summary:"
        )
        self.summary = chat_fn([{"role": "user", "content": prompt}]).strip()
        log.info("compacted %d old turns into the summary", len(old))

    def messages(self, system: str, memories: str = "") -> list[dict]:
        """Assemble the model input: system prompt, plus the running summary and any
        retrieved long-term memories as context, then the recent turns."""
        head = system
        if memories:
            head += f"\n\nRelevant long-term memory:\n{memories}"
        if self.summary:
            head += f"\n\nConversation so far (summary):\n{self.summary}"
        return [{"role": "system", "content": head}] + self.turns
