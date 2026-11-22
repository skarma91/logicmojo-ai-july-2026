"""Researcher agent: build the pool of real, named places the plan may use.

1. Retrieval (no model): for each interest, TF-IDF finds the destination's best chunks
   in the See, Do, Eat, and Drink sections. Each chunk remembers which interests found it.
2. Extraction (one model call): the model lists the named places in those chunks.
3. Grounding (code): a name is kept only if it appears word for word in its chunk.

Big-city Wikivoyage pages name places inside prose rather than as listings, which is
why a model does the extraction and code does the verification.
"""

from __future__ import annotations
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import llm
from guides import CHUNKS, INTERESTS, PLACE_SECTIONS
from schemas import Place, Pool, TripRequest

CHUNKS_PER_INTEREST = 4
MAX_PLACES = 30

SYSTEM = (
    "You extract named places a visitor can go to (sights, neighbourhoods, museums, markets, "
    "parks, restaurants, bars) from travel-guide text. Reply with one JSON object only:\n"
    '{"places": [{"name": "<copied exactly from the text>", "chunk": "<chunk id>"}]}\n'
    "Copy each name exactly as written in that chunk. Skip generic phrases such as "
    '"the old town" unless the text names it. At most 4 places per chunk.'
)


class ResearcherAgent:
    name = "researcher"

    def retrieve(self, req: TripRequest) -> dict[str, dict]:
        """chunk_id -> chunk, each tagged with the interests whose query found it."""
        chunks = [c for c in CHUNKS if c["destination"] == req.destination and c["section"] in PLACE_SECTIONS]
        if not chunks:
            return {}
        vec = TfidfVectorizer(stop_words="english", sublinear_tf=True)
        matrix = vec.fit_transform([c["text"] for c in chunks])
        found: dict[str, dict] = {}
        for interest in req.interests:
            sims = cosine_similarity(vec.transform([INTERESTS[interest]]), matrix)[0]
            for i in sims.argsort()[::-1][:CHUNKS_PER_INTEREST]:
                if sims[i] <= 0:
                    break
                c = found.setdefault(chunks[i]["id"], {**chunks[i], "interests": []})
                c["interests"].append(interest)
        return found

    def run(self, req: TripRequest) -> Pool:
        found = self.retrieve(req)
        if not found:
            return Pool(req.destination, [])
        listing = "\n\n".join(f"[{cid}] ({c['section']})\n{c['text']}" for cid, c in found.items())
        raw = llm.chat_json([{"role": "system", "content": SYSTEM},
                             {"role": "user", "content": listing}], agent=self.name)
        places, seen = [], set()
        for item in raw.get("places") or []:
            if not isinstance(item, dict):
                continue
            name, cid = str(item.get("name", "")).strip(), str(item.get("chunk", "")).strip()
            chunk = found.get(cid)
            key = name.lower()
            # Grounding: the name must appear, word for word, in the chunk it claims.
            if not chunk or len(name) < 3 or key in seen:
                continue
            if not re.search(r"(?<!\w)" + re.escape(name) + r"(?!\w)", chunk["text"], flags=re.I):
                continue
            seen.add(key)
            places.append(Place(id=f"P{len(places) + 1}", name=name, section=chunk["section"],
                                interests=list(chunk["interests"]), chunk_id=cid, url=chunk["url"]))
            if len(places) >= MAX_PLACES:
                break
        return Pool(req.destination, places)
