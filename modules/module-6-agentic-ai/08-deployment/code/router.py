"""Semantic router: choose a path by meaning, not keywords.

Each route has a handful of example utterances (data/routes.json). We embed them
with the same MiniLM model the retriever uses (class 4.4, Embeddings and vector
databases), average each route's vectors into a CENTROID, and route a request to
the closest centroid:

    route = argmax_r cos(e_q, c_r)      if  max_r cos(e_q, c_r) >= TAU
    route = FALLBACK                    otherwise

e_q is the request's embedding, c_r the centroid of route r, and TAU the minimum
similarity we trust. Below TAU the request is not close to anything we handle, so
it goes to a fallback path instead of being forced into the nearest route.

TAU is tuned on examples, not derived. Print the scores (main.py does), look at
where in-scope and out-of-scope requests fall, and set TAU between them.
"""

from __future__ import annotations
import json
import pathlib

import numpy as np

import retriever                      # reuse its embedding model and normalization

ROUTES_FILE = pathlib.Path(__file__).parent / "data" / "routes.json"
TAU = 0.35                            # tuned on the examples in data/routes.json; see the README
FALLBACK = "fallback"


class SemanticRouter:
    def __init__(self, routes_file=ROUTES_FILE, tau: float = TAU):
        spec = json.loads(pathlib.Path(routes_file).read_text())
        self.tau = tau
        self.names = list(spec)
        self.actions = {name: spec[name]["action"] for name in self.names}
        model = retriever._embedder()
        centroids = []
        for name in self.names:
            vecs = retriever._normalize(np.asarray(model.encode(spec[name]["examples"])))
            centroids.append(vecs.mean(axis=0))
        # Re-normalize: the mean of unit vectors is shorter than 1, and cosine needs unit length.
        self.centroids = retriever._normalize(np.vstack(centroids))

    def scores(self, text: str) -> dict:
        q = retriever._normalize(np.asarray(retriever._embedder().encode([text])))[0]
        return {name: float(s) for name, s in zip(self.names, self.centroids @ q)}

    def route(self, text: str) -> tuple[str, float, dict]:
        """Return (route name or FALLBACK, best score, all scores)."""
        sc = self.scores(text)
        best = max(sc, key=sc.get)
        return (best if sc[best] >= self.tau else FALLBACK), sc[best], sc
