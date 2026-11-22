"""Knowledge agent: find the help article that answers the ticket. No model call.

It gives the supervisor a second, independent opinion. Triage (a model) says what
the customer wants; retrieval (TF-IDF over the 27 articles) says which article the
words of the ticket are closest to. Tickets are short and full of typos ("cancel
oorder", "refudn"), so we compare character n-grams (3 to 5 letters inside each
word) instead of whole words: a typo still shares most of its n-grams.

When the two opinions agree, we trust the article. When they disagree, someone is
wrong, and the supervisor may hand the ticket to a human.
"""

from __future__ import annotations
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from data import load_kb
from schemas import Article


class KnowledgeAgent:
    name = "knowledge"

    def __init__(self):
        self.kb = load_kb()
        # the title and intent name are repeated so they outweigh the long, chatty body
        docs = [f"{a['title']}. {a['intent'].replace('_', ' ')}. " * 3 + a["body"] for a in self.kb]
        self.vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True)
        self.matrix = self.vectorizer.fit_transform(docs)
        self.index = {a["intent"]: i for i, a in enumerate(self.kb)}

    def scores(self, text: str):
        """Cosine similarity between the ticket and every article."""
        return cosine_similarity(self.vectorizer.transform([text]), self.matrix)[0]

    def run(self, text: str, intent: str) -> Article:
        sims = self.scores(text)
        best = int(sims.argmax())
        chosen = self.index.get(intent, best)       # trust triage's intent when it names a real one
        a = self.kb[chosen]
        return Article(intent=a["intent"], title=a["title"], body=a["body"],
                       score=float(sims[chosen]), agrees=(chosen == best))
