"""Reply writer agent: draft a short reply grounded in one help article."""

from __future__ import annotations
import llm
from schemas import Article, Draft

SYSTEM = (
    "You are a customer-support agent. Write a short, friendly reply (under 120 words) to the "
    "customer, using ONLY the help article you are given. Keep placeholders such as "
    "{{Order Number}} and <PERSON_1> exactly as written. Never invent order numbers, dates, "
    "prices, or policies, and never promise anything the article does not describe."
)


class ReplyWriterAgent:
    name = "writer"

    def run(self, text: str, article: Article, feedback: list[str] | None = None) -> Draft:
        prompt = f"Customer message:\n{text}\n\nHelp article: {article.title}\n{article.body}"
        if feedback:                                 # a second attempt: say exactly what to fix
            prompt += "\n\nYour previous draft was rejected. Fix these problems:\n- " + "\n- ".join(feedback)
        reply = llm.chat([{"role": "system", "content": SYSTEM},
                          {"role": "user", "content": prompt}], agent=self.name)
        return Draft(text=reply.strip())
