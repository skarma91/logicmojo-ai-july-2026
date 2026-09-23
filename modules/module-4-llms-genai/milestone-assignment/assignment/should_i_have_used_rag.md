# Should I have used RAG instead? (your note)

Write a short, honest argument (roughly 200 to 400 words) for why fine-tuning was,
or was not, the right lever for the behavior you added, and how it divides work with
RAG. Do not just praise both; take a position and defend it with what you observed.

Use these prompts to structure it (delete them in your final note):

1. **What each lever changed.** Which requirement did RAG solve, and which did
   fine-tuning solve? Tie each to the "behavior, not facts" idea from class 4.9.

2. **Why not fine-tune on the documents and skip RAG?** Address staleness,
   citations, and hallucination.

3. **Why not RAG-only and skip fine-tuning?** Be honest: would a good prompt have
   carried the style? What specifically made you add the tune (token cost of the
   repeated style prompt, inconsistent formatting, class 4.7)?

4. **Evidence from your build.** Compare the base model and your tuned model
   (`--model dka-assistant`) on a few questions. What stayed the same (hint:
   faithfulness comes from the retrieved context) and what changed (hint: the
   house-style adherence metric from class 4.10)?

5. **Verdict.** One or two sentences: your final recommendation for this assistant.
