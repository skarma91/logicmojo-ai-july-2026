# Module 4 milestone: the Domain Knowledge Assistant

## Goal

Build one assistant that answers questions from a document corpus, **with citations**,
in a **consistent house style**, and can say "I do not find that in the documents"
when the answer is not there. It is the capstone of Module 4: it combines retrieval
(classes 4.5 and 4.6), the shared model wrapper (`llm.py`, from class 4.1 on), and
your fine-tuned component (classes 4.9 and 4.10) into a single working tool.

The one sentence to remember:

> **RAG supplies the facts (retrieved, cited, updatable). Fine-tuning supplies the
> style (concise, always cites the source and section).**

This is a build, plus an evaluation, plus one short written note. Complete the stubs in
`assignment/assistant.py`, run the evaluation, then write
`assignment/should_i_have_used_rag.md`. A worked reference is in `solution/`.

## How to approach this (read first)

1. **Get real data.** Run `python fetch_data.py` to download a few dozen MedlinePlus
   topics into `data/corpus.jsonl`. Everything falls back to the committed
   `data/example_corpus.jsonl` (18 real records) if you skip this, but the evaluation,
   and the honest RAG-versus-fine-tuning verdict, only mean something on the larger
   corpus: a dozen passages cannot settle which lever matters, a few dozen can.
2. **Build the assistant** (Part 1): complete the stubs in `assistant.py`.
3. **Evaluate it** (Part 3): build an eval set and score retrieval and answers. Do this
   with the base model and again with your tuned model, and compare.
4. **Add and compare the fine-tune** (Part 2), then **write the note**.

## The three parts

### Part 1: the RAG assistant (from class 4.5, 4.6)

Answer questions from the corpus (`data/corpus.jsonl` if you fetched it, else the
committed `data/example_corpus.jsonl`: real public-domain health passages from
MedlinePlus, the U.S. National Library of Medicine) with grounded, cited answers. In
`assignment/assistant.py`, complete:

- `build_store(corpus)`: a persisted **Chroma** store (vectors + text + metadata),
  embedded with `all-MiniLM-L6-v2`, built once and reopened on later runs.
- `retrieve(store, query, k, pool, rerank)`: top-k by similarity, and with
  `rerank=True` a wider pool re-scored by a **cross-encoder**.
- `format_context(docs)` and `sources_list(docs)`: number the passages so the
  model can cite `[1]`, and produce a traceable source line (source, title, section,
  and URL) each.
- `answer(store, question, model, k, rerank)`: retrieve, build the grounded
  prompt, call the model through `llm.get_chat_model()`, return the answer and its
  sources.

**Expected output:** for "What is a normal blood pressure reading?", the assistant
retrieves the blood-pressure-categories passage, answers concisely (a normal reading
is below 120/80 mm Hg), cites it as `[1]`, and lists the source as the MedlinePlus
topic and section with its URL. For a question the corpus does not cover, it declines
instead of inventing an answer. Every response ends with the general-information,
not-medical-advice disclaimer.

### Part 2: add your fine-tuned component (from class 4.9, 4.10)

Swap the generator for the house-style model you trained in class 4.10, without
changing the assistant code. Convert your merged model to GGUF, register it with
Ollama (`ollama create dka-assistant -f Modelfile`), then run with `PROVIDER=ollama`
and `MODEL_NAME=dka-assistant` (or `--model dka-assistant`). If you did not finish a
real fine-tune, use the instructor's pre-baked model or describe the expected
difference from the class 4.10 results.

**Expected output:** the same facts and citations (RAG is unchanged), but the tuned
model holds the concise, always-cited style without a long style prompt, where the
base model needs the instruction and still drifts. The house-style adherence metric
from class 4.10 is higher for the tuned model; faithfulness is about the same,
because faithfulness comes from the retrieved context, not the tune.

### Part 3: evaluate it (required)

You will not trust an assistant you have not measured, and you cannot claim RAG or
fine-tuning "won" without numbers. RAG does **not** train on the corpus, so we do not
hold documents out of the retriever. Instead `build_eval.py` holds out a **20% slice of
items** to write a fixed eval set (`data/eval.jsonl`): answerable questions with a known
gold passage, plus a few unanswerable ones to test correct refusal.

```
python build_eval.py     # writes data/eval.jsonl (20% holdout + refusal probes)
python eval.py           # base model
python eval.py --model dka-assistant   # your tuned model, to compare
```

`eval.py` reports, reusing class 4.8 (Evaluating LLM and RAG):

- **hit@1, hit@3** (no model needed): is the gold passage in the top-1 / top-3
  retrieved? Retrieval quality, the foundation everything else stands on.
- **keyword recall** (needs a provider): does the grounded answer contain the expected
  fact word? A cheap, deterministic outcome check.
- **faithfulness** (needs a provider): an LLM judge checks the answer is supported by
  the retrieved passages, not invented.
- **refusal accuracy** (needs a provider): on unanswerable questions, does it decline?

This is where the RAG-versus-fine-tuning question gets settled with data: run eval with
the base model and the tuned model. Expect **hit@k and faithfulness about the same**
(both use the same RAG context) and the **house-style adherence** better for the tuned
model. That is the whole thesis: RAG moved the facts, fine-tuning moved the style.

### The note: "should I have used RAG instead?"

Write `assignment/should_i_have_used_rag.md` (about 200 to 400 words): an honest
argument for why fine-tuning was, or was not, the right lever for the behavior you
added, and how the work divides between RAG (facts) and fine-tuning (style). Take a
position; the prompts in the file guide the structure.

## Optional: a small UI

`solution/app.py` is a thin Streamlit interface over the same `answer()` function
(`streamlit run app.py`). Reuse it, do not rewrite it, if you want to see the
assistant answer with its sources.

## What ships here

```
assignment/
  assistant.py                  starter with TODO stubs (complete this)
  fetch_data.py                 download the full MedlinePlus corpus (given)
  build_eval.py                 build the eval set with a 20% holdout (given)
  eval.py                       score retrieval + answers (given)
  should_i_have_used_rag.md      the note template (write this)
  llm.py                        the shared model wrapper (given)
  data/example_corpus.jsonl     18 real MedlinePlus records, committed
solution/                       worked reference for all of the above, plus
  app.py                        thin Streamlit UI over the same functions
```

The corpus is real, public-domain health information from MedlinePlus (U.S. National
Library of Medicine). Each record keeps its source topic and URL, so every answer is
traceable. `fetch_data.py` (needs network) pulls the larger corpus into
`data/corpus.jsonl`; the assistant uses it automatically when present, else the
committed sample. This assistant provides general health information only, not medical
advice, and prints that disclaimer with every answer.

## Run

```
pip install langchain langchain-community langchain-chroma langchain-huggingface \
            sentence-transformers chromadb python-dotenv requests
# plus your provider, e.g.:  pip install langchain-ollama
python fetch_data.py                                 # download the full corpus (recommended)
python assistant.py "What is a normal blood pressure reading?"
python assistant.py --demo
python build_eval.py                                 # build the eval set
python eval.py                                       # score it (add --model dka-assistant to compare)
```

Set `PROVIDER` / `MODEL_NAME` / keys in a `.env` at the repo root (copy
`.env.example`). Chroma persists to `data/chroma/` on first build (git-ignored) and
reopens it after that, so the workflow is identical whether you use the shipped
corpus or your own documents.

## What this becomes next

In Module 5 (AI agents) this assistant becomes the agent's first tool: the thing it
calls to look something up. Build it well here and you reuse it there.

## How this is checked

A reference solution is in the `solution/` folder. Compare your outputs and your
note to it.
