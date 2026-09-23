# Class 4.10 code: fine-tuning II, hands-on QLoRA and alignment

`notebook.ipynb` runs the QLoRA fine-tune prepared in class 4.9, evaluates the tuned
model against the base, and merges it into one servable model. Alignment is taught in
the slides; the notebook ends by briefing milestone part 2.

## What ships here

```
notebook.ipynb          the class build (open in Colab, GPU runtime)
data/dataset.jsonl      the house-style SFT set from class 4.9 (72 examples)
data/make_dataset.py    regenerates dataset.jsonl by paraphrase augmentation
data/eval_set.jsonl     the class 4.8 eval questions, used here as the held-out test
```

## Run

Open on a Colab **GPU** runtime. The Setup cell installs `transformers datasets peft
trl bitsandbytes accelerate`. The flow: kick off the real QLoRA run, then, rather
than wait, load a **pre-baked checkpoint** to evaluate, merge, and serve (the
pre-bake pattern). Set `ADAPTER` in the notebook to the instructor's pre-baked
adapter, or to your own `dka-lora` output once your run finishes.

## Evaluating honestly

The notebook scores a deterministic **house-style adherence** metric (does the answer
cite `(Pub NNN, YYYY)` and stay concise) for base vs tuned on the held-out class 4.8
eval questions, which are disjoint from the training set. You can also run the full
class 4.8 metrics (faithfulness, relevance) here. Numbers are real only when you run
it; watch that general ability did not drop (overfitting).

## Model

Uses `Qwen/Qwen2.5-0.5B-Instruct`, small enough for a free Colab GPU under QLoRA.
Change `MODEL_ID` for a different base.

## No separate micro-assignment

This class briefs **milestone part 2** instead of a micro: add the fine-tuned
component to the assistant and write the "should I have used RAG?" note. See the
`milestone-assignment/` folder for the Module 4 milestone.
