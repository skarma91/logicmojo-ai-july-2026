# Class 4.9 code: fine-tuning I, the decision and the data

`notebook.ipynb` prepares a small supervised fine-tuning (SFT) dataset that teaches
the Domain Knowledge Assistant a consistent **house style** (concise answers that
cite the IRS publication and year, and decline when out of scope), then defines the
LoRA and 4-bit QLoRA training configuration. It stops before training; class 4.10
runs the QLoRA job.

## What ships here

```
notebook.ipynb          the class build (open in Colab, GPU runtime)
data/dataset.jsonl      72 chat-format house-style examples (behavior, not facts)
data/make_dataset.py    regenerates dataset.jsonl by paraphrase augmentation
```

## Run

Open the notebook on a Colab **GPU** runtime (the `bitsandbytes` 4-bit config needs
a GPU). The Setup cell installs `transformers datasets peft trl bitsandbytes
accelerate`. The notebook only loads the tokenizer (to show the chat formatting);
the 4-bit model is loaded and trained in class 4.10.

## The dataset

`dataset.jsonl` is chat-format: each row has a `system` house-style rule, a `user`
question, and the `assistant` target answer. It teaches a **behavior** (style,
citation, refusal), not tax facts, which remain RAG's job (class 4.5). It is built
by **paraphrase augmentation** (`make_dataset.py`): one house-style answer per fact,
several question phrasings each, which keeps every target consistent in style and
grows the set cheaply. The questions are deliberately **disjoint** from the class
4.8 eval set, which stays the held-out test so it can honestly judge the fine-tune
in class 4.10. This shipped set is a teaching size (72 examples); a real behavior
tune wants a few hundred to about a thousand clean examples.

## Model

The example uses `Qwen/Qwen2.5-0.5B-Instruct`, a small open instruct model that
fits a free Colab GPU under QLoRA. Any small instruct model with a chat template
works; change `MODEL_ID` in the notebook.
