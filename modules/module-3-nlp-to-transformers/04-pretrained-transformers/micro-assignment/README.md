# Micro-assignment 3.4: Using pretrained transformers

## Goal

Reason about using pretrained models: pipelines, the tokenizer-model pairing, and batched inference.

## Rules

Use only what class 3.4 (Using pretrained transformers) and earlier cover, in plain Python. These problems need no model downloads. Match each expected output exactly.

## Problems

### Problem 1: Task to pipeline name

```python
mapping = {"sentiment": "sentiment-analysis", "generate": "text-generation", "embed": "feature-extraction"}
tasks   = ["sentiment", "embed"]
```

Print the pipeline names for the given tasks.

```
['sentiment-analysis', 'feature-extraction']
```

### Problem 2: Why load the matching tokenizer

Print one line on why the tokenizer and model must be the matching pair.

```
the model learned what each token id means; a mismatched tokenizer assigns different ids, so the input is meaningless to the model. Load the tokenizer and model by the same name
```

### Problem 3: A padded batch's shape

```python
lengths = [4, 6, 5]   # token counts of three inputs
```

Print the shape of the padded batch as `(num_texts, max_len)`.

```
batch shape: (3, 6)
```

### Problem 4: Inference settings

Print one line on what `model.eval()` and `torch.no_grad()` do for inference.

```
model.eval() and torch.no_grad(): eval switches layers like dropout to scoring mode, and no_grad skips gradient tracking for a faster forward pass
```

### Problem 5: Tokens vs words

```python
s = "the cat sat"
```

Print the number of whitespace words.

```
words: 3
```

### Problem 6: Why use a pretrained model

Print one line on why we use a pretrained model instead of training our own.

```
pretrained models already learned language from huge corpora, so you get strong results without the data, compute, or time to train one yourself
```

## Deliverable

The working notebook and its output. Fill in `assignment.ipynb` and run it.

## How this is checked

A reference solution is released in the `solution/` folder. Compare your output to the expected output above.
