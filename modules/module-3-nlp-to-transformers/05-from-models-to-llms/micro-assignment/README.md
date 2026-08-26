# Micro-assignment 3.5: From models to LLMs

## Goal

Work the decoding math by hand: temperature, top-p, and how the settings shape generation.

## Rules

Use only what class 3.5 (From models to LLMs) and earlier cover, with NumPy. Temperature divides the logits by `T` before softmax; top-p keeps the smallest set of tokens reaching probability `p`. Match each expected output exactly.

## Problems

### Problem 1: Softmax with temperature

Write `softmax_T(logits, T) = softmax(logits / T)`. Print the probabilities for `[2, 1, 0]` at `T = 1` (three decimals).

```
probs: [0.665 0.245 0.09 ]
```

### Problem 2: Lower temperature is sharper

Print the maximum probability of `[2, 1, 0]` at `T = 0.5` and at `T = 2` (two decimals).

```
T=0.5 max: 0.87
T=2   max: 0.51
```

### Problem 3: Nucleus size for top-p

```python
probs = [0.5, 0.25, 0.15, 0.1]   # sorted descending
```

Print how many tokens are in the nucleus for `p = 0.9`.

```
nucleus size (p=0.9): 3
```

### Problem 4: Temperature 0 takes the top token

```python
logits = [2., 1., 3.]
```

Print the index temperature 0 would choose.

```
chosen index: 2
```

### Problem 5: Low vs high temperature

Print one line on when to use low versus high temperature.

```
use low temperature for exact tasks (extract a value, strict format) and higher temperature for creative tasks (brainstorming, writing); it is a dial, not a default
```

### Problem 6: What the context window is

Print one line describing the context window.

```
the context window is how many tokens the model can consider at once, its working memory; prompt, retrieved text, and conversation all must fit inside it
```

## Deliverable

The working notebook and its output. Fill in `assignment.ipynb` and run it.

## How this is checked

A reference solution is released in the `solution/` folder. Compare your output to the expected output above.
