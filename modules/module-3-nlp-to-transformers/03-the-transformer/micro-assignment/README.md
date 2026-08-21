# Micro-assignment 3.3: The Transformer

## Goal

Reason about the transformer's supporting parts: parameter counts, head dimensions, residuals, and LayerNorm.

## Rules

Use only what class 3.3 (The Transformer) and earlier cover, in plain Python and NumPy. A feed-forward layer is `Linear(d, 4d)` then `Linear(4d, d)`. Match each expected output exactly.

## Problems

### Problem 1: Feed-forward parameter count

For `d = 8`, a feed-forward layer is `Linear(8, 32)` then `Linear(32, 8)`, each with weights and biases. Print the total parameter count.

```
feed-forward params: 552
```

### Problem 2: Head dimension

```python
d_model, n_heads = 12, 4
```

Print the per-head dimension.

```
per-head dimension: 3
```

### Problem 3: A residual connection

```python
x = np.array([1., 2., 3.])
layer_out = np.array([0.1, 0.1, 0.1])
```

Print `x + layer_out`.

```
residual: [1.1, 2.1, 3.1]
```

### Problem 4: Concatenate two heads

```python
h1 = np.array([1., 2.])
h2 = np.array([3., 4.])
```

Print the two head outputs concatenated.

```
combined: [1.0, 2.0, 3.0, 4.0]
```

### Problem 5: LayerNorm centering

```python
v = np.array([2., 4., 6.])
```

Print the mean, then the centered vector (`v - mean`).

```
mean: 4.0
centered: [-2.0, 0.0, 2.0]
```

### Problem 6: What positional encoding fixes

Print one line on what problem positional encoding solves.

```
attention treats the sentence as a set and ignores order; positional encoding adds a position signal so the model can tell word order
```

## Deliverable

The working notebook and its output. Fill in `assignment.ipynb` and run it.

## How this is checked

A reference solution is released in the `solution/` folder. Compare your output to the expected output above.
