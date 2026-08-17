# Micro-assignment 3.2: Attention

## Goal

Work the attention math by hand: a similarity score, a softmax over scores, scaling, and a blended context vector.

## Rules

Use only what class 3.2 (Attention) and earlier cover, with NumPy. A score is a dot product; attention weights are a softmax of scores; the context vector is `weights @ values`. Match each expected output exactly.

## Problems

### Problem 1: A similarity score (dot product)

```python
q = np.array([2., 0., 1.])
k = np.array([1., 3., 1.])
```

Print the dot-product score of `q` and `k`.

```
score: 3.0
```

### Problem 2: Softmax over scores

```python
s = np.array([4., 1., 0.])
```

Print the softmax weights (three decimals) and their sum.

```
weights: [0.936 0.047 0.017]
sum: 1.0
```

### Problem 3: Scale by sqrt(d_k)

```python
scores = np.array([8., 4., 2.])
d_k = 4
```

Print `scores / sqrt(d_k)`.

```
scaled: [4.0, 2.0, 1.0]
```

### Problem 4: The context vector (weighted blend)

```python
weights = np.array([0.9, 0.1])
values  = np.array([[1., 0.], [0., 1.]])
```

Print the weighted blend `weights @ values`.

```
context: [0.9, 0.1]
```

### Problem 5: Where does this token attend?

```python
row = np.array([0.1, 0.7, 0.2])
```

Print the index of the most-attended token.

```
most-attended index: 1
```

### Problem 6: Why scale the scores?

Print one line on why we divide the scores by `sqrt(d_k)`.

```
dividing by sqrt(d_k) keeps dot products from growing large with vector length, so softmax stays smooth and training is stable
```

### Problem 7: Causal mask (no peeking ahead)

```python
scores = np.array([2.0, 1.0, 3.0, 0.5])
```

A query at position 1 may attend to positions 0 and 1 only. Set the future scores (positions 2 and 3) to `-inf`, take the softmax, and print the weights (three decimals). The future positions should come out as 0.

```
weights: [0.731 0.269 0.    0.   ]
```

## Deliverable

The working notebook and its output. Fill in `assignment.ipynb` and run it.

## How this is checked

A reference solution is released in the `solution/` folder. Compare your output to the expected output above.
