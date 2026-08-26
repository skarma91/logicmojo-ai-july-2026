# Interview Preparation Guide, Module 2: ML and DL essentials

Questions and answers for the machine learning and deep learning foundations taught in Module 2.

**Coverage map.** The questions follow the five classes of the module: class 2.1 (What is ML), class 2.2 (How models learn), class 2.3 (Neural networks), class 2.4 (PyTorch fundamentals), and class 2.5 (Training in practice).

## How to use this guide

Read the question, answer it out loud or on paper first, then expand the answer to check yourself. Each question is tagged by difficulty so you can target your level:

- **[Warm-up]** a definition or one-liner an interviewer opens with.
- **[Core]** the standard question you are expected to answer cleanly.
- **[Deep]** mechanism, trade-off, or a "why" that separates strong candidates.
- **[Applied]** a scenario or short design or debugging prompt.

Where a question turns on numbers, work it by hand before expanding. The "Strong answer adds" and "Common wrong answer" notes are what interviewers actually listen for.

---

## Part 1: Foundations of ML (class 2.1)

### Q1. [Warm-up] What is machine learning, in one sentence?

<details><summary>Answer</summary>

Instead of writing the rules by hand, you show the algorithm labeled examples and it finds the rule itself, then you test whether that rule generalizes to data it never saw.

**Strong answer adds:** the reason to reach for ML is that the rule is too complex or too fuzzy to write by hand (spam, images, language). If a few clear if-statements would do the job, ML is the wrong tool.

**Common wrong answer:** calling it "a computer that thinks" or "AI magic." It is pattern-fitting from examples, nothing more mystical.
</details>

### Q2. [Core] Contrast a hand-coded rule system with a learned model using spam filtering.

<details><summary>Answer</summary>

A hand-coded filter is an ever-growing list of if-statements ("if the text contains 'free money', mark spam"). It is brittle: spammers write "fr33 m0ney" and it slips through, and you patch rules forever. A learned filter is given many emails, each labeled spam or not, and it discovers the pattern that separates them, including combinations a human would never enumerate.

**Strong answer adds:** the learned model degrades more gracefully on novel spam because it keys on statistical patterns, not exact strings, though it still needs retraining as the distribution drifts.
</details>

### Q3. [Warm-up] Define features and labels, and what is available at prediction time.

<details><summary>Answer</summary>

Features are the inputs (X), the label is the answer we want (y). The model learns a mapping from X to y. At prediction time you have the features but not the label; the label is exactly what you are predicting.

**Common wrong answer:** assuming the label is available for new data. If it were, you would not need the model.
</details>

### Q4. [Core] Regression versus classification. Give two examples of each.

<details><summary>Answer</summary>

Regression predicts a number: house price, tomorrow's temperature. Classification predicts a category: spam or not spam, which of ten digits an image shows. The task type is decided by the label, not the features.

**Strong answer adds:** the loss follows the task, mean squared error for regression, cross-entropy for classification, a point that returns in class 2.3.
</details>

### Q5. [Core] Why split data into train, validation, and test rather than just train and test?

<details><summary>Answer</summary>

Train is where the model learns. Validation is where you tune your choices (model size, learning rate, when to stop). Test is touched once, at the end, for an honest final number. If you tuned against the test set, it would stop being honest, because you would be fitting your choices to it; validation exists to absorb that tuning so the test set stays untouched.

**Analogy:** practice questions (train), a mock exam you retake while studying (validation), the real exam sat once (test).

**Common wrong answer:** tuning hyperparameters on the test set and reporting that number. That is a leaked, optimistic estimate.
</details>

### Q6. [Core] Define overfitting and underfitting and how each shows up in the numbers.

<details><summary>Answer</summary>

Overfitting is memorizing the training data including its noise: high training accuracy, low test accuracy, a large gap. Underfitting is a model too simple to catch the real pattern: low accuracy on both train and test. The goal is the sweet spot between them.

**Strong answer adds:** overfitting is a variance problem (fix with more data, regularization, a simpler model); underfitting is a bias problem (fix with a more expressive model or better features).
</details>

### Q7. [Deep] "Higher training accuracy is always better." True or false, and why?

<details><summary>Answer</summary>

False. Past the sweet spot, rising training accuracy while validation or test accuracy falls is the signature of memorization, not learning. What you care about is generalization, measured on held-out data, so the honest number is the held-out one, not the training one.
</details>

### Q8. [Deep] What is data leakage? Give a concrete example and the fix.

<details><summary>Answer</summary>

Leakage is when information from outside the training set sneaks into training, so the score is optimistic and does not hold up in production. The classic example: scaling or imputing using statistics (mean, standard deviation) computed over the whole dataset before you split, so the training rows already "know" something about the test rows. The fix: split first, then fit any transformation on the training part only and apply it to validation and test.

**Strong answer adds:** other leakage sources include target-derived features, time leakage (using future data to predict the past), and duplicate rows spanning the split.
</details>

### Q9. [Applied] A model reports 97 percent accuracy on a dataset where 97 percent of examples are the majority class. What do you conclude?

<details><summary>Answer</summary>

Almost nothing. A model that always predicts the majority class scores 97 percent while learning nothing. Accuracy is blind on imbalanced data. You need precision, recall, F1, or a confusion matrix, and often the minority class is the one you care about (fraud, disease).

**Strong answer adds:** report per-class metrics and pick the metric that matches the cost of each error type; consider class weighting or resampling.
</details>

### Q10. [Numerical] Of 20 predictions, 18 match the true label and all 20 true labels are the majority class. What is the accuracy, and why does it flatter the model?

<details><summary>Answer</summary>

Accuracy is 18 / 20 = 0.90. It flatters the model because the data has no minority examples at all here, so a constant majority guesser would score 1.00; the number measures the class balance more than the model's skill. Always ask what the score is measured on and whether the classes are balanced.
</details>

---

## Part 2: How models learn (class 2.2)

### Q11. [Warm-up] What is a loss function?

<details><summary>Answer</summary>

A single number that scores how wrong the model currently is. Lower is better; zero would be perfect. Training is the process of adjusting the model's parameters to make this number small.
</details>

### Q12. [Core] Write mean squared error and decode every symbol.

<details><summary>Answer</summary>

L = (1/n) * sum over i of (yhat_i - y_i)^2

- L is the loss, one number for total wrongness.
- n is the number of examples scored.
- y_i is the true answer for example i; yhat_i is the model's prediction for example i.
- (yhat_i - y_i) is the error on one example.
- squaring makes every error positive and punishes large misses far more than small ones.
- the sum adds the per-example errors and (1/n) averages them.

Read aloud: "on average, how far off are we, with big misses counting extra."
</details>

### Q13. [Deep] Loss versus accuracy. Why do we optimize loss instead of accuracy directly?

<details><summary>Answer</summary>

Loss is a smooth, continuous number, so a small change in a parameter produces a small, measurable change in loss, which is exactly what gradient descent needs. Accuracy is a blunt count that jumps in steps and is flat almost everywhere, so its gradient is zero or undefined and gives no direction to move. We minimize a smooth loss in order to improve the accuracy we actually care about.
</details>

### Q14. [Core] What is a gradient, and how is it used to reduce loss?

<details><summary>Answer</summary>

The gradient is the slope of the loss with respect to the parameters: the direction that increases loss fastest. To reduce loss we step in the opposite direction, downhill.

**Analogy:** fog on a hillside. You cannot see the bottom, but you can feel the slope under your feet and step downhill.
</details>

### Q15. [Core] Write the gradient descent update rule and decode it.

<details><summary>Answer</summary>

theta_new = theta_old - eta * grad_L

- theta is a parameter, one knob the model can turn (a weight).
- grad_L is the gradient of the loss with respect to that knob.
- the minus sign moves opposite the slope, that is, downhill.
- eta is the learning rate, the step size.

Read aloud: "new knob equals old knob minus a small step in the downhill direction."
</details>

### Q16. [Numerical] Given theta = 2.0, grad_L = 0.5, eta = 0.1, compute one update step.

<details><summary>Answer</summary>

theta_new = 2.0 - 0.1 * 0.5 = 2.0 - 0.05 = 1.95. The knob moved a small amount opposite the gradient, reducing the loss.
</details>

### Q17. [Deep] What does the learning rate control, and what happens if it is too small or too large?

<details><summary>Answer</summary>

It is the step size. Too small and learning crawls, taking many steps to make progress. Too large and you overshoot the valley, bouncing around or diverging so the loss blows up. There is a sweet spot.

**Common wrong answer:** "a bigger learning rate always learns faster." Past a point it destabilizes and the loss increases every step.
</details>

### Q18. [Applied] During training the loss increases every step. What is the most likely cause and fix?

<details><summary>Answer</summary>

The learning rate is too high, so each step overshoots and climbs the loss surface. Lower the learning rate (often by 3x or 10x) and rerun. Other candidates to rule out: a sign error in the update, bad input scaling, or exploding gradients, but the learning rate is the first knob to check.
</details>

### Q19. [Core] Full-batch, stochastic, and mini-batch gradient descent. What differs, and which is standard?

<details><summary>Answer</summary>

Only how much data feeds each gradient step. Full-batch uses the whole dataset (accurate direction, slow, memory-hungry). Stochastic uses one example at a time (fast, very noisy). Mini-batch uses a small group and is the practical standard: it fits in memory and its mild noise even helps escape shallow dips. The update rule is identical in all three.

**Common wrong answer:** "more data per step is always better." Mini-batches win on memory, speed, and often generalization.
</details>

### Q20. [Deep] Does gradient descent find the lowest possible loss?

<details><summary>Answer</summary>

Not necessarily. The loss landscape can have many valleys, so descent finds a low point, not guaranteed the global lowest. In practice that is usually good enough, and for large neural networks most local minima reach similar, acceptable loss. Learning rate schedules, momentum, and good initialization help avoid poor ones.
</details>

### Q21. [Core] Describe the training loop conceptually, in four moves.

<details><summary>Answer</summary>

Predict, measure the loss, compute the gradient, take a downhill step, then repeat many times until the loss settles. This exact cycle is what class 2.4 codes in PyTorch and is the engine behind every model in the course.
</details>

---

## Part 3: Neural networks (class 2.3)

### Q22. [Core] What does a single neuron compute? Write it and decode it.

<details><summary>Answer</summary>

z = (w_1*x_1 + ... + w_m*x_m) + b, then a = f(z)

- x_1..x_m are the inputs; w_1..w_m are the weights (how much each input matters).
- the weighted sum is a dot product, a weighted vote.
- b is the bias, a shift that makes the neuron easier or harder to activate.
- z is the pre-activation total, f is the activation, a is the output.

Read aloud: "weigh the inputs, add a bias, then pass through a gate."
</details>

### Q23. [Numerical] A layer maps 10 inputs to 4 neurons. How many parameters does it have?

<details><summary>Answer</summary>

Weights: 10 * 4 = 40. Biases: one per neuron = 4. Total = 44. In general a layer from m inputs to k neurons has m*k weights plus k biases.
</details>

### Q24. [Deep] Why do neural networks need nonlinear activations? What happens without them?

<details><summary>Answer</summary>

Without a nonlinearity, stacking layers still produces only a straight-line (linear) function, no matter how many layers, because a composition of linear maps is itself linear. The activation is what lets the network bend to model curves and complex boundaries. Nonlinearity, not depth alone, is what gives a network its expressive power.

**Common wrong answer:** "more layers alone make it powerful." Stacked linear layers collapse to a single linear map.
</details>

### Q25. [Core] Compare ReLU and sigmoid. When is each used?

<details><summary>Answer</summary>

ReLU is f(z) = max(0, z): keep positives, zero out negatives. It is the default in hidden layers because it is cheap and trains fast. Sigmoid is 1 / (1 + e^(-z)): it squashes any number into 0 to 1, useful for outputting a probability. Sigmoid is mostly an output-layer choice for binary problems, not a hidden-layer default.
</details>

### Q26. [Deep] What is the vanishing gradient problem with sigmoid, and how does ReLU help?

<details><summary>Answer</summary>

For large positive or negative inputs, sigmoid flattens and its slope is nearly zero. In a deep stack, gradients are products of these small slopes, so they shrink toward zero as they propagate back and early layers barely learn. ReLU has a slope of exactly 1 for positive inputs, so it does not squash the gradient in that region, which is a major reason it is the hidden-layer default.

**Strong answer adds:** ReLU has its own failure mode, dead units (stuck at zero), which variants like Leaky ReLU address.
</details>

### Q27. [Core] What does softmax do? Write it and decode it.

<details><summary>Answer</summary>

softmax(z)_j = e^(z_j) / sum over k of e^(z_k)

It turns a vector of raw scores (logits) into probabilities that are all positive and sum to 1. Exponentiating makes every score positive and amplifies larger scores; dividing by the sum over all classes normalizes them to add to 1.

Read aloud: "exponentiate each score, then share out to probabilities."
</details>

### Q28. [Numerical] Apply softmax to logits [2, 1, 0]. Which class wins, and roughly what probability?

<details><summary>Answer</summary>

e^2 = 7.39, e^1 = 2.72, e^0 = 1.00, sum = 11.11. Probabilities are about [0.665, 0.245, 0.090]. Class 0 wins with about 0.67. Note softmax preserves the order of the logits; the largest logit always gets the largest probability.
</details>

### Q29. [Core] What is cross-entropy loss, and why does it punish confident mistakes hardest?

<details><summary>Answer</summary>

For a single example, L = -log(p_true), where p_true is the probability the model assigned to the correct class. When p_true is near 1 (confident and right), -log is near 0. When p_true is near 0 (confident and wrong), -log shoots toward infinity, so a confident mistake is punished hard.

Read aloud: "how surprised the model was by the right answer."
</details>

### Q30. [Numerical] The model gives the true class a probability of 0.1. What is the cross-entropy loss? What if it gave 0.9?

<details><summary>Answer</summary>

-log(0.1) = 2.303 (natural log). -log(0.9) = 0.105. The confident-and-right case costs about 22x less. This steep curve is what pushes the model toward calibrated confidence on the correct class.
</details>

### Q31. [Warm-up] Which loss for which task: MSE or cross-entropy?

<details><summary>Answer</summary>

MSE for regression (predicting a number), cross-entropy for classification (predicting a class). Matching the loss to the task is a basic correctness requirement.
</details>

### Q32. [Deep] What is backpropagation, and how does it relate to gradient descent?

<details><summary>Answer</summary>

Backpropagation is how the gradients are computed efficiently. After the forward pass and the loss, the error is passed backward through the layers using the chain rule, giving each weight its own gradient. Gradient descent is the separate step that then uses those gradients to update the weights. They are two parts of one loop: backprop computes the direction, the update rule takes the step.

**Common wrong answer:** "backprop is a different learning algorithm from gradient descent." It is the gradient-computation half of the same process.
</details>

---

## Part 4: PyTorch fundamentals (class 2.4)

### Q33. [Warm-up] What is a tensor, and how does it differ from a NumPy array?

<details><summary>Answer</summary>

A tensor is like a NumPy array with two extra powers: it can live on a GPU for speed, and it can track the operations performed on it so gradients can be computed automatically (autograd). Everything in PyTorch flows as tensors.
</details>

### Q34. [Core] What is autograd, and what does calling .backward() do?

<details><summary>Answer</summary>

Autograd records every operation performed on tensors that require gradients, building a computation graph. Calling .backward() on the loss walks that graph backward and computes the gradient of the loss with respect to every parameter, storing it in each parameter's .grad. That .grad is exactly the grad_L from the class 2.2 update rule.

**Common wrong answer:** "PyTorch learns on its own." It computes gradients; you still write the loop that uses them.
</details>

### Q35. [Deep] Why do you pass raw logits, not softmax probabilities, to nn.CrossEntropyLoss?

<details><summary>Answer</summary>

nn.CrossEntropyLoss applies softmax (log-softmax) internally and expects raw logits plus the integer class index. If you softmax first and pass probabilities, softmax is applied twice, the numbers are wrong, and training degrades. It is also numerically more stable to combine log and softmax in one step.

**Common wrong answer:** adding a softmax layer before CrossEntropyLoss "to be safe." That is the bug.
</details>

### Q36. [Core] SGD versus Adam. What is the difference and which is a common default?

<details><summary>Answer</summary>

SGD applies the plain class 2.2 update, one shared learning rate for all parameters. Adam adapts the effective step size per parameter using running averages of past gradients, which usually converges faster with less tuning, so it is the common default. SGD with momentum can generalize better in some vision tasks, so it is still used.
</details>

### Q37. [Core] Write the five-line PyTorch training loop and say what each line does.

<details><summary>Answer</summary>

- pred = model(x): forward pass, predict.
- loss = loss_fn(pred, y): measure wrongness.
- optimizer.zero_grad(): clear old gradients.
- loss.backward(): compute new gradients (autograd).
- optimizer.step(): take the downhill step (the update rule).

Each maps directly to the class 2.2 cycle: predict, measure, compute gradient, step.
</details>

### Q38. [Deep] Why is optimizer.zero_grad() necessary? What happens if you forget it?

<details><summary>Answer</summary>

PyTorch accumulates gradients by default: each .backward() adds to the existing .grad rather than replacing it. If you never zero them, gradients from previous steps pile up, so each step uses a corrupted, inflated gradient and training breaks. Call zero_grad() once per step before backward().

**Strong answer adds:** the accumulation default is deliberately useful for simulating large batches (gradient accumulation across mini-batches before one step).
</details>

### Q39. [Warm-up] What do model.train() and model.eval() switch?

<details><summary>Answer</summary>

They toggle layers that behave differently during training versus inference, chiefly dropout (active in train, off in eval) and batch normalization (uses batch statistics in train, running statistics in eval). Set eval() before scoring and train() before training.
</details>

---

## Part 5: Training in practice (class 2.5)

### Q40. [Core] What is a DataLoader and why batch at all?

<details><summary>Answer</summary>

Real datasets are too large to process at once, so we feed the model small groups called batches. A DataLoader hands out shuffled batches from a Dataset. Batching fits memory and gives smoother, more stable updates than one example at a time. Batch size is a real knob trading off memory, speed, and stability.
</details>

### Q41. [Core] What is an epoch, and what makes an evaluation pass "honest"?

<details><summary>Answer</summary>

One epoch is one full pass over the training data; you usually run several. An evaluation pass runs the model on held-out data with learning turned off: model.eval() mode and torch.no_grad() (no gradient tracking). That gives an honest generalization score and is also faster and lighter on memory.

**Common wrong answer:** evaluating with training behavior still on (dropout active, gradients tracked).
</details>

### Q42. [Deep] How do you read overfitting off a learning curve?

<details><summary>Answer</summary>

Plot training loss and validation loss per epoch. While both fall, the model is still learning. When validation loss turns upward while training loss keeps falling, that upturn marks the onset of overfitting: the model is now fitting training noise. The best checkpoint is around the validation minimum.
</details>

### Q43. [Core] Name three ways to control overfitting and how each works.

<details><summary>Answer</summary>

Early stopping: keep the model from the epoch with the best validation loss instead of the last one. Weight decay: a gentle penalty on large weights (the optimizer's weight_decay) that discourages memorizing. Dropout: randomly zero some activations during training so the network cannot lean on any single path; it is on in train() mode and off in eval(), which is why the mode switch matters. More or augmented data also helps.
</details>

### Q44. [Applied] Your model runs on the GPU but a batch is on the CPU. What happens and how do you fix it?

<details><summary>Answer</summary>

You get a runtime device-mismatch error: operations need both tensors on the same device. Move both the model and each batch to the same device with .to(device), where device is chosen once near the top (cuda if torch.cuda.is_available(), else cpu), and write the rest of the code device-agnostic.
</details>

### Q45. [Core] How do you make a training run reproducible?

<details><summary>Answer</summary>

Set a fixed seed (torch.manual_seed, plus NumPy and Python seeds), which makes random initialization and shuffling repeat. Two runs with the same seed should give the same first-batch loss. Full determinism on a GPU may also require deterministic algorithm flags, at some speed cost.
</details>

### Q46. [Deep] What does state_dict save, and what does it not save?

<details><summary>Answer</summary>

state_dict saves the learned numbers (weights and biases, and buffers like batch-norm running stats). It does not save the model's code or architecture. To load it back you must first construct the same model class, then load the state_dict into it.

**Common wrong answer:** "saving the model saves the code too." You still need the model definition.
</details>

### Q47. [Deep] What is meant by a "learned representation," and why does it bridge to NLP?

<details><summary>Answer</summary>

The numbers a trained network produces in its inner layers are a learned representation: a vector that captures something meaningful about the input, shaped by the task. That idea, meaning encoded as a vector, is exactly what powers embeddings in Module 3 (NLP to Transformers) and Module 4 (LLMs and GenAI). Similar inputs land at nearby vectors.
</details>

---

## Rapid-fire (mixed tiers)

<details><summary>Reveal all rapid-fire answers</summary>

1. **Bias term, in one line?** A learnable shift that lets a neuron activate independently of the weighted input.
2. **Why square the error in MSE instead of taking absolute value?** Squaring is smoothly differentiable everywhere and penalizes large errors more; absolute error (MAE) is more robust to outliers but has a kink at zero.
3. **Logits?** The raw, un-normalized scores from the last layer before softmax.
4. **One-hot label?** A vector with 1 at the true class index and 0 elsewhere.
5. **Hyperparameter versus parameter?** Parameters are learned (weights, biases); hyperparameters are set by you (learning rate, batch size, layer count).
6. **What is a parameter count good for?** A rough proxy for model capacity and memory; more parameters can overfit more easily.
7. **Why shuffle the training data?** To break ordering artifacts so batches are representative and updates are less biased.
8. **Validation set in one line?** Held-out data used to tune choices without touching the test set.
9. **What does .to(device) do?** Moves a tensor or model to CPU or GPU memory.
10. **Why is Adam popular?** Per-parameter adaptive step sizes converge fast with little tuning.
11. **Confusion matrix?** A table of predicted versus true classes, exposing which errors the model makes.
12. **Underfitting fix in one line?** A more expressive model or better features (reduce bias).
</details>

---

## Whiteboard drills (do these on paper)

<details><summary>Reveal drills and solutions</summary>

1. **Compute MSE.** yhat = [3, 5], y = [2, 5]. Errors 1 and 0, squared 1 and 0, mean = 0.5.
2. **One descent step.** theta = 1.0, eta = 0.2, grad = 4. theta_new = 1.0 - 0.2*4 = 0.2.
3. **Param count.** Network 4 -> 8 -> 3 (two layers). Layer 1: 4*8 + 8 = 40. Layer 2: 8*3 + 3 = 27. Total 67.
4. **Softmax order.** Argue why the class with the largest logit always has the largest softmax probability. Because e^x is monotonically increasing and the denominator is shared, ordering is preserved.
5. **Cross-entropy sanity.** If a 3-class model outputs a uniform [1/3, 1/3, 1/3], the loss on any example is -log(1/3) = 1.099. This is the "no information" baseline for 3 classes.
6. **Double softmax bug.** Explain in two sentences what goes wrong if you apply softmax then pass to nn.CrossEntropyLoss. Softmax runs twice, flattening the distribution, so gradients weaken and the model learns slowly or wrongly.
</details>

---

## Senior / stretch questions (beyond course coverage)

These go past what Module 2 teaches directly. They are the kind of question a senior or staff candidate should handle. Each has a working answer, and a "Go deeper" pointer into the module's suggested reading list for the full treatment.

### S1. [Senior] State the bias-variance decomposition and what each term means.

<details><summary>Answer</summary>

For squared-error loss, the expected test error of a model decomposes into three parts: bias squared, variance, and irreducible noise. Bias is the error from wrong assumptions (too simple a model, underfitting). Variance is how much the fitted model changes across different training samples (too flexible a model, overfitting). Noise is the floor you cannot beat. Lowering one often raises the other, which is why model selection is a search for the balance, not the elimination, of both.

**Strong answer adds:** deep networks complicate the classic U-curve (see the double-descent question), so treat the decomposition as intuition, not a law for large models.

**Go deeper:** Dive into Deep Learning (d2l.ai), the model-selection and generalization chapter.
</details>

### S2. [Senior] Compare L1 and L2 regularization. How do they relate to weight decay?

<details><summary>Answer</summary>

Both add a penalty on weight magnitude to the loss to discourage overfitting. L2 (ridge) penalizes the sum of squared weights, shrinking all weights smoothly toward zero without making them exactly zero. L1 (lasso) penalizes the sum of absolute weights, which drives some weights to exactly zero, giving sparse models and a form of feature selection. L2 regularization is equivalent to weight decay in plain SGD (the optimizer's weight_decay), though with adaptive optimizers like Adam the two differ subtly, which is why decoupled weight decay (AdamW) exists.

**Go deeper:** Dive into Deep Learning (d2l.ai), the weight-decay and regularization sections.
</details>

### S3. [Senior] Overparameterized networks can memorize the training set, yet often generalize well. Why?

<details><summary>Answer</summary>

Two ideas. First, SGD has an implicit regularization effect: among the many settings that fit the training data, it tends to find flatter, simpler solutions that generalize better. Second, modern networks show "double descent": as capacity grows past the point of interpolating the training data, test error can fall again rather than keep rising, breaking the classic bias-variance U-curve. So raw parameter count is a poor proxy for overfitting risk in the deep-learning regime; the optimizer, data size, and regularization matter as much as capacity.

**Go deeper:** Neural Networks and Deep Learning (Michael Nielsen), the chapters on overfitting and why deep nets learn; Dive into Deep Learning (d2l.ai) on generalization.
</details>

### S4. [Senior] Why does weight initialization matter in a deep network, and what do Xavier and He initialization do?

<details><summary>Answer</summary>

If initial weights are too large, activations and gradients explode as they pass through layers; too small and they vanish, so early layers never learn. Good initialization keeps the variance of activations and gradients roughly constant across layers at the start of training. Xavier (Glorot) initialization scales the initial weights by the number of input and output units and suits symmetric activations like tanh. He initialization scales by the number of inputs and accounts for ReLU zeroing half its inputs, so it is the default for ReLU networks.

**Go deeper:** Dive into Deep Learning (d2l.ai), the numerical-stability and initialization section; Neural Networks and Deep Learning (Nielsen) on the vanishing-gradient problem.
</details>

### S5. [Senior] Explain batch normalization: what it normalizes, why it helps, and how train and inference differ.

<details><summary>Answer</summary>

Batch norm normalizes each feature across the examples in a mini-batch to zero mean and unit variance, then applies a learned scale and shift. It helps by keeping the distribution of layer inputs stable during training, which allows higher learning rates and faster, more robust convergence, and it adds mild regularizing noise. At training time it uses the current batch's statistics; at inference it uses running averages accumulated during training, because a single example has no meaningful batch statistics. This is one reason model.eval() matters, and why batch norm behaves poorly with very small batch sizes (LayerNorm or GroupNorm are used instead in those cases, including transformers).

**Go deeper:** Dive into Deep Learning (d2l.ai), the batch-normalization chapter; PyTorch tutorials for the API behavior.
</details>

### S6. [Senior] Go beyond plain SGD: what do momentum, Adam, and learning-rate warmup add?

<details><summary>Answer</summary>

Momentum accumulates a running average of past gradients so updates keep moving through small, noisy, or flat regions and dampen oscillation across steep valleys. Adam combines momentum (a first-moment estimate) with a per-parameter scale from a second-moment estimate (average squared gradient), so each parameter gets an adapted step size; it converges fast with little tuning. Learning-rate schedules decay the rate over training so early steps move fast and later steps settle; warmup starts the rate small for the first steps to avoid early instability, which is especially important for training transformers and with adaptive optimizers.

**Go deeper:** Dive into Deep Learning (d2l.ai), the optimization-algorithms chapter.
</details>

### S7. [Senior] What are vanishing and exploding gradients in deep networks, and what are the standard fixes?

<details><summary>Answer</summary>

Gradients are products of many per-layer terms as they propagate backward. If those terms are consistently below 1, the product shrinks toward zero (vanishing) and early layers stop learning; if consistently above 1, it grows without bound (exploding) and training diverges. Fixes stack together: ReLU-family activations and careful initialization keep per-layer factors near 1, normalization layers stabilize the scale, residual (skip) connections give gradients a direct path around each block, and gradient clipping caps the norm to tame explosions (common in recurrent and transformer training).

**Go deeper:** Neural Networks and Deep Learning (Nielsen), the vanishing-gradient chapter; Dive into Deep Learning (d2l.ai) on numerical stability.
</details>

### S8. [Senior] In production, accuracy is not enough. Discuss probability calibration and handling class imbalance.

<details><summary>Answer</summary>

A model can rank examples well yet output miscalibrated probabilities, meaning its stated 0.9 confidence is not right 90 percent of the time; you check this with a reliability diagram and fix it with Platt scaling or isotonic regression, or temperature scaling for neural nets. On imbalanced data, accuracy is dominated by the majority class, so you evaluate with precision, recall, F1, and especially the precision-recall curve and its area, and you tune the decision threshold to the cost of each error rather than defaulting to 0.5. Training-side levers include class weighting and resampling. In deployment you also monitor for data and concept drift, because a model that was calibrated at launch degrades as the input distribution shifts.

**Go deeper:** scikit-learn user guide, the sections on probability calibration, model evaluation metrics, and imbalanced classification.
</details>
