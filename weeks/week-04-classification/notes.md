# Week 04 — Decision Boundary Lab Notes

## Goal

Build intuition for binary classification by training a model that separates 2D points into class `0` or class `1`.

Each row has:

- `x_position`
- `y_position`
- `class_label`

## Project root and saved files

The notebook resolves the project root by looking for `pyproject.toml`.

This keeps generated files saving to the intended repo folders, even if Jupyter starts from a different working directory.

## Dataset findings

We used `make_blobs` to create a reproducible 2D dataset.

Important details:

- `random_state=42` makes the same points each run.
- `n_samples=300` creates 300 rows.
- The saved CSV has 3 columns.
- The pandas display index is not part of the CSV because we save with `index=False`.

`make_blobs` assigns labels by center order:

- first center -> class `0`
- second center -> class `1`

Changing centers changes the meaning of the labels.

## Train/test split findings

The data was split into:

- 240 training rows
- 60 test rows

`stratify=labels` preserved class balance:

- train: 120 class `0`, 120 class `1`
- test: 30 class `0`, 30 class `1`

`np.bincount` counts how many times each integer label appears.

## Tensor shape findings

The model input tensor has shape:

```text
(rows, 2)
```

because each point has two input features.

The label tensor has shape:

```text
(rows, 1)
```

because each row has one binary answer.

## Linear classifier intuition

The model uses:

```python
nn.Linear(in_features=2, out_features=1)
```

`in_features=2` means the model receives:

- `x_position`
- `y_position`

The bias is not an input feature. It is an extra learned number.

The model computes:

```text
score = weight_1 * x_position + weight_2 * y_position + bias
```

## Logits and probabilities

The linear model outputs a raw score called a logit.

A logit becomes a probability with sigmoid:

```text
probability = sigmoid(logit)
```

Interpretation:

- probability near `0` -> class `0`
- probability near `1` -> class `1`
- probability near `0.5` -> unsure

## Loss function finding

We use:

```python
nn.BCEWithLogitsLoss()
```

This means PyTorch combines:

```text
sigmoid(logit) + binary cross entropy
```

It rewards confident correct predictions and punishes confident wrong predictions.

A prediction alone is not enough to calculate BCE.

For example, if the model predicts:

```text
p(class 1) = 0.95
```

that prediction is:

- very good if the true label is `1`
- very bad if the true label is `0`

So BCE always needs both:

```text
model prediction + true label -> wrongness score
```

In code, that is why we call:

```python
loss = loss_function(logits, y_train_tensor)
```

The logits say what the model guessed. The labels say what the correct answers were.

### BCE math notation

For one training example:

```text
y = true label, either 0 or 1
p = model's predicted probability that y is 1
```

Binary cross entropy is:

```text
BCE(y, p) = -[y * log(p) + (1 - y) * log(1 - p)]
```

This single formula handles both label cases.

If the true label is `1`, then `y = 1`:

```text
BCE(1, p) = -log(p)
```

So the loss is small when `p` is close to `1`.

If the true label is `0`, then `y = 0`:

```text
BCE(0, p) = -log(1 - p)
```

So the loss is small when `p` is close to `0`.

Short version:

```text
BCE = -log(probability assigned to the true class)
```

For a batch of rows, PyTorch averages this penalty across the rows.

Rough intuition:

```text
~0.69 = random-ish
~0.22 = already pretty good
2.40  = bad / confidently wrong
near 0 = very confident and correct
```

## Initial model findings

An untrained linear model is not blank. It starts with random weights and bias.

Those random starting values already create a decision boundary.

Depending on the dataset layout, the untrained model can be:

- accidentally good
- mostly wrong
- uncertain

This is why measuring initial accuracy and initial loss is useful before training.

## Current lesson takeaway

Classification is not just “predict a number.”

For binary classification:

1. the model outputs a raw score,
2. sigmoid turns the score into a probability,
3. a threshold turns the probability into class `0` or class `1`,
4. BCE loss measures how wrong and how confident the prediction is,
5. training changes weights and bias to move the decision boundary.
