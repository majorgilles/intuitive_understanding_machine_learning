# Week 05 — Hidden Layer Lab Notes

## Goal

Compare a linear classifier with a one-hidden-layer neural network on the same nonlinear moon-shaped dataset.

## Dataset findings

We used `make_moons` with:

- `n_samples=500`
- `noise=0.25`
- `random_state=42`

The generated table has shape `(500, 3)`:

- `x_position`
- `y_position`
- `class_label`

## Train/test split findings

The dataset was split into:

- 400 training rows
- 100 test rows

`X_train` has shape `(400, 2)` because each row has two input features.

`y_train` has shape `(400,)` before tensor conversion because each row has one class label.

## Tensor findings

The tensors have shapes:

- `X_train_tensor`: `[400, 2]`
- `y_train_tensor`: `[400, 1]`
- `X_test_tensor`: `[100, 2]`
- `y_test_tensor`: `[100, 1]`

The labels were reshaped to column tensors so they match the model output shape.

## Logits and sigmoid

The linear model outputs logits.

Sigmoid converts logits into probabilities between `0` and `1`.

Example observed probabilities:

- `0.578782`
- `0.668075`
- `0.418066`
- `0.393962`
- `0.539528`

## Linear classifier findings

The linear model uses:

```text
score = weight_1 * x_position + weight_2 * y_position + bias
```

It has:

- 2 weights
- 1 bias

After the current notebook run, the linear model reached:

- final train loss: `0.359879`
- train accuracy: `0.825`
- test accuracy: `0.810`

The loss kept decreasing slowly, but the model still appears limited.

Current interpretation: this is probably a model-capacity limit because the curved moon pattern cannot be cleanly separated by one straight boundary.

## MLP interpretation

An MLP stands for multilayer perceptron, but the biological language is only a metaphor.

In code, the model is a chain of functions:

```text
logit = linear_2(relu(linear_1(input_point)))
```

The model contains stored weights and biases. Training changes those stored numbers.

## Hidden units

`HIDDEN_UNITS = 16` means the first linear layer creates 16 intermediate output numbers for each input row.

It does not mean there are only 16 weights.

Because each input row has 2 features, each hidden unit needs 2 weights and 1 bias:

```text
hidden_raw_i = weight_ix * x_position + weight_iy * y_position + bias_i
```

So the first layer has:

- `16 * 2 = 32` weights
- `16` biases
- `48` learned numbers total

## ReLU and named parameters

The model structure is:

```text
0: Linear(2 -> 16)
1: ReLU()
2: Linear(16 -> 1)
```

`ReLU` does not appear in `named_parameters()` because it has no learned weights or biases.

It still matters because it changes the function from one big linear recipe into a nonlinear chain that can bend the decision boundary.

## Connection-first diagram

The notebook draws a Pyplot diagram saved to:

```text
artifacts/week-05-small-neural-networks/mlp_connections.png
```

The diagram shows the model as:

```text
2 inputs -> 16 hidden values -> 1 logit
```

Weighted-operation counts:

- `2 * 16 = 32` input-to-hidden weighted operations
- `16 * 1 = 16` hidden-to-output weighted operations
- `48` visible weighted operations total

The circles are values.

The lines are weighted operations: each line contains a learned weight and the multiplication that uses it.

For one hidden output:

```text
hidden_raw_i = weight_ix * x_position + weight_iy * y_position + bias_i
```

So the input-to-hidden lines are not only stored weights. They are also the math operations that multiply an input value by its learned weight before the results are added together.

This better matches the concrete code view:

```text
logit = linear_2(relu(linear_1(input_point)))
```

## MLP training findings

The MLP was trained for 2000 epochs with Adam.

Observed result:

- final MLP train loss: `0.0869`
- MLP train accuracy: `0.960`
- MLP test accuracy: `0.930`

This is much better than the linear baseline.

## Model comparison findings

Current comparison table:

- linear train accuracy: `0.825`
- linear test accuracy: `0.810`
- linear final loss: `0.359879`
- MLP train accuracy: `0.960`
- MLP test accuracy: `0.930`
- MLP final loss: `0.086871`

The MLP has lower loss and higher test accuracy, so it learned the curved pattern better.

## Decision boundary findings

The linear model produced a mostly straight boundary.

The MLP produced a nonlinear boundary that bends around the moons and covers most of the red/class `1` points better.

This supports the main lesson: hidden intermediate features plus ReLU let the model build a more flexible function than one direct linear recipe.
