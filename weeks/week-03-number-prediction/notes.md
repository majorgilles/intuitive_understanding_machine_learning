# Week 3 — Real Number Predictor Notes

## What I learned today

### Dataset, features, and target

The California Housing dataset is a table of examples.

- Each row is one housing example.
- The feature columns are the inputs the model can use.
- The target column is the answer the model tries to predict.

For this project:

- `MedHouseVal` is the target.
- The other columns are features.
- This is a regression problem because the target is a number, not a category.

`MedHouseVal` is measured in units of $100,000. For example, a prediction of `2.35`
means about `$235,000`.

### Why ML examples often use `X` and `y`

`X` usually means the feature matrix: many rows and many input columns.

`y` usually means the target vector: one answer value for each row.

In this project:

- `X` has 8 feature columns.
- `y` has 1 target value per row.

### Train/test split

The training data is used to adjust the model.

The test data is held back so I can check whether the model works on examples it
did not train on.

My split created:

- `X_train`: 16,512 rows and 8 feature columns
- `X_test`: 4,128 rows and 8 feature columns
- `y_train`: 16,512 target values
- `y_test`: 4,128 target values

### Normalization

Different feature columns can have very different number ranges. A model trains
more smoothly when the input features are on similar scales.

The normalization formula is:

```text
normalized_value = (original_value - mean) / standard_deviation
```

I calculated the mean and standard deviation from the training data only. Then I
used those same training statistics to normalize both the training inputs and the
test inputs.

This avoids data leakage, which means accidentally letting information from the
test set influence training.

After normalization, the mean of each training feature was approximately `0`.

### PyTorch tensors and shapes

A tensor is PyTorch's container for numbers.

The input tensors have this shape:

```text
X_train_tensor: [16512, 8]
X_test_tensor: [4128, 8]
```

That means:

- 16,512 training rows
- 4,128 test rows
- 8 input features per row

The target tensors have this shape:

```text
y_train_tensor: [16512, 1]
y_test_tensor: [4128, 1]
```

The `.reshape(-1, 1)` step turns a plain list of target values into a one-column
table of target values.

The `-1` means: “PyTorch, figure out the number of rows automatically.”

### The first model

The first model is:

```text
Linear(in_features=8, out_features=1)
```

This means the model receives 8 input numbers per row and outputs 1 predicted
number per row.

The model learns 8 weights and 1 bias.

Conceptually, the prediction looks like:

```text
prediction =
    weight_1 * MedInc
  + weight_2 * HouseAge
  + ...
  + weight_8 * Longitude
  + bias
```

Because this is regression, the output is not a logit. It is directly interpreted
as the predicted `MedHouseVal`.

## Next learning step

The next concept is loss.

Loss is one number that measures how wrong the model's predictions are. For this
regression project, I will use mean squared error.
