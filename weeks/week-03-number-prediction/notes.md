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

A normalized value is not a percentage and not a maximum. It tells me how far a
feature is from the training average:

- `0.0` means about average
- `+1.0` means one standard deviation above average
- `-2.0` means two standard deviations below average

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

### How the model learns the target units

The input features are normalized, but the target `MedHouseVal` is not.

That means the model is trained to answer:

```text
given normalized inputs, output a number close to the original MedHouseVal
```

PyTorch does not know about dollars. The unit comes from the target numbers used
in the loss:

```python
loss = loss_function(predictions, y_train_tensor)
```

Because `y_train_tensor` is in `MedHouseVal` units, the model's predictions learn
to be in `MedHouseVal` units too.

The normalized input values are unitless. A value like `+1.0` means one standard
deviation above the training average, not the maximum.

## Final summary

### Loss and training

Loss is one number that measures how wrong the model's predictions are.

For this regression project, I used mean squared error:

```text
error = prediction - true_value
squared_error = error * error
loss = average squared_error
```

The training loop repeated this pattern:

```text
predict -> measure loss -> compute gradients -> update weights and bias
```

#### How PyTorch computes the 8 weight gradients

In Week 1, the model had one input `x`, so the hand-written weight gradient was:

```text
weight_gradient = average(2 * errors * x)
```

Week 3 uses the same idea, but the model has 8 input columns instead of 1.
Each input column gets its own weight, so PyTorch computes one adjustment hint
for each weight:

```text
gradient_for_weight_1 = average(2 * errors * MedInc)
gradient_for_weight_2 = average(2 * errors * HouseAge)
gradient_for_weight_3 = average(2 * errors * AveRooms)
gradient_for_weight_4 = average(2 * errors * AveBedrms)
gradient_for_weight_5 = average(2 * errors * Population)
gradient_for_weight_6 = average(2 * errors * AveOccup)
gradient_for_weight_7 = average(2 * errors * Latitude)
gradient_for_weight_8 = average(2 * errors * Longitude)
```

So the general rule is:

```text
gradient_for_one_weight = average(2 * errors * that_weight's_input_column)
```

The bias does not belong to any input column, so its gradient is:

```text
bias_gradient = average(2 * errors)
```

`loss.backward()` does this math automatically. It traces how each prediction
used the 8 weights and the bias, then fills in `model.weight.grad` and
`model.bias.grad` with the gradients.

The important simple idea is that Week 3 is not a different kind of gradient.
It is the Week 1 gradient formula applied once per feature column.

The loss curve went down strongly, which shows that the model learned a better
set of weights and bias than its random starting values.

### Evaluation

The final losses were close:

```text
train loss: about 0.56
test loss: about 0.58
```

The closeness matters because the test set contains rows the model did not train
on. Similar train and test loss suggest the model generalized reasonably for a
simple baseline.

I also computed RMSE, which is easier to interpret because it is back in
`MedHouseVal`-like units:

```text
test RMSE: about 0.76 to 0.80 MedHouseVal units
test RMSE in dollars: about $76,000 to $80,000
```

That means this first baseline is often off by roughly that amount on held-back
examples.

### Prediction plot

The predicted-vs-actual plot compares one test example per dot:

- dots on the red diagonal line are perfect predictions
- dots above the line are overestimates
- dots below the line are underestimates
- dots far from the line have larger errors

The plot shows a general upward trend, so the model learned a useful relationship.
It also shows wide spread, so the model is still imperfect.

The vertical stripe near actual `5.0` suggests the dataset has capped high-value
homes, which is one reason real data can behave differently from clean toy data.

### Saved artifacts

This lab produced:

- `data/raw/week-03-california-housing.csv`
- `models/week-03-number-predictor.pt`
- `artifacts/week-03-number-prediction/loss_curve.png`
- `artifacts/week-03-number-prediction/predicted_vs_actual.png`
- `artifacts/week-03-number-prediction/example_predictions.csv`

### Main lesson

A PyTorch model does not understand dollars or houses. It only sees numbers.
The meaning comes from the dataset and the target column.

Because the input features were normalized but the target stayed in original
`MedHouseVal` units, the model learned to map normalized inputs to original-unit
house value predictions.

The model learned something real, but a single linear layer is only a baseline.
Real housing values depend on messy factors, capped values, and patterns that may
not be fully captured by a straight weighted recipe over 8 columns.
