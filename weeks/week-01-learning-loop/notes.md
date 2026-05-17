# Manual Line Learner Notes

## What the model started with

The hidden rule for the clean dataset was:

```text
y = 2x + 1
```

The model started with intentionally wrong values:

```text
weight = 0.0
bias = 0.0
```

With those starting values, every prediction was `0.0`, so the model was wrong for most inputs.

## What changed during training

During training, the model repeatedly:

1. made predictions using the current weight and bias,
2. measured how wrong those predictions were,
3. computed gradients for the weight and bias,
4. adjusted the weight and bias a little,
5. repeated the process.

The weight moved toward `2`, which is the slope of the hidden rule.
The bias moved toward `1`, which is the value of the rule when `x = 0`.

## Why the loss went down

The loss went down because each update moved the model parameters in a direction that reduced the average squared error.

If a gradient was positive, subtracting it made the parameter smaller.
If a gradient was negative, subtracting it made the parameter larger.

The learning rate made each update small, so the model improved gradually instead of jumping too far.

## Final result

After training, the model reached approximately:

```text
weight = 1.9999999999999996
bias = 0.9824120533942785
```

These values are very close to the hidden rule:

```text
weight = 2
bias = 1
```

## What I learned

Machine learning can be understood as repeated correction of adjustable knobs.
In this small example, the knobs were only `weight` and `bias`.
More complex models can have many more knobs, but the basic loop is similar:

```text
predict -> measure error -> compute gradients -> update parameters -> repeat
```
