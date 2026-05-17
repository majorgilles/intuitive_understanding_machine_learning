# PyTorch Gradient Learner Notes

## What tensors are

A tensor is PyTorch's version of an array of numbers.
In this project, `x` and `y` are tensors so PyTorch can run the math on the selected device, either CPU or CUDA.

## What the model learned

The model used the same simple line rule shape as Week 1:

```text
prediction = weight * x + bias
```

It started with `weight = 0` and `bias = 0`, so every prediction was wrong at first.
After training, the weight moved close to `2` and the bias moved close to `1`, matching the hidden rule `y = 2x + 1`.

## What gradients mean

A gradient is an adjustment hint for one parameter.
If a gradient is negative, subtracting it makes the parameter go up.
If a gradient is positive, subtracting it makes the parameter go down.

## What backward() does

`backward()` asks PyTorch to calculate the gradients for the parameters that affected the loss.
In simple words, PyTorch traces backward from the loss and fills in adjustment hints like `weight.grad` and `bias.grad`.

## What the optimizer does

The optimizer updates the parameters it was given when it was created.
In this project, `torch.optim.SGD([weight, bias], LEARNING_RATE)` was given `weight` and `bias`, so `optimizer.step()` updates those two values using their gradients.

For SGD, the update idea is:

```text
new parameter = old parameter - learning rate * gradient
```

## How this compares to Week 1

The visible learning loop stayed the same:

```text
predict -> measure wrongness -> get adjustment hints -> adjust -> repeat
```

The difference is that Week 1 calculated the gradients by hand.
This PyTorch version lets `backward()` calculate the gradients and lets the optimizer apply the updates.
