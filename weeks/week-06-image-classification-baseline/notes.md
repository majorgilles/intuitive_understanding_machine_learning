# Week 06 — Fashion-MNIST Baseline Notes

## What exists so far

We created the notebook:

`weeks/week-06-image-classification-baseline/train_fashion_mnist_baseline.ipynb`

So far it:

- finds the project root
- downloads/loads Fashion-MNIST with `torchvision`
- creates a `DataLoader`
- inspects one batch of images and labels
- saves `sample_images.png`
- defines a first baseline model
- runs one optimizer update on one batch

## Image shape

A batch has shape:

`[32, 1, 28, 28]`

Meaning:

- `32`: number of images in the batch
- `1`: one grayscale channel
- `28`: image height in pixels
- `28`: image width in pixels

One image has shape:

`[1, 28, 28]`

The visible pixel grid is:

`[28, 28]`

## Pixel values

`ToTensor()` converts each image into numbers from `0.0` to `1.0`.

Rough meaning:

- `0.0`: dark/background pixel
- `1.0`: bright pixel
- values between are gray levels

## Labels

Each image has a label number.

`train_dataset.classes` maps label numbers to class names such as:

- `Trouser`
- `Shirt`
- `Sneaker`

## Baseline model shape story

The baseline model does:

`[32, 1, 28, 28] -> [32, 784] -> [32, 10]`

Explanation:

1. `nn.Flatten()` turns each image into 784 pixel values.
2. `nn.Linear(784, 10)` produces 10 raw class scores.
3. The highest score will become the predicted clothing class later.

## Logits vs probabilities

The model outputs raw class scores called logits.

For one batch, the shape is:

`[32, 10]`

Meaning:

- `32`: images in the batch
- `10`: one raw score for each clothing class

These are not probabilities yet. They do not have to be between `0.0` and `1.0`.

## Cross-entropy

Cross-entropy punishes the model when it gives low probability to the correct class.

For one example:

`cross entropy = -log(probability assigned to the correct class)`

Examples:

- correct class probability `0.90` -> low loss
- correct class probability `0.10` -> high loss
- correct class probability `0.01` -> very high loss

In PyTorch, `nn.CrossEntropyLoss()` expects raw logits. It applies the needed softmax-like probability calculation internally.

## CrossEntropyLoss vs BCE

Use `CrossEntropyLoss` when each example has exactly one correct class.

Fashion-MNIST fits this:

`one image -> one clothing class`

Use BCE, Binary Cross Entropy, for yes/no questions or multi-label problems.

Examples:

- yes/no: `is this a shirt?`
- multi-label: `has shirt`, `has hat`, `has shoe`

Fashion-MNIST is not multi-label because one image is labeled as one class.

## One-batch sanity check

The one-batch update produced:

- loss before one update: `2.3484`
- loss after one update: `2.2696`

This does not prove the whole model is trained yet. It only confirms the basic learning pipeline works:

`logits -> loss -> gradients -> optimizer step -> changed weights`

## Current checkpoint

Completed:

- loaded Fashion-MNIST
- inspected image and batch shapes
- saved sample image grid
- built a flatten-then-linear baseline model
- ran one training update on one batch

Next learning step:

- write a full training loop over many batches
- track average train loss
- measure test accuracy
