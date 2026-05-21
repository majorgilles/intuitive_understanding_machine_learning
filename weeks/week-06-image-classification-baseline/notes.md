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

## Current checkpoint

We have not trained the model yet.

Next learning step:

- define a loss function
- define an optimizer
- run a small training loop
- measure test accuracy
