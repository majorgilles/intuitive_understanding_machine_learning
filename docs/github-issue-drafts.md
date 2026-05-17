# GitHub Issue Drafts

These are the GitHub issue bodies for a project about understanding how machines work and learn by building small local ML projects.

Each issue should be direct, concrete, and focused on observable behavior: data, tensors, guesses, loss, gradients, boundaries, images, mistakes, or debugging.

Every issue is **HITL** because the learner must run the work, inspect outputs, make a small judgment, and write a short explanation.

Labels to use on all issues:

- `HITL`
- `course`
- `beginner`

Other labels used where useful:

- `setup`
- `learning`
- `pytorch`
- `weekly-project`
- `debugging`
- `computer-vision`
- `capstone`

---

## Issue 1: Set up the local ML lab and smoke-test PyTorch/CUDA

**Type:** HITL  
**Labels:** `HITL`, `course`, `setup`, `pytorch`, `beginner`  
**Time estimate:** 2–4 hours  
**Blocked by:** None

### Project name

**Local ML Lab Smoke Test**

### Why this exists

Before learning ML, make sure the local machine can run the course. The goal is not to perfect the environment forever. The goal is to get a working local path with CPU fallback and optional CUDA on the RTX 4070 SUPER.

### What to build

Create a tiny environment-check project that proves Python, uv, PyTorch, torchvision, Jupyter, and device selection work.

### Exact files to create

- `weeks/week-00-setup/check_environment.py`
- `weeks/week-00-setup/setup_report.md`
- `src/mlcourse/__init__.py`
- `src/mlcourse/device.py`
- `artifacts/week-00-setup/device_check.txt`

### Online resources

- uv docs: https://docs.astral.sh/uv/
- PyTorch Start Locally: https://docs.pytorch.org/get-started/locally/
- PyTorch Learn the Basics: https://docs.pytorch.org/tutorials/beginner/basics/intro.html

### Step-by-step work

- [ ] Confirm `uv --version` works.
- [ ] Create a Python 3.12 virtual environment with `uv`.
- [ ] Install learning packages: `numpy`, `matplotlib`, `pandas`, `scikit-learn`, `jupyter`, `ipykernel`, `tqdm`, `torchmetrics`.
- [ ] Install PyTorch and torchvision using the official PyTorch selector.
- [ ] Create `src/mlcourse/device.py` with a function named `get_device()`.
- [ ] `get_device()` should return CUDA if available, otherwise CPU.
- [ ] Create `check_environment.py` that prints:
  - Python version
  - PyTorch version
  - torchvision version
  - whether CUDA is available
  - GPU name if available
  - result of a tiny tensor calculation
- [ ] Save the same output into `artifacts/week-00-setup/device_check.txt`.
- [ ] Write `setup_report.md` with:
  - what worked
  - what failed
  - whether CUDA works
  - what command should be used to run Python files in this repo

### Command that should work

```bash
uv run python weeks/week-00-setup/check_environment.py
```

### Human checkpoint

Read the output and decide:

- Is the repo ready for CPU learning?
- Is CUDA working?
- If CUDA is not working, is that okay for now?

### Debugging checklist

- [ ] If `uv` fails, check it is on PATH.
- [ ] If `torch` import fails, reinstall PyTorch from the official selector.
- [ ] If CUDA is false, confirm the NVIDIA driver is installed, but do not block the course forever.
- [ ] If Jupyter is annoying, continue with scripts first.

### Acceptance criteria

- [ ] `uv run python weeks/week-00-setup/check_environment.py` runs.
- [ ] `artifacts/week-00-setup/device_check.txt` exists.
- [ ] `setup_report.md` says whether CUDA works.
- [ ] `src/mlcourse/device.py` exists and provides `get_device()`.
- [ ] The learner can explain the difference between CPU and GPU in one or two plain sentences.

### Expertise gained

After this issue, I can run a local PyTorch project, check whether my GPU is available, and continue safely even if CUDA is temporarily not working.

---

## Issue 2: Week 1 — Manual Line Learner: watch numbers learn

**Type:** HITL  
**Labels:** `HITL`, `course`, `learning`, `beginner`, `weekly-project`  
**Time estimate:** 4–6 hours  
**Blocked by:** #1

### Project name

**Manual Line Learner**

### Why this exists

Frameworks can hide the learning process. This project makes learning visible by using plain Python/NumPy first.

### What to build

Build a tiny model that learns the rule:

```text
y = 2x + 1
```

The model should start with bad guesses for two knobs:

- `weight`
- `bias`

Then it should repeatedly improve those knobs and show the loss going down.

### Dataset

Generate the data inside the script.

Use this exact clean dataset first:

```python
x = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]
y = 2 * x + 1
```

No download needed.

### Exact files to create

- `weeks/week-01-learning-loop/manual_line_learner.py`
- `weeks/week-01-learning-loop/notes.md`
- `artifacts/week-01-learning-loop/loss_curve.png`
- `artifacts/week-01-learning-loop/before_after_predictions.csv`

### Online resources

- ML4A: How neural networks are trained: https://ml4a.github.io/ml4a/how_neural_networks_are_trained/
- TensorFlow Neural Network Playground: https://playground.tensorflow.org/
- Google Machine Learning Crash Course: https://developers.google.com/machine-learning/crash-course

### Step-by-step work

- [ ] Create the dataset in code.
- [ ] Start with intentionally wrong values, for example `weight = 0.0` and `bias = 0.0`.
- [ ] Write a `predict(x, weight, bias)` function.
- [ ] Write a simple loss function: average squared wrongness.
- [ ] Print the first predictions before learning.
- [ ] Run a training loop for about 100–500 steps.
- [ ] At each step, adjust `weight` and `bias` a little.
- [ ] Save loss values.
- [ ] Plot the loss curve to `loss_curve.png`.
- [ ] Save before/after predictions to `before_after_predictions.csv`.
- [ ] Write `notes.md` explaining what changed and why loss went down.

### Important constraint

Do **not** use PyTorch autograd in this issue. The point is to see the adjustment process without framework magic.

### Modification challenge

Change the hidden rule to:

```text
y = -3x + 2
```

Before running, write a prediction in `notes.md`:

- What should the final `weight` become?
- What should the final `bias` become?

Then run it and compare.

### Debugging checklist

- [ ] If loss goes up, the adjustment direction may be backwards.
- [ ] If loss changes too slowly, the step size may be too small.
- [ ] If loss explodes, the step size may be too large.
- [ ] If final predictions are bad, print the first five predictions every 50 steps.

### Acceptance criteria

- [ ] `manual_line_learner.py` runs with `uv run python weeks/week-01-learning-loop/manual_line_learner.py`.
- [ ] The script prints starting and ending `weight` and `bias`.
- [ ] Final `weight` and `bias` are close to the hidden rule.
- [ ] `loss_curve.png` exists and shows loss going down.
- [ ] `before_after_predictions.csv` exists.
- [ ] `notes.md` explains the learning loop in plain words.

### Expertise gained

After this issue, I can explain machine learning as repeated correction of adjustable knobs.

---

## Issue 3: Week 2 — PyTorch Gradient Learner: same line project with `backward()`

**Type:** HITL  
**Labels:** `HITL`, `course`, `pytorch`, `learning`, `beginner`, `weekly-project`  
**Time estimate:** 4–6 hours  
**Blocked by:** #2

### Project name

**PyTorch Gradient Learner**

### Why this exists

Week 1 showed learning by hand. This week repeats the same idea in PyTorch so the framework feels less magical.

### What to build

Rebuild the line learner using PyTorch tensors, parameters, loss, `backward()`, and an optimizer.

The visible idea should stay the same:

```text
predict -> measure wrongness -> ask PyTorch for adjustment hints -> adjust -> repeat
```

### Dataset

Generate the same clean line dataset inside the script:

```python
x = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]
y = 2 * x + 1
```

### Exact files to create

- `weeks/week-02-pytorch-basics/torch_line_learner.py`
- `weeks/week-02-pytorch-basics/notes.md`
- `artifacts/week-02-pytorch-basics/gradient_trace.csv`
- `artifacts/week-02-pytorch-basics/loss_curve.png`
- `artifacts/week-02-pytorch-basics/cpu_or_cuda.txt`

### Online resources

- Learn PyTorch — 00 PyTorch Fundamentals: https://www.learnpytorch.io/00_pytorch_fundamentals/
- PyTorch Tensors: https://docs.pytorch.org/tutorials/beginner/basics/tensorqs_tutorial.html
- PyTorch Autograd: https://docs.pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html

### Step-by-step work

- [ ] Create tensors for `x` and `y`.
- [ ] Use `get_device()` from `src/mlcourse/device.py` if it exists.
- [ ] Create `weight` and `bias` tensors with `requires_grad=True`, or use `torch.nn.Linear` after first trying raw tensors.
- [ ] Compute predictions.
- [ ] Compute mean squared error loss.
- [ ] Call `loss.backward()`.
- [ ] Print gradients for the first few training steps.
- [ ] Use an optimizer such as `torch.optim.SGD`.
- [ ] Save a CSV with columns: `epoch`, `loss`, `weight`, `bias`, `weight_grad`, `bias_grad`.
- [ ] Plot loss.
- [ ] Write notes explaining `backward()` as “PyTorch calculating adjustment hints.”

### Modification challenge

Run the project twice:

1. learning rate `0.001`
2. learning rate `0.1`

Before running, predict which one will learn faster or become unstable. Then explain what actually happened.

### Debugging checklist

- [ ] If gradients are `None`, check `requires_grad=True`.
- [ ] If loss does not change, check that `optimizer.step()` is called.
- [ ] If loss explodes, lower the learning rate.
- [ ] If CPU/GPU errors happen, check all tensors are on the same device.

### Acceptance criteria

- [ ] `torch_line_learner.py` runs.
- [ ] It prints device information.
- [ ] It calls `backward()`.
- [ ] `gradient_trace.csv` exists and contains gradient values.
- [ ] `loss_curve.png` exists.
- [ ] `notes.md` explains tensors, gradients, and `backward()` in simple words.

### Expertise gained

After this issue, I can use PyTorch tensors and explain autograd at a practical beginner level.

---

## Issue 4: Week 3 — Number Predictor Lab: train/evaluate/save a regression model

**Type:** HITL  
**Labels:** `HITL`, `course`, `pytorch`, `learning`, `beginner`, `weekly-project`  
**Time estimate:** 4–6 hours  
**Blocked by:** #3

### Project name

**House-ish Price Predictor**

### Why this exists

This moves from a perfect toy line to a more realistic number-prediction workflow: data file, train/test split, model, evaluation, saved model, and plots.

### What to build

Generate a tiny fake housing-style dataset and train a model that predicts a price-like number.

This is not a real estate model. It is a controlled dataset for learning.

### Dataset

Create a CSV file in code:

- `data/generated/week-03-house-ish-prices.csv`

Columns:

- `size_m2`
- `bedrooms`
- `distance_from_center_km`
- `price_k`

Suggested hidden rule:

```text
price_k = 50 + 3 * size_m2 + 25 * bedrooms - 4 * distance_from_center_km + noise
```

### Exact files to create

- `weeks/week-03-number-prediction/train_price_predictor.py`
- `weeks/week-03-number-prediction/notes.md`
- `data/generated/week-03-house-ish-prices.csv`
- `models/week-03-number-predictor.pt`
- `artifacts/week-03-number-prediction/loss_curve.png`
- `artifacts/week-03-number-prediction/predicted_vs_actual.png`
- `artifacts/week-03-number-prediction/example_predictions.csv`

### Online resources

- Learn PyTorch — 01 PyTorch Workflow: https://www.learnpytorch.io/01_pytorch_workflow/
- PyTorch Optimization: https://docs.pytorch.org/tutorials/beginner/basics/optimization_tutorial.html
- PyTorch Save and Load Model: https://docs.pytorch.org/tutorials/beginner/basics/saveloadrun_tutorial.html
- Google ML Crash Course exercises: https://developers.google.com/machine-learning/crash-course/exercises

### Step-by-step work

- [ ] Generate 200 fake examples and save them as CSV.
- [ ] Load the CSV back from disk.
- [ ] Split into 80% train and 20% test.
- [ ] Normalize input features if needed.
- [ ] Build a small PyTorch regression model.
- [ ] Train for a fixed number of epochs.
- [ ] Save loss values.
- [ ] Evaluate on test data.
- [ ] Save the model to `models/week-03-number-predictor.pt`.
- [ ] Plot predicted price vs actual price.
- [ ] Save 10 example predictions to CSV.
- [ ] Write notes explaining train data vs test data.

### Modification challenge

Increase the noise in the generated data.

Before running, write a prediction:

- Will test error get better or worse?
- Will the plot look tighter or messier?

Then run and explain.

### Debugging checklist

- [ ] If loss is huge, inspect feature scales.
- [ ] If train loss goes down but test loss is bad, explain overfitting in simple words.
- [ ] If predictions are all similar, check learning rate and input normalization.
- [ ] If model saving fails, check the `models/` folder exists.

### Acceptance criteria

- [ ] The script creates the generated dataset CSV.
- [ ] The script trains and evaluates a model.
- [ ] The model file exists in `models/`.
- [ ] `predicted_vs_actual.png` exists.
- [ ] `example_predictions.csv` exists.
- [ ] `notes.md` explains what train/test split means and why it matters.

### Expertise gained

After this issue, I can build a basic supervised regression workflow in PyTorch.

---

## Issue 5: Week 4 — Decision Boundary Lab: classify 2D points

**Type:** HITL  
**Labels:** `HITL`, `course`, `pytorch`, `learning`, `beginner`, `weekly-project`  
**Time estimate:** 4–6 hours  
**Blocked by:** #4

### Project name

**Decision Boundary Lab**

### Why this exists

Classification is easier to understand when I can see the model drawing a border between groups.

### What to build

Train a classifier on 2D points and visualize the decision boundary.

The model should answer:

```text
Is this point class 0 or class 1?
```

### Dataset

Use scikit-learn to generate a toy dataset:

```python
from sklearn.datasets import make_blobs
```

Start with two easy clusters, then make the problem harder with more overlap.

### Exact files to create

- `weeks/week-04-classification/train_2d_classifier.py`
- `weeks/week-04-classification/notes.md`
- `data/generated/week-04-2d-points.csv`
- `artifacts/week-04-classification/data_plot.png`
- `artifacts/week-04-classification/decision_boundary.png`
- `artifacts/week-04-classification/confusion_counts.json`

### Online resources

- Learn PyTorch — 02 Neural Network Classification: https://www.learnpytorch.io/02_pytorch_classification/
- Google ML Crash Course — Classification: https://developers.google.com/machine-learning/crash-course/classification
- TensorFlow Neural Network Playground: https://playground.tensorflow.org/

### Step-by-step work

- [ ] Generate 2D points and labels.
- [ ] Save the points to CSV.
- [ ] Plot the raw points by color.
- [ ] Split into train and test data.
- [ ] Build a simple PyTorch binary classifier.
- [ ] Train with a classification loss.
- [ ] Measure accuracy.
- [ ] Count true positives, true negatives, false positives, and false negatives.
- [ ] Plot the decision boundary.
- [ ] Write notes explaining what the boundary means.

### Modification challenge

Increase cluster overlap.

Before running, predict:

- Will accuracy go down?
- Will the boundary become less useful?
- Which mistakes do I expect to see?

Then run and explain.

### Debugging checklist

- [ ] If accuracy is around 50%, check labels and loss function.
- [ ] If shapes fail, print `X.shape` and `y.shape`.
- [ ] If the boundary plot is blank, check the grid generation.
- [ ] If loss does not move, lower the learning rate or inspect logits/probabilities.

### Acceptance criteria

- [ ] The script trains a classifier.
- [ ] `data_plot.png` shows the generated dataset.
- [ ] `decision_boundary.png` shows the learned boundary.
- [ ] `confusion_counts.json` exists.
- [ ] `notes.md` explains at least one wrong prediction type in plain words.

### Expertise gained

After this issue, I can explain classification, accuracy, and simple classification mistakes visually.

---

## Issue 6: Week 5 — Hidden Layer Lab: beat the straight-line model

**Type:** HITL  
**Labels:** `HITL`, `course`, `pytorch`, `learning`, `beginner`, `weekly-project`  
**Time estimate:** 4–6 hours  
**Blocked by:** #5

### Project name

**Hidden Layer Lab**

### Why this exists

A straight line cannot solve every classification problem. A hidden layer gives the model more flexible ways to bend the decision boundary.

### What to build

Train two models on the same nonlinear dataset:

1. A simple linear classifier.
2. A small neural network with one hidden layer.

Compare them side by side.

### Dataset

Use scikit-learn:

```python
from sklearn.datasets import make_moons
```

Generate 500 two-dimensional points with moderate noise.

### Exact files to create

- `weeks/week-05-small-neural-networks/compare_linear_vs_mlp.py`
- `weeks/week-05-small-neural-networks/notes.md`
- `data/generated/week-05-moons.csv`
- `artifacts/week-05-small-neural-networks/linear_boundary.png`
- `artifacts/week-05-small-neural-networks/mlp_boundary.png`
- `artifacts/week-05-small-neural-networks/model_comparison.csv`
- `artifacts/week-05-small-neural-networks/loss_curves.png`

### Online resources

- Google ML Crash Course — Neural Networks: https://developers.google.com/machine-learning/crash-course/neural-networks
- 3Blue1Brown — But what is a Neural Network?: https://www.3blue1brown.com/lessons/neural-networks/
- PyTorch — What is `torch.nn` really?: https://docs.pytorch.org/tutorials/beginner/nn_tutorial.html

### Step-by-step work

- [ ] Generate the moon-shaped dataset.
- [ ] Save it to CSV.
- [ ] Plot the raw data.
- [ ] Train a linear classifier.
- [ ] Train a small MLP with one hidden layer.
- [ ] Plot both decision boundaries.
- [ ] Save a comparison table with train accuracy, test accuracy, and final loss.
- [ ] Write notes explaining why the MLP can solve a problem the line struggles with.

### Modification challenge

Change hidden units:

- 4 hidden units
- 16 hidden units
- 64 hidden units

Before running, predict whether more hidden units will always be better. Then compare.

### Debugging checklist

- [ ] If both models perform badly, inspect the labels and plots.
- [ ] If the MLP does not improve, try more epochs or a smaller learning rate.
- [ ] If train accuracy is high but test accuracy drops, explain overfitting.
- [ ] If decision-boundary code is confusing, save intermediate grid predictions.

### Acceptance criteria

- [ ] Both models train in one script.
- [ ] There are two boundary plots.
- [ ] `model_comparison.csv` exists.
- [ ] `notes.md` explains hidden layers as extra flexible knobs.
- [ ] The modification challenge result is recorded.

### Expertise gained

After this issue, I can explain why neural networks are useful for nonlinear patterns.

---

## Issue 7: Week 6 — Fashion-MNIST Baseline: train a real image classifier

**Type:** HITL  
**Labels:** `HITL`, `course`, `pytorch`, `computer-vision`, `beginner`, `weekly-project`  
**Time estimate:** 4–6 hours  
**Blocked by:** #6

### Project name

**Fashion-MNIST Baseline Classifier**

### Why this exists

This is the first real image project. The goal is not state-of-the-art accuracy. The goal is to see that an image is numbers, and the same learning loop still works.

### What to build

Train a baseline Fashion-MNIST classifier and save examples, predictions, metrics, and the model.

### Dataset

Use torchvision to download Fashion-MNIST:

```python
from torchvision.datasets import FashionMNIST
```

Store downloaded data under:

- `data/fashion-mnist/`

### Exact files to create

- `weeks/week-06-image-classification-baseline/train_fashion_mnist_baseline.py`
- `weeks/week-06-image-classification-baseline/notes.md`
- `models/week-06-fashion-mnist-baseline.pt`
- `artifacts/week-06-image-classification-baseline/sample_images.png`
- `artifacts/week-06-image-classification-baseline/loss_accuracy_curves.png`
- `artifacts/week-06-image-classification-baseline/sample_predictions.png`
- `artifacts/week-06-image-classification-baseline/metrics.json`

### Online resources

- Learn PyTorch — 03 Computer Vision: https://www.learnpytorch.io/03_pytorch_computer_vision/
- PyTorch Quickstart: https://docs.pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html
- torchvision datasets: https://docs.pytorch.org/vision/stable/datasets.html

### Step-by-step work

- [ ] Download Fashion-MNIST with torchvision.
- [ ] Show a grid of example images and labels.
- [ ] Build a simple baseline model, such as flatten image -> linear layers.
- [ ] Train for a small number of epochs.
- [ ] Log train loss and test accuracy.
- [ ] Save the model.
- [ ] Save a plot of loss and accuracy.
- [ ] Save a grid of sample predictions.
- [ ] Write notes explaining how an image becomes a tensor.

### Modification challenge

Change one of these:

- batch size
- learning rate
- number of epochs

Before running, predict what should change. Then run and explain.

### Debugging checklist

- [ ] If download fails, retry or check network access.
- [ ] If tensors have wrong shape, print one batch shape.
- [ ] If CUDA fails, run on CPU.
- [ ] If accuracy is very low, inspect image labels and loss function.

### Acceptance criteria

- [ ] The script downloads/loads Fashion-MNIST.
- [ ] `sample_images.png` exists.
- [ ] A model file exists in `models/`.
- [ ] `metrics.json` includes train loss and test accuracy.
- [ ] `sample_predictions.png` exists.
- [ ] `notes.md` explains image tensors in plain words.

### Expertise gained

After this issue, I can train a small image classifier and explain the image classification workflow.

---

## Issue 8: Week 7 — Model Improvement Lab: debug and compare experiments

**Type:** HITL  
**Labels:** `HITL`, `course`, `pytorch`, `debugging`, `computer-vision`, `weekly-project`  
**Time estimate:** 4–6 hours  
**Blocked by:** #7

### Project name

**Image Classifier Improvement Lab**

### Why this exists

Improving a model should not mean randomly changing things. This project teaches small experiments and a repeatable debugging ritual.

### What to build

Take the Week 6 Fashion-MNIST baseline and run controlled improvement experiments.

Examples:

- baseline MLP vs tiny CNN
- learning rate A vs learning rate B
- 3 epochs vs 6 epochs
- with or without simple normalization

### Dataset

Reuse Fashion-MNIST from Week 6.

### Exact files to create

- `weeks/week-07-improving-and-debugging/run_fashion_mnist_experiments.py`
- `weeks/week-07-improving-and-debugging/debug_report.md`
- `models/week-07-best-fashion-mnist.pt`
- `artifacts/week-07-improving-and-debugging/experiment_log.csv`
- `artifacts/week-07-improving-and-debugging/loss_accuracy_comparison.png`
- `artifacts/week-07-improving-and-debugging/tiny_overfit_check.txt`
- `artifacts/week-07-improving-and-debugging/best_vs_baseline.md`

### Online resources

- Andrej Karpathy — A Recipe for Training Neural Networks: http://karpathy.github.io/2019/04/25/recipe/
- Google ML Crash Course — Interpreting Loss Curves: https://developers.google.com/machine-learning/crash-course/overfitting/interpreting-loss-curves
- PyTorch — Training a Classifier: https://docs.pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
- PyTorch — TensorBoard tutorial: https://docs.pytorch.org/tutorials/intermediate/tensorboard_tutorial.html

### Step-by-step work

- [ ] Load or recreate the Week 6 baseline.
- [ ] Create an experiment table with at least three runs.
- [ ] For each run, record:
  - model name
  - learning rate
  - batch size
  - epochs
  - final train loss
  - final test accuracy
  - notes
- [ ] Add a tiny-overfit check: train on a very small subset and confirm the model can memorize it.
- [ ] Save the experiment log to CSV.
- [ ] Plot loss/accuracy comparison.
- [ ] Save the best model.
- [ ] Write `debug_report.md` using the debugging ritual.

### Required debugging ritual

Complete this checklist in `debug_report.md`:

- [ ] I checked input shapes.
- [ ] I inspected raw images and labels.
- [ ] I ran one tiny batch.
- [ ] I tried to overfit a tiny dataset.
- [ ] I printed losses.
- [ ] I printed sample predictions.
- [ ] I checked CPU/GPU device placement.
- [ ] I tried or considered a smaller learning rate.
- [ ] I compared expected behavior with actual behavior.

### Modification challenge

Pick two improvement ideas before running them.

For each idea, write:

- what I changed
- why I think it might help
- what I expect to happen
- what actually happened

### Acceptance criteria

- [ ] At least three experiment runs are recorded.
- [ ] `experiment_log.csv` exists.
- [ ] `tiny_overfit_check.txt` exists.
- [ ] `loss_accuracy_comparison.png` exists.
- [ ] `week-07-best-fashion-mnist.pt` exists.
- [ ] `debug_report.md` contains the debugging ritual checklist.

### Expertise gained

After this issue, I can improve a model using controlled experiments and debug common training failures.

---

## Issue 9: Week 8 — Mistake Explorer Capstone: inspect confident wrong images

**Type:** HITL  
**Labels:** `HITL`, `course`, `pytorch`, `computer-vision`, `capstone`, `weekly-project`  
**Time estimate:** 5–7 hours  
**Blocked by:** #8

### Project name

**Fashion-MNIST Mistake Explorer**

### Why this exists

Accuracy is only one number. Real understanding comes from looking at what the model gets wrong.

This capstone turns the image classifier into an inspection tool.

### What to build

Build a script or notebook that loads the best image classifier, runs it on test images, finds mistakes, and creates a small mistake report.

The report should show:

- correct predictions
- wrong predictions
- confident wrong predictions
- a confusion matrix
- the most confused class pairs
- one before/after comparison after an improvement attempt

### Dataset

Reuse Fashion-MNIST.

### Exact files to create

- `weeks/week-08-mistake-explorer-capstone/mistake_explorer.py`
- `weeks/week-08-mistake-explorer-capstone/final_reflection.md`
- `artifacts/week-08-mistake-explorer-capstone/confusion_matrix.png`
- `artifacts/week-08-mistake-explorer-capstone/confident_wrong_predictions.png`
- `artifacts/week-08-mistake-explorer-capstone/correct_predictions.png`
- `artifacts/week-08-mistake-explorer-capstone/class_pair_mistakes.csv`
- `artifacts/week-08-mistake-explorer-capstone/before_after_improvement.md`
- `artifacts/week-08-mistake-explorer-capstone/final_metrics.json`

### Online resources

- Learn PyTorch — 03 Computer Vision: https://www.learnpytorch.io/03_pytorch_computer_vision/
- TorchMetrics — Confusion Matrix: https://lightning.ai/docs/torchmetrics/latest/classification/confusion_matrix.html
- PyTorch — Visualizing Models, Data, and Training with TensorBoard: https://docs.pytorch.org/tutorials/intermediate/tensorboard_tutorial.html

### Step-by-step work

- [ ] Load the best Week 7 model.
- [ ] Run predictions on the Fashion-MNIST test set.
- [ ] Store true label, predicted label, and confidence for each image.
- [ ] Save final metrics as JSON.
- [ ] Create a confusion matrix plot.
- [ ] Find the top confident wrong predictions.
- [ ] Save a grid of confident wrong images.
- [ ] Save a grid of correct predictions for comparison.
- [ ] Count the most common class-pair mistakes.
- [ ] Pick one improvement attempt and compare before vs after.
- [ ] Write a final reflection in plain language.

### Modification challenge

Pick one mistake pattern, for example:

- shirts confused with t-shirts
- sandals confused with sneakers
- coats confused with pullovers

Before changing anything, write:

- why I think the model confuses these classes
- what change might help
- what result I expect

Then run the change and explain what happened.

### Debugging checklist

- [ ] If confidence values look wrong, check softmax is applied on the correct dimension.
- [ ] If the confusion matrix labels are wrong, check class-name order.
- [ ] If image grids show wrong labels, inspect indexing.
- [ ] If before/after is unfair, make sure both models use the same test set.

### Acceptance criteria

- [ ] `mistake_explorer.py` runs.
- [ ] `confusion_matrix.png` exists.
- [ ] `confident_wrong_predictions.png` exists.
- [ ] `class_pair_mistakes.csv` exists.
- [ ] `before_after_improvement.md` exists.
- [ ] `final_reflection.md` answers: “What do I now understand about how computers learn?”

### Expertise gained

After this issue, I can inspect model failures, reason about mistakes, and improve a model using evidence instead of vibes.

---

# Expertise gained after all issues

After all issues, I should be able to:

- Set up and run a local PyTorch project.
- Explain learning as repeated correction.
- Build a tiny learner by hand.
- Rebuild the same idea in PyTorch.
- Train and evaluate a regression model.
- Train and inspect classification models.
- Explain why hidden layers help with nonlinear patterns.
- Train a small Fashion-MNIST image classifier.
- Run controlled improvement experiments.
- Use a repeatable debugging ritual.
- Build a mistake explorer and explain model failures in plain words.
