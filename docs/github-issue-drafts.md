# GitHub Issue Drafts

These are drafts for GitHub issues. They are written in simple words on purpose.

Labels to use:

- `course`
- `learning`
- `pytorch`
- `beginner`
- `weekly-project`
- `HITL`

Optional labels:

- `setup`
- `debugging`
- `computer-vision`
- `capstone`

---

## Issue 1: Set up the local uv + PyTorch course repo

**Type:** HITL  
**Labels:** `HITL`, `course`, `setup`, `pytorch`, `beginner`  
**Time estimate:** 2–4 hours  
**Blocked by:** None

### What to do

Create the local environment for the whole course.

This issue is not about learning ML yet. It is about making sure the machine can run the lessons.

### Online resources

- uv docs: https://docs.astral.sh/uv/
- PyTorch Start Locally: https://docs.pytorch.org/get-started/locally/
- PyTorch Learn the Basics: https://docs.pytorch.org/tutorials/beginner/basics/intro.html

### Steps

- [ ] Confirm `uv` works.
- [ ] Create a Python 3.12 virtual environment with `uv`.
- [ ] Install basic tools: `numpy`, `matplotlib`, `pandas`, `scikit-learn`, `jupyter`, `ipykernel`, `tqdm`, `torchmetrics`.
- [ ] Install PyTorch and torchvision using the official PyTorch install selector.
- [ ] Check whether CUDA sees the RTX 4070 SUPER.
- [ ] Make sure CPU fallback works.
- [ ] Create folders: `weeks/`, `src/`, `data/`, `models/`, `artifacts/`.
- [ ] Write down setup problems and fixes in a short note.

### Acceptance criteria

- [ ] `uv run python -c "import torch; print(torch.__version__)"` works.
- [ ] `torch.cuda.is_available()` result is known.
- [ ] The repo has the planned folder structure.
- [ ] The setup note says what works and what still needs fixing.

### Expertise gained

After this issue, I can set up and run a local PyTorch learning project without using Colab.

---

## Issue 2: Week 1 — See learning happen by hand

**Type:** HITL  
**Labels:** `HITL`, `course`, `learning`, `beginner`, `weekly-project`  
**Time estimate:** 4–6 hours  
**Blocked by:** Issue 1

### Simple goal

Understand the core loop:

```text
predict -> measure wrongness -> adjust -> repeat
```

No PyTorch magic yet.

### Online resources

- ML4A: How neural networks are trained: https://ml4a.github.io/ml4a/how_neural_networks_are_trained/
- TensorFlow Neural Network Playground: https://playground.tensorflow.org/
- Google Machine Learning Crash Course: https://developers.google.com/machine-learning/crash-course

### Mini-project

Build a tiny number predictor by hand.

Example: the real rule is `y = 2x + 1`. Start with bad guesses for the model's knobs. Measure wrongness. Change the knobs. Watch the model get less wrong.

### Steps

- [ ] Make a tiny table of `x` and `y` values.
- [ ] Write a tiny model with one or two knobs.
- [ ] Make predictions.
- [ ] Measure wrongness with a simple loss number.
- [ ] Adjust the knobs manually or with a simple loop.
- [ ] Plot wrongness over time.
- [ ] Write a short note: “What does learning mean here?”

### Modification challenge

Change the hidden rule from `y = 2x + 1` to another simple line. Predict what should change. Run it. Explain what happened.

### Acceptance criteria

- [ ] There is a notebook or script in `weeks/week-01-learning-loop/`.
- [ ] It prints predictions before and after learning.
- [ ] It shows loss going down.
- [ ] The weekly note explains learning without using heavy math.

### Expertise gained

After this issue, I can explain machine learning as repeated correction, not magic.

---

## Issue 3: Week 2 — Learn PyTorch tensors and `backward()` gently

**Type:** HITL  
**Labels:** `HITL`, `course`, `pytorch`, `learning`, `beginner`, `weekly-project`  
**Time estimate:** 4–6 hours  
**Blocked by:** Issue 2

### Simple goal

Learn the PyTorch building blocks without panic:

- tensor = number box / number table
- device = CPU or GPU
- gradient = hint about which way to move
- `backward()` = ask PyTorch to compute those hints

### Online resources

- Learn PyTorch — 00 PyTorch Fundamentals: https://www.learnpytorch.io/00_pytorch_fundamentals/
- PyTorch Tensors: https://docs.pytorch.org/tutorials/beginner/basics/tensorqs_tutorial.html
- PyTorch Autograd: https://docs.pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html

### Mini-project

Rebuild the Week 1 tiny predictor in PyTorch.

This time, let PyTorch compute the adjustment hints with `backward()`.

### Steps

- [ ] Create tensors.
- [ ] Move tensors to CPU or CUDA device.
- [ ] Make predictions with tensor math.
- [ ] Compute loss.
- [ ] Call `loss.backward()`.
- [ ] Print parameter values and gradients.
- [ ] Explain gradients as simple “move this way” hints.

### Modification challenge

Run the same code on CPU and GPU if CUDA works. Confirm the answer is basically the same.

### Acceptance criteria

- [ ] There is a notebook or script in `weeks/week-02-pytorch-basics/`.
- [ ] It uses tensors.
- [ ] It calls `backward()`.
- [ ] It prints gradients.
- [ ] The weekly note explains `backward()` in simple words.

### Expertise gained

After this issue, I can use basic PyTorch tensors and explain autograd at a beginner intuition level.

---

## Issue 4: Week 3 — Train a number prediction model with a real PyTorch loop

**Type:** HITL  
**Labels:** `HITL`, `course`, `pytorch`, `learning`, `beginner`, `weekly-project`  
**Time estimate:** 4–6 hours  
**Blocked by:** Issue 3

### Simple goal

Build a real training loop:

```text
for each epoch:
    predict
    calculate loss
    clear old gradients
    backward
    optimizer step
```

### Online resources

- Learn PyTorch — 01 PyTorch Workflow: https://www.learnpytorch.io/01_pytorch_workflow/
- PyTorch Optimization: https://docs.pytorch.org/tutorials/beginner/basics/optimization_tutorial.html
- PyTorch Save and Load Model: https://docs.pytorch.org/tutorials/beginner/basics/saveloadrun_tutorial.html
- Google ML Crash Course — loss / gradient descent exercises: https://developers.google.com/machine-learning/crash-course/exercises

### Mini-project

Train a small model to predict numbers. Use train/test split. Plot predictions against true answers.

### Steps

- [ ] Create or load a tiny numeric dataset.
- [ ] Split it into train and test data.
- [ ] Build a simple PyTorch model.
- [ ] Train with loss and optimizer.
- [ ] Evaluate on test data.
- [ ] Save the model.
- [ ] Plot actual values vs predicted values.

### Modification challenge

Change the learning rate. Before running, predict whether training will be slower, faster, or unstable. Then run and explain.

### Acceptance criteria

- [ ] There is a notebook or script in `weeks/week-03-number-prediction/`.
- [ ] It has a train/test split.
- [ ] It has a visible training loop.
- [ ] It saves a model file into `models/`.
- [ ] It saves a plot into `artifacts/`.

### Expertise gained

After this issue, I can train and evaluate a basic supervised PyTorch model.

---

## Issue 5: Week 4 — Classification: teach a model to choose a label

**Type:** HITL  
**Labels:** `HITL`, `course`, `pytorch`, `learning`, `beginner`, `weekly-project`  
**Time estimate:** 4–6 hours  
**Blocked by:** Issue 4

### Simple goal

Understand classification.

Regression predicts a number. Classification chooses a class, like “red dot” or “blue dot.”

### Online resources

- Learn PyTorch — 02 Neural Network Classification: https://www.learnpytorch.io/02_pytorch_classification/
- Google ML Crash Course — Classification: https://developers.google.com/machine-learning/crash-course/classification
- TensorFlow Neural Network Playground: https://playground.tensorflow.org/

### Mini-project

Train a model to classify simple 2D points. Draw the decision boundary so I can see what the model learned.

### Steps

- [ ] Create a tiny 2D dataset.
- [ ] Plot the points with colors for labels.
- [ ] Build a simple classifier.
- [ ] Train it.
- [ ] Measure accuracy.
- [ ] Plot the decision boundary.
- [ ] Explain false positives / false negatives in plain words.

### Modification challenge

Change the dataset shape or the number of training points. Predict whether the model will do better or worse. Run and explain.

### Acceptance criteria

- [ ] There is a notebook or script in `weeks/week-04-classification/`.
- [ ] It trains a classifier.
- [ ] It reports accuracy.
- [ ] It shows a decision-boundary plot.
- [ ] The weekly note explains classification mistakes simply.

### Expertise gained

After this issue, I can explain how a model chooses between labels and how accuracy can hide some mistakes.

---

## Issue 6: Week 5 — Build a small neural network for nonlinear data

**Type:** HITL  
**Labels:** `HITL`, `course`, `pytorch`, `learning`, `beginner`, `weekly-project`  
**Time estimate:** 4–6 hours  
**Blocked by:** Issue 5

### Simple goal

Understand hidden layers.

A hidden layer gives the model more flexible knobs. This helps when a straight line is not enough.

### Online resources

- Google ML Crash Course — Neural Networks: https://developers.google.com/machine-learning/crash-course/neural-networks
- 3Blue1Brown — But what is a Neural Network?: https://www.3blue1brown.com/lessons/neural-networks/
- PyTorch — What is `torch.nn` really?: https://docs.pytorch.org/tutorials/beginner/nn_tutorial.html

### Mini-project

Train two models on a toy nonlinear dataset:

1. A too-simple model.
2. A small neural network with a hidden layer.

Compare them visually.

### Steps

- [ ] Create or load a nonlinear 2D dataset.
- [ ] Train a simple model.
- [ ] Train a small neural network.
- [ ] Plot both decision boundaries.
- [ ] Compare train/test accuracy.
- [ ] Explain why the hidden layer helped or did not help.

### Modification challenge

Change the number of hidden units. Predict what should happen. Run and explain.

### Acceptance criteria

- [ ] There is a notebook or script in `weeks/week-05-small-neural-networks/`.
- [ ] It compares a simple model and a neural network.
- [ ] It includes plots.
- [ ] It explains hidden layers in simple words.

### Expertise gained

After this issue, I can explain why neural networks are useful for problems that are not simple straight lines.

---

## Issue 7: Week 6 — Train an image classifier baseline

**Type:** HITL  
**Labels:** `HITL`, `course`, `pytorch`, `computer-vision`, `beginner`, `weekly-project`  
**Time estimate:** 4–6 hours  
**Blocked by:** Issue 6

### Simple goal

Move from toy data to real images without losing the core loop.

An image is just numbers arranged in a grid. The model learns patterns in those numbers.

### Online resources

- Learn PyTorch — 03 Computer Vision: https://www.learnpytorch.io/03_pytorch_computer_vision/
- PyTorch Quickstart: https://docs.pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html
- PyTorch torchvision datasets: https://docs.pytorch.org/vision/stable/datasets.html

### Mini-project

Train a baseline image classifier on Fashion-MNIST.

### Steps

- [ ] Download Fashion-MNIST with torchvision.
- [ ] Show several example images and labels.
- [ ] Build a small baseline model.
- [ ] Train it.
- [ ] Measure train and test accuracy.
- [ ] Print a few predictions.
- [ ] Save a plot or example grid into `artifacts/`.

### Modification challenge

Change batch size or learning rate. Predict what might happen. Run and explain.

### Acceptance criteria

- [ ] There is a notebook or script in `weeks/week-06-image-classification-baseline/`.
- [ ] It loads Fashion-MNIST locally.
- [ ] It shows example images.
- [ ] It trains a classifier.
- [ ] It reports test accuracy.

### Expertise gained

After this issue, I can train a small image classifier and explain the image training loop.

---

## Issue 8: Week 7 — Improve the model and practice debugging

**Type:** HITL  
**Labels:** `HITL`, `course`, `pytorch`, `debugging`, `computer-vision`, `weekly-project`  
**Time estimate:** 4–6 hours  
**Blocked by:** Issue 7

### Simple goal

Learn that improving a model is not random guessing.

Use small experiments and the debugging ritual.

### Online resources

- Andrej Karpathy — A Recipe for Training Neural Networks: http://karpathy.github.io/2019/04/25/recipe/
- Google ML Crash Course — Interpreting Loss Curves: https://developers.google.com/machine-learning/crash-course/overfitting/interpreting-loss-curves
- PyTorch — Training a Classifier: https://docs.pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
- PyTorch — TensorBoard tutorial: https://docs.pytorch.org/tutorials/intermediate/tensorboard_tutorial.html

### Mini-project

Take the Week 6 image classifier and improve it with one or two careful changes.

Possible changes:

- Better learning rate.
- More epochs.
- Small CNN instead of flat model.
- Simple data transform.
- Better train/test logging.

### Debugging ritual to practice

- [ ] Check shapes.
- [ ] Inspect raw examples and labels.
- [ ] Run one tiny batch.
- [ ] Try to overfit a tiny dataset.
- [ ] Print loss values.
- [ ] Print sample predictions.
- [ ] Check CPU/GPU device placement.
- [ ] Try a smaller learning rate.
- [ ] Compare expected behavior vs actual behavior.

### Modification challenge

Run two experiments. Predict which one should help more. Then compare results.

### Acceptance criteria

- [ ] There is a notebook or script in `weeks/week-07-improving-and-debugging/`.
- [ ] It compares at least two runs.
- [ ] It includes loss/accuracy logs or plots.
- [ ] It includes a debugging note.
- [ ] It explains which change helped and why.

### Expertise gained

After this issue, I can improve a model using small experiments and debug common training problems.

---

## Issue 9: Week 8 — Final capstone: image classifier mistake explorer

**Type:** HITL  
**Labels:** `HITL`, `course`, `pytorch`, `computer-vision`, `capstone`, `weekly-project`  
**Time estimate:** 5–7 hours  
**Blocked by:** Issue 8

### Simple goal

Stop only looking at accuracy.

Look at mistakes. Ask:

- What did the model get wrong?
- Was it confidently wrong?
- Are some classes confused more than others?
- Did my improvement actually help?

### Online resources

- Learn PyTorch — 03 Computer Vision: https://www.learnpytorch.io/03_pytorch_computer_vision/
- Learn PyTorch — 04 Custom Datasets, optional: https://www.learnpytorch.io/04_pytorch_custom_datasets/
- TorchMetrics — Confusion Matrix: https://lightning.ai/docs/torchmetrics/latest/classification/confusion_matrix.html
- PyTorch — Visualizing Models, Data, and Training with TensorBoard: https://docs.pytorch.org/tutorials/intermediate/tensorboard_tutorial.html

### Mini-project

Build a mistake explorer for the image classifier.

The explorer should show:

- Correct predictions.
- Wrong predictions.
- Confident wrong predictions.
- A confusion matrix.
- Example images for the worst mistakes.
- A short before/after comparison after one improvement.

### Steps

- [ ] Load the best Week 7 model.
- [ ] Run predictions on the test set.
- [ ] Store true labels, predicted labels, and confidence scores.
- [ ] Show the most confident wrong predictions.
- [ ] Create a confusion matrix.
- [ ] Pick one improvement attempt.
- [ ] Compare before and after.
- [ ] Write a final plain-language explanation.

### Modification challenge

Change one training setting or model detail. Predict which mistakes should improve. Run and explain what actually happened.

### Acceptance criteria

- [ ] There is a notebook or script in `weeks/week-08-mistake-explorer-capstone/`.
- [ ] It shows confident wrong predictions.
- [ ] It creates a confusion matrix.
- [ ] It compares before and after one improvement.
- [ ] It includes a final note: “What I now understand about how computers learn.”

### Expertise gained

After this issue, I can inspect model failures, explain them in simple words, and improve a model with evidence instead of vibes.

---

# Expertise gained after all issues

After all issues, I should be able to:

- Explain machine learning as repeated correction.
- Build small PyTorch training loops.
- Use tensors, loss, optimizers, datasets, and dataloaders.
- Explain `backward()` without heavy derivative formulas.
- Train number prediction and image classification models.
- Use CPU or CUDA when available.
- Debug models with a repeatable ritual.
- Look at image-classifier mistakes and learn from them.
- Read beginner ML/PyTorch resources without feeling totally lost.
