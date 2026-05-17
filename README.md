# Intuitive Understanding of Machine Learning

A simple, practical, local-first course for understanding how computers learn.

This course is for **me**. It assumes I can program, but it assumes I am **new to machine learning, deep learning, neural networks, and PyTorch**.

It uses simple words on purpose. No shame. No pretending. If a word is important, the course should explain it before using it too much.

## The Big Idea

A model is a guesser.

Training means:

1. The model makes a guess.
2. We measure how wrong the guess is.
3. We change the model a little.
4. We try again.
5. If the wrongness goes down, the model is learning.

Short version:

```text
predict -> measure wrongness -> adjust -> repeat
```

This course keeps coming back to that loop.

## What I Should Be Able To Do By The End

By the end, I should be able to:

- Explain, in plain words, how a computer can “learn” from examples.
- Build small PyTorch models locally.
- Train a model that predicts numbers.
- Train a model that classifies images.
- Look at model mistakes and explain what probably went wrong.
- Use a simple debugging ritual when training fails.
- Understand PyTorch basics like tensors, datasets, dataloaders, loss, optimizers, and `backward()` at an intuitive level.
- Use my local machine, including my NVIDIA RTX 4070 SUPER when CUDA works, but keep CPU fallbacks.

This is **not** a heavy math course. We can talk about derivatives as “which way should the model move?”, but we will not drown in formulas.

## Main Free Learning Spine

Main spine:

- [Learn PyTorch for Deep Learning: Zero to Mastery](https://www.learnpytorch.io/)

Why this is the main spine:

- It is free to read online.
- It is beginner-friendly.
- It uses PyTorch.
- It has real code and real projects.
- It covers the pieces this course needs: tensors, workflows, classification, and computer vision.

Important: this repo does **not** blindly copy that course. It uses it as a guide, then adds simpler explanations, smaller local projects, weekly mini-projects, and debugging habits.

## Other Important Free Resources

Setup and PyTorch basics:

- [uv documentation](https://docs.astral.sh/uv/)
- [PyTorch: Start Locally](https://docs.pytorch.org/get-started/locally/)
- [PyTorch Learn the Basics](https://docs.pytorch.org/tutorials/beginner/basics/intro.html)
- [PyTorch Quickstart](https://docs.pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html)
- [PyTorch Tensors](https://docs.pytorch.org/tutorials/beginner/basics/tensorqs_tutorial.html)
- [PyTorch Autograd](https://docs.pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html)
- [PyTorch Optimization](https://docs.pytorch.org/tutorials/beginner/basics/optimization_tutorial.html)
- [What is `torch.nn` really?](https://docs.pytorch.org/tutorials/beginner/nn_tutorial.html)

Visual intuition:

- [TensorFlow Neural Network Playground](https://playground.tensorflow.org/)
- [Google Machine Learning Crash Course](https://developers.google.com/machine-learning/crash-course)
- [Google ML Crash Course: Neural Networks](https://developers.google.com/machine-learning/crash-course/neural-networks)
- [3Blue1Brown: But what is a Neural Network?](https://www.3blue1brown.com/lessons/neural-networks/)
- [ML4A: How neural networks are trained](https://ml4a.github.io/ml4a/how_neural_networks_are_trained/)

Computer vision and mistake analysis:

- [Learn PyTorch: Computer Vision](https://www.learnpytorch.io/03_pytorch_computer_vision/)
- [PyTorch: Training a Classifier](https://docs.pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html)
- [PyTorch: Visualizing Models, Data, and Training with TensorBoard](https://docs.pytorch.org/tutorials/intermediate/tensorboard_tutorial.html)
- [TorchMetrics: Confusion Matrix](https://lightning.ai/docs/torchmetrics/latest/classification/confusion_matrix.html)

Debugging:

- [Andrej Karpathy: A Recipe for Training Neural Networks](http://karpathy.github.io/2019/04/25/recipe/)
- [Google ML Crash Course: Interpreting Loss Curves](https://developers.google.com/machine-learning/crash-course/overfitting/interpreting-loss-curves)

## Local Project Setup

This course is local-first. No Google Colab.

Planned tools:

- Python
- uv
- PyTorch
- torchvision
- NumPy
- Matplotlib
- Jupyter notebooks
- scikit-learn only when useful for tiny toy datasets or metrics
- torchmetrics for final confusion matrix / mistake analysis

### Suggested setup steps

Install `uv` first:

```bash
# See: https://docs.astral.sh/uv/
uv --version
```

Create or use this repo:

```bash
git clone <this-repo-url>
cd intuitive_understanding_machine_learning
```

Create a Python environment:

```bash
uv python install 3.12
uv venv --python 3.12
```

Install normal learning tools:

```bash
uv pip install numpy matplotlib pandas scikit-learn jupyter ipykernel tqdm torchmetrics
```

Install PyTorch using the official selector:

- Go to [PyTorch Start Locally](https://docs.pytorch.org/get-started/locally/).
- Pick Windows, pip, Python, CUDA if available.
- Replace `pip3 install ...` with `uv pip install ...`.

Example shape of the command:

```bash
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/<current-cuda-wheel>
```

If CUDA is annoying, use CPU first. Learning matters more than perfect GPU setup.

Check PyTorch:

```bash
uv run python - <<'PY'
import torch
print('torch:', torch.__version__)
print('cuda available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('gpu:', torch.cuda.get_device_name(0))
PY
```

## Repo Layout

Planned layout:

```text
.
├── README.md
├── docs/
│   └── github-issue-drafts.md
├── weeks/
│   ├── week-01-learning-loop/
│   ├── week-02-pytorch-basics/
│   ├── week-03-number-prediction/
│   ├── week-04-classification/
│   ├── week-05-small-neural-networks/
│   ├── week-06-image-classification-baseline/
│   ├── week-07-improving-and-debugging/
│   └── week-08-mistake-explorer-capstone/
├── src/
│   └── shared helper code later
├── data/
│   └── downloaded datasets, not committed
├── models/
│   └── trained model files, not committed
└── artifacts/
    └── plots, reports, generated outputs, not committed
```

## Weekly Rhythm

Each week should follow this pattern:

1. Read or watch a short resource.
2. Write down the idea in simple words.
3. Build something locally.
4. Change one or two things.
5. Predict what the change should do.
6. Run it.
7. Explain what happened.
8. Save a small artifact: notebook, script, plot, model, or note.

## The Debugging Ritual

When training fails, do not panic. Use this ritual:

1. Check shapes.
2. Look at a few raw examples and labels.
3. Run one tiny batch.
4. Try to overfit a tiny dataset.
5. Print loss values.
6. Print a few predictions.
7. Check CPU/GPU device placement.
8. Try a smaller learning rate.
9. Compare what I expected with what actually happened.

The goal is to make failure boring and diagnosable.

## 8-Week Plan

Each week is about **4–6 hours**. The final capstone may take **5–7 hours**.

### Week 1 — See Learning Happen By Hand

Goal: understand the learning loop before PyTorch hides anything.

Mini-project: build a tiny model with one or two numbers as “knobs.” Make it predict a simple line like `y = 2x + 1`. Show guesses, wrongness, and improvement.

Expertise gained: I can explain “learning” as repeated adjustment, not magic.

### Week 2 — PyTorch Basics Without Panic

Goal: learn tensors, devices, and autograd in simple words.

Mini-project: recreate the tiny learning loop in PyTorch and inspect gradients as “hints for how to move.”

Expertise gained: I can use tensors and understand what `backward()` is doing at a high level.

### Week 3 — Predict Numbers With A Real Training Loop

Goal: train/test split, loss, optimizer, epochs, and evaluation.

Mini-project: train a simple PyTorch model to predict numbers, then plot predictions vs actual values.

Expertise gained: I can build a basic supervised learning workflow.

### Week 4 — Classification: Teaching A Model To Choose A Category

Goal: understand classification as choosing between labels.

Mini-project: classify simple 2D points and visualize the decision boundary.

Expertise gained: I can explain probabilities, labels, accuracy, and classification mistakes.

### Week 5 — Small Neural Networks

Goal: understand hidden layers as extra flexible “knobs.”

Mini-project: train a small neural network on a toy nonlinear dataset and compare it to a too-simple model.

Expertise gained: I can explain why hidden layers help with non-straight-line problems.

### Week 6 — Image Classification Baseline

Goal: classify images with a small PyTorch model.

Mini-project: train a Fashion-MNIST image classifier, inspect images, and print sample predictions.

Expertise gained: I can train a small image classifier and understand the data/model/loss loop.

### Week 7 — Improve The Model And Debug It

Goal: learn that better training comes from careful experiments, not random guessing.

Mini-project: change learning rate, model size, epochs, or architecture. Compare results. Practice the debugging ritual.

Expertise gained: I can run small experiments and diagnose common training problems.

### Week 8 — Final Capstone: Image Classifier + Mistake Explorer

Goal: inspect model mistakes instead of only celebrating accuracy.

Mini-project: build a mistake explorer that shows confident wrong predictions, confusion matrix, example images, and a short explanation of what changed after one improvement attempt.

Expertise gained: I can train a model, inspect its failures, improve it carefully, and explain what I learned in plain words.

## GitHub Issues

Issue drafts are in:

- [`docs/github-issue-drafts.md`](docs/github-issue-drafts.md)

Each issue includes:

- Time estimate
- Simple goal
- Online resources
- Work steps
- Acceptance criteria
- Expertise gained

## Final Expertise After The Whole Course

At the end, I should have beginner-level practical intuition for:

- What a model is
- What training means
- What loss means
- What an optimizer does
- Why learning rate matters
- What tensors are
- What `backward()` roughly does
- How a neural network can classify images
- Why models make mistakes
- How to debug a model that does not learn
- How to explain ML without hiding behind math words
