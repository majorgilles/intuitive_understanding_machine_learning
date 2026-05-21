# Understanding How Machines Work and Learn

This project exists to make machine learning inspectable.

Random note: every model starts as a guess.

Test note: small experiments make big ideas visible.

Each lab builds a small local system, runs it, looks at what the machine did, and records what changed. The focus is the mechanics: data becomes tensors, models make guesses, loss measures error, optimizers adjust parameters, and debugging turns failures into evidence.

## The Big Idea

A machine learning model is a guesser with adjustable knobs.

A computer does not learn by magic. It follows a repeatable process: turn data into numbers, make a guess, measure how wrong the guess was, and adjust the knobs.

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

This course keeps coming back to that loop so the machine feels inspectable instead of mysterious.

## Global Time Estimate

This is an **8-week course** after the local setup is done.

| Part | Time |
| --- | ---: |
| One-time local setup | 2–4 hours |
| Weeks 1–7 | 4–6 hours each |
| Week 8 capstone | 5–7 hours |
| Main 8-week course total | 33–49 hours |
| Total including setup | 35–53 hours |

Practical calendar estimate:

- Normal pace: **8 weeks**, about **4–6 hours per week**.
- If setup is annoying: add one setup day or stretch to **9 weeks**.
- If a week feels hard: repeat that week. Repeating is not failure. It is how learning works.

## What I Should Be Able To Do By The End

By the end, I should be able to:

- Explain, in plain words, how a machine can learn from examples by adjusting numbers.
- Build small PyTorch models locally.
- Train a model that predicts numbers.
- Train a model that classifies 2D points.
- Train a model that classifies images.
- Look at model mistakes and explain what probably went wrong.
- Use a repeatable debugging ritual when training fails.
- Understand PyTorch basics like tensors, datasets, dataloaders, loss, optimizers, and `backward()` at an intuitive level.
- Describe how data, tensors, model code, loss, optimizer, CPU/GPU device placement, and saved artifacts fit together.
- Use my local machine, including my NVIDIA RTX 4070 SUPER when CUDA works, while keeping CPU fallbacks.

This is **not** a heavy math course. We can talk about derivatives as “which way should the model move?”, but we will not drown in formulas.

## Main Free Learning Spine

Main spine:

- [Learn PyTorch for Deep Learning: Zero to Mastery](https://www.learnpytorch.io/)

Why this is the main spine:

- It is free to read online.
- It is beginner-friendly.
- It uses PyTorch.
- It has real code and real projects.
- It covers tensors, workflows, classification, and computer vision.

Important: this repo does **not** blindly copy that course. It uses it as a guide, then adds smaller local projects, explicit deliverables, reflection notes, and debugging habits.

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
- pandas
- Jupyter notebooks if useful
- scikit-learn for small generated toy datasets
- torchmetrics for the final confusion matrix / mistake analysis

### Suggested setup steps

Install `uv` first:

```bash
uv --version
```

Create or use this repo:

```bash
git clone https://github.com/majorgilles/intuitive_understanding_machine_learning.git
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
│   ├── week-00-setup/
│   ├── week-01-learning-loop/
│   ├── week-02-pytorch-basics/
│   ├── week-03-number-prediction/
│   ├── week-04-classification/
│   ├── week-05-small-neural-networks/
│   ├── week-06-image-classification-baseline/
│   ├── week-07-improving-and-debugging/
│   └── week-08-mistake-explorer-capstone/
├── src/
│   └── mlcourse/
│       └── shared helper code
├── data/
│   └── generated/downloaded datasets, not committed
├── models/
│   └── trained model files, not committed
└── artifacts/
    └── plots, reports, generated outputs, not committed
```

## Weekly Rhythm

Each week should produce something concrete.

Pattern:

1. Read or watch a short resource.
2. Build the required project files.
3. Run the project locally.
4. Save the required artifacts.
5. Change one or two things.
6. Predict what the change should do before running.
7. Run it.
8. Explain what happened in `notes.md` or a report.

## The Debugging Ritual

When training fails, use this ritual:

1. Check shapes.
2. Look at a few raw examples and labels.
3. Run one tiny batch.
4. Try to overfit a tiny dataset.
5. Print loss values.
6. Print a few predictions.
7. Check CPU/GPU device placement.
8. Try a smaller learning rate.
9. Compare what I expected with what actually happened.

The goal is to make failure diagnosable instead of mysterious.

## Tangible Project Plan

The detailed GitHub issues are the source of truth for each project. This table shows the concrete deliverables.

| Issue | Time | Project | Main files/artifacts | Expertise gained |
| --- | ---: | --- | --- | --- |
| #1 | 2–4h | Local ML Lab Smoke Test | `weeks/week-00-setup/check_environment.py`, `src/mlcourse/device.py`, `artifacts/week-00-setup/device_check.txt` | Run local PyTorch and check CPU/CUDA |
| #2 | 4–6h | Manual Line Learner | `manual_line_learner.py`, `loss_curve.png`, `before_after_predictions.csv` | Explain learning as repeated knob adjustment |
| #3 | 4–6h | PyTorch Gradient Learner | `torch_line_learner.py`, `gradient_trace.csv`, `loss_curve.png` | Understand tensors and `backward()` intuitively |
| #4 | 4–6h | House-ish Price Predictor | generated CSV, trained model, predicted-vs-actual plot | Build a regression train/test workflow |
| #5 | 4–6h | Decision Boundary Lab | 2D points CSV, decision-boundary plot, confusion counts | Explain classification visually |
| #6 | 4–6h | Hidden Layer Lab | linear vs MLP boundary plots, comparison CSV | Explain why hidden layers help nonlinear data |
| #7 | 4–6h | Fashion-MNIST Baseline | sample image grid, model file, metrics JSON, predictions grid | Train a real image classifier |
| #8 | 4–6h | Image Classifier Improvement Lab | experiment log, tiny-overfit check, best model, debug report | Improve and debug models systematically |
| #9 | 5–7h | Fashion-MNIST Mistake Explorer | confusion matrix, confident-wrong grid, class-pair mistakes, final reflection | Inspect failures and improve with evidence |

## GitHub Issues

The course issues are HITL because understanding how machines work and learn requires human inspection, judgment, and a short explanation.

Each issue is a small lab: build something, run it locally, look at what the machine did, and explain the result in plain words.

Issue drafts are in:

- [`docs/github-issue-drafts.md`](docs/github-issue-drafts.md)

Each issue includes:

- project name
- exact files to create
- dataset to use
- online resources
- step-by-step work
- modification challenge
- debugging checklist
- acceptance criteria
- time estimate
- expertise gained

## Final Expertise At The End Of The Course

At the end of this course, I should **not** pretend to be an ML expert yet.

I should be a **practical beginner with real intuition about how machines work and learn**. That means I can build small things, explain what is happening, and debug common problems without panicking.

### I should understand these ideas in clear words

- A **model** is a guesser with adjustable knobs.
- **Training** means making guesses, measuring wrongness, and adjusting the knobs.
- **Loss** is the number that says “how wrong was the model?”
- An **optimizer** is the tool that changes the knobs.
- **Learning rate** controls how big each change is.
- A **tensor** is a box/table/grid of numbers.
- `backward()` asks PyTorch to calculate adjustment hints.
- A **neural network** is a stack of adjustable steps that can learn flexible patterns.
- **Accuracy** is useful, but it does not tell the whole story.
- **Mistakes** are clues. They show what the model has not learned well.

### I should be able to build these things

- A tiny learning loop by hand.
- A PyTorch version of the same learning loop.
- A number prediction model.
- A classification model for simple 2D points.
- A small neural network for nonlinear data.
- A Fashion-MNIST image classifier.
- A mistake explorer that shows confident wrong predictions and a confusion matrix.

### I should be able to debug these things

When a model does not learn, I should know how to:

- Check shapes.
- Inspect raw examples and labels.
- Run one tiny batch.
- Overfit a tiny dataset.
- Print losses and predictions.
- Check CPU/GPU placement.
- Lower the learning rate.
- Compare what I expected with what actually happened.

### My level after the course

After the course, my expected level is:

> **Beginner-to-lower-intermediate practical ML learner.**

That means:

- I can follow beginner PyTorch tutorials without feeling totally lost.
- I can explain the basic learning loop to another beginner.
- I can train small local models.
- I can inspect model behavior instead of only looking at final accuracy.
- I can continue into deeper topics later: CNNs, transfer learning, embeddings, transformers, or the math behind gradients.

It does **not** mean:

- I am ready to build production ML systems alone.
- I deeply understand calculus-based backpropagation.
- I can design large models from scratch.
- I can skip careful debugging.

The win condition is simple: **machines that learn should feel less like magic and more like systems I can inspect.**
