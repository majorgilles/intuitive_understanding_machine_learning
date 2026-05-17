# Week 00 Setup Report

## What worked

- `uv` is installed and available.
- A Python 3.12 virtual environment was created successfully.
- Project dependencies are managed through `pyproject.toml` and `uv.lock`.
- PyTorch and torchvision import successfully.
- CUDA is available through PyTorch.
- The environment check script runs with:

```text
uv run python weeks/week-00-setup/check_environment.py
```

- The script writes its output to `artifacts/week-00-setup/device_check.txt`.

## What failed or was confusing

- The first package import check failed because `src/mlcourse` was not yet exposed as an installable package.
- This was fixed by enabling package mode in `pyproject.toml` and adding Hatchling build configuration.
- The device helper file was initially misspelled as `devicy.py`, then renamed to `device.py`.

## CUDA status

CUDA works.

PyTorch reports:

- PyTorch version: `2.11.0+cu128`
- torchvision version: `0.26.0+cu128`
- CUDA available: `True`
- CUDA version: `12.8`
- GPU: `NVIDIA GeForce RTX 4070 SUPER`

## Command to run Python files in this repo

Use:

```text
uv run python path/to/script.py
```

For example:

```text
uv run python weeks/week-00-setup/check_environment.py
```

## CPU vs GPU in plain words

The CPU is the general-purpose processor that can run everything but may be slower for large tensor work. The GPU is built to do many numerical operations in parallel, so PyTorch can use it to speed up machine learning calculations when CUDA is working.
