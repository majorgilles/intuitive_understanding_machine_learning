# Agent Instructions

## Python code style

This repository is for learning machine learning mechanics, so Python code should
favor clarity, explicitness, and strict typing over clever shortcuts.

### Typing rules

- Add type annotations to every function parameter and return value.
  - Why: readers should be able to understand the expected data flow without
    mentally executing the program.

- Avoid `Any`. If `Any` is truly necessary, add a short comment explaining why.
  - Why: `Any` disables useful type checking and can hide beginner-confusing bugs.

- Prefer precise built-in types such as `list[float]`, `dict[str, float]`,
  `tuple[float, float]`, and `float` instead of vague containers.
  - Why: precise types make examples easier to debug and refactor.

- For NumPy arrays, use `numpy.typing.NDArray` with a dtype when practical, for example:

  ```python
  import numpy as np
  from numpy.typing import NDArray

  FloatArray = NDArray[np.float64]
  ```

  - Why: many ML mistakes come from unclear array-like values, dtypes, or shapes.

- Use named type aliases when a concept appears repeatedly, such as `FloatArray`
  or `LossHistory`.
  - Why: domain names make learning code read like the concept being taught.

- Keep functions small and typed at the boundaries.
  - Why: small typed functions make the learning loop easier to inspect:
    predict -> measure wrongness -> adjust -> repeat.

- Do not introduce untyped helper functions, even in scripts.
  - Why: examples in this repo are teaching material, so shortcuts become patterns.

- Prefer `pathlib.Path` over raw string paths for filesystem values.
  - Why: path types clarify when a value is a file path rather than arbitrary text.
