import numpy as np
from numpy.typing import NDArray
from pathlib import Path
import matplotlib.pyplot as plt

FloatArray = NDArray[np.float64]


def predict(inputs: FloatArray, weight: float, bias: float) -> FloatArray:
    """Model that predict the target using the current weight and bias."""
    return inputs * weight + bias


def mean_squared_error(predictions: FloatArray, targets: FloatArray) -> float:
    """Loss function"""
    errors = predictions - targets
    return float(np.mean(errors**2))


x = np.array([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5], dtype=np.float64)
y = 2 * x + 1

current_weight = 0.0
current_bias = 0.0

starting_predictions = predict(x, current_weight, current_bias)
starting_loss = mean_squared_error(starting_predictions, y)

print("Starting weight:", current_weight)
print("Starting bias:", current_bias)
print("Starting loss:", starting_loss)
print("First predictions:")

for input_value, target, prediction in zip(x[:5], y[:5], starting_predictions[:5]):
    print(f"x={input_value: .1f}, target={target: .1f}, prediction={prediction: .1f}")

errors = starting_predictions - y

weight_gradient = float(np.mean(2 * errors * x))
bias_gradient = float(np.mean(2 * errors))

learning_rate = 0.01
loss_history: list[float] = []
for step in range(200):
    predictions = predict(x, current_weight, current_bias)
    loss = mean_squared_error(predictions, y)
    loss_history.append(loss)

    errors = predictions - y

    # We compute gradients to decide in which direction to adjust the weights for the next pass
    # e.g. if the weight gradient is negative, then that means the current_weight will go up
    # if the gradient is positive, the current_weight will go down
    weight_gradient = float(np.mean(2 * errors * x))
    bias_gradient = float(np.mean(2 * errors))

    current_weight = current_weight - learning_rate * weight_gradient
    current_bias = current_bias - learning_rate * bias_gradient

    print(f"Current weight gradient after update at {step}: {weight_gradient:.12f}")
    print(f"Current bias gradient after update at {step}: {bias_gradient:.12f}")
    print(f"Current weight after update at {step}: {current_weight:.12f}")
    print(f"Current bias after update at {step}: {current_bias:.12f}")
    print(f"Loss: {loss:.12f}")
    print("-" * 20)

project_root = Path(__file__).resolve().parents[2]
artifact_dir = project_root / "artifacts" / "week-01-learning-loop"
artifact_dir.mkdir(parents=True, exist_ok=True)

plt.plot(loss_history)
plt.xlabel("Training step")
plt.ylabel("Mean squared error")
plt.title("Manual Line Learner loss curve")
plt.savefig(artifact_dir / "loss_curve.png")
plt.close()


ending_predictions = predict(x, current_weight, current_bias)

comparison_rows = np.column_stack((x, y, starting_predictions, ending_predictions))

np.savetxt(
    artifact_dir / "before_after_predictions.csv",
    comparison_rows,
    delimiter=",",
    header="x,target,before_prediction,after_prediction",
    comments="",
)

print(f"Ending weight: {current_weight:.12f}")
print(f"Ending bias: {current_bias:.12f}")
