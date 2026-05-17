import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]

def predict(inputs: FloatArray, weight: float, bias: float) -> FloatArray:
    return inputs * weight + bias

def mean_squared_error(predictions: FloatArray, targets: FloatArray) -> float:
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
lost_history: list[float] = []
for step in range(200):
    predictions = predict(x, current_weight, current_bias)
    loss = mean_squared_error(predictions, y)
    lost_history.append(loss)

    errors = predictions - y

    weight_gradient = float(np.mean(2 * errors * x))
    bias_gradient = float(np.mean(2 * errors))

    current_weight = current_weight - learning_rate * weight_gradient
    current_bias = current_bias - learning_rate * bias_gradient

    print(f"After update weight at {step}:", current_weight)
    print(f"After update bias at {step}:", current_bias)
    print(f"Loss: {loss}")
    print("-" * 20)