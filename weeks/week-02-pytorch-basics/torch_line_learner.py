import torch
from numpy import dtype

from mlcourse.device import get_device

def main() -> None:
    device = get_device()
    print(f"Using device: {device}")

    x = torch.tensor(
        [-5, -4, -3, -2 , -1, 0, 1, 2, 3, 4, 5],
        dtype=torch.float32,
        device=device
    )
    y = 2 * x + 1
    print(f"x:{x}")
    print(f"y:{y}")

    weight = torch.tensor(0.0, dtype=torch.float32, device=device, requires_grad=True)
    bias = torch.tensor(0.0, dtype=torch.float32, device=device, requires_grad=True)

    predictions = weight * x + bias
    errors = predictions - y
    loss = torch.mean(errors**2)

    print("Starting weight:", weight)
    print("Starting bias:", bias)
    print("Starting predictions:", predictions)
    print("Starting loss:", loss)

    loss.backward()
    print("Weight gradient:", weight.grad)
    print("Bias gradient:", bias.grad)


if __name__ == "__main__":
    main()