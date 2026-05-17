from pathlib import Path
import numpy as np
import torch

import matplotlib.pyplot as plt
from mlcourse.device import get_device

LEARNING_RATE = 0.01
NUMBER_EPOCHS = 200


def main() -> None:
    device = get_device()
    print(f"Using device: {device}")

    x = torch.tensor(
        [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5], dtype=torch.float32, device=device
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

    optimizer = torch.optim.SGD([weight, bias], LEARNING_RATE)
    trace_rows: list[list[float]] = []
    for epoch in range(NUMBER_EPOCHS):
        optimizer.zero_grad()  # clears old gradient hints so each update uses only the current loss

        predictions = weight * x + bias  # the model
        errors = predictions - y
        loss = torch.mean(errors**2)

        loss.backward()

        print(f"Epoch {epoch}")
        print(f"  loss: {loss.item():.6f}")
        print(f"  weight before update: {weight.item():.6f}")
        print(f"  bias before update: {bias.item():.6f}")
        print(f"  weight gradient: {weight.grad.item():.6f}")
        print(f"  bias gradient: {bias.grad.item():.6f}")

        trace_rows.append(
            [
                float(epoch),
                loss.item(),
                weight.item(),
                bias.item(),
                weight.grad.item(),
                bias.grad.item(),
            ]
        )

        optimizer.step()

        print(f"  weight after update: {weight.item():.6f}")
        print(f"  bias after update: {bias.item():.6f}")
        print("-" * 20)
    print("First trace row:", trace_rows[0])
    print("Last trace row:", trace_rows[-1])

    project_root = Path(__file__).resolve().parents[2]
    artifact_dir = project_root / "artifacts" / "week-02-pytorch-basics"
    artifact_dir.mkdir(parents=True, exist_ok=True)

    np.savetxt(
        artifact_dir / "gradient_trace.csv",
        np.array(trace_rows, dtype=np.float64),
        delimiter=",",
        header="epoch,loss,weight,bias,weight_grad,bias_grad",
        comments="",
    )

    loss_history = [row[1] for row in trace_rows]

    plt.plot(loss_history)
    plt.xlabel("Epoch")
    plt.ylabel("Mean squared error")
    plt.title("PyTorch Line Learner loss curve")
    plt.savefig(artifact_dir / "loss_curve.png")
    plt.close()


if __name__ == "__main__":
    main()
