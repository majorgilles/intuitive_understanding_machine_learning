from pathlib import Path
import platform

import torch
import torchvision

from mlcourse.device import get_device


def build_report() -> str:
   device = get_device()
   lines = [
       f"Python version: {platform.python_version()}",
       f"PyTorch version: {torch.__version__}",
       f"torchvision version: {torchvision.__version__}",
       f"CUDA available: {torch.cuda.is_available()}",
       f"Selected device: {device}",
   ]

   if torch.cuda.is_available():
       lines.append(f"GPU name: {torch.cuda.get_device_name(0)}")

   x = torch.tensor([1.0, 2.0, 3.0], device=device)
   lines.append(f"Tensor calculation: {x} * 2 = {x * 2}")

   return "\n".join(lines)


def main() -> None:
   report = build_report()
   print(report)

   output_path = Path("artifacts/week-00-setup/device_check.txt")
   output_path.parent.mkdir(parents=True, exist_ok=True)
   output_path.write_text(report + "\n", encoding="utf-8")


if __name__ == "__main__":
   main()