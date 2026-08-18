"""
Phase 0 Day 3 — R1 risk validation
Goal: train a tiny CNN on MNIST, export ONNX, verify operators,
      then import into GD32 Embedded AI Tool to check code generation.

Run (in mace_env or any env with torch + torchvision + onnx + onnxruntime):
    cd CIMC/model/ai1_vision_cnn
    python train_mnist_hello.py

Output: tiny_cnn_mnist.onnx  (~55 KB)
Next:   import that file into GD32 Embedded AI Tool (see README_gdai.md)
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

try:
    import onnx
    import onnxruntime as ort
except ImportError:
    print("pip install onnx onnxruntime")
    raise


# Tiny CNN — same operator set that AI-1 vision CNN will use on GD32H759:
#   Conv2d + ReLU + MaxPool2d + Flatten + Linear
# Flash footprint target: <80 KB weights (INT8 after quantization)
class TinyCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=3, padding=1),   # 28x28 -> 28x28, 4ch
            nn.ReLU(),
            nn.MaxPool2d(2, 2),                           # -> 14x14
            nn.Conv2d(4, 8, kernel_size=3, padding=1),   # -> 14x14, 8ch
            nn.ReLU(),
            nn.MaxPool2d(2, 2),                           # -> 7x7
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(8 * 7 * 7, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")

    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    train_ds = datasets.MNIST("./data", train=True,  download=True, transform=tfm)
    test_ds  = datasets.MNIST("./data", train=False, download=True, transform=tfm)
    train_dl = DataLoader(train_ds, batch_size=256, shuffle=True,  num_workers=0)
    test_dl  = DataLoader(test_ds,  batch_size=256, shuffle=False, num_workers=0)

    model = TinyCNN().to(device)
    print(f"Params : {count_params(model):,}")
    print(f"FP32 weight size : {count_params(model)*4/1024:.1f} KB")

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, 4):
        model.train()
        running_loss = 0.0
        for data, target in train_dl:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            loss = criterion(model(data), target)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        model.eval()
        correct = 0
        with torch.no_grad():
            for data, target in test_dl:
                data, target = data.to(device), target.to(device)
                correct += model(data).argmax(1).eq(target).sum().item()
        acc = 100.0 * correct / len(test_ds)
        avg_loss = running_loss / len(train_dl)
        print(f"Epoch {epoch}/3  loss={avg_loss:.4f}  test_acc={acc:.1f}%")

    # Save PyTorch weights
    torch.save(model.state_dict(), "tiny_cnn_mnist.pt")
    print("Saved: tiny_cnn_mnist.pt")

    # Export ONNX
    model.eval()
    dummy = torch.zeros(1, 1, 28, 28, device=device)

    onnx_path = "tiny_cnn_mnist.onnx"
    torch.onnx.export(
        model,
        dummy,
        onnx_path,
        opset_version=11,           # opset 11 — widest tool compatibility
        input_names=["input"],
        output_names=["output"],
        do_constant_folding=True,
        dynamic_axes=None,          # static shape: GD32 AI Tool needs fixed shape
    )
    print(f"\nONNX exported: {onnx_path}  ({os.path.getsize(onnx_path)/1024:.1f} KB)")

    # Validate with onnx checker
    m = onnx.load(onnx_path)
    onnx.checker.check_model(m)
    print("onnx.checker : OK")

    # Validate with onnxruntime
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    x_np = np.zeros((1, 1, 28, 28), dtype=np.float32)
    out  = sess.run(None, {"input": x_np})
    pred = int(np.argmax(out[0]))
    print(f"ORT inference: output shape={out[0].shape}  dummy_pred={pred}  OK")

    # Print operators — paste this into README_gdai.md for compatibility tracking
    print("\n=== ONNX operators (paste into README_gdai.md) ===")
    ops = sorted(set(n.op_type for n in m.graph.node))
    for op in ops:
        print(f"  {op}")

    print("\n=== Next: GD32 Embedded AI Tool ===")
    print(f"  1. Open GD32 Embedded AI Tool")
    print(f"  2. New Project -> select GD32H759")
    print(f"  3. Import ONNX: {os.path.abspath(onnx_path)}")
    print(f"  4. Input name='input'  shape=[1,1,28,28]  dtype=float32")
    print(f"  5. Compression: None (float32 first, INT8 later)")
    print(f"  6. Generate Code -> copy network*.c/h to firmware/ai_models_c/")
    print(f"  7. Open Keil, add network*.c to project, F7 build -> check 0 error")


if __name__ == "__main__":
    train()
