"""
Phase 0 Day 3 — Convert TinyCNN to TFLite via TinyNeuralNetwork
Run: cd CIMC/model/ai1_vision_cnn && python convert_to_tflite.py
Output: tiny_cnn_mnist_float32.tflite
"""

import torch
import torch.nn as nn
from tinynn.converter import TFLiteConverter


class TinyCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(4, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(8 * 7 * 7, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def main():
    model = TinyCNN()
    model.load_state_dict(
        torch.load("tiny_cnn_mnist.pt", map_location="cpu", weights_only=True)
    )
    model.eval()

    dummy_input = torch.zeros(1, 1, 28, 28)
    tflite_path = "tiny_cnn_mnist_float32.tflite"

    converter = TFLiteConverter(
        model=model,
        dummy_input=dummy_input,
        tflite_path=tflite_path,
        nchw_transpose=True,   # NCHW -> NHWC for TFLite
    )
    converter.convert()

    import os
    size_kb = os.path.getsize(tflite_path) / 1024
    print(f"\nTFLite exported: {tflite_path}  ({size_kb:.1f} KB)")
    print("Next: copy to C:\\GD32AI\\User_model\\cur_float32_tflite\\")
    print("      then run GD32 AI Tool quantize script or use this TFLite directly in Keil")


if __name__ == "__main__":
    main()
