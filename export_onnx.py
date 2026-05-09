import os
import torch
import torch.nn as nn


MODEL_DIR = "models"
PT_MODEL_PATH = os.path.join(MODEL_DIR, "mnist_mlp.pt")
ONNX_MODEL_PATH = os.path.join(MODEL_DIR, "mnist_mlp.onnx")


class SmallMNISTMLP(nn.Module):
    """
    Same architecture as in train_model.py.
    This class must match the saved PyTorch checkpoint.
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 16),
            nn.ReLU(),
            nn.Linear(16, 10)
        )

    def forward(self, x):
        return self.net(x)


def export_to_onnx():
    os.makedirs(MODEL_DIR, exist_ok=True)

    if not os.path.exists(PT_MODEL_PATH):
        raise FileNotFoundError(
            f"PyTorch model not found: {PT_MODEL_PATH}. "
            f"Run train_model.py first."
        )

    model = SmallMNISTMLP()
    model.load_state_dict(torch.load(PT_MODEL_PATH, map_location="cpu"))
    model.eval()

    dummy_input = torch.randn(1, 1, 28, 28)

    torch.onnx.export(
        model,
        dummy_input,
        ONNX_MODEL_PATH,
        input_names=["input"],
        output_names=["output"],
        opset_version=11,
        do_constant_folding=True
    )

    print(f"Exported ONNX model to {ONNX_MODEL_PATH}")


if __name__ == "__main__":
    export_to_onnx()