import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


SEED = 42
BATCH_SIZE = 128
EPOCHS = 5
LR = 1e-3

MODEL_DIR = "models"
DATA_DIR = "data"

MODEL_PATH = os.path.join(MODEL_DIR, "mnist_mlp.pt")
SAMPLE_INPUT_PATH = os.path.join(DATA_DIR, "sample_input.npy")
SAMPLE_LABEL_PATH = os.path.join(DATA_DIR, "sample_label.npy")


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class SmallMNISTMLP(nn.Module):
    """
    Small fully connected neural network for MNIST classification.

    Input:  1 x 28 x 28 image
    Output: 10 logits for digits 0-9

    The model is intentionally small so that Marabou can handle the exported ONNX network.
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


def train():
    set_seed(SEED)

    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    train_dataset = datasets.MNIST(
        root=DATA_DIR,
        train=True,
        download=False,
        transform=transform
    )

    test_dataset = datasets.MNIST(
        root=DATA_DIR,
        train=False,
        download=False,
        transform=transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    model = SmallMNISTMLP().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            logits = model(images)
            loss = criterion(logits, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * images.size(0)

            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_loss = total_loss / total
        train_acc = correct / total

        print(
            f"Epoch [{epoch + 1}/{EPOCHS}] "
            f"Loss: {train_loss:.4f} "
            f"Train Acc: {train_acc:.4f}"
        )

    # Test accuracy
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            preds = logits.argmax(dim=1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    test_acc = correct / total
    print(f"Test Accuracy: {test_acc:.4f}")

    # Save model
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Saved model to {MODEL_PATH}")

    # Save one correctly classified sample for Marabou verification
    sample_saved = False

    model.eval()
    with torch.no_grad():
        for image, label in test_dataset:
            image_batch = image.unsqueeze(0).to(device)
            label_int = int(label)

            logits = model(image_batch)
            pred = int(logits.argmax(dim=1).item())

            if pred == label_int:
                np.save(SAMPLE_INPUT_PATH, image.numpy().astype(np.float32))
                np.save(SAMPLE_LABEL_PATH, np.array(label_int, dtype=np.int64))

                print(f"Saved verification sample.")
                print(f"Sample label: {label_int}")
                print(f"Model prediction: {pred}")
                print(f"Sample input path: {SAMPLE_INPUT_PATH}")
                print(f"Sample label path: {SAMPLE_LABEL_PATH}")

                sample_saved = True
                break

    if not sample_saved:
        raise RuntimeError("No correctly classified test sample was found.")


if __name__ == "__main__":
    train()