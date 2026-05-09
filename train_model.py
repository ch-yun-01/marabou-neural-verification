"""
train_model.py
Train a small fully-connected network (784 -> 32 -> 16 -> 10) on MNIST,
then export it to ONNX format for Marabou verification.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import os


# -------------------------------------------------------------------
# Model definition
# -------------------------------------------------------------------
class SmallFC(nn.Module):
    """784 -> ReLU(32) -> ReLU(16) -> 10 (logits, no softmax)."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(784, 32),   # 입력층 -> 은닉층1
            nn.ReLU(),            # 활성화 함수 (Marabou가 ReLU를 지원함)
            nn.Linear(32, 16),    # 은닉층1 -> 은닉층2
            nn.ReLU(),            # 활성화 함수
            nn.Linear(16, 10),    # 은닉층2 -> 출력층 (클래스 10개)
        )

    def forward(self, x):
        # x 형태: (배치, 1, 28, 28) 또는 (배치, 784)
        # Marabou는 평탄화된 1차원 입력을 요구하므로 reshape 수행
        x = x.view(x.size(0), -1)
        return self.net(x)


# -------------------------------------------------------------------
# Training helpers
# -------------------------------------------------------------------
def train(model, loader, optimizer, criterion, device):
    """Run one epoch of training; return average loss and accuracy."""
    model.train()
    total_loss, correct = 0.0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()           # 이전 배치의 기울기 초기화
        outputs = model(images)         # 순전파
        loss = criterion(outputs, labels)
        loss.backward()                 # 역전파
        optimizer.step()                # 가중치 업데이트
        total_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
    n = len(loader.dataset)
    return total_loss / n, correct / n  # 평균 손실, 정확도 반환


def evaluate(model, loader, criterion, device):
    """Evaluate the model on a dataset; return average loss and accuracy."""
    model.eval()
    total_loss, correct = 0.0, 0
    with torch.no_grad():   # 평가 시에는 기울기 계산 불필요
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            total_loss += criterion(outputs, labels).item() * images.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
    n = len(loader.dataset)
    return total_loss / n, correct / n


def main():
    # GPU가 있으면 CUDA 사용, 없으면 CPU 사용
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 픽셀값을 [0, 1] 범위로 정규화 — epsilon 제약 조건 정의에 중요
    transform = transforms.Compose([transforms.ToTensor()])

    # MNIST 데이터셋 로드 (없으면 자동 다운로드)
    train_dataset = datasets.MNIST("./data", train=True,  download=True, transform=transform)
    test_dataset  = datasets.MNIST("./data", train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    test_loader  = DataLoader(test_dataset,  batch_size=256, shuffle=False)

    # 모델, 손실 함수, 옵티마이저 초기화
    model     = SmallFC().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # 학습 루프
    epochs = 10
    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = train(model, train_loader, optimizer, criterion, device)
        te_loss, te_acc = evaluate(model, test_loader, criterion, device)
        print(
            f"Epoch {epoch:2d}/{epochs}  "
            f"train loss={tr_loss:.4f} acc={tr_acc:.4f}  "
            f"test loss={te_loss:.4f} acc={te_acc:.4f}"
        )

    # ------------------------------------------------------------------
    # Export to ONNX
    # ------------------------------------------------------------------
    model.eval()
    # ※ GPU로 학습한 모델을 반드시 CPU로 이동한 뒤 export해야 한다.
    #   dummy 텐서가 CPU인데 모델이 CUDA에 남아 있으면
    #   torch.onnx.export 내부 FakeTensor에서 device 충돌이 발생한다.
    model_cpu = model.cpu()
    dummy     = torch.zeros(1, 784)  # Marabou는 평탄화된 784차원 입력을 사용
    onnx_path = "mnist_fc.onnx"
    torch.onnx.export(
        model_cpu,
        dummy,
        onnx_path,
        input_names=["input"],
        output_names=["output"],
        opset_version=11,
        dynamic_axes=None,  # 배치 크기를 1로 고정 (Marabou 요구사항)
    )
    print(f"\nModel exported to {onnx_path}")

    # ------------------------------------------------------------------
    # Save one correctly-classified test sample per digit (0-9)
    # ------------------------------------------------------------------
    # model_cpu는 위 export 단계에서 이미 CPU로 이동되어 있음
    samples = {}  # {숫자 레이블: 평탄화된 numpy 배열 (784,)}
    for images, labels in DataLoader(test_dataset, batch_size=1, shuffle=False):
        label = labels.item()
        flat  = images.view(1, 784)
        pred  = model_cpu(flat).argmax(1).item()
        # 모델이 올바르게 분류한 샘플만 저장 (레이블당 1개)
        if pred == label and label not in samples:
            samples[label] = flat.squeeze().numpy()  # shape: (784,)
        if len(samples) == 10:  # 0~9 모두 수집되면 종료
            break

    np.save("sample_inputs.npy", samples)
    print("Sample inputs saved to sample_inputs.npy")
    print("Saved one correctly-classified sample per digit:")
    for d in sorted(samples):
        print(f"  digit {d}: min={samples[d].min():.3f} max={samples[d].max():.3f}")


if __name__ == "__main__":
    main()