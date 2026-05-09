# Marabou MNIST 신경망 강건성 검증

**Reliable and Trustworthy Artificial Intelligence — Assignment #3**

본 프로젝트는 [Marabou](https://github.com/NeuralNetworkVerification/Marabou)를 활용하여 MNIST 데이터셋으로 학습한 소형 완전연결 신경망의 국소 강건성(local robustness)을 형식적으로 검증하는 것을 목표로 합니다.

---

## 1. 프로젝트 개요

본 프로젝트에서는 MNIST 분류용 소형 Fully Connected Neural Network를 학습한 뒤, 특정 입력 주변의 `L∞` perturbation 영역 안에서 오분류를 유발하는 입력이 존재하는지 Marabou를 통해 검증합니다.

검증하고자 하는 속성은 다음과 같습니다.

> 기준 입력 `x`가 숫자 `d`로 분류될 때,  
> `||x' − x||∞ ≤ ε`를 만족하는 모든 입력 `x'` 역시 `d`로 분류되는가?

즉, 입력 `x` 주변의 작은 perturbation 범위 안에서도 모델의 예측이 유지되는지를 확인합니다.

- **UNSAT**: 해당 `L∞` 볼 안에 오분류를 유발하는 반례가 없음  
  → 국소 강건성 증명
- **SAT**: 오분류를 유발하는 반례가 존재함  
  → 적대적 입력 발견

---

## 2. 모델 구조

본 프로젝트에서 사용한 MNIST 분류 모델은 다음과 같은 소형 완전연결 신경망입니다.

| Layer | Input | Output | Activation |
|---|---:|---:|---|
| FC 1 | 784 | 32 | ReLU |
| FC 2 | 32 | 16 | ReLU |
| FC 3 | 16 | 10 | None |

학습 설정은 다음과 같습니다.

- Optimizer: Adam
- Learning rate: `1e-3`
- Epochs: 10
- Test accuracy: 약 95%

학습된 모델은 Marabou에서 사용할 수 있도록 ONNX 형식(`mnist_fc.onnx`)으로 내보냅니다.

---

## 3. 파일 구조

```text
marabou-neural-verification/
├── train_model.py       # MNIST 모델 학습 및 ONNX 내보내기
├── test.py              # 메인 검증 실험: digit × epsilon
├── smoke_test.py        # 빠른 동작 확인용 스모크 테스트
├── analyze_results.py   # 결과 분석 및 시각화
├── requirements.txt     # Python 의존성 목록
├── report.pdf           # 과제 보고서
└── README.md
