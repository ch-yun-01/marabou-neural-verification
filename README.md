# marabou-neural-verification
Reliable and Trustworthy Artificial Intelligence Assignment #3

# Marabou MNIST 신경망 강건성 검증

[Marabou](https://github.com/NeuralNetworkVerification/Marabou)를 사용해 MNIST로 학습한 소형 완전연결 신경망의 국소 강건성(local robustness)을 형식적으로 검증하는 프로젝트입니다.

---

## 개요

작은 FC 신경망을 MNIST로 학습한 뒤, 특정 입력 주변의 L-inf 볼 안에서 perturbation이 오분류를 유발할 수 있는지 Marabou로 검증합니다.

**검증 속성:**  
기준 입력 **x**가 숫자 **d**로 분류될 때, `||x' − x||∞ ≤ ε`를 만족하는 모든 **x'**도 **d**로 분류됨을 증명한다.

- **UNSAT** → 해당 L-inf 볼 안에 반례 없음 (강건성 증명)
- **SAT** → 반례(적대적 입력) 발견

---

## 모델 구조

| 레이어 | 입력 | 출력 | 활성화 |
|--------|------|------|--------|
| FC 1   | 784  | 32   | ReLU   |
| FC 2   | 32   | 16   | ReLU   |
| FC 3   | 16   | 10   | — (로짓) |

Adam (lr=1e-3), 10 에폭 학습, 테스트 정확도 약 95%.

---

## 파일 구조

```
marabou-neural-verification/
├── train_model.py       # MNIST 학습 및 ONNX 내보내기
├── test.py              # 메인 검증 실험 (전 digit × 여러 ε)
├── smoke_test.py        # 빠른 스모크 테스트 (전 digit, ε=0.01)
├── analyze_results.py   # 결과 분석 및 시각화 (기본 + PCA)
├── requirements.txt     # Python 의존성
├── report.pdf           # 과제 보고서
└── README.md
```

**실행 후 생성되는 파일:**
```
├── mnist_fc.onnx                        # 내보낸 ONNX 모델
├── sample_inputs.npy                    # digit별 기준 샘플
└── results/
    ├── summary.json                     # 전체 실험 결과 (digit × ε)
    ├── digit{d}_eps{e}/
    │   ├── verification_result.txt      # 실험별 검증 로그
    │   ├── adversarial_example.npy      # 적대적 입력 (SAT일 때만)
    │   └── adversarial_visualisation.png
    └── analysis/
        ├── 01_robustness_heatmap.png
        ├── 02_sat_rate_by_epsilon.png
        ├── 03_sat_rate_by_digit.png
        ├── 04_time_heatmap.png
        ├── 05_time_by_epsilon.png
        ├── 06_robustness_boundary.png
        ├── 07_per_digit_sat_vs_epsilon.png
        ├── 08_stacked_results_by_epsilon.png
        ├── summary_table.csv
        ├── 10_pca_variance.png
        ├── 11_pca2d_scatter.png
        ├── 12_pca3d_scatter.png
        ├── 13_cluster_distance_heatmap.png
        └── 14_distance_vs_epsilon.png
```

---

## 설치

### 1. Python 의존성 설치

```bash
pip install -r requirements.txt
```

PCA 분석을 위해 scikit-learn도 설치하세요:

```bash
pip install scikit-learn
```

### 2. Marabou 설치

Marabou는 PyPI에 없으므로 소스 빌드가 필요합니다. 설치 방법은 공식 저장소를 참고하세요:  
[https://github.com/NeuralNetworkVerification/Marabou](https://github.com/NeuralNetworkVerification/Marabou)

설치 후 확인:

```bash
python -c "from maraboupy import Marabou; print('Marabou OK')"
```

---

## 실행 방법

### Step 1 — 모델 학습 및 내보내기

```bash
python train_model.py
```

FC 신경망을 학습하고 `mnist_fc.onnx`로 내보냅니다.  
digit별 테스트 샘플 1개씩을 `sample_inputs.npy`에 저장합니다.

### Step 2 — 검증 실험 실행

```bash
python test.py
```

**digit 0~9** × **ε ∈ {0.01, 0.03, 0.05, 0.1, 0.2}** 총 50개 실험을 순서대로 실행합니다.

- 실험별 결과는 `results/digit{d}_eps{e}/`에 저장
- SAT이면 적대적 입력과 시각화 이미지 자동 생성
- 전체 결과는 `results/summary.json`에 누적

실험 조건을 변경하려면 `test.py` 안의 아래 두 줄을 수정하세요:

```python
EPSILONS = [0.01, 0.03, 0.05, 0.1, 0.2]   # 실험할 epsilon 값
for digit in range(0, 10):                  # 실험할 digit 범위
```

### Step 3 — 결과 분석

```bash
python analyze_results.py
```

`results/summary.json`을 읽어 14개 그래프와 요약 CSV를 `results/analysis/`에 저장합니다.  
scikit-learn이 설치되어 있으면 PCA 분석(10~14번)도 함께 실행됩니다.

### Step 4 — (선택) 스모크 테스트

```bash
python smoke_test.py
```

전 digit에 대해 ε=0.01로 빠르게 동작을 확인합니다.

---

## 인코딩 방식

`addDisjunctionConstraint` 대신 **pairwise `addInequality`** 방식을 사용합니다.  
(Marabou 내부 `EquationType` 생성자 버그를 우회하기 위함)

각 타겟 클래스 `j ≠ d`에 대해 아래 조건을 만족하는 x'이 존재하는지 검사합니다:

```
clip(x_i − ε, 0, 1) ≤ x'_i ≤ clip(x_i + ε, 0, 1)
output[d] − output[j] ≤ 0
```

9개 pairwise 쿼리가 모두 UNSAT이면 해당 지점에서 강건성이 증명됩니다.

---

## 주요 실험 결과

| ε    | SAT 비율 | 해석                     |
|------|----------|--------------------------|
| 0.01 | 0%       | 전체 강건                |
| 0.03 | 50%      | 강건성 경계 지점         |
| 0.05 | 80%      | 대부분 취약              |
| 0.1  | 100%     | 전체 취약                |
| 0.2  | 100%     | 전체 취약                |

**취약한 digit** (ε=0.03에서 SAT): 1, 3, 4, 5, 8  
**강건한 digit** (ε=0.1에서야 SAT): 0, 2

---
