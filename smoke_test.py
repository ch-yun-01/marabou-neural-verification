"""
test.py
Demonstrate running Marabou on the trained MNIST model.
Required by the assignment submission checklist.

Steps:
  1. Load the exported ONNX model (mnist_fc.onnx).
  2. Load one test sample per digit (0-9) from sample_inputs.npy.
  3. For each digit, run 9 pairwise L-inf robustness queries (epsilon=0.01).
  4. Print a summary table of results.

For the main experiment (epsilon=0.1, digit=3) see verify.py.
"""

import numpy as np
import os
import sys
import time

try:
    from maraboupy import Marabou
except ImportError:
    print(
        "[ERROR] maraboupy is not installed.\n"
        "Build Marabou and install maraboupy. See README.md."
    )
    sys.exit(1)

ONNX_PATH     = "mnist_fc.onnx"      # 검증 대상 ONNX 모델
SAMPLES_PATH  = "sample_inputs.npy"  # 각 숫자별 기준 샘플
SMOKE_EPSILON = 0.01   # 스모크 테스트용 소형 epsilon (빠른 검증)
TIMEOUT       = 120    # 쿼리당 타임아웃(초)


def pairwise_robustness(x: np.ndarray, digit: int, target: int, epsilon: float) -> str:
    """
    Run a single pairwise query: can target class beat digit?

    addInequality로 인코딩하여 Marabou 내부 Equation 생성자 버그를 우회한다.
    disjunction 대신 개별 쿼리를 실행하는 방식.
    """
    # 쿼리마다 네트워크를 새로 로드해야 한다 (Marabou가 객체를 수정함)
    network     = Marabou.read_onnx(ONNX_PATH)
    input_vars  = network.inputVars[0].flatten()   # 784개 입력 변수
    output_vars = network.outputVars[0].flatten()  # 10개 출력 변수

    # 입력 범위 설정: L-inf 볼을 [0, 1]로 클리핑
    x_flat = x.flatten()
    for i, var in enumerate(input_vars):
        lo = float(np.clip(x_flat[i] - epsilon, 0.0, 1.0))
        hi = float(np.clip(x_flat[i] + epsilon, 0.0, 1.0))
        network.setLowerBound(var, lo)
        network.setUpperBound(var, hi)

    # 출력 제약: output[digit] - output[target] <= 0
    # "target 클래스가 digit 클래스를 이기는 상황"을 탐색
    network.addInequality(
        [output_vars[digit], output_vars[target]],
        [1.0, -1.0],
        0.0
    )

    options = Marabou.createOptions(timeoutInSeconds=TIMEOUT, verbosity=0)
    result  = network.solve(options=options)

    # 반환 형태 정규화: maraboupy 버전에 따라 다를 수 있음
    if isinstance(result, (list, tuple)):
        first = result[0]

        # 예: ["sat", vals, stats] 또는 ["unsat", vals, stats]
        if isinstance(first, str):
            status = first.upper()
            if status in ["SAT", "UNSAT", "TIMEOUT", "UNKNOWN"]:
                return status
            else:
                # 예상치 못한 문자열 — 그대로 보존해 디버깅에 활용
                return f"UNKNOWN({status})"

        # 예: [vals, stats] 또는 (vals, stats)
        vals = first

    else:
        vals = result

    return "SAT" if vals else "UNSAT"


def run_digit(digit: int, x: np.ndarray, epsilon: float) -> str:
    """
    digit에 대해 9개 타겟 클래스 전부 pairwise 쿼리를 실행한다.
    하나라도 SAT이면 'SAT', 모두 UNSAT이면 'UNSAT' 반환.
    TIMEOUT/UNKNOWN이 있으면 해당 상태를 반환한다.
    """
    saw_timeout = False
    saw_unknown = False

    for target in range(10):
        if target == digit:
            continue  # 자기 자신은 스킵

        res = pairwise_robustness(x, digit, target, epsilon)

        if res == "SAT":
            return "SAT"       # 반례 발견 — 즉시 종료
        elif res == "TIMEOUT":
            saw_timeout = True # 타임아웃 발생 기록 — 계속 진행
        elif res != "UNSAT":
            saw_unknown = True # 예상치 못한 결과 기록

    # 모든 쿼리가 끝난 뒤 최악 상태를 반환
    if saw_timeout:
        return "TIMEOUT"
    if saw_unknown:
        return "UNKNOWN"
    return "UNSAT"


def main():
    # 필수 파일 존재 여부 확인
    for path in (ONNX_PATH, SAMPLES_PATH):
        if not os.path.exists(path):
            print(f"[ERROR] {path} not found. Run train_model.py first.")
            sys.exit(1)

    samples = np.load(SAMPLES_PATH, allow_pickle=True).item()
    digits  = sorted(samples.keys())

    print(f"Marabou smoke test  (epsilon={SMOKE_EPSILON})")
    print(f"Model  : {ONNX_PATH}")
    print(f"Digits : {digits}\n")
    print(f"{'Digit':>6}  {'Result':>8}  {'Time(s)':>8}")
    print("-" * 30)

    for digit in digits:
        x = samples[digit].astype(np.float64)

        t0      = time.time()
        result  = run_digit(digit, x, SMOKE_EPSILON)
        elapsed = time.time() - t0

        print(f"{digit:>6}  {result:>8}  {elapsed:>8.2f}")

    print("\nSmoke test complete.")
    print("For the full experiment (epsilon=0.1, digit=3), run test.py.")


if __name__ == "__main__":
    main()