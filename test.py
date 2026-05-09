"""
test.py

Run the main Marabou local robustness verification experiment.
If a counterexample is found (SAT), automatically visualises the result.

Verification property:
  For a reference input x classified as digit d,
  verify whether every x' with ||x' - x||_inf <= epsilon
  is also classified as d.

Encoding:
  Instead of using a disjunction constraint, this script runs separate
  pairwise queries for each target class j != d.

  For each target j, Marabou checks whether there exists x' such that:

      output[j] >= output[d]

  This is encoded as:

      output[d] - output[j] <= 0

Interpretation:
  - If any target query is SAT:
      A counterexample exists. The property is violated.
      -> adversarial_example.npy and adversarial_visualisation.png are saved.
  - If all target queries are UNSAT:
      The model is locally robust for this sample and epsilon.
"""

import os
import sys
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")   # 디스플레이 없는 환경(서버 등)에서도 동작하도록 설정
import matplotlib.pyplot as plt

try:
    from maraboupy import Marabou
except ImportError:
    print(
        "[ERROR] maraboupy is not installed.\n"
        "Please build Marabou from source and install maraboupy.\n"
        "See README.md for instructions."
    )
    sys.exit(1)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
ONNX_PATH    = "mnist_fc.onnx"
SAMPLES_PATH = "sample_inputs.npy"

EPSILON      = 0.1   # L-inf 허용 perturbation 반경
TARGET_DIGIT = 3     # 강건성을 검증할 숫자
TIMEOUT      = 300   # 쿼리당 타임아웃(초)

RESULTS_DIR     = "results"   # 검증 결과물을 저장할 폴더
RESULT_LOG_PATH = os.path.join(RESULTS_DIR, "verification_result.txt")
ADV_PATH        = os.path.join(RESULTS_DIR, "adversarial_example.npy")
VIS_PATH        = os.path.join(RESULTS_DIR, "adversarial_visualisation.png")


# ---------------------------------------------------------------------------
# Result parsing
# ---------------------------------------------------------------------------
def parse_solve_result(result):
    """
    Normalize different maraboupy solve() return formats.

    Possible formats:
      [exit_code, vals, stats]
      (exit_code, vals, stats)
      [vals, stats]
      (vals, stats)

    Return:
      result_str, vals, stats
    """
    if isinstance(result, (list, tuple)):
        if len(result) == 3:
            a, b, c = result
            if isinstance(a, str):
                # (result_str, vals, stats) 형태
                result_str = a.upper()
                vals, stats = b, c
            else:
                # (vals, stats, ?) 형태 — 첫 원소가 문자열이 아님
                vals, stats = a, b
                result_str = "SAT" if vals else "UNSAT"

        elif len(result) == 2:
            a, b = result
            if isinstance(a, str):
                # (result_str, vals) 형태
                result_str = a.upper()
                vals, stats = b, None
            else:
                # (vals, stats) 형태
                vals, stats = a, b
                result_str = "SAT" if vals else "UNSAT"

        else:
            raise RuntimeError(f"Unexpected Marabou solve result length: {len(result)}")

    else:
        # 딕셔너리 또는 기타 단일 반환값
        vals  = result
        stats = None
        result_str = "SAT" if vals else "UNSAT"

    if vals is None:
        vals = {}

    return result_str, vals, stats


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_sample(digit):
    """요청한 숫자의 평탄화된 테스트 샘플(784,)을 반환한다."""
    samples = np.load(SAMPLES_PATH, allow_pickle=True).item()
    if digit not in samples:
        raise ValueError(
            f"No sample for digit {digit}. "
            f"Available digits: {sorted(samples.keys())}"
        )
    return samples[digit].astype(np.float64)


def add_input_bounds(network, input_vars, x, epsilon):
    """L-inf 볼 입력 제약을 네트워크에 추가한다."""
    x = x.flatten()
    for i, var in enumerate(input_vars):
        lo = float(np.clip(x[i] - epsilon, 0.0, 1.0))
        hi = float(np.clip(x[i] + epsilon, 0.0, 1.0))
        network.setLowerBound(var, lo)
        network.setUpperBound(var, hi)


def add_target_counterexample_constraint(network, output_vars, digit, target):
    """
    Search for a counterexample where:

        output[target] >= output[digit]

    Marabou addInequality encodes:

        sum(coeff_i * var_i) <= scalar

    So we encode:

        output[digit] - output[target] <= 0

    disjunction 대신 addInequality를 사용해
    Marabou 내부 Equation 생성자 버그를 우회한다.
    """
    network.addInequality(
        [output_vars[digit], output_vars[target]],
        [1.0, -1.0],
        0.0
    )


def extract_adversarial_input(vals, input_vars):
    """Marabou 변수 할당 딕셔너리에서 적대적 입력 벡터를 추출한다."""
    adv_values = []
    for var in input_vars:
        var_id = int(var)
        if var_id in vals:
            adv_values.append(vals[var_id])
        elif var in vals:
            adv_values.append(vals[var])
        else:
            # Marabou가 고정 변수를 명시적으로 할당하지 않는 드문 경우 — 0.0으로 대체
            adv_values.append(0.0)
    return np.array(adv_values, dtype=np.float64)


# ---------------------------------------------------------------------------
# Core verification
# ---------------------------------------------------------------------------
def run_single_target_query(x, digit, target, epsilon):
    """
    Run one pairwise query: does target class beat digit?
    네트워크는 쿼리마다 새로 로드해야 한다 (Marabou가 객체를 수정하기 때문).
    """
    network     = Marabou.read_onnx(ONNX_PATH)
    input_vars  = network.inputVars[0].flatten()   # shape: (784,)
    output_vars = network.outputVars[0].flatten()  # shape: (10,)

    add_input_bounds(network, input_vars, x, epsilon)
    add_target_counterexample_constraint(network, output_vars, digit, target)

    options = Marabou.createOptions(timeoutInSeconds=TIMEOUT, verbosity=0)

    print(f"  target={target}: running Marabou...")
    t0      = time.time()
    result  = network.solve(options=options)
    elapsed = time.time() - t0

    result_str, vals, stats = parse_solve_result(result)

    # SAT이면 적대적 입력 추출
    adv = extract_adversarial_input(vals, input_vars) if result_str == "SAT" else None

    return {
        "target": target,
        "result": result_str,
        "time":   elapsed,
        "vals":   vals,
        "adv":    adv,
    }


def run_verification(digit, epsilon):
    """
    Run pairwise robustness verification for one digit against all other classes.
    하나라도 SAT이면 즉시 반환하고 반례를 보고한다.
    """
    x = load_sample(digit)

    print("=" * 60)
    print("Verifying local robustness")
    print("=" * 60)
    print(f"Model        : {ONNX_PATH}")
    print(f"Target digit : {digit}")
    print(f"Epsilon      : {epsilon}")
    print(f"Timeout      : {TIMEOUT}s per query")
    print()

    results    = []
    total_time = 0.0

    for target in range(10):
        if target == digit:
            continue  # 자기 자신 클래스는 스킵

        result      = run_single_target_query(x=x, digit=digit, target=target, epsilon=epsilon)
        results.append(result)
        total_time += result["time"]

        print(f"    result={result['result']}, time={result['time']:.2f}s")

        if result["result"] == "SAT":
            print()
            print(f"[SAT] Counterexample found: digit {digit} -> target {target}")
            return "SAT", total_time, results, result["adv"]

    print()
    print("[UNSAT] All pairwise target queries were UNSAT.")
    return "UNSAT", total_time, results, None


# ---------------------------------------------------------------------------
# Visualisation (SAT일 때 자동 실행)
# ---------------------------------------------------------------------------
def visualise_counterexample(x, adv, digit):
    """
    원본 샘플, perturbation, 적대적 입력을 나란히 시각화하고 PNG로 저장한다.
    SAT 결과가 나왔을 때 main()에서 자동으로 호출된다.
    """
    delta = adv - x   # perturbation 벡터

    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5))

    # 원본 이미지
    axes[0].imshow(x.reshape(28, 28), cmap="gray", vmin=0, vmax=1)
    axes[0].set_title(f"Original (digit {digit})")
    axes[0].axis("off")

    # Perturbation (시각적으로 잘 보이도록 최댓값 기준으로 스케일 조정)
    scale = np.abs(delta).max()
    axes[1].imshow(delta.reshape(28, 28), cmap="RdBu", vmin=-scale, vmax=scale)
    axes[1].set_title("Perturbation (delta)")
    axes[1].axis("off")

    # 적대적 입력
    axes[2].imshow(adv.reshape(28, 28), cmap="gray", vmin=0, vmax=1)
    axes[2].set_title("Adversarial input")
    axes[2].axis("off")

    # 전체 제목에 perturbation 크기 표시
    fig.suptitle(
        f"L-inf: {np.abs(delta).max():.5f}  |  L2: {np.linalg.norm(delta):.5f}",
        fontsize=11,
    )
    plt.tight_layout()
    plt.savefig(VIS_PATH, dpi=150)
    plt.close()
    print(f"Visualisation saved to {VIS_PATH}")


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def save_log(result_str, total_time, results, digit, epsilon):
    """검증 결과를 텍스트 파일로 저장한다."""
    lines = [
        "Marabou MNIST Local Robustness Verification",
        "=" * 60,
        f"Model: {ONNX_PATH}",
        f"Digit: {digit}",
        f"Epsilon: {epsilon}",
        f"Final result: {result_str}",
        f"Total time: {total_time:.2f}s",
        "",
        "Pairwise results:",
    ]
    for r in results:
        lines.append(
            f"  target={r['target']}, "
            f"result={r['result']}, "
            f"time={r['time']:.2f}s"
        )
    with open(RESULT_LOG_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Log saved to {RESULT_LOG_PATH}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    # results/ 폴더가 없으면 자동 생성
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # 필수 파일 존재 여부 확인
    if not os.path.exists(ONNX_PATH):
        print(f"[ERROR] {ONNX_PATH} not found. Run train_model.py first.")
        sys.exit(1)
    if not os.path.exists(SAMPLES_PATH):
        print(f"[ERROR] {SAMPLES_PATH} not found. Run train_model.py first.")
        sys.exit(1)

    digit   = TARGET_DIGIT
    epsilon = EPSILON

    result_str, total_time, results, adv = run_verification(digit, epsilon)

    print()
    print("=" * 60)
    print("Final summary")
    print("=" * 60)
    print(f"Result     : {result_str}")
    print(f"Total time : {total_time:.2f}s")

    if result_str == "UNSAT":
        print(
            f"\nInterpretation:\n"
            f"  No target class could exceed the digit-{digit} logit\n"
            f"  within L-inf radius {epsilon}.\n"
            f"  Therefore, the model is locally robust for this sample."
        )

    elif result_str == "SAT":
        x     = load_sample(digit)
        delta = adv - x

        print(
            f"\nInterpretation:\n"
            f"  Marabou found an input within L-inf radius {epsilon}\n"
            f"  that changes the model's decision away from digit {digit}."
        )
        print()
        print("Counterexample summary:")
        print(f"  L-inf perturbation : {np.abs(delta).max():.6f}")
        print(f"  L2 perturbation    : {np.linalg.norm(delta):.6f}")
        print(f"  Adversarial range  : [{adv.min():.4f}, {adv.max():.4f}]")
        print(f"  Changed pixels     : {(np.abs(delta) > 1e-6).sum()}")

        # 적대적 입력 저장
        np.save(ADV_PATH, adv)
        print(f"\nAdversarial input saved to {ADV_PATH}")

        # SAT이면 자동으로 시각화 실행
        visualise_counterexample(x, adv, digit)

    save_log(result_str, total_time, results, digit, epsilon)


if __name__ == "__main__":
    main()