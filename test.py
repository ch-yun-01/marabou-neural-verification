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
      -> results/digit{d}_eps{e}/ 폴더에 결과 저장
  - If all target queries are UNSAT:
      The model is locally robust for this sample and epsilon.

Usage:
  # 기본값 (digit=3, epsilon=0.1)
  python test.py

  # 조건 변경
  python test.py --digit 5 --epsilon 0.05
  python test.py --digit 7 --epsilon 0.2
"""

import os
import sys
import time
import argparse
import json
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
# Configuration (기본값 — 커맨드라인 인자로 덮어쓸 수 있음)
# ---------------------------------------------------------------------------
ONNX_PATH    = "mnist_fc.onnx"
SAMPLES_PATH = "sample_inputs.npy"

DEFAULT_DIGIT   = 3      # 강건성을 검증할 숫자
DEFAULT_EPSILON = 0.1    # L-inf 허용 perturbation 반경
TIMEOUT         = 300    # 쿼리당 타임아웃(초)


def make_results_dir(digit: int, epsilon: float) -> str:
    """
    실험 조건별 결과 폴더 경로를 반환하고 생성한다.
    예: results/digit3_eps0.100/
    덮어쓰기를 방지해 각 실험 결과를 독립적으로 보존한다.
    """
    folder = os.path.join("results", f"digit{digit}_eps{epsilon:.3f}")
    os.makedirs(folder, exist_ok=True)
    return folder


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
def load_sample(digit: int) -> np.ndarray:
    """요청한 숫자의 평탄화된 테스트 샘플(784,)을 반환한다."""
    samples = np.load(SAMPLES_PATH, allow_pickle=True).item()
    if digit not in samples:
        raise ValueError(
            f"No sample for digit {digit}. "
            f"Available digits: {sorted(samples.keys())}"
        )
    return samples[digit].astype(np.float64)


def add_input_bounds(network, input_vars, x: np.ndarray, epsilon: float):
    """L-inf 볼 입력 제약을 네트워크에 추가한다."""
    x = x.flatten()
    for i, var in enumerate(input_vars):
        lo = float(np.clip(x[i] - epsilon, 0.0, 1.0))
        hi = float(np.clip(x[i] + epsilon, 0.0, 1.0))
        network.setLowerBound(var, lo)
        network.setUpperBound(var, hi)


def add_target_counterexample_constraint(network, output_vars, digit: int, target: int):
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


def extract_adversarial_input(vals: dict, input_vars) -> np.ndarray:
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
def run_single_target_query(x: np.ndarray, digit: int, target: int, epsilon: float) -> dict:
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


def run_verification(digit: int, epsilon: float):
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

        result     = run_single_target_query(x=x, digit=digit, target=target, epsilon=epsilon)
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
def visualise_counterexample(x: np.ndarray, adv: np.ndarray, digit: int, out_path: str):
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
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Visualisation saved to {out_path}")


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def save_log(result_str: str, total_time: float, results: list,
             digit: int, epsilon: float, out_path: str):
    """검증 결과를 텍스트 파일로 저장한다."""
    lines = [
        "Marabou MNIST Local Robustness Verification",
        "=" * 60,
        f"Model  : {ONNX_PATH}",
        f"Digit  : {digit}",
        f"Epsilon: {epsilon}",
        f"Result : {result_str}",
        f"Time   : {total_time:.2f}s",
        "",
        "Pairwise results:",
    ]
    for r in results:
        lines.append(
            f"  target={r['target']}, "
            f"result={r['result']}, "
            f"time={r['time']:.2f}s"
        )
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Log saved to {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    # 커맨드라인 인자 파싱 — 실험 조건을 바꿀 때 사용
    parser = argparse.ArgumentParser(description="Marabou MNIST robustness verification")
    parser.add_argument("--digit",   type=int,   default=DEFAULT_DIGIT,
                        help=f"Target digit to verify (default: {DEFAULT_DIGIT})")
    parser.add_argument("--epsilon", type=float, default=DEFAULT_EPSILON,
                        help=f"L-inf perturbation radius (default: {DEFAULT_EPSILON})")
    args = parser.parse_args()

    digit   = args.digit
    epsilon = args.epsilon

    # 실험 조건별 결과 폴더 생성 (예: results/digit3_eps0.100/)
    out_dir = make_results_dir(digit, epsilon)
    print(f"Results will be saved to: {out_dir}/\n")

    # 필수 파일 존재 여부 확인
    if not os.path.exists(ONNX_PATH):
        print(f"[ERROR] {ONNX_PATH} not found. Run train_model.py first.")
        sys.exit(1)
    if not os.path.exists(SAMPLES_PATH):
        print(f"[ERROR] {SAMPLES_PATH} not found. Run train_model.py first.")
        sys.exit(1)

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
        adv_path = os.path.join(out_dir, "adversarial_example.npy")
        np.save(adv_path, adv)
        print(f"\nAdversarial input saved to {adv_path}")

        # SAT이면 자동으로 시각화 실행
        vis_path = os.path.join(out_dir, "adversarial_visualisation.png")
        visualise_counterexample(x, adv, digit, vis_path)

    # 검증 로그 저장
    log_path = os.path.join(out_dir, "verification_result.txt")
    save_log(result_str, total_time, results, digit, epsilon, log_path)


# ---------------------------------------------------------------------------
# Batch runner — digit 1~9, epsilon 0.1 고정
# ---------------------------------------------------------------------------
def run_one(digit: int, epsilon: float):
    """단일 실험을 실행하고 결과를 저장한다."""
    out_dir = make_results_dir(digit, epsilon)
    print(f"\nResults will be saved to: {out_dir}/")

    result_str, total_time, results, adv = run_verification(digit, epsilon)

    print()
    print("=" * 60)
    print("Final summary")
    print("=" * 60)
    print(f"Digit      : {digit}")
    print(f"Epsilon    : {epsilon}")
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
        adv_path = os.path.join(out_dir, "adversarial_example.npy")
        np.save(adv_path, adv)
        print(f"\nAdversarial input saved to {adv_path}")

        # SAT이면 자동으로 시각화 실행
        vis_path = os.path.join(out_dir, "adversarial_visualisation.png")
        visualise_counterexample(x, adv, digit, vis_path)

    # 검증 로그 저장
    log_path = os.path.join(out_dir, "verification_result.txt")
    save_log(result_str, total_time, results, digit, epsilon, log_path)

    return result_str, total_time


if __name__ == "__main__":
    # 필수 파일 존재 여부 확인
    for _path in (ONNX_PATH, SAMPLES_PATH):
        if not os.path.exists(_path):
            print(f"[ERROR] {_path} not found. Run train_model.py first.")
            sys.exit(1)

    # 실험할 epsilon 값 목록 — 강건 / 경계 / 취약 구간을 커버
    EPSILONS = [0.01, 0.03, 0.05, 0.1, 0.2]

    # digit 1~9 x epsilon 3개 조합 순서대로 실험 실행
    summary = []
    for epsilon in EPSILONS:
        for digit in range(0, 10):
            print("\n" + "#" * 60)
            print(f"# Experiment: digit={digit}, epsilon={epsilon}")
            print("#" * 60)
            result, elapsed = run_one(digit, epsilon)
            summary.append({
                "digit":   digit,
                "epsilon": epsilon,
                "result":  result,
                "time":    round(elapsed, 2),
            })

    # 전체 실험 요약 출력
    print("\n" + "=" * 60)
    print("All experiments complete — summary")
    print("=" * 60)
    print(f"{'Digit':>6}  {'Epsilon':>8}  {'Result':>8}")
    print("-" * 30)
    for row in summary:
        print(f"{row['digit']:>6}  {row['epsilon']:>8.3f}  {row['result']:>8}")

    # 전체 실험 결과를 JSON으로 저장
    os.makedirs("results", exist_ok=True)
    json_path = os.path.join("results", "summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {json_path}")