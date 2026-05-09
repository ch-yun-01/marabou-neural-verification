"""
analyze_results.py

results/summary.json 을 읽어 다음 분석 및 시각화를 수행한다.

[기본 분석]
1.  Robustness heatmap            — digit x epsilon, SAT/UNSAT 색상
2.  SAT rate by epsilon           — epsilon별 SAT 비율 막대그래프
3.  SAT rate by digit             — digit별 SAT 비율 막대그래프
4.  Verification time heatmap     — digit x epsilon, 소요 시간
5.  Verification time by epsilon  — epsilon별 평균/최대 시간 꺾은선
6.  Robustness boundary curve     — epsilon 증가에 따른 UNSAT 비율 감소
7.  Per-digit SAT rate vs epsilon — digit별 꺾은선
8.  Stacked bar by epsilon        — SAT/UNSAT/OTHER 누적 막대
9.  Summary CSV                   — 보고서용 표

[PCA 분석]
10. PCA explained variance        — 누적 분산 설명률
11. PCA 2D scatter                — 클러스터, 취약도별 색상
12. PCA 3D scatter                — 클러스터 3D
13. Cluster distance heatmap      — digit 간 클러스터 거리 행렬
14. Distance vs first SAT epsilon — 최근접 거리 vs 첫 SAT ε 산점도

모든 출력은 results/analysis/ 폴더에 저장된다.
"""

import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")   # 디스플레이 없는 환경(서버 등)에서도 동작하도록 설정
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D   # noqa: F401  3D scatter 등록용

try:
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    SKLEARN_OK = True
except ImportError:
    SKLEARN_OK = False
    print("[WARN] scikit-learn not installed — PCA analyses will be skipped.")
    print("       Install with: pip install scikit-learn --break-system-packages")

try:
    from torchvision import datasets, transforms
    TORCH_OK = True
except ImportError:
    TORCH_OK = False
    print("[WARN] torchvision not installed — PCA analyses will be skipped.")

SUMMARY_JSON = os.path.join("results", "summary.json")
OUT_DIR      = os.path.join("results", "analysis")

# 그래프 공통 색상
COLOR_UNSAT = "#4e91d6"
COLOR_SAT   = "#e05c5c"

# digit 취약 그룹별 색상
VULN_COLOR = {
    "high":   "#e05c5c",   # ε ≤ 0.03 에서 SAT
    "medium": "#f4a460",   # ε = 0.05 에서 SAT
    "low":    "#4e91d6",   # ε ≥ 0.1  에서 SAT
}


# ===========================================================================
# Shared helpers
# ===========================================================================
def load_summary() -> list:
    """results/summary.json 을 로드한다."""
    if not os.path.exists(SUMMARY_JSON):
        print(f"[ERROR] {SUMMARY_JSON} not found. Run test.py first.")
        sys.exit(1)
    with open(SUMMARY_JSON, encoding="utf-8") as f:
        return json.load(f)


def get_axes(summary: list):
    """digit 목록과 epsilon 목록을 정렬해서 반환한다."""
    digits   = sorted(set(r["digit"]   for r in summary))
    epsilons = sorted(set(r["epsilon"] for r in summary))
    return digits, epsilons


def build_matrix(summary: list, field: str):
    """
    digit(행) x epsilon(열) 행렬을 만든다.
    field: 'result' (SAT=1, UNSAT=0) 또는 'time'
    """
    digits, epsilons = get_axes(summary)
    matrix = np.full((len(digits), len(epsilons)), np.nan)
    for r in summary:
        i = digits.index(r["digit"])
        j = epsilons.index(r["epsilon"])
        if field == "result":
            matrix[i, j] = 1.0 if r["result"] == "SAT" else (
                            0.0 if r["result"] == "UNSAT" else np.nan)
        elif field == "time":
            matrix[i, j] = r.get("time", np.nan)
    return matrix, digits, epsilons


def sat_rate(rows: list) -> float:
    """주어진 행 목록의 SAT 비율(%)을 반환한다."""
    if not rows:
        return 0.0
    return sum(1 for r in rows if r["result"] == "SAT") / len(rows) * 100


def first_sat_epsilon(summary: list) -> dict:
    """
    summary에서 digit별 첫 SAT epsilon을 계산한다.
    모든 epsilon에서 UNSAT이면 inf.
    """
    _, epsilons = get_axes(summary)
    result = {}
    for d in range(10):
        for e in epsilons:
            row = next((r for r in summary if r["digit"] == d and r["epsilon"] == e), None)
            if row and row["result"] == "SAT":
                result[d] = e
                break
        else:
            result[d] = float("inf")
    return result


def vuln_group(digit: int, first_sat: dict) -> str:
    """digit의 취약 그룹(high/medium/low)을 반환한다."""
    e = first_sat.get(digit, float("inf"))
    if e <= 0.03:   return "high"
    elif e <= 0.05: return "medium"
    else:           return "low"


# ===========================================================================
# 1. Robustness heatmap
# ===========================================================================
def plot_robustness_heatmap(summary: list, out_dir: str):
    """digit x epsilon 격자에 SAT(빨강) / UNSAT(파랑)을 표시한다."""
    matrix, digits, epsilons = build_matrix(summary, "result")

    fig, ax = plt.subplots(figsize=(len(epsilons) * 1.8 + 2, len(digits) * 0.75 + 2))
    cmap = mcolors.ListedColormap([COLOR_UNSAT, COLOR_SAT])
    ax.imshow(matrix, cmap=cmap, vmin=0, vmax=1, aspect="auto")

    for i in range(len(digits)):
        for j in range(len(epsilons)):
            val = matrix[i, j]
            if not np.isnan(val):
                ax.text(j, i, "SAT" if val == 1 else "UNSAT",
                        ha="center", va="center", fontsize=9,
                        color="white", fontweight="bold")

    ax.set_xticks(range(len(epsilons)))
    ax.set_yticks(range(len(digits)))
    ax.set_xticklabels([f"ε={e}" for e in epsilons])
    ax.set_yticklabels([f"digit {d}" for d in digits])
    ax.set_xlabel("Epsilon (L-inf radius)")
    ax.set_ylabel("Digit")
    ax.set_title("Robustness Verification Results\n(SAT = counterexample found, UNSAT = robust)")
    legend = [Patch(color=COLOR_UNSAT, label="UNSAT (robust)"),
              Patch(color=COLOR_SAT,   label="SAT (vulnerable)")]
    ax.legend(handles=legend, loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0)
    plt.tight_layout()

    path = os.path.join(out_dir, "01_robustness_heatmap.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


# ===========================================================================
# 2. SAT rate by epsilon
# ===========================================================================
def plot_sat_rate_by_epsilon(summary: list, out_dir: str):
    """epsilon별 SAT 비율을 막대그래프로 표시한다."""
    _, epsilons = get_axes(summary)
    rates = [sat_rate([r for r in summary if r["epsilon"] == e]) for e in epsilons]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar([str(e) for e in epsilons], rates,
                  color=[COLOR_SAT if r >= 50 else COLOR_UNSAT for r in rates],
                  edgecolor="white", width=0.5)
    for bar, rate in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                f"{rate:.0f}%", ha="center", va="bottom", fontsize=10)
    ax.set_ylim(0, 115)
    ax.set_xlabel("Epsilon (L-inf radius)")
    ax.set_ylabel("SAT rate (%)")
    ax.set_title("SAT Rate by Epsilon\n(higher = more vulnerable to perturbation)")
    ax.axhline(50, color="gray", linestyle="--", linewidth=0.8, label="50% line")
    ax.legend()
    plt.tight_layout()

    path = os.path.join(out_dir, "02_sat_rate_by_epsilon.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


# ===========================================================================
# 3. SAT rate by digit
# ===========================================================================
def plot_sat_rate_by_digit(summary: list, out_dir: str):
    """digit별 SAT 비율을 막대그래프로 표시한다."""
    digits, _ = get_axes(summary)
    rates = [sat_rate([r for r in summary if r["digit"] == d]) for d in digits]

    fig, ax = plt.subplots(figsize=(9, 4))
    bars = ax.bar([str(d) for d in digits], rates,
                  color=[COLOR_SAT if r >= 50 else COLOR_UNSAT for r in rates],
                  edgecolor="white", width=0.6)
    for bar, rate in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                f"{rate:.0f}%", ha="center", va="bottom", fontsize=9)
    ax.set_ylim(0, 115)
    ax.set_xlabel("Digit")
    ax.set_ylabel("SAT rate (%)")
    ax.set_title("SAT Rate by Digit\n(across all epsilon values)")
    ax.axhline(50, color="gray", linestyle="--", linewidth=0.8, label="50% line")
    ax.legend()
    plt.tight_layout()

    path = os.path.join(out_dir, "03_sat_rate_by_digit.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


# ===========================================================================
# 4. Verification time heatmap
# ===========================================================================
def plot_time_heatmap(summary: list, out_dir: str):
    """digit x epsilon 격자에 검증 소요 시간을 히트맵으로 표시한다."""
    if not any("time" in r for r in summary):
        print("[SKIP] No time data — skipping time heatmap.")
        return

    matrix, digits, epsilons = build_matrix(summary, "time")
    fig, ax = plt.subplots(figsize=(len(epsilons) * 1.8 + 2, len(digits) * 0.75 + 2))
    im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto")

    for i in range(len(digits)):
        for j in range(len(epsilons)):
            val = matrix[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.1f}s", ha="center", va="center",
                        fontsize=8, color="black")

    ax.set_xticks(range(len(epsilons)))
    ax.set_yticks(range(len(digits)))
    ax.set_xticklabels([f"ε={e}" for e in epsilons])
    ax.set_yticklabels([f"digit {d}" for d in digits])
    ax.set_xlabel("Epsilon (L-inf radius)")
    ax.set_ylabel("Digit")
    ax.set_title("Verification Time per Query (seconds)")
    plt.colorbar(im, ax=ax, label="Time (s)")
    plt.tight_layout()

    path = os.path.join(out_dir, "04_time_heatmap.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


# ===========================================================================
# 5. Verification time by epsilon (mean & max)
# ===========================================================================
def plot_time_by_epsilon(summary: list, out_dir: str):
    """epsilon별 평균/최대 검증 시간을 꺾은선 그래프로 표시한다."""
    if not any("time" in r for r in summary):
        print("[SKIP] No time data — skipping time by epsilon plot.")
        return

    _, epsilons = get_axes(summary)
    means, maxes = [], []
    for e in epsilons:
        times = [r["time"] for r in summary if r["epsilon"] == e and "time" in r]
        means.append(np.mean(times) if times else 0)
        maxes.append(np.max(times)  if times else 0)

    x = range(len(epsilons))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(x, means, marker="o", color="#2c7bb6", label="Mean time")
    ax.plot(x, maxes, marker="s", color="#d7191c", linestyle="--", label="Max time")
    ax.fill_between(x, means, maxes, alpha=0.15, color="#abd9e9")

    ax.set_xticks(list(x))
    ax.set_xticklabels([str(e) for e in epsilons])
    ax.set_xlabel("Epsilon (L-inf radius)")
    ax.set_ylabel("Time (seconds)")
    ax.set_title("Verification Time by Epsilon\n(mean and max across all digits)")
    ax.legend()
    plt.tight_layout()

    path = os.path.join(out_dir, "05_time_by_epsilon.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


# ===========================================================================
# 6. Robustness boundary curve
# ===========================================================================
def plot_robustness_boundary(summary: list, out_dir: str):
    """epsilon 증가에 따라 UNSAT 비율이 어떻게 감소하는지 꺾은선으로 표시한다."""
    _, epsilons = get_axes(summary)
    unsat_rates = [100 - sat_rate([r for r in summary if r["epsilon"] == e])
                   for e in epsilons]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(epsilons, unsat_rates, marker="o", color="#2c7bb6", linewidth=2)
    ax.fill_between(epsilons, unsat_rates, alpha=0.15, color="#2c7bb6")
    ax.axhline(50, color="gray", linestyle="--", linewidth=0.8, label="50% threshold")

    # 처음으로 50% 아래로 내려가는 epsilon 표시
    for e, u in zip(epsilons, unsat_rates):
        if u < 50:
            ax.axvline(e, color=COLOR_SAT, linestyle=":", linewidth=1,
                       label=f"First <50% at ε={e}")
            break

    ax.set_ylim(0, 105)
    ax.set_xlabel("Epsilon (L-inf radius)")
    ax.set_ylabel("UNSAT rate (%)")
    ax.set_title("Robustness Boundary Curve\n(UNSAT rate vs epsilon)")
    ax.legend()
    plt.tight_layout()

    path = os.path.join(out_dir, "06_robustness_boundary.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


# ===========================================================================
# 7. Per-digit SAT rate vs epsilon
# ===========================================================================
def plot_per_digit_lines(summary: list, out_dir: str):
    """digit별로 epsilon 증가에 따른 SAT 비율 변화를 꺾은선으로 겹쳐 표시한다."""
    digits, epsilons = get_axes(summary)
    cmap_lines = plt.get_cmap("tab10")

    fig, ax = plt.subplots(figsize=(9, 5))
    for idx, d in enumerate(digits):
        rates = [sat_rate([r for r in summary if r["digit"] == d and r["epsilon"] == e])
                 for e in epsilons]
        ax.plot(epsilons, rates, marker="o", linewidth=1.8,
                color=cmap_lines(idx), label=f"digit {d}")

    ax.axhline(50, color="gray", linestyle="--", linewidth=0.8)
    ax.set_ylim(-5, 105)
    ax.set_xlabel("Epsilon (L-inf radius)")
    ax.set_ylabel("SAT rate (%)")
    ax.set_title("SAT Rate per Digit vs Epsilon\n(each line = one digit)")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    plt.tight_layout()

    path = os.path.join(out_dir, "07_per_digit_sat_vs_epsilon.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


# ===========================================================================
# 8. Stacked bar — SAT / UNSAT / OTHER per epsilon
# ===========================================================================
def plot_stacked_bar(summary: list, out_dir: str):
    """epsilon별 SAT / UNSAT / TIMEOUT+UNKNOWN 건수를 누적 막대로 표시한다."""
    _, epsilons = get_axes(summary)
    sat_counts, unsat_counts, other_counts = [], [], []
    for e in epsilons:
        rows = [r for r in summary if r["epsilon"] == e]
        sat_counts.append(sum(1 for r in rows if r["result"] == "SAT"))
        unsat_counts.append(sum(1 for r in rows if r["result"] == "UNSAT"))
        other_counts.append(sum(1 for r in rows if r["result"] not in ("SAT", "UNSAT")))

    x     = np.arange(len(epsilons))
    width = 0.5

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x, unsat_counts, width, label="UNSAT", color=COLOR_UNSAT)
    ax.bar(x, sat_counts,   width, bottom=unsat_counts, label="SAT", color=COLOR_SAT)
    ax.bar(x, other_counts, width,
           bottom=[u + s for u, s in zip(unsat_counts, sat_counts)],
           label="TIMEOUT/UNKNOWN", color="#aaaaaa")

    for i, (u, s, o) in enumerate(zip(unsat_counts, sat_counts, other_counts)):
        if u > 0: ax.text(i, u/2,       str(u), ha="center", va="center", color="white", fontsize=9)
        if s > 0: ax.text(i, u+s/2,     str(s), ha="center", va="center", color="white", fontsize=9)
        if o > 0: ax.text(i, u+s+o/2,   str(o), ha="center", va="center", color="black", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels([str(e) for e in epsilons])
    ax.set_xlabel("Epsilon (L-inf radius)")
    ax.set_ylabel("Number of queries")
    ax.set_title("Query Results by Epsilon\n(stacked: UNSAT / SAT / Other)")
    ax.legend()
    plt.tight_layout()

    path = os.path.join(out_dir, "08_stacked_results_by_epsilon.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


# ===========================================================================
# 9. Summary CSV
# ===========================================================================
def save_summary_csv(summary: list, out_dir: str):
    """보고서용 CSV 표를 저장한다."""
    digits, epsilons = get_axes(summary)
    header = ["Digit"] + [f"eps={e}" for e in epsilons]
    rows   = [header]

    for d in digits:
        row = [str(d)]
        for e in epsilons:
            match = next((r for r in summary if r["digit"] == d and r["epsilon"] == e), None)
            row.append(match["result"] if match else "N/A")
        rows.append(row)

    rows.append(["---"] * len(header))
    for label, res in [("SAT count", "SAT"), ("UNSAT count", "UNSAT")]:
        row = [label]
        for e in epsilons:
            col = [r for r in summary if r["epsilon"] == e]
            row.append(str(sum(1 for r in col if r["result"] == res)))
        rows.append(row)

    sat_rates_row = ["SAT rate (%)"]
    for e in epsilons:
        col = [r for r in summary if r["epsilon"] == e]
        sat_rates_row.append(f"{sat_rate(col):.1f}%")
    rows.append(sat_rates_row)

    path = os.path.join(out_dir, "summary_table.csv")
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(",".join(row) + "\n")
    print(f"Saved: {path}")

    print()
    col_w = 12
    for row in rows:
        print("  ".join(cell.ljust(col_w) for cell in row))


# ===========================================================================
# PCA helpers
# ===========================================================================
def load_mnist_samples(n_per_digit: int = 200):
    """MNIST 테스트셋에서 digit별 n_per_digit개 샘플을 로드한다."""
    from torchvision import datasets, transforms
    transform = transforms.Compose([transforms.ToTensor()])
    dataset   = datasets.MNIST("./data", train=False, download=True, transform=transform)

    X_list, y_list = [], []
    counts = {d: 0 for d in range(10)}
    for img, label in dataset:
        if counts[label] < n_per_digit:
            X_list.append(img.view(-1).numpy())
            y_list.append(label)
            counts[label] += 1
        if all(v >= n_per_digit for v in counts.values()):
            break
    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=int)


# ===========================================================================
# 10. PCA explained variance
# ===========================================================================
def plot_pca_variance(pca_full, out_dir: str):
    """PCA 누적 분산 설명률을 표시한다."""
    cumvar = np.cumsum(pca_full.explained_variance_ratio_) * 100
    xs = range(1, len(cumvar) + 1)

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(xs, cumvar, color="#2c7bb6", linewidth=1.5)
    ax.fill_between(xs, cumvar, alpha=0.15, color="#2c7bb6")

    for threshold, color in [(90, "orange"), (95, "red")]:
        idx = next((i for i, v in enumerate(cumvar) if v >= threshold), None)
        if idx is not None:
            ax.axhline(threshold, color=color, linestyle="--",
                       linewidth=0.8, label=f"{threshold}% @ PC{idx+1}")
            ax.axvline(idx + 1, color=color, linestyle=":", linewidth=0.8)

    ax.set_xlabel("Number of principal components")
    ax.set_ylabel("Cumulative explained variance (%)")
    ax.set_title("PCA Explained Variance\n(how many PCs needed to explain MNIST variance)")
    ax.set_ylim(0, 101)
    ax.legend()
    plt.tight_layout()

    path = os.path.join(out_dir, "10_pca_variance.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


# ===========================================================================
# 11. PCA 2D scatter
# ===========================================================================
def plot_pca2d(X_pca: np.ndarray, y: np.ndarray,
               first_sat: dict, out_dir: str):
    """PCA 2D 공간에서 digit 클러스터를 취약도에 따라 색칠해 표시한다."""
    fig, ax = plt.subplots(figsize=(10, 8))
    marker_map  = {"high": "^", "medium": "s", "low": "o"}
    group_label = {
        "high":   "High vulnerability (SAT at ε≤0.03)",
        "medium": "Medium vulnerability (SAT at ε=0.05)",
        "low":    "Low vulnerability (SAT at ε≥0.1)",
    }

    for d in range(10):
        mask  = (y == d)
        group = vuln_group(d, first_sat)
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
                   c=VULN_COLOR[group], marker=marker_map[group],
                   alpha=0.45, s=18)
        # 클러스터 중심에 digit 번호 표시
        cx, cy = X_pca[mask, 0].mean(), X_pca[mask, 1].mean()
        ax.text(cx, cy, str(d), fontsize=13, fontweight="bold",
                color=VULN_COLOR[group], ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7))

    legend_elems = [
        Line2D([0], [0], marker="^", color="w", markerfacecolor=VULN_COLOR["high"],
               markersize=10, label=group_label["high"]),
        Line2D([0], [0], marker="s", color="w", markerfacecolor=VULN_COLOR["medium"],
               markersize=10, label=group_label["medium"]),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=VULN_COLOR["low"],
               markersize=10, label=group_label["low"]),
    ]
    ax.legend(handles=legend_elems, loc="lower right", fontsize=9)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("PCA 2D Projection of MNIST Test Samples\n"
                 "(colour/shape = vulnerability group)")
    plt.tight_layout()

    path = os.path.join(out_dir, "11_pca2d_scatter.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


# ===========================================================================
# 12. PCA 3D scatter
# ===========================================================================
def plot_pca3d(X_pca: np.ndarray, y: np.ndarray,
               first_sat: dict, out_dir: str):
    """PCA 3D 공간에서 digit 클러스터를 시각화한다."""
    fig = plt.figure(figsize=(11, 8))
    ax  = fig.add_subplot(111, projection="3d")

    for d in range(10):
        mask  = (y == d)
        group = vuln_group(d, first_sat)
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1], X_pca[mask, 2],
                   c=VULN_COLOR[group], alpha=0.3, s=10)
        cx = X_pca[mask, 0].mean()
        cy = X_pca[mask, 1].mean()
        cz = X_pca[mask, 2].mean()
        ax.text(cx, cy, cz, str(d), fontsize=11, fontweight="bold",
                color=VULN_COLOR[group])

    ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.set_zlabel("PC3")
    ax.set_title("PCA 3D Projection of MNIST Test Samples")
    plt.tight_layout()

    path = os.path.join(out_dir, "12_pca3d_scatter.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


# ===========================================================================
# 13. Cluster distance heatmap
# ===========================================================================
def plot_cluster_distance_heatmap(X_pca: np.ndarray, y: np.ndarray,
                                   out_dir: str):
    """PCA 공간에서 digit 클러스터 중심 간 유클리드 거리를 히트맵으로 표시한다."""
    digits  = list(range(10))
    centers = np.array([X_pca[y == d].mean(axis=0) for d in digits])

    dist_matrix = np.zeros((10, 10))
    for i in range(10):
        for j in range(10):
            dist_matrix[i, j] = np.linalg.norm(centers[i] - centers[j])

    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(dist_matrix, cmap="YlOrRd_r")

    for i in range(10):
        for j in range(10):
            val = dist_matrix[i, j]
            ax.text(j, i, f"{val:.1f}", ha="center", va="center",
                    fontsize=8,
                    color="white" if val < dist_matrix.max() * 0.4 else "black")

    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    ax.set_xticklabels([f"digit {d}" for d in digits], rotation=45, ha="right")
    ax.set_yticklabels([f"digit {d}" for d in digits])
    ax.set_title("Cluster Center Distance in PCA Space\n"
                 "(smaller = closer clusters = more vulnerable to misclassification)")
    plt.colorbar(im, ax=ax, label="Euclidean distance (PCA space)")
    plt.tight_layout()

    path = os.path.join(out_dir, "13_cluster_distance_heatmap.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")

    return dist_matrix


# ===========================================================================
# 14. Min cluster distance vs first SAT epsilon
# ===========================================================================
def plot_distance_vs_epsilon(dist_matrix: np.ndarray,
                              first_sat: dict, out_dir: str):
    """
    각 digit의 최근접 클러스터 거리 vs 첫 SAT epsilon 산점도.
    거리가 짧을수록 더 작은 epsilon에서 SAT이 나온다는 상관관계를 보여준다.
    """
    digits   = list(range(10))
    min_dists, first_sats = [], []

    for d in digits:
        dists = [dist_matrix[d, j] for j in digits if j != d]
        min_dists.append(min(dists))
        fs = first_sat.get(d, float("inf"))
        first_sats.append(fs if fs != float("inf") else 0.25)

    fig, ax = plt.subplots(figsize=(7, 5))
    for d, md, fs in zip(digits, min_dists, first_sats):
        group = vuln_group(d, first_sat)
        ax.scatter(md, fs, c=VULN_COLOR[group], s=120, zorder=5)
        ax.annotate(f" digit {d}", (md, fs), fontsize=9,
                    color=VULN_COLOR[group], fontweight="bold")

    # 추세선
    coeffs = np.polyfit(min_dists, first_sats, 1)
    x_line = np.linspace(min(min_dists), max(min_dists), 100)
    ax.plot(x_line, np.polyval(coeffs, x_line),
            color="gray", linestyle="--", linewidth=1, label="trend")

    # 상관계수
    corr = np.corrcoef(min_dists, first_sats)[0, 1]
    ax.text(0.05, 0.92, f"Pearson r = {corr:.3f}", transform=ax.transAxes,
            fontsize=10, color="gray",
            bbox=dict(boxstyle="round", fc="white", alpha=0.8))

    ax.set_xlabel("Min distance to nearest cluster (PCA space)")
    ax.set_ylabel("First SAT epsilon")
    ax.set_title("Nearest Cluster Distance vs First SAT Epsilon\n"
                 "(closer clusters → more vulnerable at smaller ε)")
    ax.legend()
    plt.tight_layout()

    path = os.path.join(out_dir, "14_distance_vs_epsilon.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")
    print(f"  Pearson r (min_dist vs first_SAT_eps): {corr:.4f}")


# ===========================================================================
# Main
# ===========================================================================
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    summary = load_summary()
    print(f"Loaded {len(summary)} experiments from {SUMMARY_JSON}\n")

    # ----- 기본 분석 (1~9) -----
    print("=== Basic analysis ===")
    plot_robustness_heatmap(summary, OUT_DIR)
    plot_sat_rate_by_epsilon(summary, OUT_DIR)
    plot_sat_rate_by_digit(summary, OUT_DIR)
    plot_time_heatmap(summary, OUT_DIR)
    plot_time_by_epsilon(summary, OUT_DIR)
    plot_robustness_boundary(summary, OUT_DIR)
    plot_per_digit_lines(summary, OUT_DIR)
    plot_stacked_bar(summary, OUT_DIR)
    save_summary_csv(summary, OUT_DIR)

    # ----- PCA 분석 (10~14) -----
    if not (SKLEARN_OK and TORCH_OK):
        print("\n[SKIP] PCA analyses skipped (missing sklearn or torchvision).")
        return

    print("\n=== PCA analysis ===")
    first_sat = first_sat_epsilon(summary)
    print("First SAT epsilon per digit:")
    for d in range(10):
        e = first_sat.get(d, "never")
        print(f"  digit {d}: ε = {e}")

    print("\nLoading MNIST test samples...")
    X, y = load_mnist_samples(n_per_digit=200)
    print(f"  Loaded {len(X)} samples")

    print("Fitting PCA...")
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca2     = PCA(n_components=2)
    pca3     = PCA(n_components=3)
    pca50    = PCA(n_components=50)    # 거리 계산용
    pca_full = PCA(n_components=min(100, X.shape[1]))

    X_2d  = pca2.fit_transform(X_scaled)
    X_3d  = pca3.fit_transform(X_scaled)
    X_50d = pca50.fit_transform(X_scaled)
    pca_full.fit(X_scaled)

    print(f"  2D  explains {sum(pca2.explained_variance_ratio_)*100:.1f}% of variance")
    print(f"  3D  explains {sum(pca3.explained_variance_ratio_)*100:.1f}% of variance")
    print(f"  50D explains {sum(pca50.explained_variance_ratio_)*100:.1f}% of variance")
    print()

    plot_pca_variance(pca_full, OUT_DIR)
    plot_pca2d(X_2d,  y, first_sat, OUT_DIR)
    plot_pca3d(X_3d,  y, first_sat, OUT_DIR)
    dist_matrix = plot_cluster_distance_heatmap(X_50d, y, OUT_DIR)
    plot_distance_vs_epsilon(dist_matrix, first_sat, OUT_DIR)

    print(f"\nAll analysis outputs saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()