import os
import numpy as np
import matplotlib.pyplot as plt


ORIGINAL_PATH = "data/sample_input.npy"
COUNTEREXAMPLE_PATH = "results/counterexample.npy"
SAVE_PATH = "results/counterexample.png"


def main():
    if not os.path.exists(COUNTEREXAMPLE_PATH):
        raise FileNotFoundError(
            "No counterexample file found. "
            "Run test.py first and make sure the result is SAT."
        )

    original = np.load(ORIGINAL_PATH).reshape(28, 28)
    counterexample = np.load(COUNTEREXAMPLE_PATH).reshape(28, 28)
    diff = np.abs(counterexample - original)

    plt.figure(figsize=(9, 3))

    plt.subplot(1, 3, 1)
    plt.imshow(original, cmap="gray")
    plt.title("Original")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(counterexample, cmap="gray")
    plt.title("Counterexample")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(diff, cmap="gray")
    plt.title("Absolute Difference")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(SAVE_PATH, dpi=200)
    print(f"Saved visualization to {SAVE_PATH}")


if __name__ == "__main__":
    main()