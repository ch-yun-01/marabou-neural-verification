import os
import time
import numpy as np

from maraboupy import Marabou


ONNX_MODEL_PATH = "models/mnist_mlp.onnx"
SAMPLE_INPUT_PATH = "data/sample_input.npy"
SAMPLE_LABEL_PATH = "data/sample_label.npy"

RESULT_DIR = "results"
LOG_PATH = os.path.join(RESULT_DIR, "verification_log.txt")
COUNTEREXAMPLE_PATH = os.path.join(RESULT_DIR, "counterexample.npy")


def load_sample():
    if not os.path.exists(SAMPLE_INPUT_PATH):
        raise FileNotFoundError(
            f"Sample input not found: {SAMPLE_INPUT_PATH}. "
            f"Run train_model.py first."
        )

    if not os.path.exists(SAMPLE_LABEL_PATH):
        raise FileNotFoundError(
            f"Sample label not found: {SAMPLE_LABEL_PATH}. "
            f"Run train_model.py first."
        )

    x = np.load(SAMPLE_INPUT_PATH).astype(np.float32)
    label = int(np.load(SAMPLE_LABEL_PATH))

    return x, label


def get_prediction_with_numpy_onnx_reference(x):
    """
    This function is not used for Marabou verification.
    The saved sample was already selected as correctly classified during training.

    We keep the predicted class equal to the sample label for the verification query.
    """
    return None


def add_input_constraints(network, input_vars, x, epsilon):
    """
    Add L-infinity input perturbation constraints.

    For every pixel i:
        max(0, x_i - epsilon) <= x'_i <= min(1, x_i + epsilon)

    MNIST pixels are normalized to [0, 1].
    """
    x_flat = x.flatten()

    if len(input_vars) != len(x_flat):
        raise ValueError(
            f"Input variable size mismatch. "
            f"Marabou input vars: {len(input_vars)}, sample size: {len(x_flat)}"
        )

    for i, var in enumerate(input_vars):
        lower = max(0.0, float(x_flat[i] - epsilon))
        upper = min(1.0, float(x_flat[i] + epsilon))

        network.setLowerBound(var, lower)
        network.setUpperBound(var, upper)


def add_counterexample_constraint(network, output_vars, pred_class, target_class):
    """
    Add output constraint for searching an adversarial counterexample.

    We want to check whether there exists x' such that:

        output[target_class] >= output[pred_class]

    This means the target class logit is at least as large as the original predicted class logit.

    Marabou's addInequality encodes:

        sum(coefficients[i] * variables[i]) <= scalar

    Therefore:

        output[pred_class] - output[target_class] <= 0

    is equivalent to:

        output[target_class] >= output[pred_class]
    """
    network.addInequality(
        [output_vars[pred_class], output_vars[target_class]],
        [1.0, -1.0],
        0.0
    )


def run_single_target_query(x, pred_class, target_class, epsilon):
    """
    Run one Marabou query for a single target class.

    The query asks:
        Is there any input within the epsilon-ball around x
        that makes target_class score greater than or equal to pred_class score?
    """
    network = Marabou.read_onnx(ONNX_MODEL_PATH)

    input_vars = network.inputVars[0].flatten()
    output_vars = network.outputVars[0].flatten()

    add_input_constraints(network, input_vars, x, epsilon)
    add_counterexample_constraint(network, output_vars, pred_class, target_class)

    start_time = time.time()
    vals, stats = network.solve(verbose=False)
    runtime = time.time() - start_time

    is_sat = len(vals) > 0

    counterexample = None
    if is_sat:
        counterexample = np.array(
            [vals[var] for var in input_vars],
            dtype=np.float32
        ).reshape(1, 28, 28)

    return is_sat, runtime, counterexample


def run_verification(epsilon=0.001):
    """
    Verify local robustness for a single correctly classified MNIST sample.

    Since this is a 10-class classifier, we run 9 pairwise queries:

        target_class != pred_class

    If every query is UNSAT, then no class can overtake the predicted class
    within the epsilon-ball. In that case, the model is locally robust for this sample.

    If any query is SAT, Marabou found a counterexample.
    """
    if not os.path.exists(ONNX_MODEL_PATH):
        raise FileNotFoundError(
            f"ONNX model not found: {ONNX_MODEL_PATH}. "
            f"Run export_onnx.py first."
        )

    os.makedirs(RESULT_DIR, exist_ok=True)

    x, label = load_sample()

    # The sample was saved only when the trained model predicted it correctly.
    pred_class = label

    log_lines = []
    log_lines.append("Marabou MNIST 10-class Local Robustness Verification")
    log_lines.append("=" * 60)
    log_lines.append(f"ONNX model: {ONNX_MODEL_PATH}")
    log_lines.append(f"Sample label / predicted class: {pred_class}")
    log_lines.append(f"Epsilon: {epsilon}")
    log_lines.append("")

    print("\n".join(log_lines))

    robust = True
    total_runtime = 0.0
    found_counterexample = None
    found_target_class = None

    for target_class in range(10):
        if target_class == pred_class:
            continue

        print(f"Running query: target_class={target_class}")

        is_sat, runtime, counterexample = run_single_target_query(
            x=x,
            pred_class=pred_class,
            target_class=target_class,
            epsilon=epsilon
        )

        total_runtime += runtime

        if is_sat:
            robust = False
            found_counterexample = counterexample
            found_target_class = target_class

            line = (
                f"Target class {target_class}: SAT "
                f"(counterexample found), runtime={runtime:.4f}s"
            )
            print(line)
            log_lines.append(line)

            break

        else:
            line = (
                f"Target class {target_class}: UNSAT "
                f"(no counterexample), runtime={runtime:.4f}s"
            )
            print(line)
            log_lines.append(line)

    log_lines.append("")
    log_lines.append("=" * 60)

    if robust:
        result_line = (
            "Final result: UNSAT for all target classes. "
            "The model is locally robust for this sample and epsilon."
        )
        print(result_line)
        log_lines.append(result_line)

    else:
        result_line = (
            f"Final result: SAT. "
            f"A counterexample was found for class {pred_class} -> {found_target_class}."
        )
        print(result_line)
        log_lines.append(result_line)

        np.save(COUNTEREXAMPLE_PATH, found_counterexample)
        save_line = f"Counterexample saved to {COUNTEREXAMPLE_PATH}"
        print(save_line)
        log_lines.append(save_line)

    runtime_line = f"Total verification runtime: {total_runtime:.4f}s"
    print(runtime_line)
    log_lines.append(runtime_line)

    with open(LOG_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines))

    print(f"Verification log saved to {LOG_PATH}")


if __name__ == "__main__":
    # Start with a very small epsilon.
    # If it finishes quickly, try 0.005 or 0.01.
    run_verification(epsilon=0.001)