import argparse
import os

import matplotlib.pyplot as plt
import torch


def calibration_curve_from_probs(probs, labels, n_bins=15):
    confidences, predictions = torch.max(probs, dim=1)
    accuracies = predictions.eq(labels)

    bin_boundaries = torch.linspace(0, 1, n_bins + 1)

    bin_centers = []
    gaps = []
    counts = []

    for i in range(n_bins):
        lower = bin_boundaries[i]
        upper = bin_boundaries[i + 1]

        if i == n_bins - 1:
            in_bin = (confidences >= lower) & (confidences <= upper)
        else:
            in_bin = (confidences >= lower) & (confidences < upper)

        count = in_bin.sum().item()
        center = ((lower + upper) / 2).item()

        if count > 0:
            acc = accuracies[in_bin].float().mean().item()
            conf = confidences[in_bin].mean().item()
            gap = conf - acc
        else:
            gap = 0.0

        bin_centers.append(center)
        gaps.append(gap)
        counts.append(count)

    return bin_centers, gaps, counts


def load_prediction_file(path):
    obj = torch.load(path, map_location="cpu")

    if "probs" in obj:
        probs = obj["probs"]
    elif "logits" in obj:
        probs = torch.softmax(obj["logits"], dim=1)
    else:
        raise ValueError(f"File {path} must contain either 'probs' or 'logits'.")

    labels = obj["labels"]
    return probs, labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--save",
        type=str,
        default="outputs/figures/reliability_comparison_long.png",
        help="Where to save the reliability comparison figure",
    )
    args = parser.parse_args()

    methods = {
    "SGD": "outputs/metrics/preds_sgd_long.pt",
    "SWA": "outputs/metrics/resnet18_cifar10_swa_long_predictions.pt",
    "SWAG": "outputs/metrics/resnet18_cifar10_swag_long_predictions.pt",
}

    plt.figure(figsize=(7, 5))
    plt.axhline(0.0, linestyle="--", linewidth=1.5, label="Perfect calibration")

    for method, path in methods.items():
        if not os.path.exists(path):
            print(f"Skipping {method}, missing file: {path}")
            continue

        probs, labels = load_prediction_file(path)
        bin_centers, gaps, counts = calibration_curve_from_probs(probs, labels)

        plt.plot(bin_centers, gaps, marker="o", linewidth=2, label=method)

    plt.xlabel("Confidence")
    plt.ylabel("Confidence - Accuracy")
    plt.title("Reliability Diagram Comparison")
    plt.grid(True, alpha=0.3)
    plt.legend()

    os.makedirs(os.path.dirname(args.save), exist_ok=True)
    plt.savefig(args.save, dpi=200)
    plt.show()

    print(f"Saved reliability comparison to: {args.save}")


if __name__ == "__main__":
    main()