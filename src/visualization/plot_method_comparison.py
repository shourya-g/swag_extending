import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        type=str,
        default="outputs/metrics/sgd_swa_swag_comparison_long.csv",
        help="Path to comparison CSV",
    )
    parser.add_argument(
        "--save",
        type=str,
        default="outputs/figures/sgd_swa_swag_comparison_long.png",
        help="Path to save figure",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    methods = df["method"]

    plt.figure(figsize=(11, 4))

    plt.subplot(1, 3, 1)
    plt.bar(methods, df["accuracy"])
    plt.title("Accuracy")
    plt.ylabel("Higher is better")
    plt.ylim(0, 1)

    plt.subplot(1, 3, 2)
    plt.bar(methods, df["nll"])
    plt.title("Negative Log Likelihood")
    plt.ylabel("Lower is better")

    plt.subplot(1, 3, 3)
    plt.bar(methods, df["ece"])
    plt.title("Expected Calibration Error")
    plt.ylabel("Lower is better")

    plt.suptitle("SGD vs SWA vs SWAG")
    plt.tight_layout()

    os.makedirs(os.path.dirname(args.save), exist_ok=True)
    plt.savefig(args.save, dpi=200)
    plt.show()

    print(f"Saved plot to: {args.save}")


if __name__ == "__main__":
    main()