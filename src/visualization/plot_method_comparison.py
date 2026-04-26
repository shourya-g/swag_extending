import os

import matplotlib.pyplot as plt
import pandas as pd


CSV_PATH = "outputs/metrics/sgd_swa_swag_comparison.csv"
SAVE_PATH = "outputs/figures/sgd_swa_swag_comparison.png"


def main():
    df = pd.read_csv(CSV_PATH)

    methods = df["method"]

    plt.figure(figsize=(10, 4))

    plt.subplot(1, 3, 1)
    plt.bar(methods, df["accuracy"])
    plt.title("Accuracy")
    plt.ylabel("Higher is better")
    plt.ylim(0, 1)

    plt.subplot(1, 3, 2)
    plt.bar(methods, df["nll"])
    plt.title("NLL")
    plt.ylabel("Lower is better")

    plt.subplot(1, 3, 3)
    plt.bar(methods, df["ece"])
    plt.title("ECE")
    plt.ylabel("Lower is better")

    plt.suptitle("SGD vs SWA vs SWAG")
    plt.tight_layout()

    os.makedirs("outputs/figures", exist_ok=True)
    plt.savefig(SAVE_PATH, dpi=200)
    plt.show()

    print(f"Saved plot to: {SAVE_PATH}")


if __name__ == "__main__":
    main()