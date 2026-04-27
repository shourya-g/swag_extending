import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        type=str,
        default="outputs/metrics/ood_entropy_data.pt",
    )
    parser.add_argument(
        "--summary",
        type=str,
        default="outputs/metrics/ood_entropy_summary.csv",
    )
    parser.add_argument(
        "--save-hist",
        type=str,
        default="outputs/figures/ood_entropy_histograms.png",
    )
    parser.add_argument(
        "--save-summary",
        type=str,
        default="outputs/figures/ood_entropy_summary.png",
    )
    args = parser.parse_args()

    obj = torch.load(args.data, map_location="cpu")
    methods = obj["methods"]
    id_classes = obj["id_classes"]
    ood_classes = obj["ood_classes"]

    os.makedirs("outputs/figures", exist_ok=True)

    # Entropy histograms
    plt.figure(figsize=(14, 4))

    for idx, (method, data) in enumerate(methods.items(), start=1):
        labels = data["labels"]
        entropy = data["entropy"]

        id_mask = torch.isin(labels, torch.tensor(id_classes))
        ood_mask = torch.isin(labels, torch.tensor(ood_classes))

        id_entropy = entropy[id_mask].numpy()
        ood_entropy = entropy[ood_mask].numpy()

        plt.subplot(1, 3, idx)
        plt.hist(id_entropy, bins=30, alpha=0.6, density=True, label="ID")
        plt.hist(ood_entropy, bins=30, alpha=0.6, density=True, label="OOD")
        plt.title(f"{method}: Predictive Entropy")
        plt.xlabel("Entropy")
        plt.ylabel("Density")
        plt.legend()
        plt.grid(True, alpha=0.3)

    plt.suptitle("OOD Detection via Predictive Entropy")
    plt.tight_layout()
    plt.savefig(args.save_hist, dpi=200)
    plt.show()

    print(f"Saved entropy histograms to: {args.save_hist}")

    # Summary bar plots
    df = pd.read_csv(args.summary)

    plt.figure(figsize=(11, 4))

    plt.subplot(1, 3, 1)
    plt.bar(df["method"], df["entropy_gap_ood_minus_id"])
    plt.title("Entropy Gap")
    plt.ylabel("OOD mean - ID mean")
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 3, 2)
    plt.bar(df["method"], df["entropy_auroc"])
    plt.title("OOD AUROC")
    plt.ylabel("Higher is better")
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 3, 3)
    plt.bar(df["method"], df["sym_kl_binned_entropy"])
    plt.title("Symmetric KL")
    plt.ylabel("Higher separation")
    plt.grid(True, alpha=0.3)

    plt.suptitle("OOD Entropy Separation Summary")
    plt.tight_layout()
    plt.savefig(args.save_summary, dpi=200)
    plt.show()

    print(f"Saved entropy summary plot to: {args.save_summary}")


if __name__ == "__main__":
    main()