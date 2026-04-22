import json
import os

import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from src.utils.config import load_config
from src.models.model_factory import get_model
from src.data_loader import get_cifar10_loaders
from src.evaluation.metrics import get_calibration_bins


CHECKPOINT_PATH = "outputs/checkpoints/resnet18_cifar10_sgd.pt"
SAVE_PATH = "outputs/figures/resnet18_cifar10_sgd_reliability.png"


def collect_logits_and_labels(model, loader, device):
    model.eval()
    all_logits = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            all_logits.append(logits)
            all_labels.append(labels)

    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    return all_logits, all_labels


def main():
    config = load_config("configs/baseline.yaml")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    _, test_loader = get_cifar10_loaders(
        batch_size=config["dataset"]["batch_size"],
        num_workers=config["dataset"]["num_workers"],
    )

    model = get_model(
        name=config["model"]["name"],
        num_classes=config["model"]["num_classes"],
    ).to(device)

    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    print(f"Loaded checkpoint from: {CHECKPOINT_PATH}")

    logits, labels = collect_logits_and_labels(model, test_loader, device)

    bin_centers, bin_accuracies, bin_confidences, bin_counts = get_calibration_bins(
        logits, labels, n_bins=15
    )

    diff = [c - a for c, a in zip(bin_confidences, bin_accuracies)]

    plt.figure(figsize=(7, 5))
    plt.axhline(0.0, linestyle="--", linewidth=1.5, label="Perfect calibration")
    plt.plot(bin_centers, diff, marker="o", linewidth=2)

    plt.xlabel("Confidence")
    plt.ylabel("Confidence - Accuracy")
    plt.title("Reliability Diagram")
    plt.grid(True, alpha=0.3)
    plt.legend()

    os.makedirs("outputs/figures", exist_ok=True)
    plt.savefig(SAVE_PATH, dpi=200)
    plt.show()

    print(f"Saved reliability diagram to: {SAVE_PATH}")


if __name__ == "__main__":
    main()