import json
import os
import matplotlib.pyplot as plt


METRICS_PATH = "outputs/metrics/resnet18_cifar10_sgd_metrics.json"
SAVE_PATH = "outputs/figures/resnet18_cifar10_sgd_learning_curves.png"


def main():
    with open(METRICS_PATH, "r", encoding="utf-8") as f:
        history = json.load(f)

    epochs = list(range(1, len(history["train_loss"]) + 1))

    plt.figure(figsize=(10, 4))

    # Loss subplot
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["train_loss"], marker="o", label="Train Loss")
    plt.plot(epochs, history["test_loss"], marker="o", label="Test Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curves")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Accuracy subplot
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history["train_acc"], marker="o", label="Train Accuracy")
    plt.plot(epochs, history["test_acc"], marker="o", label="Test Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy Curves")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    os.makedirs("outputs/figures", exist_ok=True)
    plt.savefig(SAVE_PATH, dpi=200)
    plt.show()

    print(f"Saved figure to: {SAVE_PATH}")


if __name__ == "__main__":
    main()