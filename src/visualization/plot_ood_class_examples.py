import os
import argparse

import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as transforms


CIFAR10_CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


def get_first_examples_per_class(dataset, class_ids):
    """
    Returns one example image for each requested CIFAR-10 class.
    """
    examples = {}

    for image, label in dataset:
        if label in class_ids and label not in examples:
            examples[label] = image

        if len(examples) == len(class_ids):
            break

    return examples


def plot_seen_vs_unseen(save_path):
    """
    Creates a figure showing:
    - classes 0-4: seen during training
    - classes 5-9: unseen/OOD during training
    """

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    dataset = torchvision.datasets.CIFAR10(
        root="data/raw",
        train=True,
        download=True,
        transform=transform,
    )

    seen_classes = [0, 1, 2, 3, 4]
    unseen_classes = [5, 6, 7, 8, 9]

    seen_examples = get_first_examples_per_class(dataset, seen_classes)
    unseen_examples = get_first_examples_per_class(dataset, unseen_classes)

    fig, axes = plt.subplots(2, 5, figsize=(13, 5))

    fig.suptitle(
        "Five-Class OOD Setup: Classes Seen During Training vs Unseen Classes",
        fontsize=16,
        fontweight="bold",
    )

    for col, class_id in enumerate(seen_classes):
        image = seen_examples[class_id]
        image = image.permute(1, 2, 0)

        ax = axes[0, col]
        ax.imshow(image)
        ax.set_title(f"{class_id}: {CIFAR10_CLASSES[class_id]}", fontsize=11)
        ax.axis("off")

    for col, class_id in enumerate(unseen_classes):
        image = unseen_examples[class_id]
        image = image.permute(1, 2, 0)

        ax = axes[1, col]
        ax.imshow(image)
        ax.set_title(f"{class_id}: {CIFAR10_CLASSES[class_id]}", fontsize=11)
        ax.axis("off")

    axes[0, 0].set_ylabel(
        "Seen / ID\nUsed for training",
        fontsize=12,
        fontweight="bold",
        rotation=0,
        labelpad=55,
        va="center",
    )

    axes[1, 0].set_ylabel(
        "Unseen / OOD\nNot used for training",
        fontsize=12,
        fontweight="bold",
        rotation=0,
        labelpad=55,
        va="center",
    )

    plt.tight_layout(rect=[0, 0, 1, 0.90])

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=250, bbox_inches="tight")
    plt.show()

    print(f"Saved OOD class example figure to: {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--save",
        type=str,
        default="outputs/figures/ood_seen_unseen_class_examples.png",
        help="Path to save the class example figure.",
    )
    args = parser.parse_args()

    plot_seen_vs_unseen(args.save)


if __name__ == "__main__":
    main()