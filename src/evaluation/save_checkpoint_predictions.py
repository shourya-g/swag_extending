import argparse
import os

import torch

from src.utils.config import load_config
from src.models.model_factory import get_model
from src.data_loader import get_cifar10_loaders


def collect_logits_and_labels(model, loader, device):
    model.eval()
    all_logits = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)

            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())

    return torch.cat(all_logits, dim=0), torch.cat(all_labels, dim=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--save", type=str, required=True)
    args = parser.parse_args()

    config = load_config(args.config)

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

    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    print(f"Loaded checkpoint from: {args.checkpoint}")

    logits, labels = collect_logits_and_labels(model, test_loader, device)

    os.makedirs(os.path.dirname(args.save), exist_ok=True)
    torch.save(
        {
            "logits": logits,
            "labels": labels,
        },
        args.save,
    )

    print(f"Saved predictions to: {args.save}")


if __name__ == "__main__":
    main()