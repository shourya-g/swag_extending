import argparse
import json
import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from src.utils.config import load_config
from src.models.model_factory import get_model
from src.data_loader import get_cifar10_loaders
from src.evaluation.metrics import compute_ece


def ensure_dirs(config):
    os.makedirs(config["output"]["checkpoint_dir"], exist_ok=True)
    os.makedirs(config["output"]["metrics_dir"], exist_ok=True)
    os.makedirs(config["output"]["figures_dir"], exist_ok=True)


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)

        logits = model(images)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_samples += images.size(0)

    return total_loss / total_samples, total_correct / total_samples


def evaluate(model, loader, device):
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

    logits = torch.cat(all_logits, dim=0)
    labels = torch.cat(all_labels, dim=0)

    loss = F.cross_entropy(logits, labels).item()
    preds = logits.argmax(dim=1)
    acc = (preds == labels).float().mean().item()
    ece = compute_ece(logits, labels)

    return loss, acc, loss, ece


def build_optimizer(model, config):
    training_cfg = config["training"]
    optimizer_name = training_cfg.get("optimizer", "sgd").lower()

    lr = training_cfg["lr"]
    weight_decay = training_cfg.get("weight_decay", 0.0)

    if optimizer_name == "sgd":
        return optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=training_cfg.get("momentum", 0.9),
            weight_decay=weight_decay,
        )

    if optimizer_name == "adamw":
        return optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=tuple(training_cfg.get("betas", [0.9, 0.999])),
        )

    raise ValueError(f"Unknown optimizer: {optimizer_name}")


def save_metrics(metrics, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics to: {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs/baseline.yaml",
        help="Path to config file",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    ensure_dirs(config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_classes = config["dataset"].get("train_classes", None)
    test_classes = config["dataset"].get("test_classes", train_classes)

    train_loader, test_loader = get_cifar10_loaders(
        batch_size=config["dataset"]["batch_size"],
        num_workers=config["dataset"]["num_workers"],
        train_classes=train_classes,
        test_classes=test_classes,
        image_size=config["dataset"].get("image_size", 32),
        normalization=config["dataset"].get("normalization", "cifar10"),
        augment=config["dataset"].get("augment", True),
    )

    model = get_model(
        name=config["model"]["name"],
        num_classes=config["model"]["num_classes"],
        pretrained=config["model"].get("pretrained", False),
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(model, config)

    epochs = config["training"]["epochs"]

    history = {
        "experiment_name": config["experiment_name"],
        "device": str(device),
        "epochs": epochs,
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": [],
        "test_nll": [],
        "test_ece": [],
    }

    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )

        test_loss, test_acc, test_nll, test_ece = evaluate(
            model=model,
            loader=test_loader,
            device=device,
        )

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)
        history["test_nll"].append(test_nll)
        history["test_ece"].append(test_ece)

        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f} | "
            f"Test NLL: {test_nll:.4f} | Test ECE: {test_ece:.4f}"
        )

    checkpoint_path = os.path.join(
        config["output"]["checkpoint_dir"],
        f'{config["experiment_name"]}.pt',
    )

    torch.save(model.state_dict(), checkpoint_path)
    print(f"Saved checkpoint to: {checkpoint_path}")

    metrics_path = os.path.join(
        config["output"]["metrics_dir"],
        f'{config["experiment_name"]}_metrics.json',
    )

    save_metrics(history, metrics_path)


if __name__ == "__main__":
    main()