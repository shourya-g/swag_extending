import json
import os

import torch
import torch.nn as nn
import torch.optim as optim

from src.utils.config import load_config
from src.models.model_factory import get_model
from src.data_loader import get_cifar10_loaders


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        total_correct += (preds == labels).sum().item()
        total_samples += images.size(0)

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    return avg_loss, accuracy


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            total_correct += (preds == labels).sum().item()
            total_samples += images.size(0)

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    return avg_loss, accuracy


def ensure_dirs(config):
    os.makedirs(config["output"]["checkpoint_dir"], exist_ok=True)
    os.makedirs(config["output"]["metrics_dir"], exist_ok=True)
    os.makedirs(config["output"]["figures_dir"], exist_ok=True)


def save_checkpoint(model, config):
    checkpoint_path = os.path.join(
        config["output"]["checkpoint_dir"],
        f'{config["experiment_name"]}.pt'
    )
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Saved checkpoint to: {checkpoint_path}")


def save_metrics(metrics, config):
    metrics_path = os.path.join(
        config["output"]["metrics_dir"],
        f'{config["experiment_name"]}_metrics.json'
    )
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics to: {metrics_path}")


def main():
    config = load_config("configs/baseline.yaml")
    ensure_dirs(config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, test_loader = get_cifar10_loaders(
        batch_size=config["dataset"]["batch_size"],
        num_workers=config["dataset"]["num_workers"],
    )

    model = get_model(
        name=config["model"]["name"],
        num_classes=config["model"]["num_classes"],
    ).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.SGD(
        model.parameters(),
        lr=config["training"]["lr"],
        momentum=config["training"]["momentum"],
        weight_decay=config["training"]["weight_decay"],
    )

    epochs = config["training"]["epochs"]

    history = {
        "experiment_name": config["experiment_name"],
        "device": str(device),
        "epochs": epochs,
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": [],
    }

    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)

        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}"
        )

    save_checkpoint(model, config)
    save_metrics(history, config)


if __name__ == "__main__":
    main()