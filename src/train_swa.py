import json
import os

import torch
import torch.nn as nn
import torch.optim as optim

from src.utils.config import load_config
from src.models.model_factory import get_model
from src.data_loader import get_cifar10_loaders
from src.evaluation.metrics import compute_nll, compute_ece
from src.swag.swag_utils import initialize_swa_model, update_swa_model
from src.swag.bn_update import update_bn

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

    all_logits = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            total_correct += (preds == labels).sum().item()
            total_samples += images.size(0)

            all_logits.append(outputs)
            all_labels.append(labels)

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    return avg_loss, accuracy, all_logits, all_labels


def ensure_dirs(config):
    os.makedirs(config["output"]["checkpoint_dir"], exist_ok=True)
    os.makedirs(config["output"]["metrics_dir"], exist_ok=True)
    os.makedirs(config["output"]["figures_dir"], exist_ok=True)


def save_checkpoint(model, path):
    torch.save(model.state_dict(), path)
    print(f"Saved checkpoint to: {path}")


def save_metrics(metrics, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics to: {path}")


def main():
    config = load_config("configs/swa.yaml")
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

    baseline_ckpt = config["input"]["baseline_checkpoint"]
    model.load_state_dict(torch.load(baseline_ckpt, map_location=device))
    print(f"Loaded baseline checkpoint from: {baseline_ckpt}")

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.SGD(
        model.parameters(),
        lr=config["training"]["lr"],
        momentum=config["training"]["momentum"],
        weight_decay=config["training"]["weight_decay"],
    )

    swa_model = initialize_swa_model(model).to(device)
    swa_start = config["swa"]["start_epoch"]
    save_freq = config["swa"]["save_freq"]
    num_swa_models = 0
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
        "num_swa_snapshots": 0,
    }

    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        test_loss, test_acc, test_logits, test_labels = evaluate(model, test_loader, criterion, device)

        test_nll = compute_nll(test_logits, test_labels)
        test_ece = compute_ece(test_logits, test_labels)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)
        history["test_nll"].append(test_nll)
        history["test_ece"].append(test_ece)

        if epoch >= swa_start and (epoch - swa_start) % save_freq == 0:
            update_swa_model(swa_model, model, num_swa_models)
            num_swa_models += 1
            print(f"Updated SWA model with snapshot #{num_swa_models}")

        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f} | "
            f"Test NLL: {test_nll:.4f} | Test ECE: {test_ece:.4f}"
        )

    history["num_swa_snapshots"] = num_swa_models

    # Update BatchNorm statistics before final SWA evaluation
    print("\nUpdating BatchNorm statistics for SWA model.")
    update_bn(train_loader, swa_model, device)

    # Final evaluation of SWA model
    swa_test_loss, swa_test_acc, swa_test_logits, swa_test_labels = evaluate(
        swa_model, test_loader, criterion, device
    )
    swa_test_nll = compute_nll(swa_test_logits, swa_test_labels)
    swa_test_ece = compute_ece(swa_test_logits, swa_test_labels)

    history["final_swa_test_loss"] = swa_test_loss
    history["final_swa_test_acc"] = swa_test_acc
    history["final_swa_test_nll"] = swa_test_nll
    history["final_swa_test_ece"] = swa_test_ece

    print("\nFinal SWA evaluation:")
    print(
        f"SWA Test Loss: {swa_test_loss:.4f} | "
        f"SWA Test Acc: {swa_test_acc:.4f} | "
        f"SWA Test NLL: {swa_test_nll:.4f} | "
        f"SWA Test ECE: {swa_test_ece:.4f}"
    )

    checkpoint_path = os.path.join(
        config["output"]["checkpoint_dir"],
        f'{config["experiment_name"]}.pt'
    )
    metrics_path = os.path.join(
        config["output"]["metrics_dir"],
        f'{config["experiment_name"]}_metrics.json'
    )

    save_checkpoint(swa_model, checkpoint_path)
    save_metrics(history, metrics_path)


if __name__ == "__main__":
    main()