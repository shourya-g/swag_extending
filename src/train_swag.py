import json
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from src.utils.config import load_config
from src.models.model_factory import get_model
from src.data_loader import get_cifar10_loaders
from src.evaluation.metrics import compute_ece
from src.swag.swag_utils import SWAGPosterior
from src.swag.bn_update import update_bn


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        total_correct += (preds == labels).sum().item()
        total_samples += images.size(0)

    return total_loss / total_samples, total_correct / total_samples


def collect_logits_and_labels(model, loader, device):
    model.eval()
    all_logits = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            all_logits.append(logits)
            all_labels.append(labels)

    return torch.cat(all_logits, dim=0), torch.cat(all_labels, dim=0)


def evaluate_single_model(model, loader, device):
    logits, labels = collect_logits_and_labels(model, loader, device)

    nll = F.cross_entropy(logits, labels).item()
    preds = logits.argmax(dim=1)
    acc = (preds == labels).float().mean().item()
    ece = compute_ece(logits, labels)

    return acc, nll, ece


def evaluate_swag_bma(model, swag_posterior, train_loader, test_loader, device, num_samples):
    """
    Bayesian model averaging using sampled SWAG weights.

    For each sample:
    1. sample parameter vector from SWAG posterior
    2. load it into model
    3. update BN stats for sampled model
    4. collect softmax probabilities
    5. average probabilities across samples
    """
    all_probs = []
    labels_ref = None

    for sample_idx in range(num_samples):
        print(f"Evaluating SWAG sample {sample_idx + 1}/{num_samples}")

        sampled_vector = swag_posterior.sample()
        swag_posterior.set_weights(model, sampled_vector, device)

        update_bn(train_loader, model, device)

        logits, labels = collect_logits_and_labels(model, test_loader, device)
        probs = torch.softmax(logits, dim=1)

        all_probs.append(probs)

        if labels_ref is None:
            labels_ref = labels

    mean_probs = torch.stack(all_probs, dim=0).mean(dim=0)

    preds = mean_probs.argmax(dim=1)
    acc = (preds == labels_ref).float().mean().item()

    correct_probs = mean_probs[torch.arange(labels_ref.size(0), device=device), labels_ref]
    nll = -torch.log(correct_probs + 1e-12).mean().item()

    ece = compute_ece(torch.log(mean_probs + 1e-12), labels_ref)

    return acc, nll, ece, mean_probs.detach().cpu(), labels_ref.detach().cpu()


def ensure_dirs(config):
    os.makedirs(config["output"]["checkpoint_dir"], exist_ok=True)
    os.makedirs(config["output"]["metrics_dir"], exist_ok=True)
    os.makedirs(config["output"]["figures_dir"], exist_ok=True)


def save_metrics(metrics, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics to: {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
    "--config",
    type=str,
    default="configs/swag.yaml",
    help="Path to config file",
    )
    args = parser.parse_args()

    config = load_config(args.config)
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

    swag_posterior = SWAGPosterior(
        max_rank=config["swag"]["max_rank"],
        var_clamp=config["swag"]["var_clamp"],
    )

    epochs = config["training"]["epochs"]
    swag_start = config["swa"]["start_epoch"]
    save_freq = config["swa"]["save_freq"]

    history = {
        "experiment_name": config["experiment_name"],
        "device": str(device),
        "epochs": epochs,
        "train_loss": [],
        "train_acc": [],
        "single_model_test_acc": [],
        "single_model_test_nll": [],
        "single_model_test_ece": [],
        "num_swag_snapshots": 0,
    }

    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )

        test_acc, test_nll, test_ece = evaluate_single_model(
            model, test_loader, device
        )

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["single_model_test_acc"].append(test_acc)
        history["single_model_test_nll"].append(test_nll)
        history["single_model_test_ece"].append(test_ece)

        if epoch >= swag_start and (epoch - swag_start) % save_freq == 0:
            swag_posterior.collect_model(model)
            print(f"Collected SWAG snapshot #{swag_posterior.n_models}")

        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Single Test Acc: {test_acc:.4f} | "
            f"Single Test NLL: {test_nll:.4f} | "
            f"Single Test ECE: {test_ece:.4f}"
        )

    history["num_swag_snapshots"] = swag_posterior.n_models

    print("\nEvaluating SWAG with Bayesian Model Averaging...")
    swag_acc, swag_nll, swag_ece, swag_probs, swag_labels = evaluate_swag_bma(
    model=model,
    swag_posterior=swag_posterior,
    train_loader=train_loader,
    test_loader=test_loader,
    device=device,
    num_samples=config["swag"]["num_samples"],
    )

    history["final_swag_test_acc"] = swag_acc
    history["final_swag_test_nll"] = swag_nll
    history["final_swag_test_ece"] = swag_ece

    print("\nFinal SWAG evaluation:")
    print(
        f"SWAG Test Acc: {swag_acc:.4f} | "
        f"SWAG Test NLL: {swag_nll:.4f} | "
        f"SWAG Test ECE: {swag_ece:.4f}"
    )

    metrics_path = os.path.join(
        config["output"]["metrics_dir"],
        f'{config["experiment_name"]}_metrics.json'
    )

    
    preds_path = os.path.join(
    config["output"]["metrics_dir"],
    f'{config["experiment_name"]}_predictions.pt'
    )

    torch.save(
        {
            "probs": swag_probs,
            "labels": swag_labels,
        },
        preds_path,
    )
    print("\nEvaluating SWA mean from SWAG posterior...")

    # Set model weights to SWA mean
    swag_posterior.set_weights(model, swag_posterior.mean, device)

    # Recompute BatchNorm stats for the SWA mean model
    update_bn(train_loader, model, device)

    # Collect predictions for SWA mean model
    swa_logits, swa_labels = collect_logits_and_labels(model, test_loader, device)
    swa_probs = torch.softmax(swa_logits, dim=1)

    swa_preds = swa_probs.argmax(dim=1)
    swa_acc = (swa_preds == swa_labels).float().mean().item()

    swa_correct_probs = swa_probs[torch.arange(swa_labels.size(0), device=device), swa_labels]
    swa_nll = -torch.log(swa_correct_probs + 1e-12).mean().item()

    swa_ece = compute_ece(torch.log(swa_probs + 1e-12), swa_labels)

    history["final_swa_test_acc"] = swa_acc
    history["final_swa_test_nll"] = swa_nll
    history["final_swa_test_ece"] = swa_ece

    print("\nFinal SWA evaluation from SWAG trajectory:")
    print(
        f"SWA Test Acc: {swa_acc:.4f} | "
        f"SWA Test NLL: {swa_nll:.4f} | "
        f"SWA Test ECE: {swa_ece:.4f}"
    )

    # Save SWA checkpoint with updated BatchNorm stats
    swa_checkpoint_path = os.path.join(
        config["output"]["checkpoint_dir"],
        f'{config["experiment_name"].replace("swag", "swa")}.pt'
    )

    torch.save(model.state_dict(), swa_checkpoint_path)
    print(f"Saved SWA checkpoint to: {swa_checkpoint_path}")

    # Save SWA predictions
    swa_preds_path = os.path.join(
        config["output"]["metrics_dir"],
        f'{config["experiment_name"].replace("swag", "swa")}_predictions.pt'
    )

    torch.save(
        {
            "probs": swa_probs.detach().cpu(),
            "labels": swa_labels.detach().cpu(),
        },
        swa_preds_path,
    )

    print(f"Saved SWA predictions to: {swa_preds_path}")
    
    
    save_metrics(history, metrics_path)
    print(f"Saved SWAG predictions to: {preds_path}")
if __name__ == "__main__":
    main()