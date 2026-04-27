import argparse
import csv
import os

import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from src.utils.config import load_config
from src.models.model_factory import get_model
from src.data_loader import get_cifar10_loaders, get_cifar10_full_test_loader
from src.swag.swag_utils import SWAGPosterior
from src.swag.bn_update import update_bn


def predictive_entropy(probs: torch.Tensor) -> torch.Tensor:
    return -(probs * torch.log(probs + 1e-12)).sum(dim=1)


def collect_probs_and_labels(model, loader, device):
    model.eval()
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            logits = model(images)
            probs = torch.softmax(logits, dim=1)

            all_probs.append(probs.cpu())
            all_labels.append(labels.cpu())

    return torch.cat(all_probs, dim=0), torch.cat(all_labels, dim=0)


def evaluate_checkpoint_entropy(config, checkpoint_path, full_test_loader, device):
    model = get_model(
        name=config["model"]["name"],
        num_classes=config["model"]["num_classes"],
    ).to(device)

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    probs, labels = collect_probs_and_labels(model, full_test_loader, device)
    entropies = predictive_entropy(probs)

    return probs, labels, entropies


def evaluate_swag_entropy(config, posterior_path, full_test_loader, train_loader, device):
    model = get_model(
        name=config["model"]["name"],
        num_classes=config["model"]["num_classes"],
    ).to(device)

    posterior = SWAGPosterior(
        max_rank=config["swag"]["max_rank"],
        var_clamp=config["swag"]["var_clamp"],
    )

    state = torch.load(posterior_path, map_location="cpu")
    posterior.load_state_dict(state)

    all_probs = []
    labels_ref = None

    num_samples = config["swag"]["num_samples"]

    for sample_idx in range(num_samples):
        print(f"SWAG OOD entropy sample {sample_idx + 1}/{num_samples}")

        sampled_vector = posterior.sample()
        posterior.set_weights(model, sampled_vector, device)

        update_bn(train_loader, model, device)

        probs, labels = collect_probs_and_labels(model, full_test_loader, device)
        all_probs.append(probs)

        if labels_ref is None:
            labels_ref = labels

    mean_probs = torch.stack(all_probs, dim=0).mean(dim=0)
    entropies = predictive_entropy(mean_probs)

    return mean_probs, labels_ref, entropies


def summarize_entropy(method, labels, entropies, id_classes, ood_classes, n_bins=30):
    id_mask = torch.isin(labels, torch.tensor(id_classes))
    ood_mask = torch.isin(labels, torch.tensor(ood_classes))

    id_entropy = entropies[id_mask]
    ood_entropy = entropies[ood_mask]

    id_mean = id_entropy.mean().item()
    ood_mean = ood_entropy.mean().item()
    gap = ood_mean - id_mean

    # AUROC: label OOD as 1, ID as 0, entropy as score
    y_true = torch.cat([
        torch.zeros(id_entropy.numel()),
        torch.ones(ood_entropy.numel()),
    ]).numpy()

    y_score = torch.cat([id_entropy, ood_entropy]).numpy()

    auroc = roc_auc_score(y_true, y_score)

    # Symmetric KL over binned entropy distributions
    max_entropy = torch.log(torch.tensor(float(entropies.max().item() + 1e-8)))
    # Better fixed upper bound: log(num_classes)
    num_classes = int(entropies.new_tensor([]).numel())  # unused fallback

    max_val = entropies.max().item()
    bins = torch.linspace(0.0, max_val + 1e-6, n_bins + 1)

    id_hist = torch.histc(id_entropy, bins=n_bins, min=0.0, max=max_val + 1e-6)
    ood_hist = torch.histc(ood_entropy, bins=n_bins, min=0.0, max=max_val + 1e-6)

    eps = 1e-8
    p = (id_hist + eps) / (id_hist.sum() + eps * n_bins)
    q = (ood_hist + eps) / (ood_hist.sum() + eps * n_bins)

    kl_pq = (p * torch.log(p / q)).sum().item()
    kl_qp = (q * torch.log(q / p)).sum().item()
    sym_kl = kl_pq + kl_qp

    return {
        "method": method,
        "id_entropy_mean": id_mean,
        "ood_entropy_mean": ood_mean,
        "entropy_gap_ood_minus_id": gap,
        "entropy_auroc": auroc,
        "sym_kl_binned_entropy": sym_kl,
        "num_id": int(id_mask.sum().item()),
        "num_ood": int(ood_mask.sum().item()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs/swag_ood.yaml",
    )
    parser.add_argument(
        "--save-prefix",
        type=str,
        default="outputs/metrics/ood_entropy",
    )
    args = parser.parse_args()

    config = load_config(args.config)

    os.makedirs("outputs/metrics", exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    id_classes = config["ood"]["id_classes"]
    ood_classes = config["ood"]["ood_classes"]

    train_classes = config["dataset"]["train_classes"]
    test_classes = config["dataset"]["test_classes"]

    train_loader, _ = get_cifar10_loaders(
        batch_size=config["dataset"]["batch_size"],
        num_workers=config["dataset"]["num_workers"],
        train_classes=train_classes,
        test_classes=test_classes,
    )

    full_test_loader = get_cifar10_full_test_loader(
        batch_size=config["dataset"]["batch_size"],
        num_workers=config["dataset"]["num_workers"],
    )

    experiment_name = config["experiment_name"]

    sgd_checkpoint = config["input"]["baseline_checkpoint"]
    swa_checkpoint = os.path.join(
        config["output"]["checkpoint_dir"],
        f'{experiment_name.replace("swag", "swa")}.pt'
    )
    swag_posterior = os.path.join(
        config["output"]["checkpoint_dir"],
        f"{experiment_name}_posterior.pt"
    )

    methods_data = {}

    print("\nEvaluating SGD entropy...")
    sgd_probs, labels, sgd_entropy = evaluate_checkpoint_entropy(
        config, sgd_checkpoint, full_test_loader, device
    )
    methods_data["SGD"] = {
        "probs": sgd_probs,
        "labels": labels,
        "entropy": sgd_entropy,
    }

    print("\nEvaluating SWA entropy...")
    swa_probs, labels, swa_entropy = evaluate_checkpoint_entropy(
        config, swa_checkpoint, full_test_loader, device
    )
    methods_data["SWA"] = {
        "probs": swa_probs,
        "labels": labels,
        "entropy": swa_entropy,
    }

    print("\nEvaluating SWAG entropy...")
    swag_probs, labels, swag_entropy = evaluate_swag_entropy(
        config, swag_posterior, full_test_loader, train_loader, device
    )
    methods_data["SWAG"] = {
        "probs": swag_probs,
        "labels": labels,
        "entropy": swag_entropy,
    }

    summaries = []
    for method, data in methods_data.items():
        summary = summarize_entropy(
            method=method,
            labels=data["labels"],
            entropies=data["entropy"],
            id_classes=id_classes,
            ood_classes=ood_classes,
        )
        summaries.append(summary)

    pt_path = f"{args.save_prefix}_data.pt"
    csv_path = f"{args.save_prefix}_summary.csv"

    torch.save(
        {
            "methods": methods_data,
            "id_classes": id_classes,
            "ood_classes": ood_classes,
            "summaries": summaries,
        },
        pt_path,
    )

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summaries[0].keys()))
        writer.writeheader()
        writer.writerows(summaries)

    print("\nOOD entropy summary:")
    for row in summaries:
        print(row)

    print(f"\nSaved OOD entropy data to: {pt_path}")
    print(f"Saved OOD entropy summary to: {csv_path}")


if __name__ == "__main__":
    main()