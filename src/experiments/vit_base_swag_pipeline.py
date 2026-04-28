import argparse
import csv
import json
import os
import random
import time
from contextlib import contextmanager

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from src.models.model_factory import get_model
from src.data_loader import get_cifar10_loaders, get_cifar10_full_test_loader
from src.evaluation.metrics import compute_ece


# ============================================================
# Global timing
# ============================================================

TIMING_ROWS = []


def sync_cuda():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


@contextmanager
def timed_stage(stage_name: str):
    sync_cuda()
    start = time.perf_counter()

    print("\n" + "=" * 90)
    print(f"START: {stage_name}")
    print("=" * 90)

    try:
        yield
    finally:
        sync_cuda()
        end = time.perf_counter()
        elapsed = end - start

        TIMING_ROWS.append(
            {
                "stage": stage_name,
                "seconds": elapsed,
                "minutes": elapsed / 60.0,
            }
        )

        print("\n" + "=" * 90)
        print(f"END: {stage_name}")
        print(f"Time: {elapsed:.2f} seconds = {elapsed / 60.0:.2f} minutes")
        print("=" * 90)


# ============================================================
# Basic utilities
# ============================================================

def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dirs():
    for p in [
        "outputs/checkpoints",
        "outputs/metrics",
        "outputs/figures",
        "outputs/logs",
    ]:
        os.makedirs(p, exist_ok=True)


def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)

    print(f"Saved JSON: {path}")


def save_csv(rows, path):
    if not rows:
        print(f"No rows to save for: {path}")
        return

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved CSV: {path}")


def bytes_to_mb(num_bytes):
    return num_bytes / (1024 ** 2)


def swag_memory_mb(num_params, max_rank=20, bytes_per_float=4):
    """
    SWAG stores approximately:
        mean vector         = d
        second moment       = d
        low-rank deviations = Kd

    Total = (K + 2)d floats.
    """
    return bytes_to_mb((max_rank + 2) * num_params * bytes_per_float)


def adamw_memory_mb(num_params, bytes_per_float=4):
    """
    Approximate AdamW training memory:
        params      = d
        gradients   = d
        first mom   = d
        second mom  = d

    Total ~ 4d floats.
    Does not include activations or CUDA overhead.
    """
    return bytes_to_mb(4 * num_params * bytes_per_float)


def count_trainable(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_total(model):
    return sum(p.numel() for p in model.parameters())


# ============================================================
# LoRA wrapper for timm ViT qkv layers
# ============================================================

class LoRAQKVLinear(nn.Module):
    """
    Wrap timm ViT attention qkv Linear.

    Original qkv:
        x -> [q, k, v]

    LoRA update:
        q = q + B_q A_q x
        v = v + B_v A_v x

    Base qkv weights are frozen.
    Only LoRA A/B matrices are trainable.
    """

    def __init__(
        self,
        base_qkv: nn.Linear,
        rank: int = 8,
        alpha: int = 16,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.base_qkv = base_qkv
        for p in self.base_qkv.parameters():
            p.requires_grad = False

        self.in_features = base_qkv.in_features
        self.out_features = base_qkv.out_features

        if self.out_features % 3 != 0:
            raise ValueError(
                f"Expected qkv output dimension divisible by 3, got {self.out_features}"
            )

        self.hidden_dim = self.out_features // 3

        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.dropout = nn.Dropout(dropout)

        self.lora_q_A = nn.Linear(self.in_features, rank, bias=False)
        self.lora_q_B = nn.Linear(rank, self.hidden_dim, bias=False)

        self.lora_v_A = nn.Linear(self.in_features, rank, bias=False)
        self.lora_v_B = nn.Linear(rank, self.hidden_dim, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.lora_q_A.weight, a=np.sqrt(5))
        nn.init.zeros_(self.lora_q_B.weight)

        nn.init.kaiming_uniform_(self.lora_v_A.weight, a=np.sqrt(5))
        nn.init.zeros_(self.lora_v_B.weight)

    def forward(self, x):
        base_out = self.base_qkv(x)

        q_delta = self.lora_q_B(self.lora_q_A(self.dropout(x))) * self.scaling
        v_delta = self.lora_v_B(self.lora_v_A(self.dropout(x))) * self.scaling

        out = base_out.clone()
        d = self.hidden_dim

        out[..., :d] = out[..., :d] + q_delta
        out[..., 2 * d:3 * d] = out[..., 2 * d:3 * d] + v_delta

        return out


def inject_lora_into_vit_qv(model, rank=8, alpha=16, dropout=0.1):
    """
    Replace every block.attn.qkv with a LoRA-wrapped qkv layer.
    """

    if not hasattr(model, "blocks"):
        raise ValueError("Expected timm ViT model with model.blocks")

    injected = 0

    for block in model.blocks:
        block.attn.qkv = LoRAQKVLinear(
            base_qkv=block.attn.qkv,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
        )
        injected += 1

    print(f"Injected LoRA into {injected} ViT attention qkv modules.")
    return model


def freeze_all(model):
    for p in model.parameters():
        p.requires_grad = False


def unfreeze_head(model):
    for name, p in model.named_parameters():
        p.requires_grad = name.startswith("head.")


def unfreeze_lora_and_head(model):
    for name, p in model.named_parameters():
        p.requires_grad = ("lora_" in name) or name.startswith("head.")


# ============================================================
# Subset SWAG over trainable parameters
# ============================================================

class TrainableSubsetSWAG:
    """
    SWAG posterior over currently trainable parameters only.

    Head-only:
        trainable params = head.*

    LoRA:
        trainable params = lora_* + head.*
    """

    def __init__(self, model, max_rank=20, var_clamp=1e-30):
        self.max_rank = max_rank
        self.var_clamp = var_clamp

        self.names = []
        self.shapes = []
        self.numels = []

        for name, p in model.named_parameters():
            if p.requires_grad:
                self.names.append(name)
                self.shapes.append(tuple(p.shape))
                self.numels.append(p.numel())

        if not self.names:
            raise ValueError("No trainable parameters found for SWAG.")

        self.n_params = sum(self.numels)
        self.n_models = 0
        self.mean = None
        self.sq_mean = None
        self.deviations = []

        print(f"\nSWAG subset parameters: {self.n_params:,}")
        print("Selected parameters:")
        for name, n in zip(self.names, self.numels):
            print(f"  {name:<80} {n:>12,}")

    def vectorize(self, model):
        chunks = []
        param_dict = dict(model.named_parameters())

        for name in self.names:
            chunks.append(param_dict[name].detach().cpu().reshape(-1))

        return torch.cat(chunks)

    def set_weights(self, model, vector):
        param_dict = dict(model.named_parameters())
        pointer = 0

        with torch.no_grad():
            for name, shape, numel in zip(self.names, self.shapes, self.numels):
                chunk = vector[pointer:pointer + numel].view(shape)
                param_dict[name].copy_(chunk.to(param_dict[name].device))
                pointer += numel

    def collect_model(self, model):
        vector = self.vectorize(model)

        if self.mean is None:
            self.mean = torch.zeros_like(vector)
            self.sq_mean = torch.zeros_like(vector)

        self.n_models += 1

        self.mean += (vector - self.mean) / self.n_models
        self.sq_mean += (vector ** 2 - self.sq_mean) / self.n_models

        deviation = vector - self.mean
        self.deviations.append(deviation)

        if len(self.deviations) > self.max_rank:
            self.deviations.pop(0)

        print(f"Collected SWAG snapshot #{self.n_models}")

    def sample(self, scale=1.0):
        if self.mean is None:
            raise ValueError("No SWAG snapshots collected.")

        var = torch.clamp(self.sq_mean - self.mean ** 2, min=self.var_clamp)

        diag_noise = torch.randn_like(self.mean) * torch.sqrt(var)

        sample = self.mean + scale * (1.0 / np.sqrt(2.0)) * diag_noise

        if len(self.deviations) >= 2:
            D = torch.stack(self.deviations, dim=1)  # [d, K]
            z = torch.randn(D.shape[1])
            low_rank_noise = D @ z
            low_rank_noise = low_rank_noise / np.sqrt(2.0 * (D.shape[1] - 1))
            sample = sample + scale * low_rank_noise

        return sample

    def state_dict(self):
        return {
            "names": self.names,
            "shapes": self.shapes,
            "numels": self.numels,
            "max_rank": self.max_rank,
            "var_clamp": self.var_clamp,
            "n_models": self.n_models,
            "n_params": self.n_params,
            "mean": self.mean,
            "sq_mean": self.sq_mean,
            "deviations": self.deviations,
        }


# ============================================================
# Model / data
# ============================================================

def build_vit_base(num_classes, pretrained=True):
    return get_model(
        name="vit_base_patch16_224",
        num_classes=num_classes,
        pretrained=pretrained,
    )


def get_loaders(num_classes, batch_size, num_workers):
    if num_classes == 10:
        train_classes = None
        test_classes = None
    elif num_classes == 5:
        train_classes = [0, 1, 2, 3, 4]
        test_classes = [0, 1, 2, 3, 4]
    else:
        raise ValueError("Only num_classes=10 or 5 supported.")

    return get_cifar10_loaders(
        batch_size=batch_size,
        num_workers=num_workers,
        train_classes=train_classes,
        test_classes=test_classes,
        image_size=224,
        normalization="imagenet",
        augment=True,
    )


def get_full_ood_loader(batch_size, num_workers):
    return get_cifar10_full_test_loader(
        batch_size=batch_size,
        num_workers=num_workers,
        image_size=224,
        normalization="imagenet",
    )


def make_optimizer(model, lr, weight_decay):
    params = [p for p in model.parameters() if p.requires_grad]
    n_params = sum(p.numel() for p in params)

    print(f"Trainable parameters: {n_params:,}")

    if n_params == 0:
        raise ValueError("No trainable parameters for optimizer.")

    return torch.optim.AdamW(
        params,
        lr=lr,
        weight_decay=weight_decay,
        betas=(0.9, 0.999),
    )


# ============================================================
# Train / evaluation
# ============================================================

def train_one_epoch(model, loader, optimizer, device, use_amp=True):
    model.train()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    scaler = torch.amp.GradScaler(
    "cuda",
    enabled=(use_amp and device.type == "cuda"),
   )

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(
    "cuda",
    enabled=(use_amp and device.type == "cuda"),
):
            logits = model(images)
            loss = F.cross_entropy(logits, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * images.size(0)
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_samples += images.size(0)

    return total_loss / total_samples, total_correct / total_samples


@torch.no_grad()
def collect_probs_labels(model, loader, device):
    model.eval()

    logits_list = []
    probs_list = []
    labels_list = []

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        probs = torch.softmax(logits, dim=1)

        logits_list.append(logits.detach().cpu())
        probs_list.append(probs.detach().cpu())
        labels_list.append(labels.detach().cpu())

    return (
        torch.cat(logits_list, dim=0),
        torch.cat(probs_list, dim=0),
        torch.cat(labels_list, dim=0),
    )


def evaluate_single(model, loader, device):
    logits, probs, labels = collect_probs_labels(model, loader, device)

    nll = F.cross_entropy(logits, labels).item()
    acc = (logits.argmax(dim=1) == labels).float().mean().item()
    ece = compute_ece(logits.to(device), labels.to(device))

    return acc, nll, ece, probs, labels


def evaluate_swag_bma(model, swag, loader, device, num_samples, sample_scale):
    all_probs = []
    labels_ref = None

    for s in range(num_samples):
        print(f"Evaluating SWAG sample {s + 1}/{num_samples}")

        sampled = swag.sample(scale=sample_scale)
        swag.set_weights(model, sampled)

        _, probs, labels = collect_probs_labels(model, loader, device)
        all_probs.append(probs)

        if labels_ref is None:
            labels_ref = labels

    mean_probs = torch.stack(all_probs, dim=0).mean(dim=0)

    preds = mean_probs.argmax(dim=1)
    acc = (preds == labels_ref).float().mean().item()

    correct_probs = mean_probs[torch.arange(labels_ref.numel()), labels_ref]
    nll = -torch.log(correct_probs + 1e-12).mean().item()

    ece = compute_ece(torch.log(mean_probs + 1e-12).to(device), labels_ref.to(device))

    return acc, nll, ece, mean_probs, labels_ref


# ============================================================
# OOD metrics
# ============================================================

def entropy_from_probs(probs):
    return -(probs * torch.log(probs + 1e-12)).sum(dim=1)


def sym_kl_from_entropy(id_entropy, ood_entropy, bins=30):
    all_entropy = torch.cat([id_entropy, ood_entropy])
    min_v = float(all_entropy.min())
    max_v = float(all_entropy.max())

    id_hist = torch.histc(id_entropy, bins=bins, min=min_v, max=max_v)
    ood_hist = torch.histc(ood_entropy, bins=bins, min=min_v, max=max_v)

    eps = 1e-8

    p = (id_hist + eps) / (id_hist.sum() + eps * bins)
    q = (ood_hist + eps) / (ood_hist.sum() + eps * bins)

    kl_pq = torch.sum(p * torch.log(p / q)).item()
    kl_qp = torch.sum(q * torch.log(q / p)).item()

    return kl_pq + kl_qp


def compute_ood_summary(method, probs, labels):
    entropy = entropy_from_probs(probs)

    id_mask = labels < 5
    ood_mask = labels >= 5

    id_entropy = entropy[id_mask]
    ood_entropy = entropy[ood_mask]

    y_true = torch.cat(
        [
            torch.zeros(id_entropy.numel()),
            torch.ones(ood_entropy.numel()),
        ]
    ).numpy()

    y_score = torch.cat([id_entropy, ood_entropy]).numpy()

    auroc = roc_auc_score(y_true, y_score)
    sym_kl = sym_kl_from_entropy(id_entropy, ood_entropy)

    return {
        "method": method,
        "id_entropy_mean": float(id_entropy.mean()),
        "ood_entropy_mean": float(ood_entropy.mean()),
        "entropy_gap_ood_minus_id": float(ood_entropy.mean() - id_entropy.mean()),
        "entropy_auroc": float(auroc),
        "sym_kl_binned_entropy": float(sym_kl),
        "num_id": int(id_entropy.numel()),
        "num_ood": int(ood_entropy.numel()),
    }


# ============================================================
# Stages
# ============================================================

def stage_parameter_cost(max_rank):
    model = build_vit_base(num_classes=10, pretrained=False)

    total = count_total(model)

    head = sum(p.numel() for n, p in model.named_parameters() if n.startswith("head."))
    final_block = sum(p.numel() for n, p in model.named_parameters() if "blocks.11" in n)
    final_head = head + final_block

    rows = [
        {
            "parameter_group": "full_vit_base",
            "num_params": total,
            "swag_memory_mb_k20": swag_memory_mb(total, max_rank),
            "adamw_state_memory_mb": adamw_memory_mb(total),
        },
        {
            "parameter_group": "head_only",
            "num_params": head,
            "swag_memory_mb_k20": swag_memory_mb(head, max_rank),
            "adamw_state_memory_mb": adamw_memory_mb(head),
        },
        {
            "parameter_group": "final_block_plus_head",
            "num_params": final_head,
            "swag_memory_mb_k20": swag_memory_mb(final_head, max_rank),
            "adamw_state_memory_mb": adamw_memory_mb(final_head),
        },
    ]

    print("\n===== ViT-Base SWAG Cost Analysis =====")
    for r in rows:
        print(r)

    save_csv(rows, "outputs/metrics/vit_base_swag_cost_analysis.csv")


def stage_full_baseline(args, device):
    ckpt_path = "outputs/checkpoints/vit_base_full_1epoch.pt"
    metrics_path = "outputs/metrics/vit_base_full_1epoch_metrics.json"

    if os.path.exists(ckpt_path) and not args.force:
        print(f"Skipping full baseline because checkpoint exists: {ckpt_path}")
        return ckpt_path

    train_loader, test_loader = get_loaders(
        num_classes=10,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    model = build_vit_base(num_classes=10, pretrained=True).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.full_lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
    )

    history = {
        "experiment_name": "vit_base_full_1epoch",
        "train_loss": [],
        "train_acc": [],
        "test_acc": [],
        "test_nll": [],
        "test_ece": [],
        "epoch_time_seconds": [],
        "epoch_time_minutes": [],
    }

    for epoch in range(args.full_epochs):
        sync_cuda()
        epoch_start = time.perf_counter()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, device, use_amp=args.amp
        )

        test_acc, test_nll, test_ece, probs, labels = evaluate_single(
            model, test_loader, device
        )

        sync_cuda()
        epoch_time = time.perf_counter() - epoch_start

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_acc"].append(test_acc)
        history["test_nll"].append(test_nll)
        history["test_ece"].append(test_ece)
        history["epoch_time_seconds"].append(epoch_time)
        history["epoch_time_minutes"].append(epoch_time / 60.0)

        print(
            f"[Full baseline] Epoch {epoch + 1}/{args.full_epochs} | "
            f"Train Loss {train_loss:.4f} | Train Acc {train_acc:.4f} | "
            f"Test Acc {test_acc:.4f} | NLL {test_nll:.4f} | ECE {test_ece:.4f} | "
            f"Time {epoch_time / 60.0:.2f} min"
        )

    torch.save(model.state_dict(), ckpt_path)
    save_json(history, metrics_path)

    return ckpt_path


def run_subset_swag_experiment(
    run_name,
    mode,
    num_classes,
    init_checkpoint,
    epochs,
    lr,
    args,
    device,
):
    """
    mode:
        head
        lora
    """

    train_loader, test_loader = get_loaders(
        num_classes=num_classes,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    model = build_vit_base(num_classes=num_classes, pretrained=True)

    if init_checkpoint is not None and num_classes == 10:
        print(f"Loading init checkpoint: {init_checkpoint}")
        state = torch.load(init_checkpoint, map_location="cpu")
        model.load_state_dict(state)

    freeze_all(model)

    if mode == "head":
        unfreeze_head(model)

    elif mode == "lora":
        model = inject_lora_into_vit_qv(
            model,
            rank=args.lora_rank,
            alpha=args.lora_alpha,
            dropout=args.lora_dropout,
        )
        unfreeze_lora_and_head(model)

    else:
        raise ValueError(f"Unknown mode: {mode}")

    model = model.to(device)

    print(f"\n===== Running {run_name} =====")
    print(f"Mode: {mode}")
    print(f"Num classes: {num_classes}")
    print(f"Total params: {count_total(model):,}")
    print(f"Trainable params: {count_trainable(model):,}")

    optimizer = make_optimizer(model, lr=lr, weight_decay=args.weight_decay)

    swag = TrainableSubsetSWAG(
        model=model,
        max_rank=args.max_rank,
        var_clamp=1e-30,
    )

    history = {
        "experiment_name": run_name,
        "mode": mode,
        "num_classes": num_classes,
        "epochs": epochs,
        "trainable_params": count_trainable(model),
        "swag_stats_memory_mb": swag_memory_mb(count_trainable(model), args.max_rank),
        "train_loss": [],
        "train_acc": [],
        "single_test_acc": [],
        "single_test_nll": [],
        "single_test_ece": [],
        "epoch_time_seconds": [],
        "epoch_time_minutes": [],
    }

    for epoch in range(epochs):
        sync_cuda()
        epoch_start = time.perf_counter()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, device, use_amp=args.amp
        )

        test_acc, test_nll, test_ece, _, _ = evaluate_single(
            model, test_loader, device
        )

        sync_cuda()
        epoch_time = time.perf_counter() - epoch_start

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["single_test_acc"].append(test_acc)
        history["single_test_nll"].append(test_nll)
        history["single_test_ece"].append(test_ece)
        history["epoch_time_seconds"].append(epoch_time)
        history["epoch_time_minutes"].append(epoch_time / 60.0)

        if epoch >= args.swag_start_epoch and (epoch - args.swag_start_epoch) % args.save_freq == 0:
            swag.collect_model(model)

        print(
            f"[{run_name}] Epoch {epoch + 1}/{epochs} | "
            f"Train Loss {train_loss:.4f} | Train Acc {train_acc:.4f} | "
            f"Single Test Acc {test_acc:.4f} | NLL {test_nll:.4f} | ECE {test_ece:.4f} | "
            f"Time {epoch_time / 60.0:.2f} min"
        )

    # SWA mean evaluation
    print(f"\nEvaluating SWA mean for {run_name}")
    sync_cuda()
    swa_start = time.perf_counter()

    swag.set_weights(model, swag.mean)
    swa_acc, swa_nll, swa_ece, swa_probs, swa_labels = evaluate_single(
        model, test_loader, device
    )

    sync_cuda()
    swa_time = time.perf_counter() - swa_start

    history["swa_eval_time_seconds"] = swa_time
    history["swa_eval_time_minutes"] = swa_time / 60.0

    history["final_swa_test_acc"] = swa_acc
    history["final_swa_test_nll"] = swa_nll
    history["final_swa_test_ece"] = swa_ece

    print(
        f"SWA Test Acc {swa_acc:.4f} | NLL {swa_nll:.4f} | ECE {swa_ece:.4f} | "
        f"Eval Time {swa_time / 60.0:.2f} min"
    )

    # SWAG BMA
    print(f"\nEvaluating SWAG BMA for {run_name}")
    sync_cuda()
    swag_eval_start = time.perf_counter()

    swag_acc, swag_nll, swag_ece, swag_probs, swag_labels = evaluate_swag_bma(
        model=model,
        swag=swag,
        loader=test_loader,
        device=device,
        num_samples=args.num_samples,
        sample_scale=args.sample_scale,
    )

    sync_cuda()
    swag_eval_time = time.perf_counter() - swag_eval_start

    history["swag_eval_time_seconds"] = swag_eval_time
    history["swag_eval_time_minutes"] = swag_eval_time / 60.0

    history["final_swag_test_acc"] = swag_acc
    history["final_swag_test_nll"] = swag_nll
    history["final_swag_test_ece"] = swag_ece

    print(
        f"SWAG Test Acc {swag_acc:.4f} | NLL {swag_nll:.4f} | ECE {swag_ece:.4f} | "
        f"BMA Time {swag_eval_time / 60.0:.2f} min"
    )

    # Put model back to SWA mean before saving deterministic checkpoint
    swag.set_weights(model, swag.mean)

    save_json(history, f"outputs/metrics/{run_name}_metrics.json")

    torch.save(model.state_dict(), f"outputs/checkpoints/{run_name}_swa_model.pt")
    torch.save(swag.state_dict(), f"outputs/checkpoints/{run_name}_swag_posterior.pt")

    torch.save(
        {"probs": swa_probs, "labels": swa_labels},
        f"outputs/metrics/{run_name}_swa_predictions.pt",
    )

    torch.save(
        {"probs": swag_probs, "labels": swag_labels},
        f"outputs/metrics/{run_name}_swag_predictions.pt",
    )

    # OOD evaluation only for 5-class training runs
    if num_classes == 5:
        ood_rows = []

        print(f"\nRunning full CIFAR-10 OOD evaluation for {run_name}")
        full_loader = get_full_ood_loader(args.batch_size, args.num_workers)

        # SWA OOD
        sync_cuda()
        ood_swa_start = time.perf_counter()

        swag.set_weights(model, swag.mean)
        _, swa_ood_probs, full_labels = collect_probs_labels(model, full_loader, device)

        sync_cuda()
        ood_swa_time = time.perf_counter() - ood_swa_start

        swa_ood = compute_ood_summary(f"{run_name}_SWA", swa_ood_probs, full_labels)
        swa_ood["ood_eval_time_seconds"] = ood_swa_time
        swa_ood["ood_eval_time_minutes"] = ood_swa_time / 60.0
        ood_rows.append(swa_ood)

        # SWAG OOD
        sync_cuda()
        ood_swag_start = time.perf_counter()

        _, _, _, swag_ood_probs, full_labels = evaluate_swag_bma(
            model=model,
            swag=swag,
            loader=full_loader,
            device=device,
            num_samples=args.num_samples,
            sample_scale=args.sample_scale,
        )

        sync_cuda()
        ood_swag_time = time.perf_counter() - ood_swag_start

        swag_ood = compute_ood_summary(f"{run_name}_SWAG", swag_ood_probs, full_labels)
        swag_ood["ood_eval_time_seconds"] = ood_swag_time
        swag_ood["ood_eval_time_minutes"] = ood_swag_time / 60.0
        ood_rows.append(swag_ood)

        save_csv(ood_rows, f"outputs/metrics/{run_name}_ood_summary.csv")

    return history


def build_final_summary():
    rows = []

    metric_files = [
        "outputs/metrics/vit_base_full_1epoch_metrics.json",
        "outputs/metrics/vit_base_head_swag_10class_metrics.json",
        "outputs/metrics/vit_base_lora_swag_10class_metrics.json",
        "outputs/metrics/vit_base_head_swag_5class_metrics.json",
        "outputs/metrics/vit_base_lora_swag_5class_metrics.json",
    ]

    for path in metric_files:
        if not os.path.exists(path):
            continue

        with open(path, "r", encoding="utf-8") as f:
            m = json.load(f)

        exp = m["experiment_name"]

        if "test_acc" in m:
            rows.append(
                {
                    "experiment": exp,
                    "method": "single",
                    "accuracy": m["test_acc"][-1],
                    "nll": m["test_nll"][-1],
                    "ece": m["test_ece"][-1],
                    "trainable_params": "",
                    "swag_stats_memory_mb": "",
                    "train_time_minutes_total": sum(m.get("epoch_time_minutes", [])),
                    "swa_eval_time_minutes": "",
                    "swag_eval_time_minutes": "",
                }
            )

        if "final_swa_test_acc" in m:
            rows.append(
                {
                    "experiment": exp,
                    "method": "SWA",
                    "accuracy": m["final_swa_test_acc"],
                    "nll": m["final_swa_test_nll"],
                    "ece": m["final_swa_test_ece"],
                    "trainable_params": m.get("trainable_params", ""),
                    "swag_stats_memory_mb": m.get("swag_stats_memory_mb", ""),
                    "train_time_minutes_total": sum(m.get("epoch_time_minutes", [])),
                    "swa_eval_time_minutes": m.get("swa_eval_time_minutes", ""),
                    "swag_eval_time_minutes": "",
                }
            )

        if "final_swag_test_acc" in m:
            rows.append(
                {
                    "experiment": exp,
                    "method": "SWAG",
                    "accuracy": m["final_swag_test_acc"],
                    "nll": m["final_swag_test_nll"],
                    "ece": m["final_swag_test_ece"],
                    "trainable_params": m.get("trainable_params", ""),
                    "swag_stats_memory_mb": m.get("swag_stats_memory_mb", ""),
                    "train_time_minutes_total": sum(m.get("epoch_time_minutes", [])),
                    "swa_eval_time_minutes": "",
                    "swag_eval_time_minutes": m.get("swag_eval_time_minutes", ""),
                }
            )

    save_csv(rows, "outputs/metrics/vit_base_pipeline_classification_summary.csv")

    # Combine OOD summaries
    ood_rows = []
    for path in [
        "outputs/metrics/vit_base_head_swag_5class_ood_summary.csv",
        "outputs/metrics/vit_base_lora_swag_5class_ood_summary.csv",
    ]:
        if not os.path.exists(path):
            continue

        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            ood_rows.extend(list(reader))

    if ood_rows:
        save_csv(ood_rows, "outputs/metrics/vit_base_pipeline_ood_summary.csv")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)

    parser.add_argument("--full-epochs", type=int, default=1)
    parser.add_argument("--head-epochs", type=int, default=4)
    parser.add_argument("--lora-epochs", type=int, default=4)
    parser.add_argument("--ood-head-epochs", type=int, default=4)
    parser.add_argument("--ood-lora-epochs", type=int, default=4)

    parser.add_argument("--full-lr", type=float, default=2e-5)
    parser.add_argument("--head-lr", type=float, default=1e-3)
    parser.add_argument("--lora-lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)

    parser.add_argument("--max-rank", type=int, default=20)
    parser.add_argument("--num-samples", type=int, default=15)
    parser.add_argument("--sample-scale", type=float, default=1.0)
    parser.add_argument("--swag-start-epoch", type=int, default=0)
    parser.add_argument("--save-freq", type=int, default=1)

    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.1)

    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--force", action="store_true")

    args = parser.parse_args()

    seed_everything(42)
    ensure_dirs()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Args: {args}")

    with timed_stage("stage_2_vit_base_cost_analysis"):
        stage_parameter_cost(max_rank=args.max_rank)

    with timed_stage("stage_1_vit_base_full_1epoch_baseline"):
        init_ckpt = stage_full_baseline(args, device)

    with timed_stage("stage_3_head_swag_10class"):
        run_subset_swag_experiment(
            run_name="vit_base_head_swag_10class",
            mode="head",
            num_classes=10,
            init_checkpoint=init_ckpt,
            epochs=args.head_epochs,
            lr=args.head_lr,
            args=args,
            device=device,
        )

    with timed_stage("stage_5_lora_swag_10class"):
        run_subset_swag_experiment(
            run_name="vit_base_lora_swag_10class",
            mode="lora",
            num_classes=10,
            init_checkpoint=init_ckpt,
            epochs=args.lora_epochs,
            lr=args.lora_lr,
            args=args,
            device=device,
        )

    with timed_stage("stage_6_head_swag_5class_ood_training_eval"):
        run_subset_swag_experiment(
            run_name="vit_base_head_swag_5class",
            mode="head",
            num_classes=5,
            init_checkpoint=None,
            epochs=args.ood_head_epochs,
            lr=args.head_lr,
            args=args,
            device=device,
        )

    with timed_stage("stage_6_lora_swag_5class_ood_training_eval"):
        run_subset_swag_experiment(
            run_name="vit_base_lora_swag_5class",
            mode="lora",
            num_classes=5,
            init_checkpoint=None,
            epochs=args.ood_lora_epochs,
            lr=args.lora_lr,
            args=args,
            device=device,
        )

    with timed_stage("build_final_summary"):
        build_final_summary()

    save_csv(TIMING_ROWS, "outputs/metrics/vit_base_pipeline_timing_summary.csv")

    print("\nDONE. Final files:")
    print("outputs/metrics/vit_base_pipeline_classification_summary.csv")
    print("outputs/metrics/vit_base_pipeline_ood_summary.csv")
    print("outputs/metrics/vit_base_swag_cost_analysis.csv")
    print("outputs/metrics/vit_base_pipeline_timing_summary.csv")


if __name__ == "__main__":
    main()