import torch
import torch.nn.functional as F


def compute_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:

    preds = torch.argmax(logits, dim=1)
    correct = (preds == labels).float().mean().item()
    return correct


def compute_nll(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """
    Compute negative log likelihood (cross-entropy).

    Args:
        logits: Tensor of shape [N, C]
        labels: Tensor of shape [N]

    Returns:
        Mean NLL as a float
    """
    nll = F.cross_entropy(logits, labels, reduction="mean").item()
    return nll


def compute_ece(logits: torch.Tensor, labels: torch.Tensor, n_bins: int = 15) -> float:
    """
    Compute Expected Calibration Error (ECE).

    Args:
        logits: Tensor of shape [N, C]
        labels: Tensor of shape [N]
        n_bins: Number of confidence bins

    Returns:
        ECE as a float
    """
    probs = torch.softmax(logits, dim=1)
    confidences, predictions = torch.max(probs, dim=1)
    accuracies = predictions.eq(labels)

    bin_boundaries = torch.linspace(0, 1, n_bins + 1, device=logits.device)
    ece = torch.zeros(1, device=logits.device)

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        # Include right edge in the last bin
        if i == n_bins - 1:
            in_bin = (confidences >= bin_lower) & (confidences <= bin_upper)
        else:
            in_bin = (confidences >= bin_lower) & (confidences < bin_upper)

        prop_in_bin = in_bin.float().mean()

        if prop_in_bin.item() > 0:
            accuracy_in_bin = accuracies[in_bin].float().mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    return ece.item()


def get_calibration_bins(logits: torch.Tensor, labels: torch.Tensor, n_bins: int = 15):
    """
    Compute per-bin confidence and accuracy for reliability diagrams.

    Args:
        logits: Tensor of shape [N, C]
        labels: Tensor of shape [N]
        n_bins: Number of bins

    Returns:
        bin_centers: list of bin centers
        bin_accuracies: list of accuracies per bin
        bin_confidences: list of average confidences per bin
        bin_counts: list of counts per bin
    """
    probs = torch.softmax(logits, dim=1)
    confidences, predictions = torch.max(probs, dim=1)
    accuracies = predictions.eq(labels)

    bin_boundaries = torch.linspace(0, 1, n_bins + 1, device=logits.device)

    bin_centers = []
    bin_accuracies = []
    bin_confidences = []
    bin_counts = []

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        if i == n_bins - 1:
            in_bin = (confidences >= bin_lower) & (confidences <= bin_upper)
        else:
            in_bin = (confidences >= bin_lower) & (confidences < bin_upper)

        count_in_bin = in_bin.sum().item()
        bin_counts.append(count_in_bin)
        bin_centers.append(((bin_lower + bin_upper) / 2).item())

        if count_in_bin > 0:
            acc_in_bin = accuracies[in_bin].float().mean().item()
            conf_in_bin = confidences[in_bin].mean().item()
        else:
            acc_in_bin = 0.0
            conf_in_bin = 0.0

        bin_accuracies.append(acc_in_bin)
        bin_confidences.append(conf_in_bin)

    return bin_centers, bin_accuracies, bin_confidences, bin_counts