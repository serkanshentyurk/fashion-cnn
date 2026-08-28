import torch
import torch.nn.functional as F

from src.utils import get_device
from typing import Optional

def get_confidences_and_correctness(model: torch.nn.Module, loader: torch.utils.data.DataLoader, device: Optional[torch.device]=None):
    device = get_device(device)          # get the device (GPU or CPU)
    model.eval()
    all_conf, all_correct = [], []
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            probs = F.softmax(logits, dim=1)          # logits -> probabilities, YOU do the softmax
            confidence, prediction = torch.max(probs, dim=1)  # get max probability and argmax
            correct = (prediction == labels)
            all_conf.append(confidence.cpu())
            all_correct.append(correct.cpu())

    return torch.cat(all_conf), torch.cat(all_correct)


def compute_ece(confidences: torch.Tensor, correctness: torch.Tensor, n_bins: int=10) -> float:
    """
    Compute the Expected Calibration Error (ECE) of a model's predictions.
    Args:
        confidences (torch.Tensor): Array of predicted confidences (max softmax probabilities).
        correctness (torch.Tensor): Array of boolean values indicating whether each prediction was correct.
        n_bins (int): Number of bins to use for calibration.
    Returns:
        float: The computed ECE value.
    """
    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = in_bin.float().mean().item()  # proportion of samples in this bin

        if prop_in_bin > 0:
            accuracy_in_bin = correctness[in_bin].float().mean().item()
            avg_confidence_in_bin = confidences[in_bin].mean().item()
            ece += prop_in_bin * abs(avg_confidence_in_bin - accuracy_in_bin)

    return ece


def plot_reliability_diagram(confidences: torch.Tensor, correctness: torch.Tensor, n_bins: int=10) -> tuple:
    """
    Plot a reliability diagram to visualize the calibration of a model's predictions.
    Args:
        confidences (torch.Tensor): Array of predicted confidences (max softmax probabilities).
        correctness (torch.Tensor): Array of boolean values indicating whether each prediction was correct.
        n_bins (int): Number of bins to use for calibration.
    Returns:
        tuple: A tuple containing the matplotlib figure and axes objects.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    bin_centers = (bin_boundaries[:-1] + bin_boundaries[1:]) / 2
    accuracies = []
    confidences_mean = []

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)

        if in_bin.any():
            accuracy_in_bin = correctness[in_bin].float().mean().item()
            avg_confidence_in_bin = confidences[in_bin].mean().item()
            accuracies.append(accuracy_in_bin)
            confidences_mean.append(avg_confidence_in_bin)
        else:
            accuracies.append(0.0)
            confidences_mean.append(0.0)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.bar(bin_centers, accuracies, width=0.1, alpha=0.5, label='Accuracy')
    ax.plot([0, 1], [0, 1], 'r--', label='Perfect Calibration')
    ax.set_xlabel('Confidence')
    ax.set_ylabel('Accuracy')
    ax.set_title('Reliability Diagram')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()
    fig.tight_layout()
    return fig, ax
