import torch
from src.calibrate import compute_ece

def test_ece_perfect():
    # 100 predictions all at 0.9 confidence, exactly 90 correct.
    # In the (0.8, 0.9] bin: avg confidence = 0.9, accuracy = 0.9. Gap = 0. ECE = 0.
    confidences = torch.tensor([0.9] * 100)
    correctness = torch.tensor([True] * 90 + [False] * 10)
    ece = compute_ece(confidences, correctness, n_bins=10)
    assert ece < 1e-6, f"expected ~0, got {ece}"

def test_ece_worst():
    # 100 predictions all at 1.0 confidence, exactly 50 correct.
    # In the (0.9, 1.0] bin: avg confidence = 1.0, accuracy = 0.5. Gap = 0.5. ECE = 0.5.
    confidences = torch.tensor([1.0] * 100)
    correctness = torch.tensor([True] * 50 + [False] * 50)
    ece = compute_ece(confidences, correctness, n_bins=10)
    assert abs(ece - 0.5) < 1e-6, f"expected 0.5, got {ece}"
