import torch
from torch import nn
from typing import Optional

from src.utils import get_device

def evaluate(model: nn.Module, loader: torch.utils.data.DataLoader, device: Optional[torch.device]=None) -> float:
    """
    Evaluate the given model on the provided data loader.
    Args:
        model (nn.Module): The neural network model to evaluate.
        loader (DataLoader): DataLoader for the dataset to evaluate on.
        device (torch.device, optional): The device to run the evaluation on. If None, it will automatically select GPU if available, otherwise CPU.
    Returns:
        float: The accuracy of the model on the dataset.
    """

    device = get_device(device)          # get the device (GPU or CPU)
        
    model.eval()                          
    correct = 0
    total = 0
    with torch.no_grad():              
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)       # logits, [N, 10]
            predicted = torch.argmax(outputs, dim=1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    accuracy = correct / total
    return accuracy