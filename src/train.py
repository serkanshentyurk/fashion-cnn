from torch import nn
import torch
from src.model import SmallCNN
from src.data import get_data_loaders
from src.evaluate import evaluate
from src.utils import get_device
from src.calibrate import *
from typing import Optional

from sklearn.metrics import confusion_matrix, classification_report

def get_predictions(model: nn.Module, 
                    loader: torch.utils.data.DataLoader, 
                    device: Optional[torch.device]=None) -> tuple:
    device = get_device(device)          # get the device (GPU or CPU)
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            all_preds.append(preds.cpu())     
            all_labels.append(labels)
    return torch.cat(all_preds).numpy(), torch.cat(all_labels).numpy()


def train(model: nn.Module, train_loader: torch.utils.data.DataLoader, 
          epochs: int=5, lr: float=1e-3, device: Optional[torch.device]=None) -> nn.Module:
    """
    Train the given model using the provided training data loader.
    Args:
        model (nn.Module): The neural network model to train.
        train_loader (DataLoader): DataLoader for the training dataset.
        epochs (int): Number of epochs to train the model.
        lr (float): Learning rate for the optimizer.
        device (torch.device, optional): The device to run the training on. If None, it will automatically select GPU if available, otherwise CPU.
    Returns:
        nn.Module: The trained model."""
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()                      # tell the model it's training 
    
    device = get_device(device)          # get the device (GPU or CPU)
    model.to(device)                   # move the model to the device (GPU or CPU)

    for epoch in range(epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)  # move data to device
            
            optimizer.zero_grad()           # 1. zero the gradients
            outputs = model(images)         # 2. forward: get logits from the model
            loss = criterion(outputs, labels) # 3. compute the loss
            loss.backward()                 # 4. backward: compute gradients
            optimizer.step()                # 5. step: update parameters
            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        print(f"epoch {epoch+1}/{epochs}  loss: {avg_loss:.4f}")

    return model


def main():
    # Get data loaders
    train_loader, val_loader, test_loader = get_data_loaders(batch_size=64)

    # Initialize the model
    model = SmallCNN()

    # Train the model
    trained_model = train(model, train_loader, epochs=5, lr=1e-3)
    
    train_acc = evaluate(trained_model, train_loader)
    
    confidences, correctness = get_confidences_and_correctness(
        trained_model,
        val_loader)
    
    ece = compute_ece(confidences, correctness, n_bins=10)
    print(f"Validation ECE: {ece:.4f}")
    fig, ax = plot_reliability_diagram(confidences, correctness, n_bins=10)
    fig.savefig("reports/figures/reliability_diagram.png")

    val_acc   = evaluate(trained_model, val_loader)
    print(f"train accuracy: {train_acc:.4f}")
    print(f"val accuracy:   {val_acc:.4f}")
    print(f"gap:            {train_acc - val_acc:.4f}")
    
    preds, labels = get_predictions(trained_model, val_loader)
    print(confusion_matrix(labels, preds))
    print(classification_report(labels, preds))
    
    # FINAL — sealed test set, single touch, no fitting
    test_acc = evaluate(trained_model, test_loader)
    test_conf, test_correct = get_confidences_and_correctness(trained_model, test_loader)
    test_ece = compute_ece(test_conf, test_correct, n_bins=10)
    print(f"\nFINAL TEST accuracy: {test_acc:.4f}")
    print(f"FINAL TEST ECE:      {test_ece:.4f}")
    
if __name__ == "__main__":
    main()