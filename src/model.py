from torch import nn
import torch

class SmallCNN(nn.Module):
    """
    A small convolutional neural network for image classification.
    Architecture: 2 conv layers with max pooling, followed by a fully connected layer.
    """
    
    def __init__(self):
        """
        Initialize the SmallCNN.
        The architecture consists of:
        - Conv2d layer with 16 filters, kernel size 3, ReLU activation, and max pooling
        - Conv2d layer with 32 filters, kernel size 3, ReLU activation, and max pooling
        - Flatten layer to convert 2D feature maps to 1D feature vectors
        - Fully connected layer with 10 output classes (for classification)
        """
        
        super().__init__()
        self.conv1 = nn.Sequential(
			nn.Conv2d(1, 16, kernel_size=3, padding=1),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=2)
		)
        self.conv2 = nn.Sequential(
			nn.Conv2d(16, 32, kernel_size=3, padding=1),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=2)
		)
        # flatten, linear, 10 outputs
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(32 * 7 * 7, 10)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the SmallCNN.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 1, 28, 28)
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, 10) representing class logits.
        """
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x