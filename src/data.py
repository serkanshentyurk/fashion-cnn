import torch
import torchvision

def get_data_loaders(batch_size: int = 64, val_size: int = 10000, 
                     seed: int = 42, root: str = "data/"
                     ) -> tuple[torch.utils.data.DataLoader, 
                                torch.utils.data.DataLoader, 
                                torch.utils.data.DataLoader]:
    """
    Get data loaders for training, validation, and testing.
    
    Args:
		batch_size (int): Number of samples per batch.
		val_size (int): Number of samples in the validation set.
		seed (int): Random seed for reproducibility.
		root (str): Directory where the dataset will be stored.
  
	Returns:
		tuple: A tuple containing the training, validation, and test data loaders.
    """
    train_data = torchvision.datasets.FashionMNIST(root=root, train=True, download=True, transform=torchvision.transforms.ToTensor())
    test_data = torchvision.datasets.FashionMNIST(root=root, train=False, download=True, transform=torchvision.transforms.ToTensor())
    generator = torch.Generator().manual_seed(seed)
    n_total = len(train_data)
    train_set, val_set = torch.utils.data.random_split(train_data, [n_total - val_size, val_size], generator=generator) 
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader