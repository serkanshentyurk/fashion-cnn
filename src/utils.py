import torch

from typing import Optional

def get_device(device: Optional[torch.device]=None) -> torch.device:
    if device is None:
        device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    return device