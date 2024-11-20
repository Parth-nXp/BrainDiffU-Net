import numpy as np
import torch

def set_seed(seed=42):
    """
    Set seed for reproducibility across numpy and PyTorch.

    Parameters:
    seed (int): Seed value. Default is 42.
    """
    print(f"Setting random seed: {seed}")
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def initialize_device():
    """
    Initialize the device for PyTorch operations.

    Returns:
    torch.device: CUDA device if available, else CPU.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    return device