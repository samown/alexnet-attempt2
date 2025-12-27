"""
Utility functions for training
"""

import random
import numpy as np
import torch
import os
from pathlib import Path
from typing import Dict, Any


def set_seed(seed: int = 42):
    """
    Set random seed for reproducibility
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


def save_checkpoint(
    state: Dict[str, Any],
    filepath: str,
    is_best: bool = False,
    best_filepath: str = None
):
    """
    Save model checkpoint
    
    Args:
        state: Dictionary containing model state, optimizer state, epoch, etc.
        filepath: Path to save checkpoint
        is_best: Whether this is the best model so far
        best_filepath: Path to save best model (if is_best=True)
    """
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, filepath)
    
    if is_best and best_filepath:
        torch.save(state, best_filepath)


def load_checkpoint(
    filepath: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer = None,
    device: str = 'cpu'
) -> Dict[str, Any]:
    """
    Load model checkpoint
    
    Args:
        filepath: Path to checkpoint file
        model: Model to load weights into
        optimizer: Optimizer to load state into (optional)
        device: Device to load checkpoint on
        
    Returns:
        Dictionary containing checkpoint information
    """
    checkpoint = torch.load(filepath, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return checkpoint


def get_device() -> torch.device:
    """
    Get available device (CUDA if available, else CPU)
    
    Returns:
        torch.device object
    """
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


def count_parameters(model: torch.nn.Module) -> int:
    """
    Count the number of trainable parameters in a model
    
    Args:
        model: PyTorch model
        
    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


