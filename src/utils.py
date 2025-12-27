"""
Utility functions for path resolution
"""

from pathlib import Path
import os


def get_project_root() -> Path:
    """
    Get the project root directory (alexnet-and-ifood-2019-challenge-kelompok)
    
    Returns:
        Path to project root
    """
    # This file is in src/, so go up one level
    current_file = Path(__file__).resolve()
    project_root = current_file.parent.parent
    return project_root


def resolve_path(path_str: str, base_dir: Path = None) -> Path:
    """
    Resolve a path string to an absolute Path
    
    Args:
        path_str: Path string (can be relative or absolute)
        base_dir: Base directory to resolve relative paths from (default: project root)
        
    Returns:
        Resolved Path object
    """
    if base_dir is None:
        base_dir = get_project_root()
    
    path = Path(path_str)
    
    # If absolute path, return as is
    if path.is_absolute():
        return path
    
    # Resolve relative to base_dir
    return (base_dir / path).resolve()


def get_data_dir() -> Path:
    """
    Get the data directory path
    
    Returns:
        Path to data directory (folder containing train/val/test/annotations next to project root)
    """
    return get_project_root().parent


def get_results_dir() -> Path:
    """
    Get the results directory path
    
    Returns:
        Path to results directory
    """
    return get_project_root() / "results"

