"""
Evaluation utilities for AlexNet models
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
from typing import Dict, Any, Optional, Tuple
import pandas as pd

from .utils import get_device, load_checkpoint
from ..utils import resolve_path


def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    class_names: Optional[Dict[int, str]] = None
) -> Dict[str, Any]:
    """
    Evaluate model on test set
    
    Args:
        model: Model to evaluate
        test_loader: DataLoader for test data
        device: Device to evaluate on
        class_names: Dictionary mapping class_id to class_name (optional)
        
    Returns:
        Dictionary with evaluation metrics
    """
    model.eval()
    all_preds = []
    all_labels = []
    correct = 0
    total = 0
    
    with torch.no_grad():
        pbar = tqdm(test_loader, desc='Evaluating')
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            pbar.set_postfix({'acc': f'{100 * correct / total:.2f}%'})
    
    accuracy = 100 * correct / total
    
    # Per-class accuracy
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    num_classes = len(np.unique(all_labels))
    per_class_acc = {}
    
    for class_id in range(num_classes):
        mask = all_labels == class_id
        if mask.sum() > 0:
            class_correct = (all_preds[mask] == all_labels[mask]).sum()
            class_total = mask.sum()
            per_class_acc[class_id] = 100 * class_correct / class_total
    
    results = {
        'test_accuracy': accuracy,
        'per_class_accuracy': per_class_acc,
        'predictions': all_preds,
        'labels': all_labels,
        'num_samples': total
    }
    
    # Classification report
    if class_names:
        target_names = [class_names.get(i, f'Class_{i}') for i in range(num_classes)]
    else:
        target_names = [f'Class_{i}' for i in range(num_classes)]
    
    report = classification_report(
        all_labels,
        all_preds,
        target_names=target_names,
        output_dict=True,
        zero_division=0
    )
    
    results['classification_report'] = report
    
    return results


def generate_confusion_matrix(
    predictions: np.ndarray,
    labels: np.ndarray,
    class_names: Optional[Dict[int, str]] = None,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (20, 20),
    top_n: Optional[int] = None
) -> np.ndarray:
    """
    Generate and visualize confusion matrix
    
    Args:
        predictions: Array of predictions
        labels: Array of true labels
        class_names: Dictionary mapping class_id to class_name (optional)
        save_path: Path to save confusion matrix plot (optional)
        figsize: Figure size for plot
        top_n: Show only top N classes by frequency (optional)
        
    Returns:
        Confusion matrix as numpy array
    """
    num_classes = len(np.unique(labels))
    
    # Filter to top N classes if specified
    if top_n and top_n < num_classes:
        # Get most frequent classes
        unique, counts = np.unique(labels, return_counts=True)
        top_classes = unique[np.argsort(counts)[-top_n:]]
        
        mask = np.isin(labels, top_classes)
        predictions = predictions[mask]
        labels = labels[mask]
        
        # Remap class indices to 0..top_n-1 for visualization
        class_mapping = {cls: i for i, cls in enumerate(sorted(top_classes))}
        predictions = np.array([class_mapping.get(p, 0) for p in predictions])
        labels = np.array([class_mapping.get(l, 0) for l in labels])
        
        if class_names:
            class_names = {i: class_names.get(cls, f'Class_{cls}') 
                          for i, cls in enumerate(sorted(top_classes))}
        
        num_classes = top_n
    
    # Compute confusion matrix
    cm = confusion_matrix(labels, predictions, labels=range(num_classes))
    
    # Normalize confusion matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_normalized = np.nan_to_num(cm_normalized)
    
    # Plot
    plt.figure(figsize=figsize)
    
    # Create subplot for raw counts
    plt.subplot(1, 2, 1)
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=[class_names.get(i, f'Class_{i}') if class_names else f'Class_{i}' 
                     for i in range(num_classes)],
        yticklabels=[class_names.get(i, f'Class_{i}') if class_names else f'Class_{i}' 
                     for i in range(num_classes)],
        cbar_kws={'label': 'Count'}
    )
    plt.title('Confusion Matrix (Counts)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    # Create subplot for normalized
    plt.subplot(1, 2, 2)
    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt='.2f',
        cmap='Blues',
        xticklabels=[class_names.get(i, f'Class_{i}') if class_names else f'Class_{i}' 
                     for i in range(num_classes)],
        yticklabels=[class_names.get(i, f'Class_{i}') if class_names else f'Class_{i}' 
                     for i in range(num_classes)],
        cbar_kws={'label': 'Normalized'}
    )
    plt.title('Confusion Matrix (Normalized)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
    else:
        plt.show()
    
    plt.close()
    
    return cm


def evaluate_from_checkpoint(
    model: nn.Module,
    checkpoint_path: str,
    test_loader: DataLoader,
    device: torch.device,
    class_names: Optional[Dict[int, str]] = None,
    save_confusion_matrix: bool = True,
    output_dir: str = None
) -> Dict[str, Any]:
    """
    Evaluate model from checkpoint
    
    Args:
        model: Model architecture
        checkpoint_path: Path to model checkpoint
        test_loader: DataLoader for test data
        device: Device to evaluate on
        class_names: Dictionary mapping class_id to class_name (optional)
        save_confusion_matrix: Whether to save confusion matrix plot
        output_dir: Directory to save outputs (if None, uses default from config)
        
    Returns:
        Dictionary with evaluation results
    """
    # Resolve output directory
    if output_dir is None:
        output_dir = resolve_path("results/plots")
    else:
        output_dir = resolve_path(output_dir)
    
    # Load checkpoint
    checkpoint_path_resolved = resolve_path(checkpoint_path) if not Path(checkpoint_path).is_absolute() else Path(checkpoint_path)
    checkpoint = load_checkpoint(str(checkpoint_path_resolved), model, device=device)
    model = model.to(device)
    model.eval()
    
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    print(f"Checkpoint validation accuracy: {checkpoint.get('val_acc', 'N/A'):.2f}%")
    
    # Evaluate
    results = evaluate_model(model, test_loader, device, class_names)
    
    print(f"\nTest Accuracy: {results['test_accuracy']:.2f}%")
    
    # Generate confusion matrix
    if save_confusion_matrix:
        model_name = Path(checkpoint_path_resolved).stem
        cm_path = output_dir / f'{model_name}_confusion_matrix.png'
        generate_confusion_matrix(
            results['predictions'],
            results['labels'],
            class_names=class_names,
            save_path=str(cm_path),
            top_n=50  # Show top 50 classes for readability
        )
    
    return results


