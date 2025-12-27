"""
Training utilities for AlexNet models
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
from typing import Dict, Any, Optional
import time
from pathlib import Path

from .utils import save_checkpoint, get_device
from ..data.utils import calculate_class_weights, get_class_weights_tensor
from ..utils import resolve_path


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
    use_wandb: bool = True
) -> Dict[str, float]:
    """
    Train model for one epoch
    
    Args:
        model: Model to train
        train_loader: DataLoader for training data
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on
        epoch: Current epoch number
        use_wandb: Whether to log to wandb
        
    Returns:
        Dictionary with training metrics
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch} [Train]')
    
    for batch_idx, (images, labels) in enumerate(pbar):
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{running_loss / (batch_idx + 1):.4f}',
            'acc': f'{100 * correct / total:.2f}%'
        })
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100 * correct / total
    
    metrics = {
        'train_loss': epoch_loss,
        'train_accuracy': epoch_acc
    }
    
    if use_wandb:
        wandb.log({
            'epoch': epoch,
            **metrics
        })
    
    return metrics


def validate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    use_wandb: bool = True
) -> Dict[str, float]:
    """
    Validate model
    
    Args:
        model: Model to validate
        val_loader: DataLoader for validation data
        criterion: Loss function
        device: Device to validate on
        epoch: Current epoch number
        use_wandb: Whether to log to wandb
        
    Returns:
        Dictionary with validation metrics
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc=f'Epoch {epoch} [Val]')
        
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{running_loss / len(val_loader):.4f}',
                'acc': f'{100 * correct / total:.2f}%'
            })
    
    epoch_loss = running_loss / len(val_loader)
    epoch_acc = 100 * correct / total
    
    metrics = {
        'val_loss': epoch_loss,
        'val_accuracy': epoch_acc
    }
    
    if use_wandb:
        wandb.log({
            'epoch': epoch,
            **metrics
        })
    
    return metrics


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: Dict[str, Any],
    class_weights: Optional[Dict[int, float]] = None,
    resume_from: Optional[str] = None
) -> Dict[str, Any]:
    """
    Main training function
    
    Args:
        model: Model to train
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        config: Configuration dictionary
        class_weights: Dictionary mapping class_id to weight (optional)
        resume_from: Path to checkpoint to resume from (optional)
        
    Returns:
        Dictionary with training history and best model info
    """
    device = get_device()
    model = model.to(device)
    
    # Ensure paths in config are resolved (if not already)
    if 'paths' in config:
        for key in ['checkpoints_dir', 'plots_dir', 'logs_dir']:
            if key in config['paths']:
                config['paths'][key] = str(resolve_path(config['paths'][key]))
    
    # Loss function
    if class_weights:
        weights_tensor = get_class_weights_tensor(
            class_weights,
            config['model']['num_classes'],
            device
        )
        criterion = nn.CrossEntropyLoss(weight=weights_tensor)
    else:
        criterion = nn.CrossEntropyLoss()
    
    # Optimizer
    optimizer = optim.SGD(
        model.parameters(),
        lr=config['training']['learning_rate'],
        momentum=config['training']['momentum'],
        weight_decay=config['training']['weight_decay']
    )
    
    # Learning rate scheduler
    scheduler = None
    if config['training']['scheduler']['use_scheduler']:
        scheduler_type = config['training']['scheduler']['type']
        if scheduler_type == 'step':
            scheduler = optim.lr_scheduler.StepLR(
                optimizer,
                step_size=config['training']['scheduler']['step_size'],
                gamma=config['training']['scheduler']['gamma']
            )
        elif scheduler_type == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=config['training']['num_epochs']
            )
    
    # Resume from checkpoint if provided
    start_epoch = 0
    best_val_acc = 0.0
    if resume_from:
        resume_path = resolve_path(resume_from) if not Path(resume_from).is_absolute() else Path(resume_from)
        if resume_path.exists():
            checkpoint = torch.load(str(resume_path), map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_val_acc = checkpoint.get('best_val_acc', 0.0)
            print(f"Resumed from epoch {start_epoch}")
        else:
            print(f"Warning: Checkpoint not found at {resume_path}")
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    # Early stopping
    early_stopping = config['training']['early_stopping']
    patience = early_stopping['patience'] if early_stopping['use_early_stopping'] else None
    min_delta = early_stopping['min_delta'] if early_stopping['use_early_stopping'] else 0.0
    patience_counter = 0
    
    # Training loop
    num_epochs = config['training']['num_epochs']
    checkpoints_dir = resolve_path(config['paths']['checkpoints_dir'])
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    
    model_name = config['model']['name']
    best_model_path = checkpoints_dir / f'{model_name}_best.pth'
    
    print(f"\nStarting training for {num_epochs} epochs...")
    print(f"Device: {device}")
    print(f"Model: {model_name}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print("-" * 60)
    
    start_time = time.time()
    
    for epoch in range(start_epoch, num_epochs):
        epoch_start_time = time.time()
        
        # Clear GPU cache at start of epoch
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        
        # Train
        train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch,
            use_wandb=config.get('wandb', {}).get('project') is not None
        )
        
        # Validate
        val_metrics = validate(
            model, val_loader, criterion, device, epoch,
            use_wandb=config.get('wandb', {}).get('project') is not None
        )
        
        # Update learning rate
        if scheduler:
            scheduler.step()
            if config.get('wandb', {}).get('project'):
                wandb.log({'learning_rate': scheduler.get_last_lr()[0]})
        
        # Save history
        history['train_loss'].append(train_metrics['train_loss'])
        history['train_acc'].append(train_metrics['train_accuracy'])
        history['val_loss'].append(val_metrics['val_loss'])
        history['val_acc'].append(val_metrics['val_accuracy'])
        
        # Save checkpoint
        is_best = val_metrics['val_accuracy'] > best_val_acc
        if is_best:
            best_val_acc = val_metrics['val_accuracy']
            patience_counter = 0
        else:
            patience_counter += 1
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_acc': best_val_acc,
            'train_loss': train_metrics['train_loss'],
            'val_loss': val_metrics['val_loss'],
            'train_acc': train_metrics['train_accuracy'],
            'val_acc': val_metrics['val_accuracy'],
        }
        
        checkpoint_path = checkpoints_dir / f'{model_name}_epoch_{epoch}.pth'
        save_checkpoint(
            checkpoint,
            str(checkpoint_path),
            is_best=is_best,
            best_filepath=str(best_model_path)
        )
        
        epoch_time = time.time() - epoch_start_time
        print(f"Epoch {epoch}/{num_epochs-1} - "
              f"Train Loss: {train_metrics['train_loss']:.4f}, "
              f"Train Acc: {train_metrics['train_accuracy']:.2f}%, "
              f"Val Loss: {val_metrics['val_loss']:.4f}, "
              f"Val Acc: {val_metrics['val_accuracy']:.2f}%, "
              f"Time: {epoch_time:.2f}s")
        
        # Early stopping
        if patience and patience_counter >= patience:
            print(f"\nEarly stopping triggered after {epoch+1} epochs")
            print(f"Best validation accuracy: {best_val_acc:.2f}%")
            break
    
    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time/60:.2f} minutes")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    
    return {
        'history': history,
        'best_val_acc': best_val_acc,
        'best_model_path': str(best_model_path),
        'total_time': total_time
    }


