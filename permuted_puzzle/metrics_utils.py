"""
Utility functions for reading and extracting information from metrics.json files.

These functions provide a clean API for accessing training run metadata,
including split indices, permutations, model configuration, and performance metrics.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import numpy as np


def load_metrics(metrics_path: str) -> Dict[str, Any]:
    """
    Load metrics.json file.

    Args:
        metrics_path: Path to metrics.json file

    Returns:
        Dictionary containing all metrics data

    Raises:
        FileNotFoundError: If metrics file does not exist
        json.JSONDecodeError: If file is not valid JSON
    """
    metrics_path = Path(metrics_path)
    if not metrics_path.exists():
        raise FileNotFoundError(f"Metrics file not found: {metrics_path}")

    with open(metrics_path, 'r') as f:
        return json.load(f)


def get_split_indices(metrics_path: str) -> Dict[str, np.ndarray]:
    """
    Extract dataset split indices from metrics file.

    Args:
        metrics_path: Path to metrics.json file

    Returns:
        Dictionary with keys 'train', 'val', 'test' mapping to numpy arrays of indices.
        Returns empty dict if split_indices not found in metrics.

    Example:
        >>> indices = get_split_indices('results/resnet18/1/metrics.json')
        >>> train_indices = indices['train']  # array([17, 12, 2, ...])
    """
    metrics = load_metrics(metrics_path)
    split_indices = metrics.get('split_indices', {})

    # Convert lists to numpy arrays
    return {
        split: np.array(indices)
        for split, indices in split_indices.items()
    }


def get_permutation(metrics_path: str, split: Optional[str] = None) -> Optional[List[int]]:
    """
    Extract permutation from metrics file.

    Supports both legacy format (single 'permutation' key) and new format
    (separate permutations for train/val/test).

    Args:
        metrics_path: Path to metrics.json file
        split: Which split's permutation to retrieve ('train', 'val', or 'test').
               If None and using new format, returns train permutation.
               For legacy format, split parameter is ignored.

    Returns:
        List of tile indices defining the permutation, or None if no permutation found.
        For baseline (grid_size=1), returns None.

    Example:
        >>> perm = get_permutation('results/resnet18/9/metrics.json', 'train')
        >>> # [3, 1, 4, 0, 2, 5, 8, 6, 7] for 3x3 grid
    """
    metrics = load_metrics(metrics_path)

    # Check for new format first (separate permutations per split)
    if 'permutations' in metrics:
        permutations = metrics['permutations']
        if split is not None and split in permutations:
            return permutations[split]
        # Default to train if available
        return permutations.get('train')

    # Fall back to legacy format (single permutation)
    return metrics.get('permutation')


def get_all_permutations(metrics_path: str) -> Dict[str, Optional[List[int]]]:
    """
    Get all available permutations from metrics file.

    Args:
        metrics_path: Path to metrics.json file

    Returns:
        Dictionary mapping split names to permutations.
        For legacy format, returns {'train': perm, 'val': perm, 'test': perm}.
        For new format, returns actual per-split permutations.

    Example:
        >>> perms = get_all_permutations('results/resnet18/9/metrics.json')
        >>> train_perm = perms['train']
        >>> val_perm = perms['val']
    """
    metrics = load_metrics(metrics_path)

    # Check for new format
    if 'permutations' in metrics:
        return metrics['permutations']

    # Legacy format: same permutation for all splits
    perm = metrics.get('permutation')
    return {
        'train': perm,
        'val': perm,
        'test': perm
    }


def get_model_info(metrics_path: str) -> Dict[str, Any]:
    """
    Extract model configuration and metadata.

    Args:
        metrics_path: Path to metrics.json file

    Returns:
        Dictionary containing:
            - model: Model architecture name (e.g., 'resnet18')
            - grid_size: Tile grid size (1 for baseline, 3 for 3x3, etc.)
            - input_size: Input image resolution
            - mean: Normalization mean
            - std: Normalization std
            - weights_path: Absolute path to model weights
            - saved_at: Timestamp when model was saved
            - env: Python and PyTorch versions

    Example:
        >>> info = get_model_info('results/resnet18/1/metrics.json')
        >>> print(f"Model: {info['model']}, Grid: {info['grid_size']}x{info['grid_size']}")
    """
    metrics = load_metrics(metrics_path)
    metrics_path = Path(metrics_path)

    # Construct absolute path to weights
    # weights_relpath is relative to the results root (parent of metrics_path)
    weights_relpath = metrics.get('weights_relpath', 'best.pth')
    # Get the results root (e.g., results_grid/resnet18)
    results_root = metrics_path.parent.parent
    weights_path = results_root / weights_relpath

    return {
        'model': metrics.get('model'),
        'grid_size': metrics.get('grid_size'),
        'input_size': metrics.get('meta', {}).get('input_size'),
        'mean': metrics.get('meta', {}).get('mean'),
        'std': metrics.get('meta', {}).get('std'),
        'weights_path': str(weights_path.absolute()),
        'saved_at': metrics.get('saved_at'),
        'env': metrics.get('env', {}),
        'notes': metrics.get('notes', '')
    }


def get_performance_metrics(metrics_path: str) -> Dict[str, float]:
    """
    Extract performance metrics from training run.

    Args:
        metrics_path: Path to metrics.json file

    Returns:
        Dictionary containing:
            - final_train_acc: Final training accuracy
            - final_train_loss: Final training loss
            - final_val_acc: Final validation accuracy
            - final_val_loss: Final validation loss
            - best_val_acc: Best validation accuracy achieved
            - test_acc: Test accuracy (if available)
            - test_loss: Test loss (if available)

    Example:
        >>> perf = get_performance_metrics('results/resnet18/1/metrics.json')
        >>> print(f"Best val acc: {perf['best_val_acc']:.2%}")
    """
    metrics = load_metrics(metrics_path)
    history = metrics.get('history', {})

    def get_final(key):
        """Get final value from history list."""
        values = history.get(key, [])
        return values[-1] if values else None

    return {
        'final_train_acc': get_final('train_acc'),
        'final_train_loss': get_final('train_loss'),
        'final_val_acc': metrics.get('final_val_accuracy'),
        'final_val_loss': get_final('val_loss'),
        'best_val_acc': metrics.get('best_val_accuracy'),
        'test_acc': metrics.get('test_accuracy'),
        'test_loss': metrics.get('test_loss')
    }


def get_training_config(metrics_path: str) -> Dict[str, Any]:
    """
    Extract training hyperparameters and configuration.

    Args:
        metrics_path: Path to metrics.json file

    Returns:
        Dictionary containing:
            - epochs: Number of training epochs
            - batch_size: Batch size used
            - lr: Learning rate
            - wd: Weight decay
            - optimizer: Optimizer name (e.g., 'AdamW')
            - dropout: Dropout rate
            - pretrained: Whether pretrained weights were used

    Example:
        >>> config = get_training_config('results/resnet18/1/metrics.json')
        >>> print(f"LR: {config['lr']}, Batch size: {config['batch_size']}")
    """
    metrics = load_metrics(metrics_path)
    hyper = metrics.get('hyper', {})

    return {
        'epochs': hyper.get('epochs'),
        'batch_size': hyper.get('batch_size'),
        'lr': hyper.get('lr'),
        'wd': hyper.get('wd'),
        'optimizer': hyper.get('optimizer'),
        'dropout': hyper.get('dropout'),
        'pretrained': hyper.get('pretrained')
    }


def get_training_history(metrics_path: str) -> Dict[str, np.ndarray]:
    """
    Extract full training history curves.

    Args:
        metrics_path: Path to metrics.json file

    Returns:
        Dictionary with keys 'train_loss', 'train_acc', 'val_loss', 'val_acc'
        mapping to numpy arrays of per-epoch values.

    Example:
        >>> history = get_training_history('results/resnet18/1/metrics.json')
        >>> train_acc = history['train_acc']  # [0.66, 0.75, 0.82, ...]
        >>> # Plot training curves
        >>> import matplotlib.pyplot as plt
        >>> plt.plot(train_acc, label='Train')
        >>> plt.plot(history['val_acc'], label='Val')
    """
    metrics = load_metrics(metrics_path)
    history = metrics.get('history', {})

    return {
        key: np.array(values) if values else np.array([])
        for key, values in history.items()
    }


def compare_runs(metrics_paths: List[str]) -> Dict[str, List[Any]]:
    """
    Compare multiple training runs side-by-side.

    Args:
        metrics_paths: List of paths to metrics.json files

    Returns:
        Dictionary with metrics as keys and lists of values as values.
        Each list has one entry per provided metrics file.

    Example:
        >>> paths = ['results/resnet18/1/metrics.json',
        ...          'results/resnet18/9/metrics.json']
        >>> comparison = compare_runs(paths)
        >>> print(comparison['best_val_acc'])  # [0.95, 0.78]
    """
    comparison = {
        'path': [],
        'model': [],
        'grid_size': [],
        'best_val_acc': [],
        'test_acc': [],
        'epochs': [],
        'lr': []
    }

    for path in metrics_paths:
        metrics = load_metrics(path)
        comparison['path'].append(path)
        comparison['model'].append(metrics.get('model'))
        comparison['grid_size'].append(metrics.get('grid_size'))
        comparison['best_val_acc'].append(metrics.get('best_val_accuracy'))
        comparison['test_acc'].append(metrics.get('test_accuracy'))
        comparison['epochs'].append(metrics.get('epochs'))
        comparison['lr'].append(metrics.get('hyper', {}).get('lr'))

    return comparison


def get_grid_info(metrics_path: str) -> Tuple[int, int]:
    """
    Get grid size and total number of tiles.

    Args:
        metrics_path: Path to metrics.json file

    Returns:
        Tuple of (grid_size, num_tiles)

    Example:
        >>> grid_size, num_tiles = get_grid_info('results/resnet18/9/metrics.json')
        >>> print(f"{grid_size}x{grid_size} grid with {num_tiles} tiles")
        # 3x3 grid with 9 tiles
    """
    metrics = load_metrics(metrics_path)
    grid_size = metrics.get('grid_size', 1)
    num_tiles = grid_size * grid_size
    return grid_size, num_tiles
