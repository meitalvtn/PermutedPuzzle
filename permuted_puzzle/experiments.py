"""
Experiment orchestration utilities for systematic model evaluation.

This module provides high-level functions for running experiments across
multiple models and permutation configurations with proper train/val/test
evaluation methodology.
"""

import json
import traceback
from pathlib import Path
from typing import Dict, List, Any, Optional

import numpy as np
import torch

from .data import DogsVsCatsDataset, split_indices, generate_permutation, create_loader
from .models import build_model, get_model_config
from .transforms import baseline_train_transforms, baseline_val_transforms
from .train_utils import train_model, evaluate_model
from .utils_io import save_preds


def run_single_experiment(
    model_name: str,
    grid_size: int,
    data_path: str,
    results_root: Path,
    config: Dict[str, Any],
    device: str = "cuda",
    permutation: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """
    Run a single training experiment with train/val/test evaluation.

    This function encapsulates a complete experiment:
    1. Loads data and creates train/val/test splits
    2. Uses provided permutation or generates one for the specified grid size
    3. Trains model using train/val sets
    4. Evaluates on held-out test set
    5. Saves all outputs (model, metrics, predictions)

    Args:
        model_name: Name of model architecture (e.g., 'resnet18', 'resnet50')
        grid_size: Grid size for permutation (1 = baseline/no permutation)
        data_path: Path to training data directory
        results_root: Root directory for saving results
        config: Dict with training hyperparameters:
                - epochs: Number of training epochs
                - lr: Learning rate
                - weight_decay: Weight decay for optimizer
                - batch_size: Batch size
                - dropout: Dropout rate
                - pretrained: Whether to use ImageNet pretrained weights (default: True)
        device: Device to train on ('cuda' or 'cpu')
        permutation: Optional pre-generated permutation array. If provided, this will be used
                     instead of generating a new random permutation. Must have length grid_size^2.

    Returns:
        Dict containing:
            - model: Model name
            - grid_size: Grid size used
            - best_val_acc: Best validation accuracy during training
            - best_epoch: Epoch with best validation accuracy
            - test_acc: Final test set accuracy
            - test_loss: Final test set loss

    Raises:
        Exception: If experiment fails (propagated to caller)
    """
    print(f"\n{'='*80}")
    print(f"EXPERIMENT: {model_name} | Grid {grid_size}x{grid_size}")
    print(f"{'='*80}")

    # Get pretrained setting (default to True for ImageNet weights)
    pretrained = config.get('pretrained', True)
    print(f"Model: {model_name} ({'pretrained on ImageNet' if pretrained else 'training from scratch'})")

    # Setup output path: results_root/model_name/grid_size/
    out_path = results_root / model_name / str(grid_size)
    out_path.mkdir(parents=True, exist_ok=True)
    print(f"Output: {out_path}")

    # Load dataset and split (60/20/20 train/val/test)
    dataset = DogsVsCatsDataset(data_path)
    splits = split_indices(len(dataset), splits=[0.6, 0.2, 0.2])
    print(f"Split: {len(splits['train'])} train, {len(splits['val'])} val, {len(splits['test'])} test")

    # Get model config
    model_config = get_model_config(model_name)
    train_tfms = baseline_train_transforms(
        model_config['input_size'],
        model_config['mean'],
        model_config['std']
    )
    val_tfms = baseline_val_transforms(
        model_config['input_size'],
        model_config['mean'],
        model_config['std']
    )

    # Use provided permutation or generate one (None for baseline)
    if permutation is not None:
        # Validate that provided permutation matches grid_size
        expected_len = grid_size * grid_size
        if len(permutation) != expected_len:
            raise ValueError(
                f"Permutation length {len(permutation)} does not match grid_size "
                f"{grid_size} (expected {expected_len})"
            )
        perm = permutation
        print(f"Using provided permutation: {perm}")
    elif grid_size == 1:
        perm = None
    else:
        perm = generate_permutation(grid_size=grid_size)
        print(f"Generated random permutation: {perm}")

    # Create loaders (including test)
    train_loader = create_loader(
        dataset, splits['train'], train_tfms,
        permutation=perm,
        batch_size=config['batch_size'],
        shuffle=True
    )
    val_loader = create_loader(
        dataset, splits['val'], val_tfms,
        permutation=perm,
        batch_size=config['batch_size']
    )
    test_loader = create_loader(
        dataset, splits['test'], val_tfms,
        permutation=perm,
        batch_size=config['batch_size']
    )

    # Build model
    model = build_model(
        model_name,
        num_classes=2,
        pretrained=pretrained,
        dropout=config['dropout']
    )

    # Train (using train/val only for model selection)
    train_results = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=config['epochs'],
        lr=config['lr'],
        weight_decay=config['weight_decay'],
        device=device,
        out_path=str(out_path),
        config={
            'model_name': model_name,
            'grid_size': grid_size,
            'train_permutation': perm,
            'val_permutation': perm,
            'test_permutation': perm,
            'meta': model_config,
            'split_indices': splits,
            'num_classes': 2,
            'dropout': config['dropout'],
            'pretrained': pretrained
        }
    )

    # Evaluate on test set
    print(f"\n--- Evaluating on test set ---")
    test_metrics = evaluate_model(train_results['model'], test_loader, device=device)
    print(f"Test Loss: {test_metrics['loss']:.4f}, Test Acc: {test_metrics['accuracy']:.4f}")

    # Save test predictions
    save_preds(
        results_root=str(out_path),
        model_name=model_name,
        grid_size=grid_size,
        split="test",
        model=train_results['model'],
        loader=test_loader,
        device=device
    )

    # Update metrics.json to include test results
    metrics_path = out_path / "metrics.json"
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)

    metrics['test_accuracy'] = float(test_metrics['accuracy'])
    metrics['test_loss'] = float(test_metrics['loss'])

    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"\nCompleted: {model_name} | Grid {grid_size}x{grid_size}")
    print(f"  Best Val Acc: {train_results['best_val_acc']:.4f} @ epoch {train_results['best_epoch']}")
    print(f"  Test Acc:     {test_metrics['accuracy']:.4f}")

    return {
        'model': model_name,
        'grid_size': grid_size,
        'best_val_acc': train_results['best_val_acc'],
        'best_epoch': train_results['best_epoch'],
        'test_acc': test_metrics['accuracy'],
        'test_loss': test_metrics['loss']
    }


def run_experiment_grid(
    model_names: List[str],
    grid_sizes: List[int],
    data_path: str,
    results_root: Path,
    config: Dict[str, Any],
    pretrained: bool = True,
    device: str = "cuda"
) -> List[Dict[str, Any]]:
    """
    Run a grid of experiments over multiple models and grid sizes.

    Iterates through all combinations of model architectures and grid sizes,
    running a complete experiment for each combination. Failed experiments
    are logged but do not stop execution of remaining experiments.

    Args:
        model_names: List of model architecture names (e.g., ['resnet18', 'resnet50'])
        grid_sizes: List of grid sizes to test (e.g., [1, 2, 4, 7])
        data_path: Path to training data directory
        results_root: Root directory for saving results
        config: Dict with training hyperparameters (see run_single_experiment)
        pretrained: Whether to use ImageNet pretrained weights (default: True)
        device: Device to train on ('cuda' or 'cpu')

    Returns:
        List of result dicts from successful experiments. Each dict contains:
            - model: Model name
            - grid_size: Grid size used
            - best_val_acc: Best validation accuracy
            - best_epoch: Epoch with best validation accuracy
            - test_acc: Test set accuracy
            - test_loss: Test set loss
    """
    all_results = []

    for model_name in model_names:
        for grid_size in grid_sizes:
            try:
                # Create config copy and set pretrained from function parameter
                experiment_config = config.copy()
                experiment_config['pretrained'] = pretrained

                results = run_single_experiment(
                    model_name=model_name,
                    grid_size=grid_size,
                    data_path=data_path,
                    results_root=results_root,
                    config=experiment_config,
                    device=device
                )
                all_results.append(results)
            except Exception as e:
                print(f"\nERROR in {model_name} | Grid {grid_size}x{grid_size}: {e}")
                traceback.print_exc()
                continue

    return all_results


def print_experiment_summary(results: List[Dict[str, Any]]) -> None:
    """
    Print formatted summary table of experiment results.

    Displays a table showing model name, grid size, validation accuracy,
    test accuracy, and the generalization gap (val - test).

    Args:
        results: List of experiment result dicts from run_experiment_grid
    """
    if not results:
        print("\nNo results to display.")
        return

    print("\n" + "="*80)
    print("EXPERIMENT SUMMARY")
    print("="*80)
    print(f"{'Model':<12} | {'Grid':<8} | {'Best Val Acc':<12} | {'Test Acc':<10} | {'Gap':<8}")
    print("-" * 80)
    for result in results:
        gap = result['best_val_acc'] - result['test_acc']
        print(f"{result['model']:<12} | {result['grid_size']:2d}x{result['grid_size']:2d}    | "
              f"{result['best_val_acc']:.4f}       | {result['test_acc']:.4f}     | {gap:+.4f}")
    print("="*80)
