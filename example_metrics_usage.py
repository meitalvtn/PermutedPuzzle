"""
Example usage of metrics_utils in real scenarios.

This shows how to use the helper functions in typical research workflows.
"""

from permuted_puzzle.metrics_utils import (
    get_split_indices,
    get_permutation,
    get_model_info,
    get_performance_metrics
)
from permuted_puzzle.data import DogsVsCatsDataset, create_loader
from permuted_puzzle.transforms import baseline_val_transforms
from permuted_puzzle.models import build_model
import torch


def example_1_load_trained_model_and_data():
    """Example: Load a trained model and recreate its exact test set."""
    print("=" * 70)
    print("Example 1: Load Model and Recreate Test Set")
    print("=" * 70)

    metrics_path = "test_data/results_grid/resnet18/1/metrics.json"

    # Get model info
    info = get_model_info(metrics_path)
    print(f"\nModel: {info['model']}, Grid: {info['grid_size']}x{info['grid_size']}")

    # Get split indices
    indices = get_split_indices(metrics_path)
    test_indices = indices['test']
    print(f"Test set has {len(test_indices)} samples")

    # Get permutation
    perm = get_permutation(metrics_path, split='test')
    print(f"Test permutation: {perm}")

    # Load model
    model = build_model(info['model'], num_classes=2, pretrained=False)
    model.load_state_dict(torch.load(info['weights_path'], map_location='cpu'))
    model.eval()
    print("Model loaded successfully")

    # Recreate exact test dataloader
    dataset = DogsVsCatsDataset('test_data/data')
    transform = baseline_val_transforms(
        info['input_size'],
        info['mean'],
        info['std']
    )
    test_loader = create_loader(
        dataset,
        test_indices,
        transform,
        permutation=perm,
        batch_size=4,
        shuffle=False
    )
    print(f"Test loader created with {len(test_loader)} batches")

    print("\n✓ Ready to evaluate on exact test set used during training!")


def example_2_compare_baseline_vs_permuted():
    """Example: Compare baseline vs permuted model performance."""
    print("\n" + "=" * 70)
    print("Example 2: Compare Model Performance")
    print("=" * 70)

    baseline_path = "test_data/results_grid/resnet18/1/metrics.json"

    # Get performance for baseline
    perf = get_performance_metrics(baseline_path)
    info = get_model_info(baseline_path)

    print(f"\nBaseline (grid {info['grid_size']}x{info['grid_size']}):")
    print(f"  Best Val Acc:  {perf['best_val_acc']:.2%}")
    print(f"  Test Acc:      {perf['test_acc']:.2%}")

    # In a real scenario, you'd compare with permuted models:
    # permuted_path = "results/resnet18/9/metrics.json"
    # perf_permuted = get_performance_metrics(permuted_path)
    # info_permuted = get_model_info(permuted_path)
    # print(f"\nPermuted (grid {info_permuted['grid_size']}x{info_permuted['grid_size']}):")
    # print(f"  Best Val Acc:  {perf_permuted['best_val_acc']:.2%}")
    # print(f"  Test Acc:      {perf_permuted['test_acc']:.2%}")


def example_3_get_exact_training_data():
    """Example: Get the exact training indices used."""
    print("\n" + "=" * 70)
    print("Example 3: Access Exact Training Data")
    print("=" * 70)

    metrics_path = "test_data/results_grid/resnet18/1/metrics.json"

    # Get split indices
    indices = get_split_indices(metrics_path)
    train_indices = indices['train']

    print(f"\nTraining set indices ({len(train_indices)} samples):")
    print(f"  Indices: {train_indices}")

    # Load the exact training images
    dataset = DogsVsCatsDataset('test_data/data')
    print(f"\nFirst 3 training images:")
    for i, idx in enumerate(train_indices[:3]):
        img, label, filename = dataset[idx]
        class_name = 'cat' if label == 0 else 'dog'
        print(f"  {i+1}. Index {idx}: {filename} (label={label}, {class_name})")


if __name__ == "__main__":
    example_1_load_trained_model_and_data()
    example_2_compare_baseline_vs_permuted()
    example_3_get_exact_training_data()

    print("\n" + "=" * 70)
    print("All examples completed successfully!")
    print("=" * 70)
