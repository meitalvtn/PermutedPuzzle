"""
Test and demonstrate metrics_utils functionality.

This script shows how to use the helper functions to extract information
from metrics.json files.
"""

from pathlib import Path
from permuted_puzzle.metrics_utils import (
    get_split_indices,
    get_permutation,
    get_all_permutations,
    get_model_info,
    get_performance_metrics,
    get_training_config,
    get_training_history,
    get_grid_info,
    compare_runs
)


def test_single_run():
    """Test extracting information from a single metrics file."""
    print("=" * 70)
    print("TEST 1: Reading Single Metrics File")
    print("=" * 70)

    metrics_path = "test_data/results_grid/resnet18/1/metrics.json"

    # 1. Get split indices
    print("\n1. Split Indices:")
    indices = get_split_indices(metrics_path)
    for split, idx_array in indices.items():
        print(f"   {split:5s}: {len(idx_array):3d} samples - {idx_array[:5]}...")

    # 2. Get permutation
    print("\n2. Permutation:")
    perm = get_permutation(metrics_path)
    if perm is None:
        print("   None (baseline model, no permutation)")
    else:
        print(f"   {perm}")

    # 3. Get model info
    print("\n3. Model Info:")
    info = get_model_info(metrics_path)
    print(f"   Architecture:  {info['model']}")
    print(f"   Grid size:     {info['grid_size']}x{info['grid_size']}")
    print(f"   Input size:    {info['input_size']}x{info['input_size']}")
    print(f"   Weights:       {Path(info['weights_path']).name}")
    print(f"   Saved at:      {info['saved_at']}")

    # 4. Get performance metrics
    print("\n4. Performance Metrics:")
    perf = get_performance_metrics(metrics_path)
    print(f"   Best val acc:  {perf['best_val_acc']:.2%}")
    print(f"   Final val acc: {perf['final_val_acc']:.2%}")
    if perf['test_acc'] is not None:
        print(f"   Test acc:      {perf['test_acc']:.2%}")

    # 5. Get training config
    print("\n5. Training Config:")
    config = get_training_config(metrics_path)
    print(f"   Epochs:        {config['epochs']}")
    print(f"   Batch size:    {config['batch_size']}")
    print(f"   Learning rate: {config['lr']}")
    print(f"   Weight decay:  {config['wd']}")
    print(f"   Optimizer:     {config['optimizer']}")

    # 6. Get training history
    print("\n6. Training History:")
    history = get_training_history(metrics_path)
    print(f"   Train acc:     {history['train_acc']}")
    print(f"   Val acc:       {history['val_acc']}")

    # 7. Get grid info
    print("\n7. Grid Info:")
    grid_size, num_tiles = get_grid_info(metrics_path)
    print(f"   Grid: {grid_size}x{grid_size} = {num_tiles} tiles")

    print("\n" + "=" * 70)
    print("Test 1 PASSED")
    print("=" * 70)


def test_permutation_formats():
    """Test handling both legacy and new permutation formats."""
    print("\n" + "=" * 70)
    print("TEST 2: Permutation Formats")
    print("=" * 70)

    metrics_path = "test_data/results_grid/resnet18/1/metrics.json"

    print("\nAll permutations:")
    all_perms = get_all_permutations(metrics_path)
    for split, perm in all_perms.items():
        if perm is None:
            print(f"   {split:5s}: None (baseline)")
        else:
            print(f"   {split:5s}: {perm}")

    print("\nSpecific split permutations:")
    for split in ['train', 'val', 'test']:
        perm = get_permutation(metrics_path, split=split)
        if perm is None:
            print(f"   {split:5s}: None")
        else:
            print(f"   {split:5s}: {perm}")

    print("\n" + "=" * 70)
    print("Test 2 PASSED")
    print("=" * 70)


def test_compare_runs():
    """Test comparing multiple runs."""
    print("\n" + "=" * 70)
    print("TEST 3: Comparing Multiple Runs")
    print("=" * 70)

    # Use the same file multiple times for demo (normally you'd use different runs)
    metrics_paths = [
        "test_data/results_grid/resnet18/1/metrics.json",
        "test_data/results_grid/resnet18/1/metrics.json",
    ]

    comparison = compare_runs(metrics_paths)

    print("\nComparison Table:")
    print(f"{'Model':<15} {'Grid':<8} {'Best Val Acc':<15} {'Test Acc':<15}")
    print("-" * 60)
    for i in range(len(metrics_paths)):
        model = comparison['model'][i]
        grid = comparison['grid_size'][i]
        best_val = comparison['best_val_acc'][i]
        test_acc = comparison['test_acc'][i]
        best_val_str = f"{best_val:.2%}" if best_val is not None else "N/A"
        test_acc_str = f"{test_acc:.2%}" if test_acc is not None else "N/A"
        print(f"{model:<15} {grid}x{grid:<6} {best_val_str:<15} {test_acc_str:<15}")

    print("\n" + "=" * 70)
    print("Test 3 PASSED")
    print("=" * 70)


def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("METRICS UTILS TEST SUITE")
    print("=" * 70)

    try:
        test_single_run()
        test_permutation_formats()
        test_compare_runs()

        print("\n" + "=" * 70)
        print("ALL TESTS PASSED!")
        print("=" * 70)

    except Exception as e:
        print(f"\nTEST FAILED: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
