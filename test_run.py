"""
Example script demonstrating the clean API for permutation experiments.
Run with: python test_run.py
"""
from pathlib import Path
import torch

from permuted_puzzle.data import (
    DogsVsCatsDataset,
    split_indices,
    generate_permutation,
    create_loader
)
from permuted_puzzle.models import build_model, get_model_config
from permuted_puzzle.transforms import baseline_train_transforms, baseline_val_transforms
from permuted_puzzle.train_utils import train_model, evaluate_model


def prepare_dataset(root="test_data"):
    """Prepare dataset from test_data/data/ directory"""
    root = Path(root)
    data_dir = root / "data"
    if not data_dir.exists() or not any(data_dir.iterdir()):
        raise FileNotFoundError(
            f"No test data found at {data_dir}. "
            "Please place your sample images there."
        )
    return str(data_dir)


def example_1_baseline():
    """Example 1: Baseline training (no permutation)"""
    print("\n" + "="*80)
    print("EXAMPLE 1: BASELINE (NO PERMUTATION)")
    print("="*80)

    # Setup
    data_path = prepare_dataset()
    dataset = DogsVsCatsDataset(data_path)
    indices = split_indices(len(dataset), splits=[0.7, 0.3], seed=0)  # train/val only

    config = get_model_config("resnet18")
    train_tfms = baseline_train_transforms(config['input_size'], config['mean'], config['std'])
    val_tfms = baseline_val_transforms(config['input_size'], config['mean'], config['std'])

    # Create loaders (no permutation)
    train_loader = create_loader(
        dataset, indices['train'], train_tfms,
        permutation=None, batch_size=8, shuffle=True
    )
    val_loader = create_loader(
        dataset, indices['val'], val_tfms,
        permutation=None, batch_size=8
    )

    # Build and train
    model = build_model("resnet18", num_classes=2, pretrained=True, dropout=0.2)

    results = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=1,
        lr=1e-4,
        device="cuda" if torch.cuda.is_available() else "cpu",
        out_path="test_data/results/baseline",
        config={
            'model_name': 'resnet18',
            'grid_size': 1,
            'meta': config,
            'split_indices': indices  # For reproducibility
        }
    )

    print(f"\nBaseline complete! Best val acc: {results['best_val_acc']:.4f}")
    return results


def example_2_single_permutation():
    """Example 2: Train and validate on same permutation"""
    print("\n" + "="*80)
    print("EXAMPLE 2: TRAIN WITH PERMUTATION (3x3 GRID)")
    print("="*80)

    # Setup
    data_path = prepare_dataset()
    dataset = DogsVsCatsDataset(data_path)
    indices = split_indices(len(dataset), splits=[0.7, 0.3], seed=0)

    # Generate permutation
    perm = generate_permutation(grid_size=3, seed=42)
    print(f"\nPermutation (seed=42): {perm}")

    config = get_model_config("resnet18")
    train_tfms = baseline_train_transforms(config['input_size'], config['mean'], config['std'])
    val_tfms = baseline_val_transforms(config['input_size'], config['mean'], config['std'])

    # Create loaders with same permutation
    train_loader = create_loader(
        dataset, indices['train'], train_tfms,
        permutation=perm, grid_size=3, batch_size=8, shuffle=True
    )
    val_loader = create_loader(
        dataset, indices['val'], val_tfms,
        permutation=perm, grid_size=3, batch_size=8
    )

    # Build and train
    model = build_model("resnet18", num_classes=2, pretrained=True, dropout=0.2)

    results = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=2,
        lr=1e-4,
        device="cuda" if torch.cuda.is_available() else "cpu",
        out_path="test_data/results/permuted_3x3",
        config={
            'model_name': 'resnet18',
            'grid_size': 3,
            'train_permutation': perm,
            'val_permutation': perm,
            'meta': config,
            'split_indices': indices  # For reproducibility
        }
    )

    print(f"\nPermuted training complete! Best val acc: {results['best_val_acc']:.4f}")
    return results


def example_3_transfer_test():
    """Example 3: Transfer test - train on PermA, test on PermB"""
    print("\n" + "="*80)
    print("EXAMPLE 3: TRANSFER TEST (TRAIN=PERM_A, TEST=PERM_B)")
    print("="*80)

    # Setup
    data_path = prepare_dataset()
    dataset = DogsVsCatsDataset(data_path)
    indices = split_indices(len(dataset), splits=[0.6, 0.2, 0.2], seed=0)  # train/val/test

    # Generate two different permutations
    perm_a = generate_permutation(grid_size=3, seed=0)
    perm_b = generate_permutation(grid_size=3, seed=1)
    print(f"\nPermutation A (train/val): {perm_a}")
    print(f"Permutation B (test):      {perm_b}")

    config = get_model_config("resnet18")
    train_tfms = baseline_train_transforms(config['input_size'], config['mean'], config['std'])
    val_tfms = baseline_val_transforms(config['input_size'], config['mean'], config['std'])

    # Create loaders
    train_loader = create_loader(
        dataset, indices['train'], train_tfms,
        permutation=perm_a, grid_size=3, batch_size=8, shuffle=True
    )
    val_loader = create_loader(
        dataset, indices['val'], val_tfms,
        permutation=perm_a, grid_size=3, batch_size=8
    )
    test_loader = create_loader(
        dataset, indices['test'], val_tfms,
        permutation=perm_b, grid_size=3, batch_size=8
    )

    # Build and train on PermA
    model = build_model("resnet18", num_classes=2, pretrained=True, dropout=0.2)

    results = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=1,
        lr=1e-4,
        device="cuda" if torch.cuda.is_available() else "cpu",
        out_path="test_data/results/transfer_test",
        config={
            'model_name': 'resnet18',
            'grid_size': 3,
            'train_permutation': perm_a,
            'val_permutation': perm_a,
            'test_permutation': perm_b,  # Save test permutation too
            'meta': config,
            'split_indices': indices  # For reproducibility
        }
    )

    # Transfer test on PermB
    print("\n--- Evaluating on transfer test set (Perm B) ---")
    test_metrics = evaluate_model(results['model'], test_loader,
                                  device="cuda" if torch.cuda.is_available() else "cpu")

    print(f"\nTransfer test complete!")
    print(f"   Val Acc (Perm A):  {results['best_val_acc']:.4f}")
    print(f"   Test Acc (Perm B): {test_metrics['accuracy']:.4f}")
    print(f"   Transfer Gap:      {(results['best_val_acc'] - test_metrics['accuracy']):.4f}")

    return results, test_metrics


def example_4_multiple_grids():
    """Example 4: Compare different grid sizes"""
    print("\n" + "="*80)
    print("EXAMPLE 4: COMPARE DIFFERENT GRID SIZES")
    print("="*80)

    data_path = prepare_dataset()
    dataset = DogsVsCatsDataset(data_path)
    indices = split_indices(len(dataset), splits=[0.7, 0.3], seed=0)

    config = get_model_config("resnet18")
    train_tfms = baseline_train_transforms(config['input_size'], config['mean'], config['std'])
    val_tfms = baseline_val_transforms(config['input_size'], config['mean'], config['std'])

    results_summary = []

    for grid_size in [1, 2, 3]:
        print(f"\n--- Training with grid size {grid_size}x{grid_size} ---")

        if grid_size == 1:
            perm = None
        else:
            perm = generate_permutation(grid_size=grid_size, seed=0)

        train_loader = create_loader(
            dataset, indices['train'], train_tfms,
            permutation=perm, grid_size=grid_size, batch_size=8, shuffle=True
        )
        val_loader = create_loader(
            dataset, indices['val'], val_tfms,
            permutation=perm, grid_size=grid_size, batch_size=8
        )

        model = build_model("resnet18", num_classes=2, pretrained=True, dropout=0.2)

        results = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=1,  # Quick comparison
            lr=1e-4,
            device="cuda" if torch.cuda.is_available() else "cpu",
            out_path=f"test_data/results/grid_{grid_size}x{grid_size}",
            config={
                'model_name': 'resnet18',
                'grid_size': grid_size,
                'train_permutation': perm,
                'val_permutation': perm,
                'meta': config,
                'split_indices': indices  # For reproducibility
            }
        )

        results_summary.append((grid_size, results['best_val_acc']))

    print("\nGrid comparison complete!")
    print("\nResults Summary:")
    for grid_size, acc in results_summary:
        grid_str = f"{grid_size}x{grid_size}" if grid_size > 1 else "baseline"
        print(f"   Grid {grid_str:8s}: {acc:.4f}")


if __name__ == "__main__":
    print("\n" + "="*80)
    print(" PERMUTED PUZZLE - API EXAMPLES")
    print("="*80)

    try:
        # Run all examples
        print("\n[1/4] Running baseline example...")
        example_1_baseline()

        print("\n[2/4] Running single permutation example...")
        example_2_single_permutation()

        print("\n[3/4] Running transfer test example...")
        example_3_transfer_test()

        print("\n[4/4] Running grid comparison example...")
        example_4_multiple_grids()

        print("\n" + "="*80)
        print(" ALL EXAMPLES COMPLETED SUCCESSFULLY!")
        print("="*80 + "\n")

    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("Please ensure test data is in test_data/data/ directory")
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        raise
