"""
Quick test of the experiments module with minimal epochs.

Run with: python test_experiments.py
"""
from pathlib import Path
import torch

from permuted_puzzle.experiments import (
    run_single_experiment,
    run_experiment_grid,
    print_experiment_summary
)


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


def test_single_experiment():
    """Test running a single experiment with 1 epoch"""
    print("\n" + "="*80)
    print("TEST 1: Single Experiment (ResNet18, 2x2 grid, 1 epoch)")
    print("="*80)

    data_path = prepare_dataset()
    results_root = Path("test_data/results")

    config = {
        "epochs": 1,  # Just 1 epoch for quick test
        "lr": 1e-4,
        "weight_decay": 1e-4,
        "batch_size": 4,  # Small batch for quick test
        "dropout": 0.2,
        "pretrained": True,  # Use ImageNet pretrained weights
        "seed": 42
    }

    device = "cuda" if torch.cuda.is_available() else "cpu"

    result = run_single_experiment(
        model_name="resnet18",
        grid_size=2,
        data_path=data_path,
        results_root=results_root,
        config=config,
        device=device
    )

    print("\nResult:")
    print(f"  Model: {result['model']}")
    print(f"  Grid Size: {result['grid_size']}x{result['grid_size']}")
    print(f"  Best Val Acc: {result['best_val_acc']:.4f}")
    print(f"  Test Acc: {result['test_acc']:.4f}")
    print(f"  Gap: {result['best_val_acc'] - result['test_acc']:+.4f}")

    # Verify files were created
    out_path = results_root / "resnet18" / "2"
    assert (out_path / "best.pth").exists(), "Model checkpoint not saved"
    assert (out_path / "metrics.json").exists(), "Metrics not saved"
    assert (out_path / "preds_val.npz").exists(), "Val predictions not saved"
    assert (out_path / "preds_test.npz").exists(), "Test predictions not saved"

    print("\nAll files created successfully!")
    return result


def test_experiment_grid():
    """Test running multiple experiments in a grid"""
    print("\n" + "="*80)
    print("TEST 2: Experiment Grid (2 models, 2 grid sizes, 1 epoch each)")
    print("="*80)

    data_path = prepare_dataset()
    results_root = Path("test_data/results_grid")

    config = {
        "epochs": 1,  # Just 1 epoch for quick test
        "lr": 1e-4,
        "weight_decay": 1e-4,
        "batch_size": 4,  # Small batch for quick test
        "dropout": 0.2,
        "pretrained": True,  # Use ImageNet pretrained weights
        "seed": 42
    }

    device = "cuda" if torch.cuda.is_available() else "cpu"

    results = run_experiment_grid(
        model_names=["resnet18"],  # Just one model for quick test
        grid_sizes=[1, 2],  # Baseline + one permuted
        data_path=data_path,
        results_root=results_root,
        config=config,
        device=device
    )

    print_experiment_summary(results)

    assert len(results) == 2, f"Expected 2 results, got {len(results)}"
    print(f"\nSuccessfully completed {len(results)} experiments!")

    return results


if __name__ == "__main__":
    print("\n" + "="*80)
    print(" EXPERIMENTS MODULE QUICK TEST")
    print("="*80)

    try:
        # Test 1: Single experiment
        print("\n[1/2] Testing single experiment...")
        test_single_experiment()

        # Test 2: Experiment grid
        print("\n[2/2] Testing experiment grid...")
        test_experiment_grid()

        print("\n" + "="*80)
        print(" ALL TESTS PASSED!")
        print("="*80 + "\n")

    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("Please ensure test data is in test_data/data/ directory")
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        raise
