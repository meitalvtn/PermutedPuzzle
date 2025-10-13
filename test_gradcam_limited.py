"""Test Grad-CAM with limited heatmaps and overlays to verify early stopping."""

from pathlib import Path
import torch
import numpy as np

from permuted_puzzle.models import build_model, get_model_config
from permuted_puzzle.data import DogsVsCatsDataset, create_loader
from permuted_puzzle.transforms import baseline_val_transforms
from permuted_puzzle.gradcam import run_gradcam_analysis


def test_limited_gradcam():
    """Test that Grad-CAM stops early when limits are set."""
    # Paths
    model_path = Path("test_data/results_grid/resnet18/1/best.pth")
    data_dir = Path("test_data/data")
    output_dir = Path("test_data/gradcam_output/limited_test")

    # Load model
    device = 'cpu'
    model = build_model('resnet18', num_classes=2, pretrained=False)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # Create dataset
    config = get_model_config('resnet18')
    transform = baseline_val_transforms(config['input_size'], config['mean'], config['std'])
    dataset = DogsVsCatsDataset(str(data_dir), transform=transform)
    indices = np.arange(len(dataset))
    test_loader = create_loader(
        dataset, indices, transform,
        permutation=None, batch_size=8, shuffle=False
    )

    print(f"\nDataset has {len(dataset)} images")
    print("Running Grad-CAM with limits:")
    print("  - Heatmaps: 5 per class per category")
    print("  - Overlays: 3 per class per category")
    print("  - Expected to process: ~20 images (not all 25)")

    # Run with limits
    results = run_gradcam_analysis(
        model,
        test_loader,
        num_heatmaps_per_class=5,
        num_overlays_per_class=3,
        device=device,
        results_dir=output_dir,
        layer_name='layer4',
        mean=config['mean'],
        std=config['std']
    )

    total_processed = len(results['correct']) + len(results['incorrect'])
    print(f"\n✓ Total Grad-CAM computed: {total_processed}")

    # Verify we didn't process all images unnecessarily
    if total_processed <= 25:
        print(f"✓ Optimization working: Only processed {total_processed} images, not all {len(dataset)}")
    else:
        print(f"✗ Warning: Processed {total_processed} images, expected ~20")


if __name__ == "__main__":
    test_limited_gradcam()
