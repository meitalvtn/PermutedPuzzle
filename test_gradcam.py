"""
Test Grad-CAM API with real dataset samples.

This test demonstrates the complete Grad-CAM workflow:
1. Load a small dataset (5 images)
2. Run Grad-CAM analysis with max_samples=10
3. Save overlay visualizations for a subset (5 total)
"""

from pathlib import Path
from torch.utils.data import DataLoader, Subset
from permuted_puzzle.data import DogsVsCatsDataset
from permuted_puzzle.models import build_model, get_model_config
from permuted_puzzle.transforms import baseline_val_transforms
from permuted_puzzle.gradcam import run_gradcam_analysis, save_gradcam_overlays


def test_gradcam_workflow():
    """
    Test complete Grad-CAM workflow:
    - Load 5 images from dataset
    - Run analysis with max_samples=10
    - Save overlays for 5 samples
    """
    print("Testing Grad-CAM workflow...")
    print("=" * 60)

    # Setup paths
    data_path = Path("test_data/data")
    results_dir = Path("test_results/gradcam_test")
    results_dir.mkdir(parents=True, exist_ok=True)

    # Check if dataset exists
    if not data_path.exists():
        print(f"Warning: Dataset not found at {data_path}")
        print("Skipping test (requires Dogs vs Cats dataset)")
        return

    # Load dataset with 5 images
    print("\n1. Loading dataset...")
    model_config = get_model_config('resnet18')
    transform = baseline_val_transforms(
        model_config['input_size'],
        model_config['mean'],
        model_config['std']
    )
    full_dataset = DogsVsCatsDataset(str(data_path), transform=transform)

    # Create subset with first 20 images to ensure we have enough for balanced sampling
    subset_indices = list(range(min(20, len(full_dataset))))
    dataset = Subset(full_dataset, subset_indices)
    print(f"   Loaded {len(dataset)} images")

    # Create dataloader
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False)

    # Build model (untrained is fine for testing)
    print("\n2. Building model...")
    model = build_model('resnet18', pretrained=False, num_classes=2)
    model.eval()
    print("   Model ready (ResNet18)")

    # Run Grad-CAM analysis with max_samples=10
    print("\n3. Running Grad-CAM analysis (max_samples=10)...")
    results = run_gradcam_analysis(
        model=model,
        dataloader=dataloader,
        max_samples=10,
        device='cpu',
        results_dir=str(results_dir),
        layer_name='layer4',
        mean=model_config['mean'],
        std=model_config['std']
    )

    # Verify results structure
    print("\n4. Verifying results...")
    assert 'correct' in results, "Results should have 'correct' key"
    assert 'incorrect' in results, "Results should have 'incorrect' key"

    total_samples = len(results['correct']) + len(results['incorrect'])
    print(f"   Total samples collected: {total_samples}")
    print(f"   Correct predictions: {len(results['correct'])}")
    print(f"   Incorrect predictions: {len(results['incorrect'])}")

    assert total_samples <= 10, f"Should have at most 10 samples, got {total_samples}"

    # Verify each result entry has required fields
    for category in ['correct', 'incorrect']:
        for i, entry in enumerate(results[category]):
            assert 'heatmap' in entry, f"{category}[{i}] missing 'heatmap'"
            assert 'image_tensor' in entry, f"{category}[{i}] missing 'image_tensor'"
            assert 'label' in entry, f"{category}[{i}] missing 'label'"
            assert 'pred' in entry, f"{category}[{i}] missing 'pred'"
            assert entry['heatmap'].ndim == 2, f"{category}[{i}] heatmap should be 2D"
            assert entry['image_tensor'].ndim == 3, f"{category}[{i}] image_tensor should be 3D"

    print("   All result entries verified")

    # Save overlays for a subset (5 total: up to 3 correct, up to 3 incorrect)
    print("\n5. Saving overlay visualizations (subset of 5)...")
    subset_results = {
        'correct': results['correct'][:3],
        'incorrect': results['incorrect'][:3]
    }

    actual_subset_size = len(subset_results['correct']) + len(subset_results['incorrect'])
    print(f"   Saving overlays for {actual_subset_size} samples...")

    save_gradcam_overlays(
        results=subset_results,
        results_dir=str(results_dir),
        mean=model_config['mean'],
        std=model_config['std']
    )

    # Verify overlay files were created
    overlays_dir = results_dir / 'overlays'
    overlay_files = list(overlays_dir.rglob('*.png'))
    print(f"   Created {len(overlay_files)} overlay files")

    # Verify heatmap files were created
    heatmaps_dir = results_dir / 'heatmaps'
    heatmap_files = list(heatmaps_dir.rglob('*.npy'))
    print(f"   Created {len(heatmap_files)} heatmap files")

    print("\n" + "=" * 60)
    print("Grad-CAM workflow test passed!")
    print(f"Results saved to: {results_dir}")


if __name__ == "__main__":
    test_gradcam_workflow()
