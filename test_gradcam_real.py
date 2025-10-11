"""
End-to-end test of Grad-CAM module with real trained model and data.

This test:
1. Loads a trained ResNet18 baseline model
2. Creates a dataloader from test images
3. Runs generate_gradcam() on individual images
4. Runs run_gradcam_analysis() for batch visualization
5. Saves outputs to test_data/gradcam_output/
"""

from pathlib import Path
import torch
import matplotlib.pyplot as plt
import numpy as np

from permuted_puzzle.models import build_model, get_model_config
from permuted_puzzle.data import DogsVsCatsDataset, create_loader, split_indices
from permuted_puzzle.transforms import baseline_val_transforms
from permuted_puzzle.gradcam import generate_gradcam, run_gradcam_analysis


def test_single_image_gradcam():
    """Test Grad-CAM on a single image."""
    print("\n" + "="*60)
    print("TEST 1: Single Image Grad-CAM")
    print("="*60)

    # Paths
    model_path = Path("test_data/results_grid/resnet18/1/best.pth")
    data_dir = Path("test_data/data")
    output_dir = Path("test_data/gradcam_output/single")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check files exist
    if not model_path.exists():
        print(f"ERROR: Model not found at {model_path}")
        return False
    if not data_dir.exists():
        print(f"ERROR: Data directory not found at {data_dir}")
        return False

    print(f"Loading model from: {model_path}")
    print(f"Loading data from: {data_dir}")

    # Load model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    model = build_model('resnet18', num_classes=2, pretrained=False)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    print("Model loaded successfully")

    # Get model config for transforms
    config = get_model_config('resnet18')
    mean = config['mean']
    std = config['std']
    input_size = config['input_size']

    # Create dataset and get a single image
    transform = baseline_val_transforms(input_size, mean, std)
    dataset = DogsVsCatsDataset(str(data_dir), transform=transform)

    print(f"Dataset size: {len(dataset)} images")

    # Test on first cat and first dog
    for idx in [0, len(dataset)//2]:  # First image and middle image (likely different class)
        image, label = dataset[idx]
        class_name = 'cat' if label == 0 else 'dog'

        print(f"\nProcessing image {idx}: {class_name}")

        # Generate Grad-CAM
        heatmap, overlay = generate_gradcam(
            model,
            image,
            target_class=None,  # Use predicted class
            layer_name='layer4',
            device=device,
            mean=mean,
            std=std
        )

        # Get prediction
        with torch.no_grad():
            output = model(image.unsqueeze(0).to(device))
            pred = output.argmax(dim=1).item()
            confidence = torch.softmax(output, dim=1)[0, pred].item()

        pred_name = 'cat' if pred == 0 else 'dog'
        correct = "CORRECT" if pred == label else "INCORRECT"

        print(f"  True: {class_name}, Predicted: {pred_name} ({confidence:.2%}) - {correct}")
        print(f"  Heatmap shape: {heatmap.shape}, range: [{heatmap.min():.3f}, {heatmap.max():.3f}]")
        print(f"  Overlay shape: {overlay.shape}, dtype: {overlay.dtype}")

        # Denormalize original image for visualization
        img_np = image.numpy().transpose(1, 2, 0)
        img_np = (img_np * np.array(std).reshape(1, 1, 3)) + np.array(mean).reshape(1, 1, 3)
        img_np = np.clip(img_np * 255, 0, 255).astype(np.uint8)

        # Create visualization
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        axes[0].imshow(img_np)
        axes[0].set_title(f'Original\nTrue: {class_name}')
        axes[0].axis('off')

        axes[1].imshow(heatmap, cmap='jet')
        axes[1].set_title('Grad-CAM Heatmap')
        axes[1].axis('off')

        axes[2].imshow(overlay)
        axes[2].set_title(f'Overlay\nPred: {pred_name} ({confidence:.2%})')
        axes[2].axis('off')

        plt.suptitle(f'Grad-CAM Analysis - Image {idx} - {correct}', fontsize=14, fontweight='bold')
        plt.tight_layout()

        output_path = output_dir / f"gradcam_{class_name}_{idx}.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"  Saved visualization to: {output_path}")

    print("\nTest 1 PASSED: Single image Grad-CAM successful")
    return True


def test_batch_analysis():
    """Test batch Grad-CAM analysis with run_gradcam_analysis()."""
    print("\n" + "="*60)
    print("TEST 2: Batch Grad-CAM Analysis")
    print("="*60)

    # Paths
    model_path = Path("test_data/results_grid/resnet18/1/best.pth")
    data_dir = Path("test_data/data")
    output_dir = Path("test_data/gradcam_output/batch")

    print(f"Loading model from: {model_path}")
    print(f"Loading data from: {data_dir}")

    # Load model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    model = build_model('resnet18', num_classes=2, pretrained=False)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # Get model config
    config = get_model_config('resnet18')
    mean = config['mean']
    std = config['std']
    input_size = config['input_size']

    # Create dataset and dataloader
    transform = baseline_val_transforms(input_size, mean, std)
    dataset = DogsVsCatsDataset(str(data_dir), transform=transform)

    # Use all images as test set
    indices = np.arange(len(dataset))
    test_loader = create_loader(
        dataset,
        indices,
        transform,
        permutation=None,  # No permutation for baseline
        batch_size=8,
        shuffle=False
    )

    print(f"Dataset size: {len(dataset)} images")
    print(f"DataLoader batches: {len(test_loader)}")

    # Run batch analysis
    print("\nRunning batch Grad-CAM analysis...")
    results = run_gradcam_analysis(
        model,
        test_loader,
        num_per_class=3,  # 3 correct + 3 incorrect per class
        device=device,
        results_dir=output_dir,
        layer_name='layer4',
        mean=mean,
        std=std
    )

    print("\nAnalysis complete!")
    print(f"Correct predictions visualized: {len(results['correct'])}")
    print(f"Incorrect predictions visualized: {len(results['incorrect'])}")

    # Print summary
    print("\nCorrect predictions:")
    for i, item in enumerate(results['correct']):
        class_name = 'cat' if item['label'] == 0 else 'dog'
        print(f"  {i+1}. True: {class_name}, Pred: {class_name}")

    print("\nIncorrect predictions:")
    for i, item in enumerate(results['incorrect']):
        true_name = 'cat' if item['label'] == 0 else 'dog'
        pred_name = 'cat' if item['pred'] == 0 else 'dog'
        print(f"  {i+1}. True: {true_name}, Pred: {pred_name}")

    # Verify outputs
    if 'path' in results['correct'][0]:
        print(f"\nVisualizations saved to: {output_dir}")
        print("  Subdirectories: correct/, incorrect/")

    # Create summary visualization
    create_summary_plot(results, output_dir)

    print("\nTest 2 PASSED: Batch analysis successful")
    return True


def create_summary_plot(results, output_dir):
    """Create a summary grid of all visualizations."""
    print("\nCreating summary visualization...")

    n_correct = len(results['correct'])
    n_incorrect = len(results['incorrect'])
    total = n_correct + n_incorrect

    if total == 0:
        print("No results to visualize")
        return

    # Create grid: 2 rows (correct/incorrect), up to 6 columns
    n_cols = min(6, max(n_correct, n_incorrect))
    fig, axes = plt.subplots(2, n_cols, figsize=(4*n_cols, 8))

    if n_cols == 1:
        axes = axes.reshape(2, 1)

    # Plot correct predictions
    for i in range(n_cols):
        if i < n_correct:
            item = results['correct'][i]
            axes[0, i].imshow(item['overlay'])
            class_name = 'cat' if item['label'] == 0 else 'dog'
            axes[0, i].set_title(f'Correct: {class_name}', color='green', fontweight='bold')
        else:
            axes[0, i].axis('off')
        axes[0, i].set_xticks([])
        axes[0, i].set_yticks([])

    # Plot incorrect predictions
    for i in range(n_cols):
        if i < n_incorrect:
            item = results['incorrect'][i]
            axes[1, i].imshow(item['overlay'])
            true_name = 'cat' if item['label'] == 0 else 'dog'
            pred_name = 'cat' if item['pred'] == 0 else 'dog'
            axes[1, i].set_title(f'True: {true_name}\nPred: {pred_name}', color='red', fontweight='bold')
        else:
            axes[1, i].axis('off')
        axes[1, i].set_xticks([])
        axes[1, i].set_yticks([])

    plt.suptitle('Grad-CAM Analysis Summary\nTop: Correct Predictions | Bottom: Incorrect Predictions',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()

    summary_path = output_dir / 'summary.png'
    plt.savefig(summary_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Summary visualization saved to: {summary_path}")


def main():
    """Run all tests."""
    print("="*60)
    print("GRAD-CAM END-TO-END TEST WITH REAL DATA")
    print("="*60)

    success = True

    # Test 1: Single image
    try:
        if not test_single_image_gradcam():
            success = False
    except Exception as e:
        print(f"\nTest 1 FAILED with error: {e}")
        import traceback
        traceback.print_exc()
        success = False

    # Test 2: Batch analysis
    try:
        if not test_batch_analysis():
            success = False
    except Exception as e:
        print(f"\nTest 2 FAILED with error: {e}")
        import traceback
        traceback.print_exc()
        success = False

    # Final summary
    print("\n" + "="*60)
    if success:
        print("ALL TESTS PASSED!")
        print("Check test_data/gradcam_output/ for visualizations")
    else:
        print("SOME TESTS FAILED - see errors above")
    print("="*60)

    return success


if __name__ == "__main__":
    main()
