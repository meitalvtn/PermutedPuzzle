"""
Simple test to verify Grad-CAM module implementation.
"""

import torch
import numpy as np
from permuted_puzzle.models import build_model
from permuted_puzzle.gradcam import generate_gradcam, GradCAMHook, compute_gradcam_heatmap


def test_imports():
    """Test that all imports work correctly."""
    print("Testing imports...")
    from permuted_puzzle.gradcam import generate_gradcam, run_gradcam_analysis
    from permuted_puzzle import generate_gradcam as api_gradcam
    from permuted_puzzle import run_gradcam_analysis as api_analysis
    print("All imports successful")


def test_gradcam_hook():
    """Test that GradCAMHook can attach to a model."""
    print("\nTesting GradCAMHook...")
    model = build_model('resnet18', pretrained=False)

    # Test hook creation
    with GradCAMHook(model, 'layer4') as hook:
        # Create dummy input
        dummy_input = torch.randn(1, 3, 224, 224)

        # Forward pass
        output = model(dummy_input)

        # Backward pass
        output[0, 0].backward()

        # Check that activations and gradients were captured
        assert hook.activations is not None, "Activations not captured"
        assert hook.gradients is not None, "Gradients not captured"
        assert hook.activations.shape[0] == 1, "Batch size should be 1"

        print(f"  Activations shape: {hook.activations.shape}")
        print(f"  Gradients shape: {hook.gradients.shape}")

    print("GradCAMHook test passed")


def test_compute_heatmap():
    """Test heatmap computation."""
    print("\nTesting compute_gradcam_heatmap...")

    # Create dummy activations and gradients
    activations = torch.randn(1, 512, 7, 7)
    gradients = torch.randn(1, 512, 7, 7)

    # Compute heatmap
    heatmap = compute_gradcam_heatmap(activations, gradients)

    # Check output properties
    assert isinstance(heatmap, np.ndarray), "Heatmap should be numpy array"
    assert heatmap.shape == (7, 7), f"Expected shape (7, 7), got {heatmap.shape}"
    assert heatmap.min() >= 0.0, "Heatmap values should be >= 0"
    assert heatmap.max() <= 1.0, "Heatmap values should be <= 1"

    print(f"  Heatmap shape: {heatmap.shape}")
    print(f"  Heatmap range: [{heatmap.min():.3f}, {heatmap.max():.3f}]")
    print("Heatmap computation test passed")


def test_generate_gradcam():
    """Test full Grad-CAM generation."""
    print("\nTesting generate_gradcam...")

    model = build_model('resnet18', pretrained=False)
    model.eval()

    # Create dummy image
    image_tensor = torch.randn(1, 3, 224, 224)

    # Generate Grad-CAM
    heatmap, overlay = generate_gradcam(
        model,
        image_tensor,
        target_class=0,
        layer_name='layer4',
        device='cpu'
    )

    # Check outputs
    assert isinstance(heatmap, np.ndarray), "Heatmap should be numpy array"
    assert isinstance(overlay, np.ndarray), "Overlay should be numpy array"
    assert heatmap.ndim == 2, f"Heatmap should be 2D, got {heatmap.ndim}D"
    assert overlay.ndim == 3, f"Overlay should be 3D (H, W, C), got {overlay.ndim}D"
    assert overlay.shape[2] == 3, f"Overlay should have 3 channels, got {overlay.shape[2]}"
    assert overlay.dtype == np.uint8, f"Overlay should be uint8, got {overlay.dtype}"

    print(f"  Heatmap shape: {heatmap.shape}")
    print(f"  Overlay shape: {overlay.shape}")
    print(f"  Overlay range: [{overlay.min()}, {overlay.max()}]")
    print("generate_gradcam test passed")


def test_layer_detection():
    """Test that invalid layer names raise errors."""
    print("\nTesting layer detection...")

    model = build_model('resnet18', pretrained=False)

    try:
        with GradCAMHook(model, 'nonexistent_layer') as hook:
            pass
        assert False, "Should have raised ValueError for nonexistent layer"
    except ValueError as e:
        print(f"  Correctly raised error: {e}")

    print("Layer detection test passed")


if __name__ == "__main__":
    print("Running Grad-CAM module tests...\n")
    print("=" * 60)

    test_imports()
    test_gradcam_hook()
    test_compute_heatmap()
    test_generate_gradcam()
    test_layer_detection()

    print("\n" + "=" * 60)
    print("All tests passed successfully!")
