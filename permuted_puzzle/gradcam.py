"""
Grad-CAM visualization module for model interpretability.

This module provides tools to visualize which regions of an image
a trained CNN attends to when making predictions.
"""

from typing import Tuple, Dict, List, Optional, Union
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import cv2
from tqdm import tqdm


class GradCAMHook:
    """
    Hook manager for capturing feature maps and gradients.

    This class registers forward and backward hooks on a target layer
    to capture the activations and gradients needed for Grad-CAM computation.

    Args:
        model: PyTorch model
        layer_name: Name of the target convolutional layer (e.g., 'layer4')

    Attributes:
        activations: Forward activations from the target layer
        gradients: Gradients flowing back through the target layer
    """

    def __init__(self, model: nn.Module, layer_name: str):
        self.model = model
        self.layer_name = layer_name
        self.activations = None
        self.gradients = None
        self.handles = []

        target_layer = self._get_layer_by_name(model, layer_name)
        if target_layer is None:
            raise ValueError(f"Layer '{layer_name}' not found in model")

        # Register hooks
        self.handles.append(
            target_layer.register_forward_hook(self._forward_hook)
        )
        self.handles.append(
            target_layer.register_full_backward_hook(self._backward_hook)
        )

    def _get_layer_by_name(self, model: nn.Module, layer_name: str) -> Optional[nn.Module]:
        """
        Retrieve a layer from the model by its name.

        Args:
            model: PyTorch model
            layer_name: Name of the layer

        Returns:
            The layer module if found, None otherwise
        """
        for name, module in model.named_modules():
            if name == layer_name:
                return module
        return None

    def _forward_hook(self, module: nn.Module, input: Tuple, output: torch.Tensor) -> None:
        """
        Hook to capture forward activations.

        Args:
            module: The layer module
            input: Input tensors to the layer
            output: Output tensor from the layer
        """
        self.activations = output.detach()

    def _backward_hook(self, module: nn.Module, grad_input: Tuple, grad_output: Tuple) -> None:
        """
        Hook to capture gradients during backpropagation.

        Args:
            module: The layer module
            grad_input: Gradients with respect to inputs
            grad_output: Gradients with respect to outputs
        """
        self.gradients = grad_output[0].detach()

    def remove_hooks(self) -> None:
        """
        Remove all registered hooks.
        """
        for handle in self.handles:
            handle.remove()
        self.handles = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.remove_hooks()


def compute_gradcam_heatmap(
    activations: torch.Tensor,
    gradients: torch.Tensor
) -> np.ndarray:
    """
    Compute Grad-CAM heatmap from activations and gradients.

    Args:
        activations: Feature maps from target layer, shape (1, C, H, W)
        gradients: Gradients of target class w.r.t. activations, shape (1, C, H, W)

    Returns:
        Heatmap as numpy array, shape (H, W), normalized to [0, 1]
    """
    # Global average pooling of gradients to get weights
    weights = torch.mean(gradients, dim=(2, 3), keepdim=True)  # (1, C, 1, 1)

    # Weighted combination of activation maps
    weighted_activations = weights * activations  # (1, C, H, W)
    cam = torch.sum(weighted_activations, dim=1).squeeze(0)  # (H, W)

    # Apply ReLU to focus on features with positive influence
    cam = F.relu(cam)

    # Normalize to [0, 1]
    cam = cam.cpu().numpy()
    if cam.max() > 0:
        cam = cam / cam.max()

    return cam


def overlay_heatmap_on_image(
    image_tensor: torch.Tensor,
    heatmap: np.ndarray,
    mean: List[float],
    std: List[float],
    colormap: int = cv2.COLORMAP_JET,
    alpha: float = 0.4
) -> np.ndarray:
    """
    Overlay Grad-CAM heatmap on original image.

    Args:
        image_tensor: Original image tensor, shape (C, H, W), normalized
        heatmap: Grad-CAM heatmap, shape (H, W), values in [0, 1]
        mean: Normalization mean used during preprocessing
        std: Normalization std used during preprocessing
        colormap: OpenCV colormap to use for heatmap
        alpha: Transparency for overlay (0=only image, 1=only heatmap)

    Returns:
        Overlay image as numpy array, shape (H, W, 3), uint8
    """
    # Denormalize image tensor
    image_np = image_tensor.detach().cpu().numpy().transpose(1, 2, 0)  # (H, W, C)
    mean_arr = np.array(mean).reshape(1, 1, 3)
    std_arr = np.array(std).reshape(1, 1, 3)
    image_np = (image_np * std_arr) + mean_arr
    image_np = np.clip(image_np * 255, 0, 255).astype(np.uint8)

    # Resize heatmap to match image dimensions
    h, w = image_np.shape[:2]
    heatmap_resized = cv2.resize(heatmap, (w, h))

    # Apply colormap
    heatmap_colored = cv2.applyColorMap(
        (heatmap_resized * 255).astype(np.uint8),
        colormap
    )
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    # Blend heatmap with original image
    overlay = (alpha * heatmap_colored + (1 - alpha) * image_np).astype(np.uint8)

    return overlay


def generate_gradcam(
    model: nn.Module,
    image_tensor: torch.Tensor,
    target_class: Optional[int] = None,
    layer_name: str = 'layer4',
    device: str = 'cuda',
    mean: List[float] = [0.485, 0.456, 0.406],
    std: List[float] = [0.229, 0.224, 0.225]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate Grad-CAM heatmap and overlay for a single image.

    Args:
        model: Trained PyTorch model
        image_tensor: Normalized image tensor, shape (1, C, H, W) or (C, H, W)
        target_class: Class index to compute Grad-CAM for. If None, uses predicted class
        layer_name: Name of target convolutional layer (e.g., 'layer4' for ResNet)
        device: Device to run computation on ('cuda' or 'cpu')
        mean: Normalization mean (for denormalization during visualization)
        std: Normalization std (for denormalization during visualization)

    Returns:
        Tuple of (heatmap, overlay_image)
            - heatmap: Grad-CAM heatmap, shape (H, W), values in [0, 1]
            - overlay_image: Overlay of heatmap on image, shape (H, W, 3), uint8
    """
    model.eval()
    model.to(device)

    # Ensure batch dimension
    if image_tensor.ndim == 3:
        image_tensor = image_tensor.unsqueeze(0)

    image_tensor = image_tensor.to(device)
    image_tensor.requires_grad_(True)

    # Set up hooks
    with GradCAMHook(model, layer_name) as hook:
        # Forward pass
        output = model(image_tensor)

        # Determine target class
        if target_class is None:
            target_class = output.argmax(dim=1).item()

        # Zero gradients
        model.zero_grad()

        # Backward pass for target class
        class_score = output[0, target_class]
        class_score.backward()

        # Compute Grad-CAM heatmap
        heatmap = compute_gradcam_heatmap(hook.activations, hook.gradients)

    # Create overlay visualization
    overlay = overlay_heatmap_on_image(
        image_tensor.squeeze(0),
        heatmap,
        mean,
        std
    )

    return heatmap, overlay


def run_gradcam_analysis(
    model: nn.Module,
    dataloader: DataLoader,
    num_heatmaps_per_class: Optional[int] = None,
    num_overlays_per_class: int = 5,
    device: str = 'cuda',
    results_dir: Optional[Union[str, Path]] = None,
    layer_name: str = 'layer4',
    mean: List[float] = [0.485, 0.456, 0.406],
    std: List[float] = [0.229, 0.224, 0.225],
    save_overlays: bool = True
) -> Dict[str, List[Dict[str, np.ndarray]]]:
    """
    Run Grad-CAM analysis on a dataset, separating quantitative and qualitative outputs.

    This function evaluates the model on the provided dataloader and generates Grad-CAM
    visualizations. Heatmaps and overlays can be saved independently with different limits
    per class per category.

    Directory structure:
        results_dir/
            heatmaps/
                correct/    # Up to num_heatmaps_per_class per class (or all if None)
                incorrect/  # Up to num_heatmaps_per_class per class (or all if None)
            overlays/
                correct/    # Up to num_overlays_per_class per class
                incorrect/  # Up to num_overlays_per_class per class

    Args:
        model: Trained PyTorch model
        dataloader: DataLoader providing test samples
        num_heatmaps_per_class: Number of heatmaps (.npy) to save per class per category.
                                If None, saves all heatmaps. Default: None (all)
        num_overlays_per_class: Number of overlays (.png) to save per class per category.
                                Default: 5
        device: Device to run computation on ('cuda' or 'cpu')
        results_dir: Optional directory to save outputs. If None, nothing is saved to disk.
        layer_name: Name of target convolutional layer for Grad-CAM
        mean: Normalization mean (for denormalization during visualization)
        std: Normalization std (for denormalization during visualization)
        save_overlays: Whether to save overlay visualizations as PNG files

    Returns:
        Dictionary with structure:
        {
            'correct': [
                {
                    'heatmap': Grad-CAM heatmap (H, W),
                    'overlay': overlay visualization (H, W, 3) or None,
                    'label': true label (int),
                    'pred': predicted label (int),
                    'original_filename': source filename (str, if provided by dataset),
                    'heatmap_path': saved heatmap path (str or None, if results_dir provided),
                    'overlay_path': saved overlay path (str or None, if results_dir provided)
                },
                ...
            ],
            'incorrect': [...]
        }
    """
    model.eval()
    model.to(device)

    # Determine max samples we need to collect per category per class
    max_needed_per_class = max(
        num_heatmaps_per_class if num_heatmaps_per_class is not None else float('inf'),
        num_overlays_per_class
    )

    # Storage for samples by category and class
    samples_by_category = {
        'correct': {0: [], 1: []},
        'incorrect': {0: [], 1: []}
    }

    # Collect predictions until we have enough samples
    print(f"Collecting samples (need {max_needed_per_class} per class per category)...")
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            # Unpack batch (handles both 2-tuple and 3-tuple returns)
            if len(batch) == 3:
                images, labels, filenames = batch
            else:
                images, labels = batch
                filenames = [None] * len(images)

            images = images.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1)

            for img, label, pred, filename in zip(images, labels, preds, filenames):
                label_val = label.item()
                pred_val = pred.item()
                category = 'correct' if label_val == pred_val else 'incorrect'

                # Only store if we need more samples for this category/class
                if len(samples_by_category[category][label_val]) < max_needed_per_class:
                    samples_by_category[category][label_val].append((
                        img.cpu(),
                        label_val,
                        pred_val,
                        filename
                    ))

            # Check if we have enough samples for all categories/classes
            all_satisfied = all(
                len(samples_by_category[cat][cls]) >= max_needed_per_class
                for cat in ['correct', 'incorrect']
                for cls in [0, 1]
            )
            if all_satisfied:
                break

    # Flatten samples into single list for processing
    all_samples = []
    for category in ['correct', 'incorrect']:
        for class_id in [0, 1]:
            all_samples.extend(samples_by_category[category][class_id])

    # Prepare output directory structure if specified
    heatmaps_dir = None
    overlays_dir = None
    if results_dir is not None:
        results_dir = Path(results_dir)
        heatmaps_dir = results_dir / 'heatmaps'
        overlays_dir = results_dir / 'overlays'

        # Create directory structure
        for category in ['correct', 'incorrect']:
            (heatmaps_dir / category).mkdir(parents=True, exist_ok=True)
            if save_overlays:
                (overlays_dir / category).mkdir(parents=True, exist_ok=True)

    # Track heatmap and overlay counts per class per category
    heatmap_counts = {
        'correct': {0: 0, 1: 0},
        'incorrect': {0: 0, 1: 0}
    }
    overlay_counts = {
        'correct': {0: 0, 1: 0},
        'incorrect': {0: 0, 1: 0}
    }

    # Counter for fallback filenames
    sample_counts = {
        'correct': {0: 0, 1: 0},
        'incorrect': {0: 0, 1: 0}
    }

    # Generate Grad-CAM visualizations for all samples
    results = {'correct': [], 'incorrect': []}

    print(f"Generating Grad-CAM for {len(all_samples)} images...")
    for img, label, pred, orig_filename in tqdm(all_samples, desc="Processing"):
        # Determine category
        category = 'correct' if label == pred else 'incorrect'
        class_id = label

        # Generate Grad-CAM heatmap
        heatmap, overlay = generate_gradcam(
            model,
            img,
            target_class=pred,
            layer_name=layer_name,
            device=device,
            mean=mean,
            std=std
        )

        result_entry = {
            'heatmap': heatmap,
            'overlay': overlay,
            'label': label,
            'pred': pred,
            'original_filename': orig_filename
        }

        # Save to disk if directory provided
        if results_dir is not None:
            # Determine filename (use original or fallback)
            if orig_filename:
                name_without_ext = Path(orig_filename).stem
            else:
                idx = sample_counts[category][class_id]
                if category == 'correct':
                    class_name = 'cat' if class_id == 0 else 'dog'
                    name_without_ext = f"class{class_id}_{class_name}_{idx:02d}"
                else:
                    true_class = 'cat' if label == 0 else 'dog'
                    pred_class = 'cat' if pred == 0 else 'dog'
                    name_without_ext = f"true{true_class}_pred{pred_class}_{idx:02d}"
                sample_counts[category][class_id] += 1

            # Save heatmap as .npy (check limit if specified)
            should_save_heatmap = (
                num_heatmaps_per_class is None or
                heatmap_counts[category][class_id] < num_heatmaps_per_class
            )

            if should_save_heatmap:
                heatmap_filename = f"{name_without_ext}.npy"
                heatmap_path = heatmaps_dir / category / heatmap_filename
                np.save(str(heatmap_path), heatmap)
                result_entry['heatmap_path'] = str(heatmap_path)
                heatmap_counts[category][class_id] += 1
            else:
                result_entry['heatmap_path'] = None

            # Save overlay as .png (check limit)
            should_save_overlay = (
                save_overlays and
                overlay_counts[category][class_id] < num_overlays_per_class
            )

            if should_save_overlay:
                overlay_filename = f"{name_without_ext}.png"
                overlay_path = overlays_dir / category / overlay_filename
                cv2.imwrite(str(overlay_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
                result_entry['overlay_path'] = str(overlay_path)
                overlay_counts[category][class_id] += 1
            else:
                result_entry['overlay_path'] = None

        results[category].append(result_entry)

    # Summary
    num_correct = len(results['correct'])
    num_incorrect = len(results['incorrect'])
    total = num_correct + num_incorrect

    print(f"\nProcessed {total} total images:")
    print(f"  - Correct predictions: {num_correct}")
    print(f"  - Incorrect predictions: {num_incorrect}")

    if results_dir is not None:
        print(f"\nResults saved to: {results_dir}")

        # Report heatmaps
        total_heatmaps_saved = sum(heatmap_counts['correct'].values()) + sum(heatmap_counts['incorrect'].values())
        if num_heatmaps_per_class is None:
            print(f"  - Heatmaps (.npy, all {total_heatmaps_saved} saved): {heatmaps_dir}")
        else:
            print(f"  - Heatmaps (.npy, {total_heatmaps_saved} saved, max {num_heatmaps_per_class} per class per category): {heatmaps_dir}")

        # Report overlays
        if save_overlays:
            total_overlays_saved = sum(overlay_counts['correct'].values()) + sum(overlay_counts['incorrect'].values())
            print(f"  - Overlays (.png, {total_overlays_saved} saved, max {num_overlays_per_class} per class per category): {overlays_dir}")

    return results
