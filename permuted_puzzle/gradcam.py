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
    num_per_class: int = 5,
    device: str = 'cuda',
    results_dir: Optional[Union[str, Path]] = None,
    layer_name: str = 'layer4',
    mean: List[float] = [0.485, 0.456, 0.406],
    std: List[float] = [0.229, 0.224, 0.225]
) -> Dict[str, List[Dict[str, np.ndarray]]]:
    """
    Run Grad-CAM analysis on a dataset, selecting correct and incorrect predictions.

    This function evaluates the model on the provided dataloader, identifies
    correct and incorrect predictions for each class, and generates Grad-CAM
    visualizations for a specified number of samples per category.

    Args:
        model: Trained PyTorch model
        dataloader: DataLoader providing test samples
        num_per_class: Number of samples per class per category (correct/incorrect)
        device: Device to run computation on ('cuda' or 'cpu')
        results_dir: Optional directory to save visualizations. If provided,
                     saves images as PNG files organized by category
        layer_name: Name of target convolutional layer for Grad-CAM
        mean: Normalization mean (for denormalization during visualization)
        std: Normalization std (for denormalization during visualization)

    Returns:
        Dictionary with structure:
        {
            'correct': [
                {
                    'image': original image (H, W, 3),
                    'heatmap': Grad-CAM heatmap (H, W),
                    'overlay': overlay visualization (H, W, 3),
                    'label': true label (int),
                    'pred': predicted label (int),
                    'path': saved file path (str, if results_dir provided)
                },
                ...
            ],
            'incorrect': [...]
        }
    """
    model.eval()
    model.to(device)

    # Storage for samples
    correct_samples = {0: [], 1: []}  # class_id -> list of (image, label, pred)
    incorrect_samples = {0: [], 1: []}

    # Collect predictions
    print("Collecting predictions...")
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Evaluating"):
            images = images.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1)

            for img, label, pred in zip(images, labels, preds):
                label_val = label.item()
                pred_val = pred.item()

                if label_val == pred_val:
                    if len(correct_samples[label_val]) < num_per_class:
                        correct_samples[label_val].append((img.cpu(), label_val, pred_val))
                else:
                    if len(incorrect_samples[label_val]) < num_per_class:
                        incorrect_samples[label_val].append((img.cpu(), label_val, pred_val))

                # Check if we have enough samples
                if all(len(correct_samples[c]) >= num_per_class for c in [0, 1]) and \
                   all(len(incorrect_samples[c]) >= num_per_class for c in [0, 1]):
                    break

    # Generate Grad-CAM visualizations
    results = {'correct': [], 'incorrect': []}

    # Prepare output directory if specified
    if results_dir is not None:
        results_dir = Path(results_dir)
        results_dir.mkdir(parents=True, exist_ok=True)
        (results_dir / 'correct').mkdir(exist_ok=True)
        (results_dir / 'incorrect').mkdir(exist_ok=True)

    print("Generating Grad-CAM visualizations...")

    # Process correct predictions
    for class_id, samples in correct_samples.items():
        for idx, (img, label, pred) in enumerate(tqdm(samples, desc=f"Correct class {class_id}")):
            heatmap, overlay = generate_gradcam(
                model,
                img,
                target_class=pred,
                layer_name=layer_name,
                device=device,
                mean=mean,
                std=std
            )

            # Denormalize original image for storage
            img_np = img.numpy().transpose(1, 2, 0)
            mean_arr = np.array(mean).reshape(1, 1, 3)
            std_arr = np.array(std).reshape(1, 1, 3)
            img_np = (img_np * std_arr) + mean_arr
            img_np = np.clip(img_np * 255, 0, 255).astype(np.uint8)

            result_entry = {
                'image': img_np,
                'heatmap': heatmap,
                'overlay': overlay,
                'label': label,
                'pred': pred
            }

            # Save to disk if directory provided
            if results_dir is not None:
                class_name = 'cat' if class_id == 0 else 'dog'
                filename = f"correct_class{class_id}_{class_name}_{idx:02d}.png"
                filepath = results_dir / 'correct' / filename

                # Create side-by-side visualization
                combined = np.hstack([img_np, overlay])
                cv2.imwrite(str(filepath), cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))
                result_entry['path'] = str(filepath)

            results['correct'].append(result_entry)

    # Process incorrect predictions
    for class_id, samples in incorrect_samples.items():
        for idx, (img, label, pred) in enumerate(tqdm(samples, desc=f"Incorrect class {class_id}")):
            heatmap, overlay = generate_gradcam(
                model,
                img,
                target_class=pred,
                layer_name=layer_name,
                device=device,
                mean=mean,
                std=std
            )

            # Denormalize original image for storage
            img_np = img.numpy().transpose(1, 2, 0)
            mean_arr = np.array(mean).reshape(1, 1, 3)
            std_arr = np.array(std).reshape(1, 1, 3)
            img_np = (img_np * std_arr) + mean_arr
            img_np = np.clip(img_np * 255, 0, 255).astype(np.uint8)

            result_entry = {
                'image': img_np,
                'heatmap': heatmap,
                'overlay': overlay,
                'label': label,
                'pred': pred
            }

            # Save to disk if directory provided
            if results_dir is not None:
                true_class = 'cat' if label == 0 else 'dog'
                pred_class = 'cat' if pred == 0 else 'dog'
                filename = f"incorrect_true{true_class}_pred{pred_class}_{idx:02d}.png"
                filepath = results_dir / 'incorrect' / filename

                # Create side-by-side visualization
                combined = np.hstack([img_np, overlay])
                cv2.imwrite(str(filepath), cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))
                result_entry['path'] = str(filepath)

            results['incorrect'].append(result_entry)

    print(f"Generated {len(results['correct'])} correct and {len(results['incorrect'])} incorrect visualizations")
    if results_dir is not None:
        print(f"Results saved to {results_dir}")

    return results
