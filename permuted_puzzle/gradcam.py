"""
Grad-CAM visualization module for model interpretability.

This module provides tools to visualize which regions of an image
a trained model attends to when making predictions.

Supports both CNN (ResNet, EfficientNet, etc.) and Transformer
(Swin Transformer) architectures with automatic format detection.
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

    Supports both CNN and Transformer architectures:
    - CNNs (ResNet, VGG, etc.): layers output (B, C, H, W)
    - Swin Transformers: layers output (B, H, W, C)

    Automatically detects and normalizes to channels-first format.

    Args:
        model: PyTorch model (CNN or Swin Transformer)
        layer_name: Name of the target layer
            - For CNNs: e.g., 'layer4' (ResNet), 'features.28' (VGG)
            - For Swin: e.g., 'features.7.1.norm1', 'features.5.5.norm2'

    Attributes:
        activations: Forward activations from target layer, shape (B, C, H, W)
        gradients: Gradients flowing back through target layer, shape (B, C, H, W)

    Example:
        # CNN (ResNet)
        with GradCAMHook(resnet_model, 'layer4') as hook:
            output = model(input)
            output[0, target_class].backward()
            heatmap = compute_gradcam_heatmap(hook.activations, hook.gradients)

        # Swin Transformer
        with GradCAMHook(swin_model, 'features.7.1.norm1') as hook:
            output = model(input)
            output[0, target_class].backward()
            heatmap = compute_gradcam_heatmap(hook.activations, hook.gradients)
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

    def _is_channels_last_format(self, tensor: torch.Tensor) -> bool:
        """
        Detect if 4D tensor is in (B, H, W, C) format vs (B, C, H, W).

        Uses heuristic: In channels-last format, the last dimension (C) is typically
        much larger than the middle dimensions (H, W). In channels-first format,
        the second dimension (C) is usually much larger than the last two (H, W).

        Examples:
            (1, 7, 7, 768)   -> channels-last  (D3=768 >> D1=7, D2=7)
            (1, 768, 7, 7)   -> channels-first (D1=768 >> D2=7, D3=7)
            (1, 3, 224, 224) -> channels-first (D1=3 << D2=224, D3=224)

        Args:
            tensor: 4D tensor to check

        Returns:
            True if tensor is in channels-last format, False otherwise
        """
        if tensor.ndim != 4:
            return False

        B, D1, D2, D3 = tensor.shape

        # Heuristic: Compare ratios to determine which dimension is likely channels
        # If last dimension is much larger than spatial dims, it's likely channels-last
        channels_last_score = D3 / max(D1, D2, 1)
        channels_first_score = D1 / max(D2, D3, 1)

        return channels_last_score > channels_first_score

    def _normalize_to_channels_first(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Convert tensor to channels-first (B, C, H, W) format if needed.

        Args:
            tensor: 4D tensor in either (B, C, H, W) or (B, H, W, C) format

        Returns:
            Tensor in (B, C, H, W) format

        Raises:
            ValueError: If tensor is not 4D
        """
        if tensor.ndim != 4:
            raise ValueError(
                f"Expected 4D tensor, got {tensor.ndim}D tensor with shape {tensor.shape}. "
                f"GradCAM requires convolutional or transformer layers that output 4D feature maps. "
                f"Layer '{self.layer_name}' may not be suitable for GradCAM."
            )

        if self._is_channels_last_format(tensor):
            # (B, H, W, C) -> (B, C, H, W)
            return tensor.permute(0, 3, 1, 2).contiguous()

        return tensor

    def _validate_tensor_shape(self, tensor: torch.Tensor, tensor_type: str) -> None:
        """
        Validate tensor has expected 4D shape.

        Args:
            tensor: The tensor to validate
            tensor_type: 'activations' or 'gradients' for error messages

        Raises:
            ValueError: If tensor is not 4D
        """
        if tensor.ndim != 4:
            raise ValueError(
                f"Expected 4D {tensor_type} tensor, got {tensor.ndim}D with shape {tensor.shape}"
            )

    def _forward_hook(self, module: nn.Module, input: Tuple, output: torch.Tensor) -> None:
        """
        Hook to capture forward activations.

        Automatically detects and normalizes tensor format:
        - CNN layers output (B, C, H, W) - stored as-is
        - Swin layers output (B, H, W, C) - converted to (B, C, H, W)

        Args:
            module: The layer module
            input: Input tensors to the layer
            output: Output tensor from the layer

        Raises:
            ValueError: If output is not a 4D tensor
        """
        output_detached = output.detach()
        normalized_output = self._normalize_to_channels_first(output_detached)
        self._validate_tensor_shape(normalized_output, 'activations')
        self.activations = normalized_output

    def _backward_hook(self, module: nn.Module, grad_input: Tuple, grad_output: Tuple) -> None:
        """
        Hook to capture gradients during backpropagation.

        Automatically detects and normalizes tensor format:
        - CNN layers: gradients in (B, C, H, W) - stored as-is
        - Swin layers: gradients in (B, H, W, C) - converted to (B, C, H, W)

        Args:
            module: The layer module
            grad_input: Gradients with respect to inputs
            grad_output: Gradients with respect to outputs (first element used)

        Raises:
            ValueError: If gradient is not a 4D tensor
        """
        grad_detached = grad_output[0].detach()
        normalized_grad = self._normalize_to_channels_first(grad_detached)
        self._validate_tensor_shape(normalized_grad, 'gradients')
        self.gradients = normalized_grad

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

    Supports both CNN and Swin Transformer architectures with automatic format detection.

    Args:
        model: Trained PyTorch model (CNN or Swin Transformer)
        image_tensor: Normalized image tensor, shape (1, C, H, W) or (C, H, W)
        target_class: Class index to compute Grad-CAM for. If None, uses predicted class
        layer_name: Name of target layer
            - For ResNet: 'layer4' (final conv layer)
            - For Swin Tiny: 'features.7.1.norm1' (stage 4, 7×7 resolution)
                           or 'features.5.5.norm2' (stage 3, 14×14 resolution)
        device: Device to run computation on ('cuda' or 'cpu')
        mean: Normalization mean (for denormalization during visualization)
        std: Normalization std (for denormalization during visualization)

    Returns:
        Tuple of (heatmap, overlay_image)
            - heatmap: Grad-CAM heatmap, shape (H, W), values in [0, 1]
            - overlay_image: Overlay of heatmap on image, shape (H, W, 3), uint8

    Example:
        # For Swin Transformer
        heatmap, overlay = generate_gradcam(
            swin_model,
            image,
            layer_name='features.7.1.norm1'
        )
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
    max_samples: Optional[int] = None,
    device: str = 'cuda',
    results_dir: Optional[Union[str, Path]] = None,
    layer_name: str = 'layer4',
    mean: List[float] = [0.485, 0.456, 0.406],
    std: List[float] = [0.229, 0.224, 0.225]
) -> Dict[str, List[Dict[str, np.ndarray]]]:
    """
    Run Grad-CAM analysis on a dataset with balanced sampling.

    This function evaluates the model on the provided dataloader and generates Grad-CAM
    heatmaps. It attempts to collect half correct and half incorrect predictions,
    but respects the max_samples limit. If balanced sampling is not possible (e.g., high
    accuracy model), it fills the remaining quota with available samples.

    Directory structure (if results_dir is provided):
        results_dir/
            heatmaps/
                correct/
                incorrect/

    Args:
        model: Trained PyTorch model
        dataloader: DataLoader providing test samples
        max_samples: Maximum total number of samples to process. If None, processes all
                     available data. The function attempts to collect max_samples/2 correct
                     and max_samples/2 incorrect predictions. Default: None (all)
        device: Device to run computation on ('cuda' or 'cpu')
        results_dir: Optional directory to save heatmaps. If None, nothing is saved to disk.
        layer_name: Name of target convolutional layer for Grad-CAM
        mean: Normalization mean (stored in results for later overlay generation)
        std: Normalization std (stored in results for later overlay generation)

    Returns:
        Dictionary with structure:
        {
            'correct': [
                {
                    'heatmap': Grad-CAM heatmap (H, W),
                    'image_tensor': original image tensor (C, H, W),
                    'label': true label (int),
                    'pred': predicted label (int),
                    'original_filename': source filename (str, if provided by dataset),
                    'heatmap_path': saved heatmap path (str or None, if results_dir provided)
                },
                ...
            ],
            'incorrect': [...]
        }
    """
    model.eval()
    model.to(device)

    # Determine target samples per category (try to balance correct/incorrect)
    if max_samples is None:
        target_per_category = float('inf')
        total_target = float('inf')
    else:
        target_per_category = max_samples // 2
        total_target = max_samples

    # Storage for samples by category
    samples_by_category = {
        'correct': [],
        'incorrect': []
    }

    # Phase 1: Balanced collection (try to get target_per_category for each category)
    if max_samples is None:
        print("Collecting all available samples...")
    else:
        print(f"Collecting samples (target: {target_per_category} correct, {target_per_category} incorrect, {total_target} total)...")

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Phase 1: Balanced collection"):
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

                # Only store if we need more samples for this category
                if len(samples_by_category[category]) < target_per_category:
                    samples_by_category[category].append((
                        img.cpu(),
                        label_val,
                        pred_val,
                        filename
                    ))

            # Check if we have enough samples for both categories
            if target_per_category != float('inf'):
                all_satisfied = all(
                    len(samples_by_category[cat]) >= target_per_category
                    for cat in ['correct', 'incorrect']
                )
                if all_satisfied:
                    break

        # Phase 2: Fill remaining quota (if we haven't reached total target)
        current_total = sum(len(samples_by_category[cat]) for cat in ['correct', 'incorrect'])

        if total_target != float('inf') and current_total < total_target:
            remaining = total_target - current_total
            print(f"Phase 2: Filling remaining {remaining} samples from available data...")

            for batch in tqdm(dataloader, desc="Phase 2: Fill remaining quota"):
                if current_total >= total_target:
                    break

                # Unpack batch
                if len(batch) == 3:
                    images, labels, filenames = batch
                else:
                    images, labels = batch
                    filenames = [None] * len(images)

                images = images.to(device)
                outputs = model(images)
                preds = outputs.argmax(dim=1)

                for img, label, pred, filename in zip(images, labels, preds, filenames):
                    if current_total >= total_target:
                        break

                    label_val = label.item()
                    pred_val = pred.item()
                    category = 'correct' if label_val == pred_val else 'incorrect'

                    # Add sample regardless of category distribution
                    samples_by_category[category].append((
                        img.cpu(),
                        label_val,
                        pred_val,
                        filename
                    ))
                    current_total += 1

    # Flatten samples into single list for processing
    all_samples = []
    for category in ['correct', 'incorrect']:
        all_samples.extend(samples_by_category[category])

    # Prepare output directory structure if specified
    heatmaps_dir = None
    if results_dir is not None:
        results_dir = Path(results_dir)
        heatmaps_dir = results_dir / 'heatmaps'

        # Create directory structure
        for category in ['correct', 'incorrect']:
            (heatmaps_dir / category).mkdir(parents=True, exist_ok=True)

    # Counter for fallback filenames
    sample_counts = {
        'correct': {0: 0, 1: 0},
        'incorrect': {0: 0, 1: 0}
    }

    # Generate Grad-CAM heatmaps for all samples
    results = {'correct': [], 'incorrect': []}

    print(f"Generating Grad-CAM for {len(all_samples)} images...")
    for img, label, pred, orig_filename in tqdm(all_samples, desc="Processing"):
        # Determine category
        category = 'correct' if label == pred else 'incorrect'
        class_id = label

        # Generate Grad-CAM heatmap (we don't need the overlay here)
        heatmap, _ = generate_gradcam(
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
            'image_tensor': img,
            'label': label,
            'pred': pred,
            'original_filename': orig_filename
        }

        # Save heatmap to disk if directory provided
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

            # Save heatmap as .npy
            heatmap_filename = f"{name_without_ext}.npy"
            heatmap_path = heatmaps_dir / category / heatmap_filename
            np.save(str(heatmap_path), heatmap)
            result_entry['heatmap_path'] = str(heatmap_path)

        results[category].append(result_entry)

    # Summary
    num_correct = len(results['correct'])
    num_incorrect = len(results['incorrect'])
    total = num_correct + num_incorrect

    print(f"\nProcessed {total} total images:")
    print(f"  - Correct predictions: {num_correct}")
    print(f"    - Class 0: {len([r for r in results['correct'] if r['label'] == 0])}")
    print(f"    - Class 1: {len([r for r in results['correct'] if r['label'] == 1])}")
    print(f"  - Incorrect predictions: {num_incorrect}")
    print(f"    - Class 0: {len([r for r in results['incorrect'] if r['label'] == 0])}")
    print(f"    - Class 1: {len([r for r in results['incorrect'] if r['label'] == 1])}")

    if results_dir is not None:
        print(f"\nHeatmaps saved to: {heatmaps_dir}")

    return results


def save_gradcam_overlays(
    results: Dict[str, List[Dict]],
    results_dir: Union[str, Path],
    mean: List[float] = [0.485, 0.456, 0.406],
    std: List[float] = [0.229, 0.224, 0.225],
    colormap: int = cv2.COLORMAP_JET,
    alpha: float = 0.4
) -> None:
    """
    Save overlay visualizations from Grad-CAM results.

    Takes the results dictionary from run_gradcam_analysis and creates
    PNG overlay visualizations for a subset of samples.

    Directory structure:
        results_dir/
            overlays/
                correct/
                incorrect/

    Args:
        results: Results dictionary from run_gradcam_analysis
        results_dir: Directory to save overlay images
        mean: Normalization mean used during preprocessing
        std: Normalization std used during preprocessing
        colormap: OpenCV colormap to use for heatmap visualization
        alpha: Transparency for overlay (0=only image, 1=only heatmap)

    Returns:
        None
    """
    results_dir = Path(results_dir)
    overlays_dir = results_dir / 'overlays'

    # Create directory structure
    for category in ['correct', 'incorrect']:
        (overlays_dir / category).mkdir(parents=True, exist_ok=True)

    # Counter for fallback filenames
    sample_counts = {
        'correct': {0: 0, 1: 0},
        'incorrect': {0: 0, 1: 0}
    }

    total_saved = 0
    print(f"Saving overlay visualizations...")

    for category in ['correct', 'incorrect']:
        for result_entry in tqdm(results[category], desc=f"Saving {category} overlays"):
            heatmap = result_entry['heatmap']
            image_tensor = result_entry['image_tensor']
            label = result_entry['label']
            pred = result_entry['pred']
            orig_filename = result_entry.get('original_filename')

            # Create overlay
            overlay = overlay_heatmap_on_image(
                image_tensor,
                heatmap,
                mean,
                std,
                colormap=colormap,
                alpha=alpha
            )

            # Determine filename (use original or fallback)
            if orig_filename:
                name_without_ext = Path(orig_filename).stem
            else:
                idx = sample_counts[category][label]
                if category == 'correct':
                    class_name = 'cat' if label == 0 else 'dog'
                    name_without_ext = f"class{label}_{class_name}_{idx:02d}"
                else:
                    true_class = 'cat' if label == 0 else 'dog'
                    pred_class = 'cat' if pred == 0 else 'dog'
                    name_without_ext = f"true{true_class}_pred{pred_class}_{idx:02d}"
                sample_counts[category][label] += 1

            # Save overlay as .png
            overlay_filename = f"{name_without_ext}.png"
            overlay_path = overlays_dir / category / overlay_filename
            cv2.imwrite(str(overlay_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
            total_saved += 1

    print(f"\nSaved {total_saved} overlay visualizations to: {overlays_dir}")
