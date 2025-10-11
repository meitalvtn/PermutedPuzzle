"""
Feature map extraction and analysis module.

This module provides tools for extracting intermediate feature representations
from trained CNNs and analyzing their spatial structure through permutation
and similarity comparison.
"""

from typing import Dict, List, Optional, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureExtractor:
    """
    Hook-based feature extractor for capturing intermediate layer activations.

    This class registers a forward hook on a target layer to capture its
    output during the forward pass. Unlike GradCAM, this does not require
    gradients or backward passes.

    Args:
        model: PyTorch model
        layer_name: Name of the target layer (e.g., 'layer4', 'features.7')

    Attributes:
        activations: Captured activations from the target layer
    """

    def __init__(self, model: nn.Module, layer_name: str):
        self.model = model
        self.layer_name = layer_name
        self.activations = None
        self.handle = None

        target_layer = self._get_layer_by_name(model, layer_name)
        if target_layer is None:
            raise ValueError(f"Layer '{layer_name}' not found in model")

        self.handle = target_layer.register_forward_hook(self._forward_hook)

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

    def _forward_hook(self, module: nn.Module, input: tuple, output: torch.Tensor) -> None:
        """
        Hook to capture forward activations.

        Args:
            module: The layer module
            input: Input tensors to the layer
            output: Output tensor from the layer
        """
        self.activations = output.detach()

    def remove_hook(self) -> None:
        """
        Remove the registered hook.
        """
        if self.handle is not None:
            self.handle.remove()
            self.handle = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.remove_hook()


def extract_features(
    model: nn.Module,
    image_tensor: torch.Tensor,
    layer_name: str = 'layer4',
    device: str = 'cuda'
) -> torch.Tensor:
    """
    Extract feature map from a specific layer of a trained model.

    Performs a forward pass through the model and captures the activations
    at the specified layer. This is useful for analyzing internal representations
    without requiring gradients.

    Args:
        model: Trained PyTorch model
        image_tensor: Normalized image tensor, shape (C, H, W) or (1, C, H, W)
        layer_name: Name of target layer to extract features from
                   (e.g., 'layer4' for ResNet, 'features.7' for VGG)
        device: Device to run computation on ('cuda' or 'cpu')

    Returns:
        Feature map tensor, shape (C, H, W) where C is the number of feature channels

    Example:
        >>> model = build_model('resnet18', pretrained=False)
        >>> model.load_state_dict(torch.load('model.pth'))
        >>> image = dataset[0][0]  # shape (3, 224, 224)
        >>> features = extract_features(model, image, layer_name='layer4')
        >>> features.shape  # (512, 7, 7) for ResNet18 layer4
    """
    model.eval()
    model.to(device)

    # Ensure batch dimension
    if image_tensor.ndim == 3:
        image_tensor = image_tensor.unsqueeze(0)

    image_tensor = image_tensor.to(device)

    # Extract features using hook
    with FeatureExtractor(model, layer_name) as extractor:
        with torch.no_grad():
            _ = model(image_tensor)
        features = extractor.activations

    # Remove batch dimension for cleaner output
    if features.shape[0] == 1:
        features = features.squeeze(0)

    return features


def permute_feature_map(
    feature_map: torch.Tensor,
    permutation: List[int],
    grid_size: Optional[int] = None
) -> torch.Tensor:
    """
    Apply spatial tile permutation to a feature map.

    Divides the spatial dimensions (H, W) of a feature map into grid_size x grid_size
    blocks and reorders them according to the permutation. Unlike permute_image_tensor,
    this works with feature maps that have an arbitrary number of channels.

    Args:
        feature_map: Feature tensor, shape (C, H, W) or (1, C, H, W)
        permutation: List of length grid_size^2 indicating new tile order
        grid_size: Number of tiles per spatial dimension. If None, inferred from permutation length

    Returns:
        Permuted feature map with same shape as input

    Raises:
        ValueError: If permutation length is not a perfect square or if
                   spatial dimensions are not divisible by grid_size

    Example:
        >>> feature_map = torch.randn(512, 14, 14)  # 512 channels, 14x14 spatial
        >>> permutation = [3, 1, 2, 0]  # 2x2 permutation
        >>> permuted = permute_feature_map(feature_map, permutation)
        >>> permuted.shape  # (512, 14, 14)
    """
    # Handle batch dimension
    squeeze_batch = False
    if feature_map.ndim == 3:
        feature_map = feature_map.unsqueeze(0)
        squeeze_batch = True

    # Infer grid_size from permutation
    if grid_size is None:
        n_tiles = len(permutation)
        grid_size = int(n_tiles ** 0.5)
        if grid_size * grid_size != n_tiles:
            raise ValueError(f"Permutation length must be a perfect square, got {n_tiles}")

    B, C, H, W = feature_map.shape

    # Check divisibility
    if H % grid_size != 0 or W % grid_size != 0:
        raise ValueError(
            f"Feature map spatial dimensions ({H}, {W}) must be divisible by grid_size ({grid_size})"
        )

    tile_h, tile_w = H // grid_size, W // grid_size

    # Reshape: (B, C, H, W) -> (B, C, grid_size, tile_h, grid_size, tile_w)
    tiles = feature_map.reshape(B, C, grid_size, tile_h, grid_size, tile_w)

    # Transpose: (B, C, grid_size, tile_h, grid_size, tile_w) -> (B, C, grid_size, grid_size, tile_h, tile_w)
    tiles = tiles.permute(0, 1, 2, 4, 3, 5)

    # Flatten tiles: (B, C, grid_size, grid_size, tile_h, tile_w) -> (B, C, N^2, tile_h, tile_w)
    tiles = tiles.reshape(B, C, grid_size * grid_size, tile_h, tile_w)

    # Apply permutation
    tiles = tiles[:, :, permutation, :, :]

    # Reshape back: (B, C, N^2, tile_h, tile_w) -> (B, C, grid_size, grid_size, tile_h, tile_w)
    tiles = tiles.reshape(B, C, grid_size, grid_size, tile_h, tile_w)

    # Transpose back: (B, C, grid_size, grid_size, tile_h, tile_w) -> (B, C, grid_size, tile_h, grid_size, tile_w)
    tiles = tiles.permute(0, 1, 2, 4, 3, 5)

    # Reconstruct: -> (B, C, H, W)
    result = tiles.reshape(B, C, H, W)

    # Remove batch dimension if it was added
    if squeeze_batch:
        result = result.squeeze(0)

    return result


def inverse_permutation(permutation: List[int]) -> List[int]:
    """
    Compute the inverse of a permutation.

    Given a permutation p where p[i] = j means element i goes to position j,
    returns the inverse permutation inv_p where inv_p[j] = i.

    Args:
        permutation: List of unique integers from 0 to N-1 representing a permutation

    Returns:
        Inverse permutation as a list of integers

    Raises:
        ValueError: If permutation is invalid (not a permutation of range(N))

    Example:
        >>> perm = [2, 0, 1]  # 0->2, 1->0, 2->1
        >>> inv = inverse_permutation(perm)
        >>> inv  # [1, 2, 0]  # position 0 came from 1, position 1 from 2, position 2 from 0
        >>> # Verify: applying perm then inv gives identity
        >>> [inv[perm[i]] for i in range(3)]  # [0, 1, 2]
    """
    n = len(permutation)

    # Validate permutation
    if set(permutation) != set(range(n)):
        raise ValueError(
            f"Invalid permutation: must contain each integer from 0 to {n-1} exactly once"
        )

    # Compute inverse
    inverse = [0] * n
    for i, j in enumerate(permutation):
        inverse[j] = i

    return inverse


def compare_features(
    feature_map1: torch.Tensor,
    feature_map2: torch.Tensor
) -> Dict[str, float]:
    """
    Compute similarity metrics between two feature maps.

    Compares two feature maps using multiple similarity measures to assess
    how similar their representations are. Useful for analyzing whether
    models learn similar internal representations despite different inputs.

    Args:
        feature_map1: First feature tensor, shape (C, H, W) or (1, C, H, W)
        feature_map2: Second feature tensor, same shape as feature_map1

    Returns:
        Dictionary containing:
            - 'cosine': Cosine similarity in [-1, 1], higher is more similar
            - 'mse': Mean squared error, lower is more similar
            - 'mae': Mean absolute error, lower is more similar
            - 'correlation': Pearson correlation in [-1, 1], higher is more similar
            - 'normalized_l2': Normalized L2 distance in [0, inf], lower is more similar

    Raises:
        ValueError: If feature maps have different shapes

    Example:
        >>> f1 = extract_features(model1, image, layer_name='layer4')
        >>> f2 = extract_features(model2, image, layer_name='layer4')
        >>> metrics = compare_features(f1, f2)
        >>> print(f"Cosine similarity: {metrics['cosine']:.4f}")
        >>> print(f"MSE: {metrics['mse']:.4f}")
    """
    # Ensure same shape
    if feature_map1.shape != feature_map2.shape:
        raise ValueError(
            f"Feature maps must have same shape, got {feature_map1.shape} and {feature_map2.shape}"
        )

    # Remove batch dimension if present
    if feature_map1.ndim == 4:
        feature_map1 = feature_map1.squeeze(0)
        feature_map2 = feature_map2.squeeze(0)

    # Flatten for easier computation
    f1_flat = feature_map1.flatten()
    f2_flat = feature_map2.flatten()

    # Cosine similarity
    cosine_sim = F.cosine_similarity(
        f1_flat.unsqueeze(0),
        f2_flat.unsqueeze(0),
        dim=1
    ).item()

    # Mean squared error
    mse = F.mse_loss(feature_map1, feature_map2).item()

    # Mean absolute error
    mae = F.l1_loss(feature_map1, feature_map2).item()

    # Pearson correlation
    f1_centered = f1_flat - f1_flat.mean()
    f2_centered = f2_flat - f2_flat.mean()
    correlation = (
        (f1_centered * f2_centered).sum() /
        (f1_centered.norm() * f2_centered.norm() + 1e-8)
    ).item()

    # Normalized L2 distance
    l2_dist = torch.dist(f1_flat, f2_flat, p=2).item()
    normalized_l2 = l2_dist / (f1_flat.norm().item() + 1e-8)

    return {
        'cosine': cosine_sim,
        'mse': mse,
        'mae': mae,
        'correlation': correlation,
        'normalized_l2': normalized_l2
    }
