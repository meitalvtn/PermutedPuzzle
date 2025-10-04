from typing import Dict, Tuple
import torch.nn as nn

from .resnet18 import build as resnet18_build
from .mobilenet_v3_large import build as mobilenet_v3_large_build
from .efficientnet_b0 import build as efficientnet_b0_build
from .convnext_tiny import build as convnext_tiny_build
from .simple_cnn import build as simple_cnn_build

REGISTRY = {
    "resnet18": resnet18_build,
    "mobilenet_v3_large": mobilenet_v3_large_build,
    "efficientnet_b0": efficientnet_b0_build,
    "convnext_tiny": convnext_tiny_build,
    "simple_cnn": simple_cnn_build,
}

# Metadata registry (avoids model instantiation for config lookup)
META_REGISTRY = {
    "resnet18": {"input_size": 224, "mean": [.485, .456, .406], "std": [.229, .224, .225]},
    "mobilenet_v3_large": {"input_size": 224, "mean": [.485, .456, .406], "std": [.229, .224, .225]},
    "efficientnet_b0": {"input_size": 224, "mean": [.485, .456, .406], "std": [.229, .224, .225]},
    "convnext_tiny": {"input_size": 224, "mean": [.485, .456, .406], "std": [.229, .224, .225]},
    "simple_cnn": {"input_size": 224, "mean": [.485, .456, .406], "std": [.229, .224, .225]},
}


def build_model(
    name: str,
    num_classes: int = 2,
    pretrained: bool = False,
    dropout: float = 0.2
) -> nn.Module:
    """
    Build model from registry.

    Args:
        name: Model name (e.g., 'resnet18', 'mobilenet_v3_large')
        num_classes: Number of output classes
        pretrained: Whether to use pretrained weights
        dropout: Dropout rate

    Returns:
        PyTorch model

    Example:
        >>> model = build_model("resnet18", num_classes=2, pretrained=True)
    """
    if name not in REGISTRY:
        raise ValueError(f"Unknown model: {name}. Available: {list(REGISTRY.keys())}")

    build_fn = REGISTRY[name]
    model, _ = build_fn(num_classes=num_classes, pretrained=pretrained, dropout=dropout)
    return model


def get_model_config(name: str) -> Dict:
    """
    Get model configuration metadata.

    Args:
        name: Model name (e.g., 'resnet18', 'mobilenet_v3_large')

    Returns:
        Dict with keys: 'input_size', 'mean', 'std'

    Example:
        >>> config = get_model_config("resnet18")
        >>> print(config['input_size'])  # 224
    """
    if name not in META_REGISTRY:
        raise ValueError(f"Unknown model: {name}. Available: {list(META_REGISTRY.keys())}")

    return META_REGISTRY[name].copy()
