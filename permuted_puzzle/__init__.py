"""
PermutedPuzzle: A library for training models on spatially permuted images.

Clean API for composable experiments with image permutations.
"""

# Data utilities
from .data import (
    DogsVsCatsDataset,
    PermutedDogsVsCatsDataset,
    split_indices,
    generate_permutation,
    create_loader,
)

# Model utilities
from .models import (
    build_model,
    get_model_config,
    REGISTRY as MODEL_REGISTRY,
)

# Transform utilities
from .transforms import (
    baseline_train_transforms,
    baseline_val_transforms,
    permute_image_tensor,
)

# Training utilities
from .train_utils import (
    train_model,
    evaluate_model,
)

# Grad-CAM utilities
from .gradcam import (
    generate_gradcam,
    run_gradcam_analysis,
)

__all__ = [
    # Data
    'DogsVsCatsDataset',
    'PermutedDogsVsCatsDataset',
    'split_indices',
    'generate_permutation',
    'create_loader',
    # Models
    'build_model',
    'get_model_config',
    'MODEL_REGISTRY',
    # Transforms
    'baseline_train_transforms',
    'baseline_val_transforms',
    'permute_image_tensor',
    # Training
    'train_model',
    'evaluate_model',
    # Grad-CAM
    'generate_gradcam',
    'run_gradcam_analysis',
]
