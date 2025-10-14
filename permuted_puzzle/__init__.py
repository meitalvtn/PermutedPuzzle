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

# Metrics utilities
from .metrics_utils import (
    load_metrics,
    get_split_indices,
    get_permutation,
    get_all_permutations,
    get_model_info,
    get_performance_metrics,
    get_training_config,
    get_training_history,
    compare_runs,
    get_grid_info,
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
    # Metrics
    'load_metrics',
    'get_split_indices',
    'get_permutation',
    'get_all_permutations',
    'get_model_info',
    'get_performance_metrics',
    'get_training_config',
    'get_training_history',
    'compare_runs',
    'get_grid_info',
]
