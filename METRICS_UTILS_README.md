# Metrics Utils - Quick Reference

Helper functions for extracting information from `metrics.json` files.

## Installation

Already included in the library:
```python
from permuted_puzzle.metrics_utils import *
```

## Functions Overview

### 1. **get_split_indices(metrics_path)**
Get train/val/test split indices used during training.

```python
indices = get_split_indices('results/resnet18/1/metrics.json')
train_indices = indices['train']  # numpy array [17, 12, 2, ...]
val_indices = indices['val']
test_indices = indices['test']
```

**Use case**: Recreate exact train/val/test splits for reproducibility.

---

### 2. **get_permutation(metrics_path, split=None)**
Get the tile permutation used for a specific split.

```python
# Get train permutation (or any split)
perm = get_permutation('results/resnet18/9/metrics.json', split='train')
# [3, 1, 4, 0, 2, 5, 8, 6, 7] for 3x3 grid

# For baseline models (grid_size=1), returns None
perm = get_permutation('results/resnet18/1/metrics.json')
# None
```

**Use case**: Apply same permutation to new images for inference.

---

### 3. **get_all_permutations(metrics_path)**
Get permutations for all splits at once.

```python
perms = get_all_permutations('results/resnet18/9/metrics.json')
train_perm = perms['train']
val_perm = perms['val']
test_perm = perms['test']
```

**Use case**: Check if different permutations were used per split.

---

### 4. **get_model_info(metrics_path)**
Get model architecture and configuration details.

```python
info = get_model_info('results/resnet18/1/metrics.json')

# Returns:
{
    'model': 'resnet18',
    'grid_size': 1,
    'input_size': 224,
    'mean': [0.485, 0.456, 0.406],
    'std': [0.229, 0.224, 0.225],
    'weights_path': '/absolute/path/to/best.pth',
    'saved_at': '2025-10-10 16:33:23',
    'env': {'python': '3.11', 'torch': '2.3.1'},
    'notes': 'Baseline, 224x224, ImageNet norm'
}
```

**Use case**: Load model with exact same configuration.

---

### 5. **get_performance_metrics(metrics_path)**
Get training and evaluation metrics.

```python
perf = get_performance_metrics('results/resnet18/1/metrics.json')

# Returns:
{
    'final_train_acc': 0.6667,
    'final_train_loss': 0.5994,
    'final_val_acc': 1.0,
    'final_val_loss': 0.3067,
    'best_val_acc': 1.0,
    'test_acc': 0.8,
    'test_loss': 0.4600
}
```

**Use case**: Compare model performance across different configurations.

---

### 6. **get_training_config(metrics_path)**
Get hyperparameters used during training.

```python
config = get_training_config('results/resnet18/1/metrics.json')

# Returns:
{
    'epochs': 1,
    'batch_size': 4,
    'lr': 0.0001,
    'wd': 0.0001,
    'optimizer': 'AdamW',
    'dropout': 0.2,
    'pretrained': True
}
```

**Use case**: Replicate training setup exactly.

---

### 7. **get_training_history(metrics_path)**
Get per-epoch training curves.

```python
history = get_training_history('results/resnet18/1/metrics.json')

train_acc = history['train_acc']  # [0.66, 0.75, 0.82, ...]
val_acc = history['val_acc']      # [0.70, 0.78, 0.85, ...]

# Plot training curves
import matplotlib.pyplot as plt
plt.plot(train_acc, label='Train')
plt.plot(val_acc, label='Val')
plt.legend()
plt.show()
```

**Use case**: Analyze training dynamics, check for overfitting.

---

### 8. **get_grid_info(metrics_path)**
Get grid configuration details.

```python
grid_size, num_tiles = get_grid_info('results/resnet18/9/metrics.json')
# (3, 9) for 3x3 grid
```

**Use case**: Quick check of permutation complexity.

---

### 9. **compare_runs(metrics_paths)**
Compare multiple training runs side-by-side.

```python
paths = [
    'results/resnet18/1/metrics.json',   # baseline
    'results/resnet18/9/metrics.json',   # 3x3
    'results/resnet18/64/metrics.json'   # 8x8
]

comparison = compare_runs(paths)

# Returns dict with lists:
{
    'model': ['resnet18', 'resnet18', 'resnet18'],
    'grid_size': [1, 3, 8],
    'best_val_acc': [0.95, 0.78, 0.65],
    'test_acc': [0.92, 0.75, 0.60],
    ...
}
```

**Use case**: Generate comparison tables for papers/presentations.

---

## Complete Example: Load Model and Data

```python
from permuted_puzzle.metrics_utils import (
    get_split_indices,
    get_permutation,
    get_model_info
)
from permuted_puzzle.data import DogsVsCatsDataset, create_loader
from permuted_puzzle.transforms import baseline_val_transforms
from permuted_puzzle.models import build_model
import torch

# 1. Get metadata
metrics_path = "results/resnet18/9/metrics.json"
info = get_model_info(metrics_path)
indices = get_split_indices(metrics_path)
perm = get_permutation(metrics_path, split='test')

# 2. Load model
model = build_model(info['model'], num_classes=2, pretrained=False)
model.load_state_dict(torch.load(info['weights_path']))
model.eval()

# 3. Recreate exact test set
dataset = DogsVsCatsDataset('data/')
transform = baseline_val_transforms(info['input_size'], info['mean'], info['std'])
test_loader = create_loader(
    dataset,
    indices['test'],
    transform,
    permutation=perm,
    batch_size=32
)

# 4. Evaluate
# Now you can evaluate on the EXACT same test set used during training
```

---

## Tips

- **Always use these helpers** instead of manually parsing JSON to avoid errors
- **Save metrics_path** in your analysis notebooks for reproducibility
- **Use compare_runs()** to generate results tables quickly
- **Check split_indices** to ensure no data leakage between train/val/test

---

## See Also

- `test_metrics_utils.py` - Full test suite
- `example_metrics_usage.py` - Real-world usage examples
