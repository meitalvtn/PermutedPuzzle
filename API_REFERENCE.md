# Permuted Puzzle API Reference

Quick reference for main library functions.

---

## Data & Loaders (`permuted_puzzle.data`)

### `DogsVsCatsDataset(img_dir, transform=None)`
**Input:** Image directory path, optional transform
**Output:** Dataset object
**Description:** Load dogs vs cats images from directory

### `generate_permutation(grid_size)`
**Input:** Grid size (N)
**Output:** List of N² integers
**Description:** Generate random tile permutation (saved to metrics.json for reproducibility)

### `split_indices(n_samples, splits)`
**Input:** Number of samples, split ratios [0.6, 0.2, 0.2]
**Output:** Dict with 'train'/'val'/'test' index arrays
**Description:** Split dataset indices randomly (saved to metrics.json for reproducibility)

### `create_loader(dataset, indices, transform, permutation=None, batch_size=64, shuffle=False)`
**Input:** Dataset, indices, transform, optional permutation, batch size, shuffle flag
**Output:** DataLoader
**Description:** Create DataLoader with optional tile permutation

---

## Models (`permuted_puzzle.models`)

### `build_model(model_name, num_classes=2, pretrained=True, dropout=0.2)`
**Input:** Model name ('resnet18'/'mobilenet_v3_large'/'efficientnet_b0'/'convnext_tiny'/'swin_tiny'/'simple_cnn'), classes, pretrained flag, dropout
**Output:** PyTorch model
**Description:** Build pretrained or from-scratch model

### `get_model_config(model_name)`
**Input:** Model name
**Output:** Dict with 'input_size', 'mean', 'std'
**Description:** Get preprocessing config for model

---

## Transforms (`permuted_puzzle.transforms`)

### `baseline_train_transforms(input_size, mean, std)`
**Input:** Image size, normalization mean/std
**Output:** torchvision.Compose transform
**Description:** Training augmentations (crop, flip, normalize)

### `baseline_val_transforms(input_size, mean, std)`
**Input:** Image size, normalization mean/std
**Output:** torchvision.Compose transform
**Description:** Validation preprocessing (resize, crop, normalize)

### `permute_image_tensor(image_tensor, permutation=None, grid_size=None)`
**Input:** Tensor (C,H,W), permutation list, optional grid_size
**Output:** Permuted tensor (C,H,W)
**Description:** Apply tile permutation to image tensor

---

## Training (`permuted_puzzle.train_utils`)

### `train_model(model, train_loader, val_loader, epochs=10, lr=1e-4, device='cuda', out_path=None, config=None)`
**Input:** Model, loaders, hyperparameters, device, output path, config dict
**Output:** Dict with 'history', 'best_val_acc', 'best_epoch', 'model'
**Description:** Train model with validation, save best checkpoint

### `evaluate_model(model, loader, device='cuda', criterion=None)`
**Input:** Model, DataLoader, device, optional loss function
**Output:** Dict with 'loss' and 'accuracy'
**Description:** Evaluate model on dataset

---

## Experiments (`permuted_puzzle.experiments`)

### `run_single_experiment(model_name, grid_size, data_path, results_root, config, device='cuda')`
**Input:** Model name, grid size, data path, results directory, config dict, device
**Output:** Dict with val/test accuracies, epochs
**Description:** Run complete train/val/test experiment with one configuration

### `run_experiment_grid(model_names, grid_sizes, data_path, results_root, config, device='cuda')`
**Input:** Lists of models/grid sizes, data path, results directory, config dict, device
**Output:** List of result dicts
**Description:** Run experiments across all model×grid combinations

### `print_experiment_summary(results)`
**Input:** List of result dicts
**Output:** None (prints table)
**Description:** Print formatted summary table

---

## Grad-CAM Visualization (`permuted_puzzle.gradcam`)

### `generate_gradcam(model, image_tensor, target_class=None, layer_name='layer4', device='cuda', mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])`
**Input:** Model, normalized image tensor (1, C, H, W) or (C, H, W), optional target class, layer name, device, normalization params
**Output:** Tuple (heatmap, overlay_image) as numpy arrays
**Description:** Compute Grad-CAM activation map for chosen layer and overlay it on original image

### `run_gradcam_analysis(model, dataloader, num_per_class=5, device='cuda', results_dir=None, layer_name='layer4', mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])`
**Input:** Model, test DataLoader, samples per class, device, optional output directory, layer name, normalization params
**Output:** Dict with 'correct' and 'incorrect' lists containing visualization data
**Description:** Automatically select correct/incorrect predictions, generate Grad-CAMs, save visualizations

---

## Feature Analysis (`permuted_puzzle.feature_analysis`)

### `extract_features(model, image_tensor, layer_name='layer4', device='cuda')`
**Input:** Model, normalized image tensor (C, H, W) or (1, C, H, W), layer name, device
**Output:** Feature map tensor (C, H, W)
**Description:** Extract intermediate feature representations from a specific layer (forward pass only, no gradients)

### `permute_feature_map(feature_map, permutation, grid_size=None)`
**Input:** Feature tensor (C, H, W) or (1, C, H, W), permutation list, optional grid size
**Output:** Permuted feature map with same shape
**Description:** Apply spatial tile permutation to feature map (works with any number of channels)

### `inverse_permutation(permutation)`
**Input:** List of integers representing a permutation
**Output:** List of integers (inverse permutation)
**Description:** Compute the inverse of a given permutation

### `compare_features(feature_map1, feature_map2)`
**Input:** Two feature tensors of same shape
**Output:** Dict with similarity metrics ('cosine', 'mse', 'mae', 'correlation', 'normalized_l2')
**Description:** Compute multiple similarity measures between two feature maps

### `get_compatible_layers(model, grid_size, input_size=224, device='cuda')`
**Input:** Model, grid size, input image size, device
**Output:** Dict mapping layer names to spatial dimensions (H, W)
**Description:** Find all layers with feature map dimensions divisible by grid_size (sorted by resolution, largest first)

---

## Config Dict Format

```python
config = {
    "epochs": 10,
    "lr": 1e-4,
    "weight_decay": 1e-4,
    "batch_size": 32,
    "dropout": 0.2,
    "pretrained": True  # Use ImageNet weights
}
```

---

## Typical Usage Pattern

### Running Experiments
```python
from permuted_puzzle.data import DogsVsCatsDataset, split_indices, generate_permutation, create_loader
from permuted_puzzle.models import build_model, get_model_config
from permuted_puzzle.transforms import baseline_train_transforms, baseline_val_transforms
from permuted_puzzle.experiments import run_experiment_grid, print_experiment_summary

# Quick experiments
results = run_experiment_grid(
    model_names=["resnet18"],
    grid_sizes=[1, 2, 4, 8],
    data_path="data/train",
    results_root=Path("results"),
    config={"epochs": 10, "lr": 1e-4, "batch_size": 32, "dropout": 0.2, "pretrained": True}
)

print_experiment_summary(results)
```

### Grad-CAM Analysis
```python
from permuted_puzzle.gradcam import run_gradcam_analysis, generate_gradcam
from permuted_puzzle.models import build_model
from permuted_puzzle.data import DogsVsCatsDataset, create_loader, split_indices, generate_permutation
from permuted_puzzle.transforms import baseline_val_transforms
import torch

# Load trained model
model = build_model('resnet18', pretrained=False)
model.load_state_dict(torch.load('results/models/resnet18_grid3.pth'))

# Create test dataloader (or load splits/permutation from metrics.json)
dataset = DogsVsCatsDataset('data/train')
splits = split_indices(len(dataset), [0.6, 0.2, 0.2])
perm = generate_permutation(grid_size=3)
val_transform = baseline_val_transforms(224, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
test_loader = create_loader(dataset, splits['test'], val_transform, permutation=perm)

# Run analysis
results = run_gradcam_analysis(
    model,
    test_loader,
    num_per_class=5,
    results_dir='results/gradcam/'
)

# Access individual visualizations
for item in results['correct']:
    print(f"Label: {item['label']}, Pred: {item['pred']}")
    # item['image'], item['heatmap'], item['overlay'] are numpy arrays ready for plotting
```

### Feature Unscrambling Analysis
```python
from permuted_puzzle.feature_analysis import (
    extract_features,
    permute_feature_map,
    inverse_permutation,
    compare_features,
    get_compatible_layers
)
from permuted_puzzle.models import build_model
from permuted_puzzle.data import DogsVsCatsDataset
from permuted_puzzle.transforms import baseline_val_transforms, permute_image_tensor
import torch

# Load baseline and permuted models
baseline_model = build_model('resnet18', pretrained=False)
baseline_model.load_state_dict(torch.load('results/baseline_model.pth'))

permuted_model = build_model('resnet18', pretrained=False)
permuted_model.load_state_dict(torch.load('results/permuted_grid4_model.pth'))

# Find compatible layers for grid_size=4
grid_size = 4
compatible = get_compatible_layers(baseline_model, grid_size=grid_size)
print(f"Compatible layers for grid_size={grid_size}:")
for layer_name, (h, w) in compatible.items():
    print(f"  {layer_name}: {h}x{w}")

# Use the best compatible layer (first in dict = largest resolution)
layer_name = list(compatible.keys())[0]
print(f"\nUsing layer: {layer_name}")

# Load test image
dataset = DogsVsCatsDataset('data/train', transform=baseline_val_transforms(224, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))
image, label = dataset[0]

# Extract features from baseline model
f_original = extract_features(baseline_model, image, layer_name=layer_name, device='cuda')

# Create permuted version and extract features
permutation = [15, 2, 8, 1, 12, 6, 3, 9, 14, 5, 0, 11, 7, 13, 4, 10]  # 4x4 permutation
permuted_image = permute_image_tensor(image, permutation=permutation)
f_permuted = extract_features(permuted_model, permuted_image, layer_name=layer_name, device='cuda')

# Unscramble feature map using inverse permutation
inv_perm = inverse_permutation(permutation)
f_unscrambled = permute_feature_map(f_permuted, inv_perm, grid_size=grid_size)

# Compare similarity
similarity = compare_features(f_original, f_unscrambled)
print(f"\nSimilarity metrics:")
print(f"  Cosine Similarity: {similarity['cosine']:.4f}")
print(f"  Correlation:       {similarity['correlation']:.4f}")
print(f"  MSE:               {similarity['mse']:.4f}")
```
