import math
import os
import random
from typing import Dict, List, Optional
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, Subset


from permuted_puzzle.transforms import permute_image_tensor


class DogsVsCatsDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.image_filenames = sorted([
            fname for fname in os.listdir(img_dir)
            if fname.endswith(".jpg")
        ])

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_name = self.image_filenames[idx]
        label = 0 if "cat" in img_name else 1  # 0: cat, 1: dog
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        return image, label

class PermutedDogsVsCatsDataset(Dataset):
    def __init__(self, base_dataset, grid_size=2, permutation=None):
        """
        Args:
            base_dataset: original DogsVsCatsDataset
            grid_size: NxN tile grid to permute
            permutation: fixed tile permutation (list of length N*N), or None for random
        """
        self.base_dataset = base_dataset
        self.grid_size = grid_size
        self.permutation = permutation

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        image, label = self.base_dataset[idx]
        permuted_image = permute_image_tensor(
            image, grid_size=self.grid_size, permutation=self.permutation
        )
        return permuted_image, label


def split_indices(
    n_samples: int,
    splits: List[float],
    seed: int = 0
) -> Dict[str, np.ndarray]:
    """
    Split dataset indices into multiple sets (e.g., train/val/test).

    Args:
        n_samples: Total number of samples
        splits: List of split ratios (must sum to 1.0), e.g., [0.6, 0.2, 0.2]
        seed: Random seed for reproducibility

    Returns:
        Dict mapping split names to index arrays, e.g.,
        {'train': array([...]), 'val': array([...]), 'test': array([...])}
    """
    if not math.isclose(sum(splits), 1.0, rel_tol=1e-5):
        raise ValueError(f"Splits must sum to 1.0, got {sum(splits)}")

    # Generate shuffled indices
    torch.manual_seed(seed)
    indices = torch.randperm(n_samples).numpy()

    # Split indices
    split_names = ['train', 'val', 'test'][:len(splits)]
    result = {}
    start = 0

    for i, (name, ratio) in enumerate(zip(split_names, splits)):
        if i == len(splits) - 1:
            # Last split gets all remaining samples
            result[name] = indices[start:]
        else:
            count = int(n_samples * ratio)
            result[name] = indices[start:start + count]
            start += count

    return result


def generate_permutation(grid_size: int, seed: int) -> List[int]:
    """
    Generate deterministic NxN tile permutation.

    Args:
        grid_size: Grid dimension (N for NxN grid)
        seed: Random seed for reproducibility

    Returns:
        List of length grid_size^2 representing tile permutation
    """
    random.seed(seed)
    n_tiles = grid_size * grid_size
    perm = list(range(n_tiles))
    random.shuffle(perm)
    return perm


def create_loader(
    dataset: Dataset,
    indices: np.ndarray,
    transform,
    permutation: Optional[List[int]] = None,
    grid_size: int = 1,
    batch_size: int = 64,
    shuffle: bool = False,
    num_workers: int = 0
) -> DataLoader:
    """
    Create DataLoader from dataset subset with optional permutation.

    Args:
        dataset: Base dataset (DogsVsCatsDataset)
        indices: Indices to include in this loader
        transform: Torchvision transforms to apply
        permutation: Optional tile permutation (if None, no permutation applied)
        grid_size: Grid dimension for permutation (only used if permutation is not None)
        batch_size: Batch size
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes

    Returns:
        DataLoader with optional permutation applied
    """
    # Create dataset with transform
    # Check if dataset has img_dir attribute (DogsVsCatsDataset)
    if hasattr(dataset, 'img_dir'):
        dataset_with_transform = type(dataset)(dataset.img_dir, transform=transform)
    else:
        # For other dataset types, assume they accept transform in constructor
        try:
            dataset_with_transform = type(dataset)(transform=transform)
        except TypeError:
            raise TypeError(
                f"create_loader requires dataset to either have 'img_dir' attribute "
                f"or accept 'transform' parameter. Got {type(dataset).__name__}"
            )

    # Subset to specified indices
    subset = Subset(dataset_with_transform, indices.tolist())

    # Apply permutation if specified
    if permutation is not None:
        subset = PermutedDogsVsCatsDataset(subset, grid_size=grid_size, permutation=permutation)

    # Create DataLoader
    loader = DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers
    )

    return loader
