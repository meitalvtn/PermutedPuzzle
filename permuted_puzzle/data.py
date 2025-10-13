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
    """
    Dataset for Dogs vs Cats classification from image directory.

    Loads images from a directory, automatically assigns labels based on filename
    (images with 'cat' in the name get label 0, others get label 1).

    Args:
        img_dir (str): Path to directory containing image files
        transform (callable, optional): Transform to apply to images

    Returns:
        (Tensor, int, str): Image tensor, label, and original filename
    """
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

        return image, label, img_name

class PermutedDogsVsCatsDataset(Dataset):
    """
    Dataset wrapper that applies a fixed N×N tile permutation.

    Divides each input image into grid_size×grid_size tiles and reorders them
    according to the provided permutation. Grid size is automatically inferred
    from the permutation length.

    Args:
        base_dataset (Dataset): Underlying dataset to wrap (e.g., DogsVsCatsDataset).
        permutation (list[int]): List of tile indices defining the permutation.
                                Length must be a perfect square (N²).

    Returns:
        (Tensor, int, str): Permuted image tensor, label, and original filename
    """
    def __init__(self, base_dataset, permutation):
        """
        Args:
            base_dataset: original DogsVsCatsDataset
            permutation: fixed tile permutation (list of length N*N)
        """
        if permutation is None:
            raise ValueError("permutation is required for PermutedDogsVsCatsDataset")

        self.base_dataset = base_dataset
        self.permutation = permutation

        # Validate and infer grid_size from permutation
        n_tiles = len(permutation)
        grid_size = int(n_tiles ** 0.5)
        if grid_size * grid_size != n_tiles:
            raise ValueError(f"Permutation length must be a perfect square, got {n_tiles}")
        self.grid_size = grid_size

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        image, label, filename = self.base_dataset[idx]
        permuted_image = permute_image_tensor(
            image, permutation=self.permutation
        )
        return permuted_image, label, filename


def split_indices(
    n_samples: int,
    splits: List[float]
) -> Dict[str, np.ndarray]:
    """
    Split dataset indices into multiple sets (e.g., train/val/test).

    Args:
        n_samples: Total number of samples
        splits: List of split ratios (must sum to 1.0), e.g., [0.6, 0.2, 0.2]

    Returns:
        Dict mapping split names to index arrays, e.g.,
        {'train': array([...]), 'val': array([...]), 'test': array([...])}

    Note:
        The generated split indices should be saved (e.g., in metrics.json)
        for reproducibility rather than relying on random seeds.
    """
    if not math.isclose(sum(splits), 1.0, rel_tol=1e-5):
        raise ValueError(f"Splits must sum to 1.0, got {sum(splits)}")

    # Generate shuffled indices
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


def generate_permutation(grid_size: int) -> List[int]:
    """
    Generate random NxN tile permutation.

    Args:
        grid_size: Grid dimension (N for NxN grid)

    Returns:
        List of length grid_size^2 representing tile permutation

    Note:
        The generated permutation should be saved (e.g., in metrics.json)
        for reproducibility rather than relying on random seeds.
    """
    n_tiles = grid_size * grid_size
    perm = list(range(n_tiles))
    random.shuffle(perm)
    return perm


def create_loader(
    dataset: Dataset,
    indices: np.ndarray,
    transform,
    permutation: Optional[List[int]] = None,
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
        permutation: Optional tile permutation (if None, no permutation applied).
                    Grid size is automatically inferred from permutation length.
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
        subset = PermutedDogsVsCatsDataset(subset, permutation=permutation)

    # Create DataLoader
    loader = DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers
    )

    return loader
