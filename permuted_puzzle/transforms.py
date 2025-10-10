import random

import torch
from torchvision import transforms

def baseline_train_transforms(input_size: int, mean, std):
    """
    Create the standard training data transformation pipeline.

    Applies random spatial augmentations and normalization consistent
    with ImageNet-style preprocessing.

    Args:
        input_size (int): Final spatial size (height and width) expected by the model.
        mean (list or tuple): Per-channel mean for normalization (e.g. ImageNet mean).
        std (list or tuple): Per-channel standard deviation for normalization.

    Returns:
        torchvision.transforms.Compose: Composed transform performing:
            - RandomResizedCrop(input_size, scale=(0.7, 1.0), ratio=(0.9, 1.1))
            - RandomHorizontalFlip()
            - ToTensor()
            - Normalize(mean, std)

    Notes:
        The crop and flip augmentations introduce moderate variation
        while preserving semantic content, helping prevent overfitting
        on small or medium-sized datasets.
    """
    return transforms.Compose([
        transforms.RandomResizedCrop(input_size, scale=(0.7, 1.0), ratio=(0.9, 1.1)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

def baseline_val_transforms(input_size: int, mean, std):
    """
        Create the standard validation data transformation pipeline.

        Applies deterministic resizing, cropping, and normalization for
        consistent evaluation. Mirrors standard ImageNet validation
        preprocessing, where images are resized slightly larger and then
        center-cropped to the model's input size.

        Args:
            input_size (int): Final spatial size (height and width) expected by the model.
            mean (list or tuple): Per-channel mean for normalization (e.g. ImageNet mean).
            std (list or tuple): Per-channel standard deviation for normalization.

        Returns:
            torchvision.transforms.Compose: Composed transform performing:
                - Resize(int(input_size * 1.14))
                - CenterCrop(input_size)
                - ToTensor()
                - Normalize(mean, std)

        Notes:
            The factor 1.14 approximates the traditional 256→224 scaling used
            in ImageNet pipelines (256/224 ≈ 1.14).
        """
    return transforms.Compose([
        transforms.Resize(int(input_size * 1.14)),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

def permute_image_tensor(
    image_tensor,
    permutation=None,
    grid_size=None,
    random_seed=None
):
    """
    Split image into N×N tiles and permute them using vectorized operations.

    Args:
        image_tensor: torch.Tensor of shape (C, H, W)
        permutation: list of length N² indicating new tile order.
                    If provided, grid_size is automatically inferred.
        grid_size: number of tiles per dimension (N). Only required if permutation=None.
        random_seed: optional seed for random permutation generation.

    Returns:
        torch.Tensor of shape (C, H, W) with permuted tiles

    Raises:
        ValueError: If both permutation and grid_size are None, or if permutation length is not a perfect square.
    """
    # Infer grid_size from permutation
    if permutation is not None:
        n_tiles = len(permutation)
        grid_size = int(n_tiles ** 0.5)
        if grid_size * grid_size != n_tiles:
            raise ValueError(f"Permutation length must be a perfect square, got {n_tiles}")
    elif grid_size is None:
        raise ValueError("Either permutation or grid_size must be provided")

    # Generate permutation if needed
    if permutation is None:
        permutation = list(range(grid_size * grid_size))
        if random_seed is not None:
            random.seed(random_seed)
            random.shuffle(permutation)

    C, H, W = image_tensor.shape
    tile_h, tile_w = H // grid_size, W // grid_size

    # Reshape: (C, H, W) -> (C, grid_size, tile_h, grid_size, tile_w)
    tiles = image_tensor.reshape(C, grid_size, tile_h, grid_size, tile_w)

    # Transpose: (C, grid_size, tile_h, grid_size, tile_w) -> (C, grid_size, grid_size, tile_h, tile_w)
    tiles = tiles.permute(0, 1, 3, 2, 4)

    # Flatten tiles: (C, grid_size, grid_size, tile_h, tile_w) -> (C, N², tile_h, tile_w)
    tiles = tiles.reshape(C, grid_size * grid_size, tile_h, tile_w)

    # Apply permutation
    tiles = tiles[:, permutation, :, :]

    # Reshape back: (C, N², tile_h, tile_w) -> (C, grid_size, grid_size, tile_h, tile_w)
    tiles = tiles.reshape(C, grid_size, grid_size, tile_h, tile_w)

    # Transpose back: (C, grid_size, grid_size, tile_h, tile_w) -> (C, grid_size, tile_h, grid_size, tile_w)
    tiles = tiles.permute(0, 1, 3, 2, 4)

    # Reconstruct: -> (C, H, W)
    return tiles.reshape(C, H, W)
