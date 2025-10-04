import random

import torch
from torchvision import transforms

def baseline_train_transforms(input_size: int, mean, std):
    return transforms.Compose([
        transforms.RandomResizedCrop(input_size, scale=(0.7, 1.0), ratio=(0.9, 1.1)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

def baseline_val_transforms(input_size: int, mean, std):
    return transforms.Compose([
        transforms.Resize(int(input_size * 1.14)),  # classic 256 → center 224 style
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

def permute_image_tensor(
    image_tensor,
    grid_size=2,
    permutation=None,
    random_seed=None,
    highlight_tile_idx=None,
    border_width=3
):
    """
    Split and permute a square image tensor (C, H, W).

    Args:
        image_tensor: torch.Tensor of shape (C, H, W)
        grid_size: number of tiles per dimension (N)
        permutation: optional list of length N*N indicating new tile order.
                    If None, uses identity permutation (no shuffle).
        random_seed: optional int, if provided generates random permutation with this seed.
                    Only used when permutation=None. Makes randomness explicit.
        highlight_tile_idx: optional int, tile index to highlight with border (for debugging)
        border_width: width of highlight border in pixels (default 3)

    Returns:
        torch.Tensor of shape (C, H, W) with permuted tiles
    """
    C, H, W = image_tensor.shape
    tile_h, tile_w = H // grid_size, W // grid_size

    # Step 1: Split into tiles
    tiles = []
    for i in range(grid_size):
        for j in range(grid_size):
            tile = image_tensor[:,
                                i * tile_h : (i + 1) * tile_h,
                                j * tile_w : (j + 1) * tile_w].clone()
            tiles.append(tile)

    # Add border to highlight tile (before permutation)
    if highlight_tile_idx is not None and 0 <= highlight_tile_idx < len(tiles):
        tile = tiles[highlight_tile_idx]
        # Add red border (RGB channels)
        if C >= 3:
            # Top border
            tile[0, :border_width, :] = 1.0  # R
            tile[1, :border_width, :] = 0.0  # G
            tile[2, :border_width, :] = 0.0  # B
            # Bottom border
            tile[0, -border_width:, :] = 1.0
            tile[1, -border_width:, :] = 0.0
            tile[2, -border_width:, :] = 0.0
            # Left border
            tile[0, :, :border_width] = 1.0
            tile[1, :, :border_width] = 0.0
            tile[2, :, :border_width] = 0.0
            # Right border
            tile[0, :, -border_width:] = 1.0
            tile[1, :, -border_width:] = 0.0
            tile[2, :, -border_width:] = 0.0

    # Step 2: Permute tiles
    if permutation is None:
        if random_seed is not None:
            # Explicit random permutation requested
            import random
            permutation = list(range(len(tiles)))
            random.seed(random_seed)
            random.shuffle(permutation)
        else:
            # No permutation specified - use identity (no shuffle)
            permutation = list(range(len(tiles)))

    permuted_tiles = [tiles[i] for i in permutation]

    # Step 3: Reconstruct image
    rows = []
    for i in range(grid_size):
        row = torch.cat(permuted_tiles[i * grid_size : (i + 1) * grid_size], dim=2)
        rows.append(row)
    new_image = torch.cat(rows, dim=1)

    return new_image
