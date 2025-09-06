import random

import torch
from torchvision import transforms

def baseline_train_transforms(input_size: int, mean, std):
    return transforms.Compose([
        transforms.RandomResizedCrop(input_size),
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

def permute_image_tensor(image_tensor, grid_size=2, permutation=None):
    """
    Split and permute a square image tensor (C, H, W).

    Args:
        image_tensor: torch.Tensor of shape (C, H, W)
        grid_size: number of tiles per dimension (N)
        permutation: optional list of length N*N indicating new tile order

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
                                j * tile_w : (j + 1) * tile_w]
            tiles.append(tile)

    # Step 2: Permute tiles
    if permutation is None:
        permutation = list(range(len(tiles)))
        random.shuffle(permutation)

    permuted_tiles = [tiles[i] for i in permutation]

    # Step 3: Reconstruct image
    rows = []
    for i in range(grid_size):
        row = torch.cat(permuted_tiles[i * grid_size : (i + 1) * grid_size], dim=2)
        rows.append(row)
    new_image = torch.cat(rows, dim=1)

    return new_image
