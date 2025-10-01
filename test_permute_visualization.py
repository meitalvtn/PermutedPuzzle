#!/usr/bin/env python3
"""
Test script to visualize permute_image_tensor transformations.
"""

import torch
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
from torchvision import transforms

from permuted_puzzle.transforms import permute_image_tensor


def load_image_as_tensor(image_path):
    """Load an image and convert to tensor."""
    img = Image.open(image_path).convert('RGB')
    to_tensor = transforms.ToTensor()
    return to_tensor(img)


def tensor_to_numpy(tensor):
    """Convert tensor (C, H, W) to numpy array (H, W, C) for display."""
    return tensor.permute(1, 2, 0).cpu().numpy()


def visualize_permutations(image_path, grid_sizes=[2, 3, 4], seed=42):
    """
    Visualize original image and permuted versions with different grid sizes.

    Args:
        image_path: Path to the image file
        grid_sizes: List of grid sizes to test
        seed: Random seed for reproducible permutations
    """
    # Load image
    img_tensor = load_image_as_tensor(image_path)

    # Create figure
    n_plots = len(grid_sizes) + 1  # original + permuted versions
    fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 5))

    # Show original
    axes[0].imshow(tensor_to_numpy(img_tensor))
    axes[0].set_title('Original', fontsize=14)
    axes[0].axis('off')

    # Show permuted versions
    for idx, grid_size in enumerate(grid_sizes, start=1):
        torch.manual_seed(seed)
        permuted = permute_image_tensor(img_tensor, grid_size=grid_size)

        axes[idx].imshow(tensor_to_numpy(permuted))
        axes[idx].set_title(f'Grid {grid_size}x{grid_size}\n({grid_size**2} tiles)', fontsize=14)
        axes[idx].axis('off')

    plt.tight_layout()
    image_name = Path(image_path).stem
    plt.suptitle(f'Tile Permutation Visualization - {image_name}', fontsize=16, y=1.02)
    plt.show()


def visualize_specific_permutation(image_path, grid_size=3, permutation=None):
    """
    Visualize a specific permutation with tile numbering.

    Args:
        image_path: Path to the image file
        grid_size: Grid size (N for NxN tiles)
        permutation: Specific permutation to apply (None for random)
    """
    # Load image
    img_tensor = load_image_as_tensor(image_path)

    # Apply permutation
    if permutation is not None:
        torch.manual_seed(42)  # For consistent random if None
    permuted = permute_image_tensor(img_tensor, grid_size=grid_size, permutation=permutation)

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Show original
    axes[0].imshow(tensor_to_numpy(img_tensor))
    axes[0].set_title('Original', fontsize=14)
    axes[0].axis('off')

    # Show permuted
    axes[1].imshow(tensor_to_numpy(permuted))
    perm_str = str(permutation) if permutation else "random"
    axes[1].set_title(f'Permuted (grid {grid_size}x{grid_size})\n{perm_str}', fontsize=14)
    axes[1].axis('off')

    plt.tight_layout()
    plt.show()


def visualize_tile_tracking(image_path, grid_size=3, permutation=None, highlight_tile_idx=0):
    """
    Visualize where a specific tile moves from original to permuted position.

    Args:
        image_path: Path to the image file
        grid_size: Grid size (N for NxN tiles)
        permutation: Specific permutation to apply (None for random)
        highlight_tile_idx: Index of tile to highlight (default 0 = top-left)
    """
    # Load image
    img_tensor = load_image_as_tensor(image_path)

    # Apply permutation WITH highlighting
    permuted = permute_image_tensor(img_tensor, grid_size=grid_size,
                                   permutation=permutation,
                                   highlight_tile_idx=highlight_tile_idx)

    # Also create version without highlighting for reference
    original_with_border = permute_image_tensor(img_tensor, grid_size=1,
                                                highlight_tile_idx=None)

    # Add border to original for comparison
    img_highlighted = img_tensor.clone()
    C, H, W = img_tensor.shape
    tile_h, tile_w = H // grid_size, W // grid_size
    border_width = 3

    # Calculate position of highlighted tile
    row = highlight_tile_idx // grid_size
    col = highlight_tile_idx % grid_size

    # Add red border to that tile in original
    y1, y2 = row * tile_h, (row + 1) * tile_h
    x1, x2 = col * tile_w, (col + 1) * tile_w

    img_highlighted[0, y1:y1+border_width, x1:x2] = 1.0  # Top
    img_highlighted[1, y1:y1+border_width, x1:x2] = 0.0
    img_highlighted[2, y1:y1+border_width, x1:x2] = 0.0

    img_highlighted[0, y2-border_width:y2, x1:x2] = 1.0  # Bottom
    img_highlighted[1, y2-border_width:y2, x1:x2] = 0.0
    img_highlighted[2, y2-border_width:y2, x1:x2] = 0.0

    img_highlighted[0, y1:y2, x1:x1+border_width] = 1.0  # Left
    img_highlighted[1, y1:y2, x1:x1+border_width] = 0.0
    img_highlighted[2, y1:y2, x1:x1+border_width] = 0.0

    img_highlighted[0, y1:y2, x2-border_width:x2] = 1.0  # Right
    img_highlighted[1, y1:y2, x2-border_width:x2] = 0.0
    img_highlighted[2, y1:y2, x2-border_width:x2] = 0.0

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Show original with highlighted tile
    axes[0].imshow(tensor_to_numpy(img_highlighted))
    axes[0].set_title(f'Original (Tile {highlight_tile_idx} highlighted)', fontsize=14)
    axes[0].axis('off')

    # Show permuted with highlighted tile in new position
    axes[1].imshow(tensor_to_numpy(permuted))
    if permutation:
        new_pos = permutation.index(highlight_tile_idx)
        perm_str = f'Tile {highlight_tile_idx} → Position {new_pos}'
    else:
        perm_str = 'Random permutation'
    axes[1].set_title(f'Permuted (grid {grid_size}x{grid_size})\n{perm_str}', fontsize=14)
    axes[1].axis('off')

    plt.tight_layout()
    plt.show()


def visualize_tile_tracking_multiple(image_paths, grid_size=3, permutation=None, highlight_tile_idx=0):
    """
    Visualize where a specific tile moves across multiple images with the same permutation.
    This verifies that the fixed permutation puts the same tile in the same position across all images.

    Args:
        image_paths: List of image file paths
        grid_size: Grid size (N for NxN tiles)
        permutation: Specific permutation to apply
        highlight_tile_idx: Index of tile to highlight (default 0 = top-left)
    """
    num_images = len(image_paths)
    fig, axes = plt.subplots(num_images, 2, figsize=(10, 5 * num_images))

    if num_images == 1:
        axes = [axes]

    for idx, image_path in enumerate(image_paths):
        # Load image
        img_tensor = load_image_as_tensor(image_path)

        # Apply permutation WITH highlighting
        permuted = permute_image_tensor(img_tensor, grid_size=grid_size,
                                       permutation=permutation,
                                       highlight_tile_idx=highlight_tile_idx)

        # Add border to original for comparison
        img_highlighted = img_tensor.clone()
        C, H, W = img_tensor.shape
        tile_h, tile_w = H // grid_size, W // grid_size
        border_width = 3

        # Calculate position of highlighted tile
        row = highlight_tile_idx // grid_size
        col = highlight_tile_idx % grid_size

        # Add red border to that tile in original
        y1, y2 = row * tile_h, (row + 1) * tile_h
        x1, x2 = col * tile_w, (col + 1) * tile_w

        img_highlighted[0, y1:y1+border_width, x1:x2] = 1.0
        img_highlighted[1, y1:y1+border_width, x1:x2] = 0.0
        img_highlighted[2, y1:y1+border_width, x1:x2] = 0.0

        img_highlighted[0, y2-border_width:y2, x1:x2] = 1.0
        img_highlighted[1, y2-border_width:y2, x1:x2] = 0.0
        img_highlighted[2, y2-border_width:y2, x1:x2] = 0.0

        img_highlighted[0, y1:y2, x1:x1+border_width] = 1.0
        img_highlighted[1, y1:y2, x1:x1+border_width] = 0.0
        img_highlighted[2, y1:y2, x1:x1+border_width] = 0.0

        img_highlighted[0, y1:y2, x2-border_width:x2] = 1.0
        img_highlighted[1, y1:y2, x2-border_width:x2] = 0.0
        img_highlighted[2, y1:y2, x2-border_width:x2] = 0.0

        # Show original with highlighted tile
        axes[idx][0].imshow(tensor_to_numpy(img_highlighted))
        axes[idx][0].set_title(f'Image {idx+1}: Original', fontsize=12)
        axes[idx][0].axis('off')

        # Show permuted with highlighted tile
        axes[idx][1].imshow(tensor_to_numpy(permuted))
        axes[idx][1].set_title(f'Image {idx+1}: Permuted', fontsize=12)
        axes[idx][1].axis('off')

    if permutation:
        new_pos = permutation.index(highlight_tile_idx)
        suptitle = f'Tile Tracking: Tile {highlight_tile_idx} → Position {new_pos} (Grid {grid_size}x{grid_size})\nSame permutation applied to all images'
    else:
        suptitle = f'Tile Tracking (Grid {grid_size}x{grid_size})'

    plt.suptitle(suptitle, fontsize=14, y=0.995)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # Find a test image
    test_data_dir = Path(__file__).parent / 'test_data' / 'data'
    sample_images = list(test_data_dir.glob('*.jpg'))[:3]  # Use first 3 images

    if not sample_images:
        print("No sample images found in test_data/data")
        exit(1)

    print(f"Found {len(sample_images)} sample images")
    print(f"Using: {sample_images[0].name}")
    print()

    # Test 1: Show different grid sizes
    print("Test 1: Visualizing different grid sizes (2x2, 3x3, 4x4)")
    visualize_permutations(sample_images[0], grid_sizes=[8,11])

    # Test 2: Show a specific permutation
    print("\nTest 2: Specific permutation for 3x3 grid")
    # Example: reverse order permutation
    specific_perm = [8, 7, 6, 5, 4, 3, 2, 1, 0]
    visualize_specific_permutation(sample_images[0], grid_size=3, permutation=specific_perm)

    # Test 3: Track a specific tile through permutation (sanity check)
    print("\nTest 3: Tile tracking - see where tile 0 (top-left) moves to")
    visualize_tile_tracking(sample_images[0], grid_size=3, permutation=specific_perm, highlight_tile_idx=0)

    # Test 4: Verify fixed permutation works consistently across multiple images
    if len(sample_images) >= 3:
        print("\nTest 4: Fixed permutation sanity check - same tile position across different images")
        visualize_tile_tracking_multiple(sample_images[:3], grid_size=3,
                                        permutation=specific_perm, highlight_tile_idx=0)

    # Test 5: Show random permutations on different images
    if len(sample_images) > 1:
        print("\nTest 5: Random permutations on different images")
        for img_path in sample_images[:2]:
            print(f"Processing {img_path.name}")
            visualize_permutations(img_path, grid_sizes=[2, 4])
