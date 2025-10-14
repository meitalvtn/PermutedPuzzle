"""
Test script for get_path_by_filename helper method.
"""

import tempfile
import os
from pathlib import Path
from PIL import Image
import numpy as np
from torch.utils.data import Subset

from permuted_puzzle.data import DogsVsCatsDataset, PermutedDogsVsCatsDataset


def create_dummy_images(img_dir, num_images=5):
    """Create dummy test images."""
    filenames = []
    for i in range(num_images):
        if i % 2 == 0:
            filename = f"cat.{i}.jpg"
        else:
            filename = f"dog.{i}.jpg"

        img_path = os.path.join(img_dir, filename)
        # Create a simple RGB image
        img = Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8))
        img.save(img_path)
        filenames.append(filename)

    return filenames


def test_dogs_vs_cats_dataset():
    """Test get_path_by_filename on DogsVsCatsDataset."""
    print("Testing DogsVsCatsDataset.get_path_by_filename...")

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create dummy images
        filenames = create_dummy_images(tmpdir, num_images=5)

        # Create dataset
        dataset = DogsVsCatsDataset(tmpdir, transform=None)

        # Test valid filenames
        for filename in filenames:
            path = dataset.get_path_by_filename(filename)
            expected_path = os.path.join(tmpdir, filename)

            assert path == expected_path, f"Expected {expected_path}, got {path}"
            assert os.path.exists(path), f"Path {path} does not exist"

        print(f"  ✓ Successfully retrieved paths for {len(filenames)} images")

        # Test invalid filename
        try:
            dataset.get_path_by_filename("nonexistent.jpg")
            assert False, "Should have raised ValueError"
        except ValueError as e:
            print(f"  ✓ Correctly raised ValueError for invalid filename")

    print("  PASS\n")


def test_permuted_dataset():
    """Test get_path_by_filename on PermutedDogsVsCatsDataset."""
    print("Testing PermutedDogsVsCatsDataset.get_path_by_filename...")

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create dummy images
        filenames = create_dummy_images(tmpdir, num_images=5)

        # Create base dataset
        base_dataset = DogsVsCatsDataset(tmpdir, transform=None)

        # Create permuted dataset
        permutation = [3, 1, 2, 0]  # 2x2 permutation
        permuted_dataset = PermutedDogsVsCatsDataset(base_dataset, permutation)

        # Test valid filenames
        for filename in filenames:
            path = permuted_dataset.get_path_by_filename(filename)
            expected_path = os.path.join(tmpdir, filename)

            assert path == expected_path, f"Expected {expected_path}, got {path}"
            assert os.path.exists(path), f"Path {path} does not exist"

        print(f"  ✓ Successfully retrieved paths through permuted wrapper")

    print("  PASS\n")


def test_subset_dataset():
    """Test get_path_by_filename on Subset(DogsVsCatsDataset)."""
    print("Testing Subset -> PermutedDogsVsCatsDataset delegation...")

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create dummy images
        filenames = create_dummy_images(tmpdir, num_images=5)

        # Create base dataset
        base_dataset = DogsVsCatsDataset(tmpdir, transform=None)

        # Create subset (simulates what create_loader does)
        subset = Subset(base_dataset, [0, 1, 2])

        # Wrap in permuted dataset
        permutation = [3, 1, 2, 0]  # 2x2 permutation
        permuted_dataset = PermutedDogsVsCatsDataset(subset, permutation)

        # Test valid filenames
        for filename in filenames:
            path = permuted_dataset.get_path_by_filename(filename)
            expected_path = os.path.join(tmpdir, filename)

            assert path == expected_path, f"Expected {expected_path}, got {path}"
            assert os.path.exists(path), f"Path {path} does not exist"

        print(f"  ✓ Successfully retrieved paths through Subset wrapper")

    print("  PASS\n")


if __name__ == "__main__":
    print("=" * 60)
    print("Testing get_path_by_filename helper method")
    print("=" * 60 + "\n")

    test_dogs_vs_cats_dataset()
    test_permuted_dataset()
    test_subset_dataset()

    print("=" * 60)
    print("ALL TESTS PASSED!")
    print("=" * 60)
