import math
import os
from PIL import Image
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


def get_dataloaders(img_dir, train_tfms, val_tfms, batch_size=64, val_split=0.2):
    # Get dataset size
    full_dataset = DogsVsCatsDataset(img_dir, transform=None)
    n_total = len(full_dataset)
    n_val = math.floor(val_split * n_total)
    n_train = n_total - n_val

    # Generate random permutation of indices (reproducible, seed set in train_model)
    indices = torch.randperm(n_total).tolist()
    train_indices = indices[:n_train]
    val_indices = indices[n_train:]

    # Create two independent base datasets with transforms
    train_base = DogsVsCatsDataset(img_dir, transform=train_tfms)
    val_base = DogsVsCatsDataset(img_dir, transform=val_tfms)

    # Subset them with the split indices
    train_ds = Subset(train_base, train_indices)
    val_ds = Subset(val_base, val_indices)

    # DataLoaders
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, val_loader