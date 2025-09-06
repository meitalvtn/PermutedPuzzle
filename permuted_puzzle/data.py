import math
import os
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, random_split


from permuted_puzzle.transforms import permute_image_tensor


class DogsVsCatsDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.image_filenames = [
            fname for fname in os.listdir(img_dir)
            if fname.endswith(".jpg")
        ]

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
    full_dataset = DogsVsCatsDataset(img_dir, transform=None)
    n_total = len(full_dataset)
    n_val = math.floor(val_split * n_total)
    n_train = n_total - n_val

    # Split
    train_ds, val_ds = random_split(full_dataset, [n_train, n_val])

    # Re-wrap with transforms
    train_ds.dataset.transform = train_tfms
    val_ds.dataset.transform = val_tfms

    # DataLoaders
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, val_loader