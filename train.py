from typing import Dict, Any
import os
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
import torchvision
import matplotlib.pyplot as plt

from permuted_puzzle.models import REGISTRY
from permuted_puzzle.transforms import baseline_train_transforms, baseline_val_transforms
from permuted_puzzle.data import get_dataloaders, DogsVsCatsDataset, PermutedDogsVsCatsDataset
from permuted_puzzle.utils_io import save_run, save_preds


def train_model(
    model_name: str,
    data_path: str,
    out_path: str = "results",
    grid: int = 1,
    epochs: int = 10,
    batch_size: int = 64,
    lr: float = 1e-4,
    wd: float = 1e-4,
    dropout: float = 0.2,
    save_preds_flag: bool = False,
    visualize: bool = True,
) -> Dict[str, Any]:
    """
    Train a classification model on Dogs vs Cats, optionally with grid permutations.

    Returns:
        dict containing training history, best accuracy, and model
    """

    # Reproducibility
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # === Training configuration logs ===
    print("\n" + "=" * 60)
    print("TRAINING CONFIGURATION")
    print("=" * 60)
    print(f"Model:           {model_name}")
    print(f"Device:          {device}")
    print(f"Grid Size:       {grid}x{grid} {'(baseline - no permutation)' if grid == 1 else f'({grid**2} tiles)'}")
    print(f"Epochs:          {epochs}")
    print(f"Batch Size:      {batch_size}")
    print(f"Learning Rate:   {lr}")
    print(f"Weight Decay:    {wd}")
    print(f"Dropout:         {dropout}")
    print(f"Data Path:       {data_path}")
    print(f"Output Path:     {out_path}")
    print(f"Save Preds:      {save_preds_flag}")
    print("=" * 60 + "\n")

    # 1. Build model
    build_fn = REGISTRY[model_name]
    model, meta = build_fn(num_classes=2, pretrained=True, dropout=dropout)
    model = model.to(device)
    print(f"Model loaded: {model_name} (pretrained=True)")
    print(f"Input size: {meta['input_size']}, Mean: {meta['mean']}, Std: {meta['std']}\n")

    # 2. Transforms
    train_tfms = baseline_train_transforms(meta["input_size"], meta["mean"], meta["std"])
    val_tfms = baseline_val_transforms(meta["input_size"], meta["mean"], meta["std"])

    # 3. Data
    if grid == 1:
        print("Loading baseline data (no permutation)...")
        train_loader, val_loader = get_dataloaders(
            data_path, train_tfms, val_tfms, batch_size=batch_size
        )
    else:
        print(f"Loading data with permutation (grid {grid}x{grid})...")
        full_dataset = DogsVsCatsDataset(data_path, transform=None)
        n_val = math.floor(0.2 * len(full_dataset))
        n_train = len(full_dataset) - n_val
        train_subset, val_subset = torch.utils.data.random_split(full_dataset, [n_train, n_val])

        train_subset.dataset.transform = train_tfms
        val_subset.dataset.transform = val_tfms

        N = grid * grid
        fixed_perm = list(range(N))
        random.seed(0)
        random.shuffle(fixed_perm)
        print(f"Fixed permutation (seed=0): {fixed_perm}")

        train_ds = PermutedDogsVsCatsDataset(train_subset, grid_size=grid, permutation=fixed_perm)
        val_ds = PermutedDogsVsCatsDataset(val_subset, grid_size=grid, permutation=fixed_perm)

        train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
        val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    print(f"Train samples: {len(train_loader.dataset)}, Val samples: {len(val_loader.dataset)}\n")

    # Sanity check
    if visualize:
        print("--- RUNNING VISUALIZATION CHECK ---")
        data_batch, labels_batch = next(iter(train_loader))
        print(f"Grid size for this run: {grid}")
        print(f"Shape of a single image tensor: {data_batch[0].shape}")
        grid_img = torchvision.utils.make_grid(data_batch[:4], nrow=4)
        mean, std = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])
        grid_img = grid_img.permute(1, 2, 0).numpy()
        grid_img = std * grid_img + mean
        grid_img = np.clip(grid_img, 0, 1)
        plt.figure(figsize=(10, 10))
        plt.title("Are these images scrambled?")
        plt.imshow(grid_img)
        plt.axis("off")
        plt.show()

    # 4. Training setup
    print("\n=== Starting Training ===")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    scaler = GradScaler()
    print(f"Optimizer: AdamW (lr={lr}, wd={wd})")
    print(f"Criterion: CrossEntropyLoss")
    print(f"Mixed Precision: Enabled (GradScaler)\n")

    history = {"train_loss": [], "val_loss": [], "val_acc": []}
    best_val_acc, best_epoch = 0.0, -1

    Path(out_path).mkdir(parents=True, exist_ok=True)
    best_tmp_path = os.path.join(out_path, f"{model_name}_best_tmp.pth")

    def train_one_epoch(model, loader):
        model.train()
        total_loss, correct, total = 0, 0, 0
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)
        return total_loss / total, correct / total

    def evaluate(model, loader):
        model.eval()
        total_loss, correct, total = 0, 0, 0
        with torch.no_grad():
            for images, labels in loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                total_loss += loss.item() * images.size(0)
                correct += (outputs.argmax(1) == labels).sum().item()
                total += labels.size(0)
        return total_loss / total, correct / total

    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader)
        val_loss, val_acc = evaluate(model, val_loader)
        history["train_loss"].append(float(train_loss))
        history["val_loss"].append(float(val_loss))
        history["val_acc"].append(float(val_acc))

        print(f"Epoch {epoch+1}/{epochs}: "
              f"Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}, "
              f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc, best_epoch = val_acc, epoch + 1
            torch.save(model.state_dict(), best_tmp_path)

    if os.path.exists(best_tmp_path):
        model.load_state_dict(torch.load(best_tmp_path, map_location=device))

    print(f"\nBest Val Accuracy: {best_val_acc:.4f} at epoch {best_epoch}\n")

    # Save run artifacts
    save_run(
        results_root=out_path,
        model_name=model_name,
        grid_size=grid,
        model=model,
        history=history,
        meta=meta,
        hyper={"epochs": epochs, "batch_size": batch_size, "lr": lr, "wd": wd,
               "optimizer": "AdamW", "dropout": dropout},
        notes="Baseline, 224x224, ImageNet norm"
    )

    if save_preds_flag:
        save_preds(
            results_root=out_path,
            model_name=model_name,
            grid_size=grid,
            split="val",
            model=model,
            loader=val_loader,
            device=device,
        )

    if os.path.exists(best_tmp_path):
        os.remove(best_tmp_path)
        print(f"Removed temp checkpoint: {best_tmp_path}")

    print(f"Saved outputs under: {out_path}")

    return {"history": history, "best_val_acc": best_val_acc, "best_epoch": best_epoch, "model": model}
