import math
import os
import argparse

import numpy
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from pathlib import Path

import torchvision
from torch.cuda.amp import GradScaler, autocast
import random, numpy as np
import matplotlib.pyplot as plt

from permuted_puzzle.models import REGISTRY
from permuted_puzzle.transforms import baseline_train_transforms, baseline_val_transforms
from permuted_puzzle.data import get_dataloaders, DogsVsCatsDataset, PermutedDogsVsCatsDataset
from permuted_puzzle.utils_io import save_run, save_preds


def visualize_samples(loader, num_samples=3, grid_size=None):
    """Show sample images as sanity check before training."""
    # Get a batch
    images, labels = next(iter(loader))
    images = images[:num_samples]
    labels = labels[:num_samples]

    fig, axes = plt.subplots(1, num_samples, figsize=(5 * num_samples, 5))
    if num_samples == 1:
        axes = [axes]

    for idx, (img, label) in enumerate(zip(images, labels)):
        # Convert from tensor (C, H, W) to numpy (H, W, C)
        img_np = img.permute(1, 2, 0).cpu().numpy()
        # Denormalize if needed (assume ImageNet normalization)
        img_np = img_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        img_np = np.clip(img_np, 0, 1)

        axes[idx].imshow(img_np)
        class_name = "Dog" if label.item() == 1 else "Cat"
        title = f"{class_name}"
        if grid_size and grid_size > 1:
            title += f" (Grid {grid_size}x{grid_size})"
        axes[idx].set_title(title, fontsize=14)
        axes[idx].axis('off')

    plt.tight_layout()
    plt.show()


def train_one_epoch(model, loader, criterion, optimizer, scaler, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        with autocast():
            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total

def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)

    return running_loss / total, correct / total

def main(args):
    # Reproducibility
    torch.manual_seed(0); random.seed(0); numpy.random.seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Training configuration logs
    print("\n" + "="*60)
    print("TRAINING CONFIGURATION")
    print("="*60)
    print(f"Model:           {args.model}")
    print(f"Device:          {device}")
    print(f"Grid Size:       {args.grid}x{args.grid} {'(baseline - no permutation)' if args.grid == 1 else f'({args.grid**2} tiles)'}")
    print(f"Epochs:          {args.epochs}")
    print(f"Batch Size:      {args.batch_size}")
    print(f"Learning Rate:   {args.lr}")
    print(f"Weight Decay:    {args.wd}")
    print(f"Dropout:         {args.dropout}")
    print(f"Data Path:       {args.data}")
    print(f"Output Path:     {args.out}")
    print(f"Save Preds:      {args.save_preds}")
    print("="*60 + "\n")

    # 1. Build model + meta
    build_fn = REGISTRY[args.model]
    model, meta = build_fn(num_classes=2, pretrained=True, dropout=args.dropout)
    model = model.to(device)
    print(f"Model loaded: {args.model} (pretrained=True)")
    print(f"Input size: {meta['input_size']}, Mean: {meta['mean']}, Std: {meta['std']}\n")

    # 2. Transforms
    train_tfms = baseline_train_transforms(meta["input_size"], meta["mean"], meta["std"])
    val_tfms   = baseline_val_transforms(meta["input_size"], meta["mean"], meta["std"])

    # 3. Data
    if args.grid == 1:
        # Baseline (no permutation)
        print("Loading baseline data (no permutation)...")
        train_loader, val_loader = get_dataloaders(
            args.data, train_tfms, val_tfms, batch_size=args.batch_size
        )
        # This is the corrected code
    else:
        # Permuted dataset
        print(f"Loading data with permutation (grid {args.grid}x{args.grid})...")

        # 1. Create ONE full dataset without transforms first
        full_dataset = DogsVsCatsDataset(args.data, transform=None)

        # 2. Perform the train/val split
        n_total = len(full_dataset)
        n_val = math.floor(0.2 * n_total)  # Assuming 20% validation split
        n_train = n_total - n_val
        train_subset, val_subset = torch.utils.data.random_split(full_dataset, [n_train, n_val])

        # 3. Now, assign the correct transforms to each subset
        # NOTE: This is how you assign transforms to a Subset object
        train_subset.dataset.transform = train_tfms
        val_subset.dataset.transform = val_tfms

        # 4. Create a fixed permutation
        N = args.grid * args.grid
        fixed_perm = list(range(N))
        random.seed(0)
        random.shuffle(fixed_perm)
        print(f"Fixed permutation (seed=0): {fixed_perm}")

        # 5. Wrap the SUBSETS with the permutation layer
        train_ds = PermutedDogsVsCatsDataset(train_subset, grid_size=args.grid, permutation=fixed_perm)
        val_ds = PermutedDogsVsCatsDataset(val_subset, grid_size=args.grid, permutation=fixed_perm)

        # 6. Create the final DataLoaders
        train_loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
        val_loader = torch.utils.data.DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    print(f"Train samples: {len(train_loader.dataset)}, Val samples: {len(val_loader.dataset)}\n")

    # Sanity check: visualize samples before training
    print("--- RUNNING VISUALIZATION CHECK ---")
    # Get one batch from the dataloader
    data_batch, labels_batch = next(iter(train_loader))
    print(f"Grid size for this run should be: {args.grid}")
    print(f"Shape of a single image tensor: {data_batch[0].shape}")

    # Create a grid of images to display
    grid_img = torchvision.utils.make_grid(data_batch[:4], nrow=4)

    # We need to un-normalize the image to see it correctly
    # IMPORTANT: Adjust these mean/std values to match your transforms
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    grid_img = grid_img.permute(1, 2, 0).numpy()
    grid_img = std * grid_img + mean
    grid_img = np.clip(grid_img, 0, 1)

    plt.figure(figsize=(10, 10))
    plt.title("Are these images scrambled?")
    plt.imshow(grid_img)
    plt.axis('off')
    plt.show()

    # 4. Training setup
    print("\n=== Starting Training ===")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scaler = GradScaler()
    print(f"Optimizer: AdamW (lr={args.lr}, wd={args.wd})")
    print(f"Criterion: CrossEntropyLoss")
    print(f"Mixed Precision: Enabled (GradScaler)\n")

    # 5. Training loop
    history = {"train_loss": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0.0
    best_epoch = -1

    Path(args.out).mkdir(parents=True, exist_ok=True)
    best_tmp_path = os.path.join(args.out, f"{args.model}_best_tmp.pth")

    for epoch in range(args.epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        history["train_loss"].append(float(train_loss))
        history["val_loss"].append(float(val_loss))
        history["val_acc"].append(float(val_acc))
        print(f"Epoch {epoch+1}/{args.epochs}: "
              f"Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}, "
              f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            torch.save(model.state_dict(), best_tmp_path)

    # Reload best weights before saving
    if os.path.exists(best_tmp_path):
        model.load_state_dict(torch.load(best_tmp_path, map_location=device))

    print(f"Best Val Accuracy: {best_val_acc:.4f} at epoch {best_epoch}")

    hyper = {
        "epochs": args.epochs, "batch_size": args.batch_size,
        "lr": args.lr, "wd": args.wd, "optimizer": "AdamW",
        "dropout": args.dropout
    }
    save_run(
        results_root=args.out,
        model_name=args.model,
        grid_size=args.grid,
        model=model,
        history=history,
        meta=meta,
        hyper=hyper,
        notes="Baseline, 224x224, ImageNet norm"
    )

    if args.save_preds:
        save_preds(
            results_root=args.out,
            model_name=args.model,
            grid_size=args.grid,
            split="val",
            model=model,
            loader=val_loader,
            device=device,
        )

    print(f"Saved outputs under: {args.out}")

    # Clean up temp checkpoint
    if os.path.exists(best_tmp_path):
        os.remove(best_tmp_path)
        print(f"Removed temp checkpoint: {best_tmp_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="resnet18", choices=list(REGISTRY.keys()))
    parser.add_argument("--data", type=str, required=True, help="Path to dataset folder")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--wd", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--out", type=str, default="results", help="Root folder to save outputs")
    parser.add_argument("--grid", type=int, default=1, help="Grid size (1 = baseline)")
    parser.add_argument("--save_preds", action="store_true", help="Also dump val preds to NPZ")

    args = parser.parse_args()

    main(args)
