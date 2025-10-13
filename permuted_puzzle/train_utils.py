from typing import Dict, Any, Optional
import os
from pathlib import Path

import logging
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
import torchvision
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, stream=sys.stdout, format="%(message)s", force=True)
logger = logging.getLogger("train")
logger.setLevel(logging.INFO)
logger.propagate = False
_handler = logging.StreamHandler(sys.stdout)
_handler.setFormatter(logging.Formatter("%(message)s"))
logger.handlers = [_handler]

from .utils_io import save_run, save_preds


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: str = "cuda",
    criterion = None
) -> Dict[str, float]:
    """
    Evaluate model on a data loader.

    Args:
        model: PyTorch model
        loader: DataLoader to evaluate on
        device: Device to use ('cuda' or 'cpu')
        criterion: Loss function (if None, uses CrossEntropyLoss)

    Returns:
        Dict with 'loss' and 'accuracy' keys
    """
    if criterion is None:
        criterion = nn.CrossEntropyLoss()

    device_obj = torch.device(device)
    model = model.to(device_obj)
    model.eval()

    total_loss, correct, total = 0, 0, 0

    with torch.no_grad():
        for batch in loader:
            # Unpack batch (handles both 2-tuple and 3-tuple returns)
            if len(batch) == 3:
                images, labels, _ = batch  # Ignore filenames
            else:
                images, labels = batch
            images, labels = images.to(device_obj), labels.to(device_obj)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)

    return {
        'loss': total_loss / total,
        'accuracy': correct / total
    }


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 10,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    device: str = "cuda",
    out_path: Optional[str] = None,
    config: Optional[Dict] = None,
) -> Dict[str, Any]:
    """
    Train a classification model.

    Args:
        model: PyTorch model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        epochs: Number of training epochs
        lr: Learning rate
        weight_decay: Weight decay for optimizer
        device: Device to use ('cuda' or 'cpu')
        out_path: Path to save outputs (checkpoints, metrics, predictions)
        config: Optional config dict for logging (can include model_name, permutations, etc.)

    Returns:
        Dict containing:
            - 'history': Training history with train/val losses and accuracies
            - 'best_val_acc': Best validation accuracy achieved
            - 'best_epoch': Epoch with best validation accuracy
            - 'model': Trained model (with best checkpoint loaded)
    """
    device_obj = torch.device(device)
    model = model.to(device_obj)

    # Extract config values
    config = config or {}
    model_name = config.get('model_name', 'model')
    grid_size = config.get('grid_size', 1)
    train_perm = config.get('train_permutation')
    val_perm = config.get('val_permutation')
    test_perm = config.get('test_permutation')
    split_indices = config.get('split_indices')
    meta = config.get('meta', {'input_size': 224, 'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]})

    # Build permutations dict for saving
    permutations = {}
    if train_perm is not None:
        permutations['train'] = train_perm
    if val_perm is not None:
        permutations['val'] = val_perm
    if test_perm is not None:
        permutations['test'] = test_perm

    # === Training configuration logs ===
    logger.info("\n" + "=" * 80)
    logger.info("TRAINING CONFIGURATION")
    logger.info("=" * 80)
    logger.info("")
    logger.info("MODEL")
    logger.info(f"  Architecture:        {model_name}")
    logger.info(f"  Pretrained:          {config.get('pretrained', 'Not specified')}")
    logger.info(f"  Number of Classes:   {config.get('num_classes', 2)}")
    logger.info(f"  Input Size:          {meta['input_size']}")
    logger.info(f"  Normalization:       Mean={meta['mean']}, Std={meta['std']}")
    logger.info("")
    logger.info("DATA")
    logger.info(f"  Train Samples:       {len(train_loader.dataset)}")
    logger.info(f"  Val Samples:         {len(val_loader.dataset)}")
    logger.info(f"  Batch Size:          {train_loader.batch_size}")
    logger.info(f"  Grid Size:           {grid_size}x{grid_size}")
    if train_perm:
        logger.info(f"  Train Permutation:   {train_perm}")
    if val_perm and val_perm != train_perm:
        logger.info(f"  Val Permutation:     {val_perm}")
    logger.info("")
    logger.info("TRAINING")
    logger.info(f"  Epochs:              {epochs}")
    logger.info(f"  Optimizer:           AdamW")
    logger.info(f"  Learning Rate:       {lr}")
    logger.info(f"  Weight Decay:        {weight_decay}")
    logger.info(f"  Loss Function:       CrossEntropyLoss")
    logger.info(f"  Mixed Precision:     {'Enabled (GradScaler)' if torch.cuda.is_available() else 'Disabled (CPU)'}")
    logger.info("")
    logger.info("DEVICE")
    logger.info(f"  Device:              {device_obj}")
    if torch.cuda.is_available():
        logger.info(f"  GPU Name:            {torch.cuda.get_device_name(0)}")
        logger.info(f"  GPU Memory:          {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    logger.info("=" * 80 + "\n")

    # Visualize sample batch
    logger.info("--- RUNNING VISUALIZATION CHECK ---")
    batch = next(iter(train_loader))
    # Unpack batch (handles both 2-tuple and 3-tuple returns)
    if len(batch) == 3:
        data_batch, labels_batch, _ = batch
    else:
        data_batch, labels_batch = batch
    logger.info(f"Shape of a single image tensor: {data_batch[0].shape}")
    grid_img = torchvision.utils.make_grid(data_batch[:4], nrow=4)
    mean, std = np.array(meta['mean']), np.array(meta['std'])
    grid_img = grid_img.permute(1, 2, 0).numpy()
    grid_img = std * grid_img + mean
    grid_img = np.clip(grid_img, 0, 1)
    plt.figure(figsize=(10, 10))
    plt.title(f"Sample Training Images (Grid {grid_size}x{grid_size})")
    plt.imshow(grid_img)
    plt.axis("off")
    plt.show()

    # Training setup
    logger.info("\n=== Starting Training ===")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scaler = GradScaler('cuda') if torch.cuda.is_available() else None
    logger.info(f"Optimizer: AdamW (lr={lr}, wd={weight_decay})")
    logger.info(f"Criterion: CrossEntropyLoss\n")

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc, best_epoch = 0.0, -1

    # Create output directory and temp checkpoint path
    if out_path:
        Path(out_path).mkdir(parents=True, exist_ok=True)
        best_tmp_path = os.path.join(out_path, f"{model_name}_best_tmp.pth")
    else:
        best_tmp_path = None

    def train_one_epoch(model, loader):
        model.train()
        total_loss, correct, total = 0, 0, 0
        for batch in loader:
            # Unpack batch (handles both 2-tuple and 3-tuple returns)
            if len(batch) == 3:
                images, labels, _ = batch  # Ignore filenames
            else:
                images, labels = batch
            images, labels = images.to(device_obj), labels.to(device_obj)
            optimizer.zero_grad()
            if scaler:
                with autocast('cuda'):
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)
        return total_loss / total, correct / total

    # Training loop
    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader)
        val_metrics = evaluate_model(model, val_loader, device=device, criterion=criterion)

        history["train_loss"].append(float(train_loss))
        history["train_acc"].append(float(train_acc))
        history["val_loss"].append(float(val_metrics['loss']))
        history["val_acc"].append(float(val_metrics['accuracy']))

        logger.info(f"Epoch {epoch+1}/{epochs}: "
                    f"Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}, "
                    f"Val Loss={val_metrics['loss']:.4f}, Val Acc={val_metrics['accuracy']:.4f}")

        if val_metrics['accuracy'] > best_val_acc:
            best_val_acc, best_epoch = val_metrics['accuracy'], epoch + 1
            if best_tmp_path:
                torch.save(model.state_dict(), best_tmp_path)

    # Load best checkpoint
    if best_tmp_path and os.path.exists(best_tmp_path):
        model.load_state_dict(torch.load(best_tmp_path, map_location=device_obj))

    logger.info(f"\nBest Val Accuracy: {best_val_acc:.4f} at epoch {best_epoch}\n")

    # Save outputs
    if out_path:
        notes = f"{'Baseline' if grid_size == 1 else f'{grid_size}x{grid_size} permuted'}, {meta['input_size']}x{meta['input_size']}, ImageNet norm"
        save_run(
            results_root=out_path,
            model_name=model_name,
            grid_size=grid_size,
            model=model,
            history=history,
            meta=meta,
            hyper={
                "epochs": epochs,
                "batch_size": train_loader.batch_size,
                "lr": lr,
                "wd": weight_decay,
                "optimizer": "AdamW",
                "dropout": config.get('dropout', 0.2),
                "pretrained": config.get('pretrained', True)
            },
            notes=notes,
            split_indices=split_indices,
            permutations=permutations if permutations else None
        )

        save_preds(
            results_root=out_path,
            model_name=model_name,
            grid_size=grid_size,
            split="val",
            model=model,
            loader=val_loader,
            device=str(device_obj),
        )

        # Clean up temp checkpoint
        if best_tmp_path and os.path.exists(best_tmp_path):
            os.remove(best_tmp_path)
            logger.info(f"Removed temp checkpoint: {best_tmp_path}")

        logger.info(f"Saved outputs under: {out_path}")

    return {
        "history": history,
        "best_val_acc": best_val_acc,
        "best_epoch": best_epoch,
        "model": model
    }
