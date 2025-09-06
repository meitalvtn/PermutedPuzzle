import os, json, time, hashlib, platform
from pathlib import Path
import torch
import numpy as np

def _ensure(p: str): Path(p).mkdir(parents=True, exist_ok=True)

def save_run(results_root: str,
             model_name: str,
             grid_size: int,
             model: torch.nn.Module,
             history: dict,
             meta: dict,
             hyper: dict,
             notes: str = "") -> dict:
    """
    history: {"train_loss": [...], "val_acc": [...], "val_loss": [...]} (any you track)
    meta:    {"input_size": 224, "mean": [...], "std": [...]}
    hyper:   {"epochs": int, "batch_size": int, "lr": float, "wd": float, "optimizer": "..."}
    """
    results_root = str(results_root)
    model_dir   = f"{results_root}/models/{model_name}/{grid_size}"
    metrics_dir = f"{results_root}/metrics/{model_name}"
    _ensure(model_dir); _ensure(metrics_dir)

    # 1) Weights
    weights_path = f"{model_dir}/best.pth"
    torch.save(model.state_dict(), weights_path)

    # 2) Metrics.json
    metrics_path = f"{metrics_dir}/{grid_size}.json"
    metrics = {
        "model": model_name,
        "grid_size": grid_size,
        "epochs": int(hyper.get("epochs", len(history.get("val_acc", [])))),
        "final_val_accuracy": float(history.get("val_acc", [0])[-1]) if history.get("val_acc") else None,
        "best_val_accuracy": float(max(history.get("val_acc", [0]))),
        "history": {k: list(map(float, v)) for k, v in history.items()},
        "meta": meta,
        "hyper": hyper,
        "notes": notes,
        "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "env": {"python": platform.python_version(), "torch": torch.__version__},
        "weights": weights_path,
    }
    with open(metrics_path, "w") as f: json.dump(metrics, f, indent=2)

    # 3) Manifest (dedupe on model+grid)
    manifest_path = f"{results_root}/manifest.json"
    if os.path.exists(manifest_path):
        with open(manifest_path, "r") as f: manifest = json.load(f)
    else:
        manifest = []

    manifest = [m for m in manifest if not (m["model"] == model_name and m["grid_size"] == grid_size)]
    manifest.append({
        "model": model_name,
        "grid_size": grid_size,
        "epochs": metrics["epochs"],
        "final_val_accuracy": metrics["final_val_accuracy"],
        "best_val_accuracy": metrics["best_val_accuracy"],
        "weights": weights_path,
        "metrics": metrics_path,
        "notes": notes,
    })
    with open(manifest_path, "w") as f: json.dump(manifest, f, indent=2)

    return {"weights_path": weights_path, "metrics_path": metrics_path}

def save_preds(results_root: str,
               model_name: str,
               grid_size: int,
               split: str,
               model: torch.nn.Module,
               loader,
               device: str):
    model.eval()
    all_logits, all_labels = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            all_logits.append(logits.cpu())
            all_labels.append(yb.cpu())
    all_logits = torch.cat(all_logits).numpy()
    all_labels = torch.cat(all_labels).numpy()
    preds = all_logits.argmax(axis=1)

    out_dir = f"{results_root}/preds"
    _ensure(out_dir)
    out_path = f"{out_dir}/{model_name}_{grid_size}_{split}.npz"
    np.savez_compressed(out_path, logits=all_logits, labels=all_labels, preds=preds)
    return out_path
