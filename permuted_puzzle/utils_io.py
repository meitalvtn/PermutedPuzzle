import os, json, time, platform
from pathlib import Path
from typing import Dict, Iterable, Tuple, Optional
import torch
import numpy as np

def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def _atomic_write_json(path: Path, obj: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    tmp.replace(path)

def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")

def _env_info() -> dict:
    return {"python": platform.python_version(), "torch": torch.__version__}

def _best_acc(history: Dict[str, Iterable[float]]) -> float:
    v = history.get("val_acc") or []
    return float(max(v)) if len(v) else 0.0

def _final_acc(history: Dict[str, Iterable[float]]) -> Optional[float]:
    v = history.get("val_acc") or []
    return float(v[-1]) if len(v) else None

def save_run(
    results_root: str,
    model_name: str,
    grid_size: int,
    model: torch.nn.Module,
    history: Dict[str, Iterable[float]],
    meta: dict,
    hyper: dict,
    notes: str = "",
    permutation: Optional[list] = None,
) -> dict:
    """
    Saves into: {results_root}/
      - best.pth
      - metrics.json
    """
    run_dir = Path(results_root)
    _ensure_dir(run_dir)

    # 1) Weights
    weights_path = run_dir / "best.pth"
    torch.save(model.state_dict(), weights_path)

    # 2) Metrics.json (self-contained and portable)
    metrics_path = run_dir / "metrics.json"
    lib_root = run_dir.parent
    weights_relpath = os.path.relpath(weights_path, start=lib_root)
    metrics_relpath = os.path.relpath(metrics_path, start=lib_root)

    metrics = {
        "model": model_name,
        "grid_size": grid_size,
        "epochs": int(hyper.get("epochs", len(history.get("val_acc", [])))),
        "final_val_accuracy": _final_acc(history),
        "best_val_accuracy": _best_acc(history),
        "history": {k: list(map(float, v)) for k, v in history.items()},
        "meta": meta,
        "hyper": hyper,
        "notes": notes,
        "saved_at": _now(),
        "env": _env_info(),
        "weights_relpath": weights_relpath,
        "metrics_relpath": metrics_relpath,
    }

    # add permutation if provided
    if permutation is not None:
        metrics["permutation"] = permutation

    _atomic_write_json(metrics_path, metrics)

    return {
        "weights_path": str(weights_path),
        "metrics_path": str(metrics_path),
        "run_dir": str(run_dir),
    }


def save_preds(
    results_root: str,
    model_name: str,
    grid_size: int,
    split: str,
    model: torch.nn.Module,
    loader,
    device: str
) -> str:
    """
    Saves into: {results_root}/preds_{split}.npz
    Contains: logits, labels, preds (argmax)
    """
    model.eval()
    all_logits, all_labels = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            all_logits.append(logits.detach().cpu())
            all_labels.append(yb.detach().cpu())

    logits = torch.cat(all_logits).numpy()
    labels = torch.cat(all_labels).numpy()
    preds = logits.argmax(axis=1)

    run_dir = Path(results_root)
    _ensure_dir(run_dir)
    out_path = run_dir / f"preds_{split}.npz"
    np.savez_compressed(out_path, logits=logits, labels=labels, preds=preds)
    return str(out_path)