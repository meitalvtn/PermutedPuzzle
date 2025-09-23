import json
from pathlib import Path


def build_manifest(results_root: str, overwrite: bool = True) -> str:
    """
    Scan results/{model}/{grid}/ folders under lib_root and rebuild manifest.json.
    Keeps only the best_val_accuracy for each (model, grid).

    Args:
        results_root: path to the results root (e.g. '/content/drive/MyDrive/MLDS_PermutedPuzzle/results')
        overwrite: if True, write manifest.json; else just return path without writing.

    Returns:
        Path to manifest.json (string)
    """
    lib_root = Path(results_root)
    manifest = []

    for model_dir in lib_root.iterdir():
        if not model_dir.is_dir():
            continue
        model = model_dir.name
        for grid_dir in model_dir.iterdir():
            if not grid_dir.is_dir():
                continue
            metrics_path = grid_dir / "metrics.json"
            if not metrics_path.exists():
                continue
            with open(metrics_path, "r") as f:
                metrics = json.load(f)
            manifest.append({
                "model": model,
                "grid_size": int(metrics.get("grid_size", grid_dir.name)),
                "best_val_accuracy": metrics.get("best_val_accuracy"),
                "epochs": metrics.get("epochs"),
                "weights_relpath": str(Path(model) / grid_dir.name / "best.pth"),
                "metrics_relpath": str(Path(model) / grid_dir.name / "metrics.json"),
                "notes": metrics.get("notes", ""),
                "updated_at": metrics.get("saved_at", "")
            })

    manifest_path = lib_root / "manifest.json"
    if overwrite:
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)
    return f"Manifest saved at: {manifest_path}"

