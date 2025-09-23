import json
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def summarize_results(results_root: str, show_curves: bool = True) -> pd.DataFrame:
    """
    Summarize all runs in a results library folder.

    Args:
        results_root: Path to results root (e.g., '/content/drive/MyDrive/MLDS_PermutedPuzzle/results')
        show_curves: If True, plot training/val curves for grid=1 baselines.

    Returns:
        Pandas DataFrame with summary of manifest.json
    """
    results_root = Path(results_root)
    manifest_path = results_root / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"No manifest.json found under {results_root}")

    # Load manifest
    with open(manifest_path, "r") as f:
        manifest = json.load(f)
    df = pd.DataFrame(manifest)

    # Print a clean text summary
    display_cols = ["model", "grid_size", "best_val_accuracy", "epochs", "notes"]
    print("=== Baseline Summary ===")
    print(df[display_cols].sort_values(["grid_size", "model"]).to_string(index=False))

    # Optionally plot curves for grid=1 runs
    if show_curves:
        for entry in manifest:
            if entry["grid_size"] != 1:
                continue
            metrics_path = results_root / entry["metrics_relpath"]
            if not metrics_path.exists():
                continue
            with open(metrics_path, "r") as f:
                metrics = json.load(f)
            hist = metrics["history"]

            plt.figure(figsize=(6,4))
            if "train_loss" in hist:
                plt.plot(hist["train_loss"], label="Train Loss")
            if "val_loss" in hist:
                plt.plot(hist["val_loss"], label="Val Loss")
            if "val_acc" in hist:
                plt.plot(hist["val_acc"], label="Val Acc")
            plt.title(f"{entry['model']} (grid=1)")
            plt.xlabel("Epoch")
            plt.ylabel("Value")
            plt.legend()
            plt.show()

    return df
