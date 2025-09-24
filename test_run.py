import os
from pathlib import Path

def prepare_small_dataset(root="test_data"):
    """
    Use a tiny sample dataset placed at test_data/data/.
    """
    root = Path(root)
    small_dir = root / "data"
    if not small_dir.exists() or not any(small_dir.iterdir()):
        raise FileNotFoundError(
            f"No test data found at {small_dir}. "
            "Please place your sample images there."
        )
    return str(small_dir)

if __name__ == "__main__":
    data_path = prepare_small_dataset()
    out_root = Path("test_data/results")
    out_root.mkdir(parents=True, exist_ok=True)

    models = [
        "simple_cnn",
        "resnet18",
        "mobilenet_v3_large",
        "efficientnet_b0",
        "convnext_tiny",
    ]
    grids = [1, 2]  # baseline and permuted

    for model in models:
        for grid in grids:
            out_dir = out_root / f"{model}_{grid}"
            out_dir.mkdir(parents=True, exist_ok=True)

            cmd = (
                f"python train.py --model {model} "
                f"--data {data_path} --epochs 1 --batch_size 8 "
                f"--out {out_dir} --grid {grid} --save_preds"
            )
            print(f"▶ Running: {cmd}")
            ret = os.system(cmd)
            if ret != 0:
                print(f"Failed: {model}, grid={grid}")
            else:
                print(f"Done: {model}, grid={grid}")
