from pathlib import Path
from train import train_model

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
        "resnet18",
    ]
    grids = [11]  # baseline and permuted

    for model in models:
        for grid in grids:
            out_dir = out_root / f"{model}_{grid}"
            out_dir.mkdir(parents=True, exist_ok=True)

            print(f"▶ Running: {model}, grid={grid}")
            try:
                train_model(
                    model_name=model,
                    data_path=data_path,
                    out_path=str(out_dir),
                    grid=grid,
                    epochs=1,
                    batch_size=8
                )
                print(f"Done: {model}, grid={grid}")
            except Exception as e:
                print(f"Failed: {model}, grid={grid} - {e}")
