import os
from pathlib import Path

def prepare_small_dataset(root="data", n_samples=200):
    """
    Instead of downloading the full Kaggle dataset, just point to
    the local test_data directory inside the project.
    """
    root = Path(root)
    small_dir = root / "test_data"

    if not small_dir.exists() or not any(small_dir.iterdir()):
        raise FileNotFoundError(
            f"No test data found at {small_dir}. "
            "Please place your sample images there."
        )

    return str(small_dir)

if __name__ == "__main__":
    data_path = prepare_small_dataset()

    # run train.py with 1 epoch
    os.system(
        f"python train.py --model resnet18 --data {data_path} --epochs 1 --batch_size 16"
    )
