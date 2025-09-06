import os
from pathlib import Path

def prepare_small_dataset(root="test_data", n_samples=200):
    """
    Instead of downloading the full Kaggle dataset, just point to
    the local test_data directory inside the project.
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
    out_dir = "test_data/results"
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # run train.py with 1 epoch
    run_simple_cnn = (
        f"python train.py --model simple_cnn "
        f"--data {data_path} --epochs 1 --batch_size 16 --lr 3e-4 "
        f"--out {out_dir} --grid 1 --save_preds"
    )
    run_resnet18 = (
        f"python train.py --model resnet18 "
        f"--data {data_path} --epochs 1 --batch_size 16 "
        f"--out {out_dir} --grid 1 --save_preds"
    )
    run_mobilenet = (
        f"python train.py --model mobilenet_v3_large "
        f"--data {data_path} --epochs 1 --batch_size 16 "
        f"--out {out_dir} --grid 1 --save_preds"
    )
    run_efficientnet = (
        f"python train.py --model efficientnet_b0 "
        f"--data {data_path} --epochs 1 --batch_size 16 "
        f"--out {out_dir} --grid 1 --save_preds"
    )
    run_convnext = (
        f"python train.py --model convnext_tiny "
        f"--data {data_path} --epochs 1 --batch_size 16 "
        f"--out {out_dir} --grid 1 --save_preds"
    )

    os.system(
        run_simple_cnn,
    )
