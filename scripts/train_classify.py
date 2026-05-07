import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ultralytics import YOLO
import torch

def get_device():
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        return 0

    print("CUDA GPU not found. Using CPU.")
    return "cpu"

def train():
    device = get_device()
    data_dir = ROOT / "data" / "processed" / "classification_dataset"
    if not data_dir.exists():
        raise FileNotFoundError(f"Dataset not found: {data_dir}")

    model = YOLO("yolo11s-cls.pt")

    results = model.train(
        task="classify",
        data=str(data_dir),

        epochs=100,
        imgsz=300,
        batch=32,

        device=device,
        amp=device != "cpu",

        optimizer="auto",
        lr0=0.01,
        weight_decay=0.0005,

        patience=20,

        project=str(ROOT / "solar_defects"),
        name="classification_run",

        exist_ok=True,
        workers=4,
        verbose=True
    )

    print("\nTraining Complete")

    return results

if __name__ == "__main__":
    train()