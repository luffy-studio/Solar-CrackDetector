"""
test_model.py — Comprehensive YOLO11 Classification Model Tester
=================================================================
Modes:
  1. Validation  : evaluates accuracy on the entire val split
  2. Single image: predict one image and show result
  3. Batch       : predict a whole folder and save a summary CSV
  4. Confusion   : plot the confusion matrix from the val set

Usage examples
--------------
# 1 — Validate on the full val set (prints top-1/top-5 accuracy)
python scripts/test_model.py --mode val

# 2 — Predict a single image
python scripts/test_model.py --mode single --source data/processed/classification_dataset/val/defective/cell0002.png

# 3 — Predict every image in a folder
python scripts/test_model.py --mode batch --source data/processed/classification_dataset/val/defective

# 4 — Draw confusion matrix
python scripts/test_model.py --mode confusion
"""

import argparse
import csv
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import torch
from ultralytics import YOLO

# ── Paths ─────────────────────────────────────────────────────────────────────
MODEL_PATH  = ROOT / "solar_defects" / "classification_run" / "weights" / "best.pt"
VAL_DIR     = ROOT / "data" / "processed" / "classification_dataset" / "val"
RESULTS_DIR = ROOT / "results" / "test_outputs"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = {0: "defective", 1: "non_defective"}   # adjust if order differs


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_model(path: Path = None) -> YOLO:
    p = path or MODEL_PATH
    if not p.exists():
        sys.exit(f"[ERROR] Model not found at {p}\n"
                 "  → Run:  python scripts/train_classify.py")
    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Loading model  : {p.name}")
    print(f"[INFO] Device         : {'GPU (CUDA)' if device == 0 else 'CPU'}")
    model = YOLO(str(p))
    return model, device


def confidence_bar(conf: float, width: int = 20) -> str:
    filled = int(conf * width)
    return f"[{'█' * filled}{'░' * (width - filled)}] {conf*100:5.1f}%"


# ── Mode 1: Full Validation ────────────────────────────────────────────────────

def run_validation(model: YOLO, device):
    print("\n" + "="*60)
    print("  MODE: Full Validation Set Evaluation")
    print("="*60)
    metrics = model.val(
        data=str(ROOT / "data" / "processed" / "classification_dataset"),
        split="val",
        device=device,
        verbose=False,
    )
    top1 = metrics.top1
    top5 = metrics.top5
    print(f"\n  ✅  Top-1 Accuracy : {top1*100:.2f}%")
    print(f"  ✅  Top-5 Accuracy : {top5*100:.2f}%")
    print(f"\n  Results saved to  : {metrics.save_dir}")
    return metrics


# ── Mode 2: Single Image ───────────────────────────────────────────────────────

def run_single(model: YOLO, device, source: str):
    source = Path(source)
    if not source.exists():
        sys.exit(f"[ERROR] File not found: {source}")

    print("\n" + "="*60)
    print(f"  MODE: Single Image Prediction")
    print(f"  File: {source.name}")
    print("="*60)

    results = model.predict(str(source), device=device, verbose=False)
    r = results[0]

    # top-5 probabilities
    probs     = r.probs           # Probs object
    top5_idx  = probs.top5        # list of 5 class indices
    top5_conf = probs.top5conf.tolist()

    print(f"\n  Ground truth class (from folder) : "
          f"{source.parent.name if source.parent.name in CLASS_NAMES.values() else 'unknown'}")
    print(f"\n  {'Rank':<6} {'Class':<16} {'Confidence'}")
    print(f"  {'-'*50}")
    for rank, (idx, conf) in enumerate(zip(top5_idx, top5_conf), 1):
        cls_name = r.names[idx]
        bar      = confidence_bar(conf)
        marker   = " ◀ PREDICTION" if rank == 1 else ""
        print(f"  #{rank:<5} {cls_name:<16} {bar}{marker}")

    # Save annotated copy
    img = cv2.imread(str(source))
    if img is not None:
        pred_cls  = r.names[top5_idx[0]]
        pred_conf = top5_conf[0]
        color     = (0, 200, 0) if pred_cls == "non_defective" else (0, 0, 220)
        label     = f"{pred_cls}  {pred_conf*100:.1f}%"
        cv2.putText(img, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, color, 2, cv2.LINE_AA)
        out_path  = RESULTS_DIR / f"pred_{source.name}"
        cv2.imwrite(str(out_path), img)
        print(f"\n  Annotated image saved → {out_path}")

    return results


# ── Mode 3: Batch Folder ──────────────────────────────────────────────────────

def run_batch(model: YOLO, device, source: str):
    source = Path(source)
    if not source.is_dir():
        sys.exit(f"[ERROR] Not a directory: {source}")

    image_exts = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}
    images     = [f for f in source.rglob("*") if f.suffix.lower() in image_exts]

    if not images:
        sys.exit(f"[ERROR] No images found in {source}")

    print("\n" + "="*60)
    print(f"  MODE: Batch Prediction")
    print(f"  Folder  : {source}")
    print(f"  Images  : {len(images)}")
    print("="*60 + "\n")

    rows      = []
    correct   = 0
    gt_folder = source.name   # folder name = ground truth class

    for img_path in images:
        results  = model.predict(str(img_path), device=device, verbose=False)
        r        = results[0]
        pred_idx = r.probs.top1
        pred_cls = r.names[pred_idx]
        pred_conf = float(r.probs.top1conf)
        is_correct = (pred_cls == gt_folder)
        if is_correct:
            correct += 1
        rows.append({
            "image":      img_path.name,
            "prediction": pred_cls,
            "confidence": f"{pred_conf*100:.2f}%",
            "gt_folder":  gt_folder,
            "correct":    "✓" if is_correct else "✗",
        })
        mark = "✓" if is_correct else "✗"
        print(f"  {mark}  {img_path.name:<25} → {pred_cls:<16} ({pred_conf*100:.1f}%)")

    # Save CSV
    csv_path = RESULTS_DIR / f"batch_{source.name}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    accuracy = correct / len(images) * 100
    print(f"\n  Folder accuracy : {accuracy:.1f}%  ({correct}/{len(images)} correct)")
    print(f"  Results CSV     : {csv_path}")


# ── Mode 4: Confusion Matrix ───────────────────────────────────────────────────

def run_confusion(model: YOLO, device):
    """
    Re-runs validation; ultralytics automatically saves confusion_matrix.png
    inside the run's results directory.
    """
    print("\n" + "="*60)
    print("  MODE: Confusion Matrix")
    print("="*60)
    metrics = model.val(
        data=str(ROOT / "data" / "processed" / "classification_dataset"),
        split="val",
        device=device,
        plots=True,
        verbose=False,
    )
    save_dir = Path(metrics.save_dir)
    cm_path  = save_dir / "confusion_matrix.png"
    cm_norm  = save_dir / "confusion_matrix_normalized.png"
    print(f"\n  Saved confusion matrix     → {cm_path}")
    print(f"  Saved normalized matrix    → {cm_norm}")
    print(f"  Top-1 Accuracy             : {metrics.top1*100:.2f}%")


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Test the trained YOLO11 solar cell classifier",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--mode", choices=["val", "single", "batch", "confusion"],
        default="val",
        help="Test mode (default: val)",
    )
    parser.add_argument(
        "--source", type=str, default=None,
        help="Path to image or folder (required for single/batch modes)",
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="Path to model weights (default: solar_defects/classification_run/weights/best.pt)",
    )

    args = parser.parse_args()

    # Allow overriding model path via CLI
    model_path = Path(args.model) if args.model else MODEL_PATH

    model, device = load_model(model_path)

    if args.mode == "val":
        run_validation(model, device)

    elif args.mode == "single":
        if not args.source:
            parser.error("--source is required for single mode")
        run_single(model, device, args.source)

    elif args.mode == "batch":
        if not args.source:
            parser.error("--source is required for batch mode")
        run_batch(model, device, args.source)

    elif args.mode == "confusion":
        run_confusion(model, device)


if __name__ == "__main__":
    main()
