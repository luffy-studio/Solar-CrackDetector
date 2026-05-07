# 🔆 Solar Panel Defect Detection — Complete Project Guide

> **Production-ready AI system** for classifying solar panel cells as **defective** or **non-defective** using YOLO11, PyTorch, FastAPI, and Docker.

---

## 1. Project Overview

This system uses **YOLO11 image classification** (Ultralytics) to detect cracks and defects in electroluminescence (EL) images of solar cells. The ELPV dataset (2,624 images from ZAE Bayern) is used for training.

| Item | Detail |
|------|--------|
| **Model** | YOLO11s-cls (5.4M parameters) |
| **Task** | Binary image classification |
| **Classes** | `defective`, `non_defective` |
| **Dataset** | ELPV — 2,624 EL images (300×300 px) |
| **GPU** | NVIDIA RTX 4050 — FP16/AMP enabled |
| **Accuracy** | 82.86% top-1, 100% top-5 |
| **Inference** | 0.7 ms/image on GPU |
| **API** | FastAPI on port 8000 |
| **Container** | Docker + NVIDIA GPU pass-through |

---

## 2. Folder Structure

```
solar-panel-defect-detection/
│
├── configs/                          # ⚙️ YAML configuration files
│   ├── dataset.yaml                  #    Dataset paths and class names
│   ├── train.yaml                    #    Training hyperparameters
│   └── inference.yaml                #    Inference & streaming settings
│
├── data/                             # 📂 Dataset directory (git-ignored)
│   ├── raw/elpv-dataset/             #    Downloaded raw ELPV repo
│   └── processed/classification_dataset/
│       ├── train/defective/          #    Training images (80%)
│       ├── train/non_defective/
│       ├── val/defective/            #    Validation images (20%)
│       └── val/non_defective/
│
├── dataset_preparation/              # 🔧 Data pipeline scripts
│   ├── download_elpv_dataset.py      #    Downloads + organizes dataset
│   ├── preprocess.py                 #    Resize & standardize images
│   └── augment.py                    #    Albumentations augmentation
│
├── scripts/                          # 🚀 Execution entry-points
│   ├── train.py                      #    Object detection training
│   ├── train_classify.py             #    Classification training (active)
│   ├── test_model.py                 #    4-mode testing suite
│   ├── evaluate.py                   #    Legacy evaluation script
│   └── realtime_inference.py         #    Webcam/video stream inference
│
├── src/                              # 📦 Core source code (library)
│   ├── api/
│   │   ├── __init__.py
│   │   └── main.py                   #    FastAPI REST server
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py                 #    YAML config loader (AppConfig)
│   │   └── logger.py                 #    Centralized logging
│   └── inference/
│       ├── __init__.py
│       ├── model.py                  #    SolarDefectModel wrapper
│       └── predictor.py              #    High-level InferencePredictor
│
├── solar_defects/                    # 📊 Training run outputs
│   └── classification_run/weights/
│       ├── best.pt                   #    Best model checkpoint
│       └── last.pt                   #    Final epoch checkpoint
│
├── runs/                             # 📊 Validation/test outputs
├── results/test_outputs/             # 📊 Test predictions & CSVs
├── logs/                             # 📝 Application log files
│
├── .github/workflows/ci_cd.yml      # 🔄 GitHub Actions CI/CD
├── Dockerfile                        # 🐳 Container image definition
├── docker-compose.yml                # 🐳 Docker Compose with GPU
├── requirements.txt                  # 📋 Python dependencies
└── README.md                         # 📖 Quick-start readme
```

---

## 3. File-by-File Explanation

### 3.1 Configuration Files (`configs/`)

| File | Purpose | When to Modify |
|------|---------|----------------|
| **`dataset.yaml`** | Defines dataset root path, train/val/test image directories, class names (`defective`, `non_defective`), and augmentation flags. | When adding new classes, changing data paths, or toggling augmentation. |
| **`train.yaml`** | All training hyperparameters: model name, epochs, batch size, learning rate, optimizer, image size, AMP toggle, early stopping patience, and logging backends. | When tuning hyperparameters or switching model sizes (yolo11n → yolo11m). |
| **`inference.yaml`** | Model weights path, confidence/IoU thresholds, device selection, FP16 toggle, and webcam/video stream settings. | When deploying with different thresholds or changing video sources. |

**How they connect:** `src/core/config.py` loads these files via `AppConfig.get_train_config()`, `get_inference_config()`, and `get_dataset_config()`. Every script reads config from here — **no hardcoded values in the codebase**.

### 3.2 Dataset Preparation (`dataset_preparation/`)

| File | What It Does | Importance |
|------|-------------|------------|
| **`download_elpv_dataset.py`** | Downloads the ELPV dataset ZIP from GitHub, extracts it, parses the space-separated `labels.csv`, classifies each image (probability > 0.0 = defective), shuffles with seed 42, splits 80/20 into `train/` and `val/` folders. | **Entry point of the entire pipeline.** Must run first before any training. |
| **`preprocess.py`** | Resizes raw images to a uniform size (default 640×640) and converts all formats to JPG. Uses OpenCV's `INTER_AREA` interpolation. | Useful when working with images of varying sizes from different sources. |
| **`augment.py`** | Applies offline data augmentation using Albumentations: horizontal/vertical flips, rotation, brightness/contrast, Gaussian/motion blur. Handles YOLO bounding box format. | Increases dataset diversity to reduce overfitting. Designed for object detection labels. |

**Execution flow:** `download_elpv_dataset.py` → (optionally) `preprocess.py` → (optionally) `augment.py` → training.

### 3.3 Training Scripts (`scripts/`)

| File | What It Does | When to Use |
|------|-------------|-------------|
| **`train_classify.py`** ⭐ | **Primary training script.** Auto-detects GPU, reports VRAM, loads `yolo11s-cls.pt` pretrained weights, trains classification on the ELPV dataset with AMP/FP16, early stopping (patience=20), saves best weights to `solar_defects/classification_run/weights/best.pt`. | **Use this** for the current classification dataset. |
| **`train.py`** | Object detection training script. Reads all parameters from `configs/train.yaml`, uses `AppConfig`, supports bounding-box detection tasks. | Use when you have a bounding-box annotated dataset (not the current ELPV data). |
| **`test_model.py`** ⭐ | Comprehensive 4-mode testing: `--mode val` (full accuracy evaluation), `--mode single` (one image prediction with confidence bars), `--mode batch` (folder prediction + CSV export), `--mode confusion` (confusion matrix plots). | **Use after training** to evaluate model quality. |
| **`evaluate.py`** | Legacy evaluation script. Loads model via `AppConfig`, runs `model.val()` and logs mAP metrics. Designed for detection tasks. | Use for detection model evaluation. |
| **`realtime_inference.py`** | Opens a webcam or video file, runs frame-by-frame inference using `InferencePredictor`, overlays FPS counter and bounding boxes, displays live OpenCV window. Press `q` to quit. | Use for live demo / real-time monitoring. |

### 3.4 Source Library (`src/`)

#### `src/core/` — Shared Infrastructure

| File | What It Does |
|------|-------------|
| **`config.py`** | Defines `load_config()` (safe YAML parser) and `AppConfig` class with `BASE_DIR`, `CONFIG_DIR`, and three class methods to load each YAML file. **Every script depends on this.** |
| **`logger.py`** | Creates a dual-output logger (console + timestamped file in `logs/`). Format: `timestamp - SolarDefectDetection - LEVEL - message`. The singleton `logger` instance is imported across the project. |

#### `src/inference/` — Prediction Engine

| File | What It Does |
|------|-------------|
| **`model.py`** | `SolarDefectModel` class — wraps `ultralytics.YOLO`. Constructor auto-detects GPU, reads confidence/IoU thresholds from config, enables FP16 half-precision when GPU is available. The `predict()` method runs inference on any source (path, numpy array, URL). |
| **`predictor.py`** | `InferencePredictor` class — higher-level wrapper. Takes a numpy image, calls `SolarDefectModel.predict()`, extracts bounding boxes with class names and confidences, returns an annotated image. Used by the API and the real-time script. |

**Dependency chain:** `predictor.py` → `model.py` → `config.py` + `logger.py`

#### `src/api/` — REST API

| File | What It Does |
|------|-------------|
| **`main.py`** | FastAPI application with two endpoints: `GET /` (health check), `POST /predict` (accepts image upload, returns JSON predictions or annotated JPEG). Lazily initializes `InferencePredictor` on startup. |

### 3.5 Deployment Files

| File | What It Does |
|------|-------------|
| **`Dockerfile`** | Builds from `pytorch/pytorch:2.0.1-cuda11.7` base image. Installs OpenCV system deps, pip dependencies, copies `src/` and `configs/`, exposes port 8000, runs uvicorn. |
| **`docker-compose.yml`** | Orchestrates the API container with GPU pass-through (`NVIDIA_VISIBLE_DEVICES=all`), maps ports 8000, mounts `runs/` and `logs/` volumes. |
| **`requirements.txt`** | All Python dependencies: `ultralytics>=8.3.0`, `torch>=2.0.0`, `fastapi`, `opencv-python`, `albumentations`, `mlflow`, `tensorboard`, etc. |
| **`.github/workflows/ci_cd.yml`** | GitHub Actions: runs `flake8` linting and `pytest` on every push/PR to `main`. On push to `main`, builds the Docker image. Contains commented-out Docker Hub push steps. |

---

## 4. Setup Instructions

### Prerequisites
- Python 3.10+
- Git
- NVIDIA GPU + CUDA drivers (optional but highly recommended)

### Installation
```bash
# 1. Clone the repository
git clone <your-repo-url>
cd solar-panel-defect-detection

# 2. Create virtual environment (recommended)
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/Mac

# 3. Install dependencies
pip install -r requirements.txt

# 4. Verify installation
python -c "from ultralytics import YOLO; print('OK')"
```

---

## 5. GPU Setup

```bash
# Check if PyTorch sees your GPU
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

| Requirement | Version |
|-------------|---------|
| NVIDIA Driver | ≥ 525.x |
| CUDA Toolkit | ≥ 11.7 |
| PyTorch | ≥ 2.0.0 (with CUDA) |

If the above command prints `False`, install the CUDA version of PyTorch:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

The training and inference scripts **auto-detect** GPU availability and fall back to CPU if no GPU is found.

---

## 6. Dataset Setup

```bash
# Downloads ELPV dataset from GitHub and organizes into train/val splits
python dataset_preparation/download_elpv_dataset.py
```

This will create:
```
data/processed/classification_dataset/
├── train/defective/      (893 images)
├── train/non_defective/  (1206 images)
├── val/defective/        (223 images)
└── val/non_defective/    (302 images)
```

**Total:** 2,624 images | **Split:** 80% train / 20% val | **Seed:** 42

---

## 7. Training

```bash
python scripts/train_classify.py
```

**What happens internally:**
1. `check_gpu()` detects hardware → prints device name and VRAM
2. Loads `yolo11s-cls.pt` pretrained weights (downloaded on first run)
3. Trains for 100 epochs with batch=32, imgsz=300, AMP enabled
4. Early stopping at patience=20 (stops if no improvement for 20 epochs)
5. Saves `best.pt` and `last.pt` to `solar_defects/classification_run/weights/`

**Expected output:** ~83% top-1 accuracy in ~15 minutes on RTX 4050.

### Training Tips
- **Larger model:** Change `yolo11s-cls.pt` → `yolo11m-cls.pt` for higher accuracy
- **More epochs:** Increase `epochs=200` for longer training
- **Batch size:** Reduce to 16 if you get CUDA out-of-memory errors

---

## 8. Testing & Validation

```bash
# Full validation set accuracy
python scripts/test_model.py --mode val

# Predict a single image
python scripts/test_model.py --mode single --source C:\path\to\image.png

# Predict all images in a folder + export CSV
python scripts/test_model.py --mode batch --source data/processed/classification_dataset/val/defective

# Generate confusion matrix
python scripts/test_model.py --mode confusion
```

**Custom model path:** `--model path/to/your/weights.pt`

---

## 9. Inference on New Images

### Quick CLI prediction
```bash
python scripts/test_model.py --mode single --source C:\Users\sarva\anigravity\image.png
```

### Real-time webcam
```bash
python scripts/realtime_inference.py
```
Press `q` to quit. Configure video source in `configs/inference.yaml`.

### Python API
```python
from ultralytics import YOLO
model = YOLO("solar_defects/classification_run/weights/best.pt")
results = model.predict("your_image.png")
print(results[0].probs.top1)      # predicted class index
print(results[0].probs.top1conf)  # confidence
```

---

## 10. API Deployment

```bash
# Start the FastAPI server
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

### Endpoints

| Method | URL | Description |
|--------|-----|-------------|
| `GET` | `/` | Health check |
| `POST` | `/predict` | Upload image → get predictions |
| `POST` | `/predict?return_image=true` | Upload image → get annotated JPEG |

### Test with curl
```bash
curl.exe -X POST "http://localhost:8000/predict" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@sample_image.png"
```

### Interactive docs
Open `http://localhost:8000/docs` in your browser for Swagger UI.

---

## 11. Docker Usage

```bash
# Build and run with GPU support
docker-compose up --build

# Or without compose
docker build -t solar-defect-api .
docker run --gpus all -p 8000:8000 solar-defect-api
```

**Important:** Copy your trained `best.pt` into the container's expected path, or mount it as a volume:
```yaml
volumes:
  - ./solar_defects:/app/solar_defects
```

---

## 12. Config File Reference

### `configs/train.yaml`
```yaml
model:
  name: yolo11s.pt       # Model size: n/s/m/l/x
  task: detect            # Task type
training:
  epochs: 100             # Max training epochs
  patience: 20            # Early stop patience
  batch: 16               # Batch size (reduce if OOM)
  imgsz: 640              # Input image size
  device: 0               # 0=GPU, 'cpu'=CPU
  amp: true               # Mixed precision (FP16)
  lr0: 0.01               # Initial learning rate
```

### `configs/inference.yaml`
```yaml
model_path: runs/train/solar_defects/exp/weights/best.pt
confidence_threshold: 0.5   # Minimum confidence to accept
iou_threshold: 0.45         # NMS IoU threshold
device: 0                   # GPU device ID
half: true                  # FP16 inference
stream:
  source: 0                 # 0=webcam, or video file path
  display: true             # Show OpenCV window
```

### `configs/dataset.yaml`
```yaml
path: ../data/processed     # Dataset root
train: images/train         # Training images
val: images/val             # Validation images
names:
  0: 'defective'
  1: 'non_defective'
```

---

## 13. System Workflow Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    DATA PIPELINE                            │
│                                                             │
│  download_elpv_dataset.py                                   │
│       │                                                     │
│       ▼                                                     │
│  GitHub ZIP ──► Extract ──► Parse labels.csv                │
│       │                         │                           │
│       ▼                         ▼                           │
│  data/raw/             Classify: prob>0 = defective         │
│                              │                              │
│                              ▼                              │
│                    Shuffle (seed=42) + Split 80/20           │
│                              │                              │
│                              ▼                              │
│              data/processed/classification_dataset/          │
│                    train/  +  val/                           │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                   TRAINING PIPELINE                         │
│                                                             │
│  train_classify.py                                          │
│       │                                                     │
│       ├── check_gpu() ──► Detects CUDA device + VRAM        │
│       ├── YOLO('yolo11s-cls.pt') ──► Load pretrained        │
│       └── model.train(                                      │
│              data=..., epochs=100, imgsz=300,               │
│              batch=32, amp=True, patience=20                │
│           )                                                 │
│       │                                                     │
│       ▼                                                     │
│  solar_defects/classification_run/weights/best.pt           │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                 TESTING & EVALUATION                        │
│                                                             │
│  test_model.py                                              │
│       │                                                     │
│       ├── --mode val       ──► Top-1/Top-5 accuracy         │
│       ├── --mode single    ──► Predict one image            │
│       ├── --mode batch     ──► Folder prediction + CSV      │
│       └── --mode confusion ──► Confusion matrix plot        │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                   DEPLOYMENT                                │
│                                                             │
│  Option A: FastAPI                                          │
│    uvicorn src.api.main:app --port 8000                     │
│    POST /predict ──► InferencePredictor ──► SolarDefectModel│
│                                                             │
│  Option B: Docker                                           │
│    docker-compose up --build  (with GPU pass-through)       │
│                                                             │
│  Option C: Real-time                                        │
│    python scripts/realtime_inference.py  (webcam/video)     │
└─────────────────────────────────────────────────────────────┘
```

---

## 14. Common Errors & Fixes

| Error | Cause | Fix |
|-------|-------|-----|
| `ModuleNotFoundError: ultralytics` | Dependencies not installed | `pip install -r requirements.txt` |
| `CUDA out of memory` | Batch size too large for GPU VRAM | Reduce `batch` to 16 or 8 |
| `Model not found at ...` | Weights file path is wrong | Check `configs/inference.yaml` → `model_path` |
| `No CUDA GPU found` | PyTorch CPU-only build | Install CUDA PyTorch: `pip install torch --index-url https://download.pytorch.org/whl/cu121` |
| `Dataset not found` | Forgot to download dataset | Run `python dataset_preparation/download_elpv_dataset.py` |
| `IndentationError` in Python | Mixed tabs/spaces | Use 4 spaces consistently; run a formatter like `black` |
| `Permission denied` (Docker) | Docker daemon not running | Start Docker Desktop |
| `'yolo' not recognized` | Ultralytics CLI not in PATH | Use `python -m ultralytics` or run via scripts |
| `labels.csv not found` | Wrong directory structure | The file is at `src/elpv_dataset/data/labels.csv` inside the extracted repo |

---

## 15. Where to Modify for Common Tasks

| Task | Files to Modify |
|------|----------------|
| Add a new class | `configs/dataset.yaml` (add to `names`), re-label data |
| Change model size | `scripts/train_classify.py` (change `yolo11s-cls.pt` → `yolo11m-cls.pt`) |
| Adjust confidence | `configs/inference.yaml` → `confidence_threshold` |
| Add new dataset | `dataset_preparation/download_elpv_dataset.py` (add new download logic) |
| Change API port | `Dockerfile` + `docker-compose.yml` + `src/api/main.py` |
| Add new API endpoint | `src/api/main.py` |
| Change logging | `src/core/logger.py` |
| Modify augmentation | `dataset_preparation/augment.py` |

---

## 16. Best Practices

1. **Never hardcode paths** — always use `configs/*.yaml` and `AppConfig`
2. **Always use virtual environments** — `python -m venv venv`
3. **Version your models** — save each `best.pt` with a date or version number
4. **Monitor with TensorBoard** — `tensorboard --logdir solar_defects/classification_run`
5. **Use early stopping** — set `patience=20` to prevent overfitting
6. **Enable AMP** — `amp=True` for 2× faster training on NVIDIA GPUs
7. **Test before deploying** — always run `test_model.py --mode val` after training
8. **Use Git LFS** for `.pt` weight files — they're too large for regular Git
9. **Log everything** — the `logger` module writes to both console and `logs/` directory

---

## 17. Future Scalability

| Area | Suggestion |
|------|-----------|
| **Multi-GPU** | Set `device=[0,1]` in train.yaml for DataParallel training |
| **Object Detection** | Annotate images with bounding boxes (LabelImg/Roboflow), use `train.py` with `yolo11s.pt` |
| **Model Export** | `model.export(format='onnx')` for TensorRT/edge deployment |
| **Cloud Deploy** | Push Docker image to AWS ECR / GCP Artifact Registry |
| **Auto-retrain** | Schedule `train_classify.py` via cron/Airflow when new data arrives |
| **Dashboard** | Add Streamlit/Gradio frontend for non-technical users |
| **Edge/Drone** | Export to TensorRT FP16 for NVIDIA Jetson deployment |
| **Data Versioning** | Use DVC (Data Version Control) alongside Git |
| **Experiment Tracking** | Enable MLflow in `train.yaml` → `mlflow: true` |

---

## 18. Quick Reference Commands

```bash
# Full pipeline (run in order)
python dataset_preparation/download_elpv_dataset.py    # Step 1: Get data
python scripts/train_classify.py                        # Step 2: Train model
python scripts/test_model.py --mode val                 # Step 3: Validate
python scripts/test_model.py --mode confusion           # Step 4: Confusion matrix
python scripts/test_model.py --mode single --source C:\path\to\image.png  # Step 5: Predict

# API server
uvicorn src.api.main:app --host 0.0.0.0 --port 8000

# Docker
docker-compose up --build
```

---

*Last updated: May 2026 | Built with YOLO11 + PyTorch + FastAPI*
