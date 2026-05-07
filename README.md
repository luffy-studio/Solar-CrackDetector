# Solar Panel Defect Detection System

An industry-level computer vision system for detecting defective solar panels using deep learning (YOLO11), PyTorch, and FastAPI.

## Folder Structure

```
solar-panel-defect-detection/
├── configs/                # Configuration files (YAML)
├── data/                   # Dataset directory (ignored in git)
├── dataset_preparation/    # Scripts for preprocessing and augmentation
├── src/                    # Core source code
│   ├── api/                # FastAPI application
│   ├── core/               # Logging and config management
│   └── inference/          # Prediction logic and model wrappers
├── scripts/                # Execution scripts (train, evaluate, realtime)
├── docker-compose.yml      # Docker compose configuration
├── Dockerfile              # Docker image definition
└── requirements.txt        # Python dependencies
```

## Setup & Installation

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Prepare Dataset**:
   Place your images in `data/raw/images` and annotations in YOLO format in `data/raw/labels`.
   Run preprocessing and augmentation:
   ```bash
   python dataset_preparation/preprocess.py
   python dataset_preparation/augment.py
   ```

## Usage

### Training
Configure hyperparameters in `configs/train.yaml` and `configs/dataset.yaml`.
```bash
python scripts/train.py
```

### Evaluation
```bash
python scripts/evaluate.py
```

### Real-time Inference (Webcam/Video)
Configure source in `configs/inference.yaml`.
```bash
python scripts/realtime_inference.py
```

### API Deployment
Start the FastAPI server:
```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```
Or using Docker:
```bash
docker-compose up --build
```

Send a test request:
```bash
curl.exe -X POST "http://localhost:8000/predict" -H "accept: application/json" -H "Content-Type: multipart/form-data" -F "file=@<sample_image.jpg>"
```


docker save -o solar_defect_image.tar solar-defect-api

## Features
- **State-of-the-art Detection**: Powered by YOLO11 with built-in mixed precision (FP16).
- **GPU Accelerated**: Native CUDA support, automatic fallback detection, multi-GPU scaling ready.
- **Config-Driven**: Easily change parameters without modifying code.
- **Production Ready**: Containerized with Docker, exposed via FastAPI.
- **Robust Pipeline**: Includes data augmentation, evaluation, and CI/CD setup.
