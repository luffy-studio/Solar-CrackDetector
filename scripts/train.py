import sys
import os
from pathlib import Path

import torch
from ultralytics import YOLO

# Add project root to Python path
ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

from src.core.config import AppConfig
from src.core.logger import logger


def train_model():
    """
    Train YOLO11 classification model
    for solar panel defect detection.
    """
    # ---------------- GPU CHECK ---------------- #
    if torch.cuda.is_available():
        device = 0
        logger.info(
            f"GPU detected: {torch.cuda.get_device_name(0)}"
        )
        logger.info(
            "Using GPU acceleration with AMP."
        )
    else:
        device = "cpu"
        logger.warning(
            "No CUDA GPU found. Using CPU."
        )
    logger.info(
        "Starting YOLO11 classification training..."
    )
    # ---------------- LOAD CONFIG ---------------- #
    train_config = AppConfig.get_train_config()
    params = train_config.get("training", {})
    # ---------------- DATASET PATH ---------------- #
    dataset_path = os.path.abspath(
        os.path.join(
            ROOT,
            "data",
            "processed",
            "classification_dataset"
        )
    )
    logger.info(
        f"Dataset path: {dataset_path}"
    )
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(
            f"Dataset not found: {dataset_path}"
        )
    # ---------------- MODEL ---------------- #
    # FORCE classification model
    model_name = "yolo11s-cls.pt"
    logger.info(
        f"Initializing model: {model_name}"
    )
    model = YOLO(model_name)
    # ---------------- TRAINING ---------------- #
    logger.info(
        "Training started..."
    )
    try:
        results = model.train(
            # IMPORTANT
            task="classify",
            # Classification dataset folder
            data=dataset_path,
            epochs=params.get("epochs",100),
            patience=params.get("patience",20),
            batch=params.get("batch",16),
            
            # Better for classification
            imgsz=params.get("imgsz",224),
            device=device,
            
            # Windows multiprocessing safer
            workers=0,

            optimizer=params.get("optimizer","auto"),
            amp=device != "cpu",
            lr0=params.get("lr0",  0.01),

            weight_decay=params.get("weight_decay",0.0005),
            project=params.get( "project", "solar_defects"),
            name=params.get("name","classification_run" ),
            exist_ok=True,
            verbose=True
        )
        
        logger.info("Training completed successfully.")
        logger.info("Best model saved at:")
        logger.info(
            f"runs/classify/"
            f"{params.get('project', 'solar_defects')}/"
            f"{params.get('name', 'classification_run')}/weights/best.pt"
        )
        return results

    except Exception as e:
        logger.error( f"Training failed: {str(e)}")
        raise

if __name__ == "__main__":

    train_model()















'''
# import sys
# import os
# from pathlib import Path
# from ultralytics import YOLO

# # Add src to path to import core modules
# sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
# from src.core.config import AppConfig
# from src.core.logger import logger

# def train_model():
#     """
#     Train the YOLO11 model for solar panel defect detection.
#     """
#     import torch
#     if torch.cuda.is_available():
#         logger.info(f"GPU detected: {torch.cuda.get_device_name(0)}")
#         logger.info("Will use GPU for accelerated training.")
#     else:
#         logger.warning("No GPU detected! Training will fall back to CPU and may be very slow.")

#     logger.info("Starting YOLO11 training pipeline...")
    
#     # Load configurations
#     train_config = AppConfig.get_train_config()
#     dataset_yaml_path = os.path.abspath(os.path.join(AppConfig.CONFIG_DIR, "dataset.yaml"))
    
#     logger.info(f"Using dataset config: {dataset_yaml_path}")
    
#     # Initialize YOLO model
#     model_name = train_config.get('model', {}).get('name', 'yolo11s.pt')
#     logger.info(f"Initializing model: {model_name}")
#     model = YOLO(model_name)
    
#     # Training parameters
#     params = train_config.get('training', {})
    
#     # Start training
#     logger.info("Training started. This may take a while depending on hardware.")
#     try:
#         results = model.train(
#             data=dataset_yaml_path,
#             epochs=params.get('epochs', 100),
#             patience=params.get('patience', 50),
#             batch=params.get('batch', 16),
#             imgsz=params.get('imgsz', 640),
#             device=params.get('device', 0),
#             workers=params.get('workers', 8),
#             optimizer=params.get('optimizer', 'auto'),
#             amp=params.get('amp', True), 
#             lr0=params.get('lr0', 0.01),
#             lrf=params.get('lrf', 0.01),
#             momentum=params.get('momentum', 0.937),
#             weight_decay=params.get('weight_decay', 0.0005),
#             project=params.get('project', 'solar_defects'),
#             name=params.get('name', 'exp'),
#             exist_ok=True
#         )
#         logger.info("Training completed successfully.")
#         return results
#     except Exception as e:
#         logger.error(f"Training failed: {str(e)}")
#         raise

# if __name__ == "__main__":
#     train_model()
'''