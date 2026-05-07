import os
from ultralytics import YOLO

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))


from src.core.config import AppConfig
from src.core.logger import logger

class SolarDefectModel:
    """
    Wrapper around YOLO11 model for inference.
    """
    def __init__(self, model_path: str = None):
        import torch
        if model_path is None:
            config = AppConfig.get_inference_config()
            self.model_path = config.get('model_path', 'runs/classify/solar_defects/classification_run/weights/best.pt')
            
            # Auto-detect GPU and configure device
            if torch.cuda.is_available():
                self.device = config.get('device', 0)
                self.half = config.get('half', True) # Use FP16 for faster inference
                logger.info(f"Using GPU device {self.device} with half-precision={self.half}")
            else:
                self.device = 'cpu'
                self.half = False
                logger.info("GPU not found. Falling back to CPU inference.")
                
            self.conf = config.get('confidence_threshold', 0.5)
            self.iou = config.get('iou_threshold', 0.45)
        else:
            self.model_path = model_path
            self.device = 0 if torch.cuda.is_available() else 'cpu'
            self.half = True if self.device != 'cpu' else False
            self.conf = 0.5
            self.iou = 0.45
            
        self.model = self._load_model()
        
    def _load_model(self):
        """Loads the YOLO11 model weights."""
        if not os.path.exists(self.model_path):
            logger.warning(f"Model not found at {self.model_path}. Please ensure model is trained.")
            return None
            
        try:
            logger.info(f"Loading YOLO11 model from {self.model_path}")
            model = YOLO(self.model_path)
            # Warmup is recommended but omitting for faster initialization here
            return model
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return None
            
    def predict(self, source):
        """
        Run prediction on a source (image path, numpy array, etc.)
        """
        if self.model is None:
            raise ValueError("Model is not loaded.")
            
        results = self.model.predict(
            source=source,
            conf=self.conf,
            iou=self.iou,
            device=self.device,
            half=self.half,
            verbose=False
        )
        return results
