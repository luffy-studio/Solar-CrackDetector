import sys
import os
from ultralytics import YOLO

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.core.config import AppConfig
from src.core.logger import logger

def evaluate_model():
    """
     trained YOLOv11 model.
    """
    logger.info("Starting evaluation pipeline...")
    
    inference_config = AppConfig.get_inference_config()
    model_path = inference_config.get('model_path')
    
    if not os.path.exists(model_path):
        logger.error(f"Model file not found at: {model_path}")
        logger.error("Please train the model first or provide a valid path.")
        return
        
    logger.info(f"Loading model from {model_path}")
    model = YOLO(model_path)
    
    dataset_yaml_path = os.path.abspath(os.path.join(AppConfig.CONFIG_DIR, "dataset.yaml"))
    
    logger.info("Running validation...")
    try:
        # Validate the model
        metrics = model.val(data=dataset_yaml_path, split='val')
        
        # Log metrics
        logger.info(f"mAP50-95: {metrics.box.map}")
        logger.info(f"mAP50: {metrics.box.map50}")
        logger.info(f"mAP75: {metrics.box.map75}")
        logger.info("Evaluation completed successfully.")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}")

if __name__ == "__main__":
    evaluate_model()
