# import cv2
# import numpy as np
# from typing import List, Dict, Any
# from src.inference.model import SolarDefectModel
# from src.core.logger import logger

# class InferencePredictor:
#     """
#     High-level predictor class that utilizes SolarDefectModel.
#     Handles input/output formatting.
#     """
#     def __init__(self):
#         self.model = SolarDefectModel()
        
#     def predict_image(self, image: np.ndarray) -> Dict[str, Any]:
#         """
#         Predicts defects on a single image.
#         Returns bounding boxes, confidences, and class names.
#         """
#         results = self.model.predict(image)
        
#         predictions = []
#         for result in results:
#             boxes = result.boxes
#             for box in boxes:
#                 x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
#                 conf = float(box.conf[0].cpu().numpy())
#                 cls_id = int(box.cls[0].cpu().numpy())
#                 cls_name = result.names[cls_id]
                
#                 predictions.append({
#                     "bbox": [int(x1), int(y1), int(x2), int(y2)],
#                     "confidence": conf,
#                     "class_id": cls_id,
#                     "class_name": cls_name
#                 })
                
#         # Optional: draw results on image
#         annotated_img = results[0].plot() if len(results) > 0 else image
        
#         return {
#             "predictions": predictions,
#             "annotated_image": annotated_img
#         }


import sys
from pathlib import Path
from typing import Dict, Any

import numpy as np

# Add project root to Python path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.inference.model import SolarDefectModel
from src.core.logger import logger


class InferencePredictor:
    """
    High-level predictor class for YOLO11 classification inference.
    """

    def __init__(self):
        self.model = SolarDefectModel()

    def predict_image(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Predict class for a single image.

        Args:
            image (np.ndarray): Input image array.

        Returns:
            Dict[str, Any]: Prediction result with class id, class name, and confidence.
        """
        if image is None:
            raise ValueError("Input image is None.")

        results = self.model.predict(image)

        if not results or results[0].probs is None:
            raise ValueError("No classification result returned by the model.")

        result = results[0]

        class_id = int(result.probs.top1)
        confidence = float(result.probs.top1conf)
        class_name = result.names[class_id]

        return {
            "class_id": class_id,
            "class_name": class_name,
            "confidence": confidence,
        }


if __name__ == "__main__":
    import cv2

    image_path = r"C:\Users\sarva\.vscode\Styarth sir dataset\data\processed\classification_dataset\train\defective\cell0001_aug_1.jpg"

    img = cv2.imread(image_path)

    if img is None:
        raise FileNotFoundError(f"Image not found at: {image_path}")

    predictor = InferencePredictor()
    prediction = predictor.predict_image(img)

    print("Prediction:", prediction)