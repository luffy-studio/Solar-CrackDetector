import cv2
import sys
import os
import time

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.inference.predictor import InferencePredictor
from src.core.config import AppConfig
from src.core.logger import logger

def run_realtime_inference():
    """
    Runs real-time inference on webcam stream or video file.
    """
    config = AppConfig.get_inference_config()
    stream_config = config.get('stream', {})
    
    source = stream_config.get('source', 0)
    display = stream_config.get('display', True)
    
    logger.info(f"Starting real-time inference on source: {source}")
    
    predictor = InferencePredictor()
    
    # Check if we should convert int string to int
    if isinstance(source, str) and source.isdigit():
        source = int(source)
        
    cap = cv2.VideoCapture(source)
    
    if not cap.isOpened():
        logger.error(f"Failed to open video source: {source}")
        return
        
    logger.info("Press 'q' to quit the stream.")
    
    fps_start_time = time.time()
    fps_counter = 0
    fps = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            logger.warning("Failed to grab frame. Exiting...")
            break
            
        # Inference
        result = predictor.predict_image(frame)
        annotated_frame = result["annotated_image"]
        
        # Calculate FPS
        fps_counter += 1
        if (time.time() - fps_start_time) > 1.0:
            fps = fps_counter / (time.time() - fps_start_time)
            fps_counter = 0
            fps_start_time = time.time()
            
        # Display FPS
        cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    
        if display:
            cv2.imshow("Solar Panel Defect Detection", annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    cap.release()
    cv2.destroyAllWindows()
    logger.info("Stream closed.")

if __name__ == "__main__":
    run_realtime_inference()
