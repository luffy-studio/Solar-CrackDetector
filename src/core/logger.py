import logging
import sys
import os
from datetime import datetime

def setup_logger(name: str = "SolarDefectDetection", log_dir: str = "logs"):
    """
    Setup a logger for the application.
    
    Args:
        name (str): Name of the logger
        log_dir (str): Directory to save log files
        
    Returns:
        logging.Logger: Configured logger
    """
    os.makedirs(log_dir, exist_ok=True)
    
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    if not logger.handlers:
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(console_format)
        
        # File handler
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        file_handler = logging.FileHandler(os.path.join(log_dir, f"app_{timestamp}.log"))
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(console_format)
        
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)
        
    return logger

# Create a default logger instance to be used across the app
logger = setup_logger()
