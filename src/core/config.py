import yaml
import os
from pathlib import Path

def load_config(config_path: str) -> dict:
    """
        Load a YAML configuration file.
        
        Args:
            config_path (str): Path to the YAML file.
            
        Returns:
            dict: Parsed configuration dictionary.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")
        
    with open(config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
            return config
        except yaml.YAMLError as exc:
            raise Exception(f"Error parsing YAML file: {exc}")

class AppConfig:
    """ Centralized config management """
    BASE_DIR = Path(__file__).resolve().parent.parent.parent
    CONFIG_DIR = BASE_DIR / "configs"
    
    @classmethod
    def get_dataset_config(cls):
        return load_config(cls.CONFIG_DIR / "dataset.yaml")
        
    @classmethod
    def get_train_config(cls):
        return load_config(cls.CONFIG_DIR / "train.yaml")
        
    @classmethod
    def get_inference_config(cls):
        return load_config(cls.CONFIG_DIR / "inference.yaml")
