import os
import cv2
import shutil
from pathlib import Path
from tqdm import tqdm

def preprocess_images(input_dir: str, output_dir: str, target_size: tuple = (640, 640)):
    """
    Preprocess images by resizing and standardizing format.
    
    Args:
        input_dir (str): Directory containing raw images.
        output_dir (str): Directory to save processed images.
        target_size (tuple): Target (width, height) for resizing.
    """
    os.makedirs(output_dir, exist_ok=True)
    input_path = Path(input_dir)
    
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    images = [f for f in input_path.rglob('*') if f.suffix.lower() in image_extensions]
    
    print(f"Found {len(images)} images for preprocessing.")
    
    for img_path in tqdm(images, desc="Processing Images"):
        try:
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"Warning: Could not read image {img_path}")
                continue
                
            # Resize image
            resized_img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
            
            # Save standardized as JPG
            output_filename = f"{img_path.stem}.jpg"
            output_filepath = os.path.join(output_dir, output_filename)
            cv2.imwrite(output_filepath, resized_img)
            
        except Exception as e:
            print(f"Error processing {img_path}: {e}")

if __name__ == "__main__":
    raw_data_dir = "C:/Users/sarva/.vscode/Styarth sir dataset/data/images"
    processed_data_dir = "C:/Users/sarva/.vscode/Styarth sir dataset/data/processed/images/all"
    
    if os.path.exists(raw_data_dir):
        preprocess_images(raw_data_dir, processed_data_dir)
    else:
        print(f"Raw data directory {raw_data_dir} does not exist. Skipping for now.")
