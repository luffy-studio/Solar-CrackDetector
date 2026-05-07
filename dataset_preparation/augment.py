# import albumentations as A
# import cv2
# import os
# from pathlib import Path
# from tqdm import tqdm

# def get_augmentation_pipeline():
#     """
#     Defines the data augmentation pipeline using Albumentations.
#     Suitable for solar panel images.
#     """
#     return A.Compose([
#         A.HorizontalFlip(p=0.5),
#         A.VerticalFlip(p=0.5),
#         A.RandomRotate90(p=0.5),
#         A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.5),
#         A.OneOf([
#             A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1),
#             A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=1)
#         ], p=0.5),
#         A.OneOf([
#             A.GaussianBlur(blur_limit=(3, 7), p=1),
#             A.MotionBlur(blur_limit=(3, 7), p=1)
#         ], p=0.3)
#     ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

# def augment_dataset(image_dir: str, label_dir: str, output_image_dir: str, output_label_dir: str, num_augs: int = 2):
#     """
#     Applies offline augmentation to a dataset (both images and YOLO format labels).
#     """
#     os.makedirs(output_image_dir, exist_ok=True)
#     os.makedirs(output_label_dir, exist_ok=True)
    
#     transform = get_augmentation_pipeline()
    
#     image_paths = list(Path(image_dir).glob('*.jpg'))
    
#     for img_path in tqdm(image_paths, desc="Augmenting Dataset"):
#         img = cv2.imread(str(img_path))
#         label_path = Path(label_dir) / f"{img_path.stem}.txt"
        
#         bboxes = []
#         class_labels = []
        
#         # Read existing labels if they exist
#         if label_path.exists():
#             with open(label_path, 'r') as f:
#                 for line in f.readlines():
#                     parts = line.strip().split()
#                     class_id = int(parts[0])
#                     # YOLO format: [x_center, y_center, width, height]
#                     bbox = [float(x) for x in parts[1:]]
#                     bboxes.append(bbox)
#                     class_labels.append(class_id)
        
#         # Save original
#         shutil.copy(img_path, os.path.join(output_image_dir, img_path.name))
#         if label_path.exists():
#             shutil.copy(label_path, os.path.join(output_label_dir, label_path.name))
            
#         # Generate augmentations
#         for i in range(num_augs):
#             try:
#                 transformed = transform(image=img, bboxes=bboxes, class_labels=class_labels)
#                 aug_img = transformed['image']
#                 aug_bboxes = transformed['bboxes']
#                 aug_labels = transformed['class_labels']
                
#                 aug_img_name = f"{img_path.stem}_aug_{i}.jpg"
#                 aug_label_name = f"{img_path.stem}_aug_{i}.txt"
                
#                 cv2.imwrite(os.path.join(output_image_dir, aug_img_name), aug_img)
                
#                 with open(os.path.join(output_label_dir, aug_label_name), 'w') as f:
#                     for label, bbox in zip(aug_labels, aug_bboxes):
#                         bbox_str = " ".join([f"{x:.6f}" for x in bbox])
#                         f.write(f"{label} {bbox_str}\n")
                        
#             except Exception as e:
#                 print(f"Failed augmentation on {img_path.name}: {e}")

# if __name__ == "__main__":
#     import shutil
#     augment_dataset('data/raw/images', 'data/raw/labels', 'data/processed/images/train', 'data/processed/labels/train')
#     print("Augmentation script ready.")
#     # Example usage:



import albumentations as A
import cv2
import os
from pathlib import Path
from tqdm import tqdm

def get_augmentation_pipeline():

    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),

        A.ShiftScaleRotate(
            shift_limit=0.0625,
            scale_limit=0.1,
            rotate_limit=15,
            p=0.5
        ),

        A.OneOf([
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=1
            ),

            A.HueSaturationValue(
                hue_shift_limit=20,
                sat_shift_limit=30,
                val_shift_limit=20,
                p=1
            )
        ], p=0.5),

        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 7), p=1),
            A.MotionBlur(blur_limit=(3, 7), p=1)
        ], p=0.3)
    ])

def augment_class_folder(folder_path, num_augs=2):

    transform = get_augmentation_pipeline()

    image_paths = list(Path(folder_path).glob('*.*'))

    for img_path in tqdm(image_paths, desc=f"Augmenting {folder_path}"):

        img = cv2.imread(str(img_path))

        if img is None:
            continue

        for i in range(num_augs):

            transformed = transform(image=img)

            aug_img = transformed['image']

            aug_name = f"{img_path.stem}_aug_{i}.jpg"

            save_path = os.path.join(folder_path, aug_name)

            cv2.imwrite(save_path, aug_img)

if __name__ == "__main__":

    augment_class_folder(
        "C:/Users/sarva/.vscode/Styarth sir dataset/data/processed/classification_dataset/train/defective"
    )

    augment_class_folder(
        "C:/Users/sarva/.vscode/Styarth sir dataset/data/processed/classification_dataset/train/non_defective"
    )

    print("Augmentation completed.")

