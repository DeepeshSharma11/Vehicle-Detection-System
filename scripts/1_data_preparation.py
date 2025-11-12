import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import shutil
from utils.config import Config

def download_sample_data():
    """
    Agar aapke paas data nahi hai toh yeh function sample images download karega
    Note: Actual mein aap apna dataset use karein
    """
    print("Creating sample data structure...")
    
    # Sample images ke liye directory banayein
    sample_images_dir = os.path.join(Config.DATA_DIR, 'raw_images')
    
    print("Please add your vehicle images to:", sample_images_dir)
    print("You can download vehicle images from:")
    print("- Kaggle: Vehicle Detection Datasets")
    print("- COCO dataset")
    print("- Or take your own photos")
    
    return sample_images_dir

def organize_dataset():
    """
    Dataset ko train/val/test mein organize karein
    """
    raw_dir = os.path.join(Config.DATA_DIR, 'raw_images')
    processed_dir = os.path.join(Config.DATA_DIR, 'processed')
    
    # Get all image files
    image_files = [f for f in os.listdir(raw_dir) 
                  if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if len(image_files) == 0:
        print("No images found in raw_images directory!")
        return
    
    # Split dataset
    train_files, temp_files = train_test_split(image_files, test_size=0.3, random_state=42)
    val_files, test_files = train_test_split(temp_files, test_size=0.5, random_state=42)
    
    # Create directories
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(processed_dir, split, 'images'), exist_ok=True)
        os.makedirs(os.path.join(processed_dir, split, 'labels'), exist_ok=True)
    
    # Copy files to respective directories
    for split, files in [('train', train_files), ('val', val_files), ('test', test_files)]:
        for file in files:
            src_path = os.path.join(raw_dir, file)
            dst_path = os.path.join(processed_dir, split, 'images', file)
            shutil.copy2(src_path, dst_path)
    
    print(f"Dataset organized:")
    print(f"Train: {len(train_files)} images")
    print(f"Validation: {len(val_files)} images") 
    print(f"Test: {len(test_files)} images")

if __name__ == "__main__":
    Config.create_directories()
    download_sample_data()
    # organize_dataset()  # Uncomment when you have actual images