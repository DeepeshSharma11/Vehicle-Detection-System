import os
import cv2
from utils.config import Config
from utils.helpers import VehicleDetector, plot_results

def detect_single_image():
    """
    Single image mein vehicle detection
    """
    detector = VehicleDetector()
    
    # Sample image path - aap yahaan apni image ka path dein
    sample_image = os.path.join(Config.DATA_DIR, 'raw_images', 'sample_vehicle.jpg')
    
    # Agar sample image nahi hai toh koi aur image use karein
    if not os.path.exists(sample_image):
        print(f"Sample image not found at {sample_image}")
        print("Please add an image to the raw_images directory")
        return
    
    print(f"Processing image: {sample_image}")
    
    # Detection run karein
    detections, result = detector.detect_vehicles(sample_image, save_result=True)
    
    # Results display karein
    print(f"\nDetection Results:")
    print(f"Total vehicles detected: {len(detections)}")
    
    for i, detection in enumerate(detections, 1):
        print(f"{i}. {detection['class']} - Confidence: {detection['confidence']:.2f}")
    
    # Plot results with bounding boxes
    if detections:
        plot_path = plot_results(sample_image, detections)
        print(f"Detailed plot saved to: {plot_path}")

def detect_multiple_images():
    """
    Multiple images par batch processing
    """
    detector = VehicleDetector()
    images_dir = os.path.join(Config.DATA_DIR, 'raw_images')
    
    image_files = [f for f in os.listdir(images_dir) 
                  if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not image_files:
        print("No images found in raw_images directory!")
        return
    
    print(f"Processing {len(image_files)} images...")
    
    total_vehicles = 0
    for image_file in image_files:
        image_path = os.path.join(images_dir, image_file)
        detections, _ = detector.detect_vehicles(image_path, save_result=True)
        total_vehicles += len(detections)
        print(f"{image_file}: {len(detections)} vehicles")
    
    print(f"\nTotal vehicles detected in all images: {total_vehicles}")

if __name__ == "__main__":
    Config.create_directories()
    
    print("Vehicle Detection System")
    print("1. Detect vehicles in single image")
    print("2. Detect vehicles in multiple images")
    
    choice = input("Enter your choice (1 or 2): ").strip()
    
    if choice == '1':
        detect_single_image()
    elif choice == '2':
        detect_multiple_images()
    else:
        print("Invalid choice!")