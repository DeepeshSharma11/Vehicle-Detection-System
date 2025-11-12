import os
import cv2
from ultralytics import YOLO
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import json

class Config:
    # Paths
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    MODELS_DIR = os.path.join(BASE_DIR, 'models')
    RESULTS_DIR = os.path.join(BASE_DIR, 'results')
    
    # Model settings
    MODEL_NAME = 'yolov8n.pt'
    IMAGE_SIZE = 640
    CONFIDENCE_THRESHOLD = 0.5
    
    # Vehicle classes (COCO dataset)
    VEHICLE_CLASSES = {
        2: 'car',
        3: 'motorcycle', 
        5: 'bus',
        7: 'truck'
    }
    
    @classmethod
    def create_directories(cls):
        """Create all necessary directories"""
        directories = [
            cls.DATA_DIR,
            os.path.join(cls.DATA_DIR, 'raw_images'),
            os.path.join(cls.DATA_DIR, 'processed'),
            cls.MODELS_DIR,
            cls.RESULTS_DIR,
            os.path.join(cls.RESULTS_DIR, 'detected_images'),
            os.path.join(cls.RESULTS_DIR, 'output_videos'),
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            print(f"✓ Created: {directory}")

class VehicleDetector:
    def __init__(self, model_path=None):
        print("Loading YOLOv8 model...")
        if model_path is None:
            self.model = YOLO('yolov8n.pt')
        else:
            self.model = YOLO(model_path)
        print("✓ Model loaded successfully!")
    
    def detect_vehicles(self, image_path):
        """Detect vehicles in a single image"""
        if not os.path.exists(image_path):
            print(f"❌ Image not found: {image_path}")
            return []
        
        print(f"Processing image: {image_path}")
        
        # Run inference
        results = self.model(image_path, conf=Config.CONFIDENCE_THRESHOLD)
        
        # Filter only vehicle detections
        vehicle_detections = []
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    class_id = int(box.cls[0])
                    if class_id in Config.VEHICLE_CLASSES:
                        vehicle_detections.append({
                            'class': Config.VEHICLE_CLASSES[class_id],
                            'confidence': float(box.conf[0]),
                            'bbox': box.xyxy[0].tolist()  # [x1, y1, x2, y2]
                        })
        
        # Save result
        output_dir = os.path.join(Config.RESULTS_DIR, 'detected_images')
        os.makedirs(output_dir, exist_ok=True)
        
        output_path = os.path.join(output_dir, f"detected_{os.path.basename(image_path)}")
        result.save(output_path)
        print(f"✓ Result saved to: {output_path}")
        
        return vehicle_detections, output_path
    
    def real_time_detection(self):
        """Real-time vehicle detection using webcam"""
        print("🚀 Starting Real-Time Vehicle Detection...")
        print("Press 'q' to quit")
        
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        if not cap.isOpened():
            print("❌ Error: Could not open webcam")
            return
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Run inference
            results = self.model(frame, conf=Config.CONFIDENCE_THRESHOLD)
            
            # Annotate frame
            annotated_frame = results[0].plot()
            
            # Count vehicles
            vehicle_count = 0
            if results[0].boxes is not None:
                for box in results[0].boxes:
                    class_id = int(box.cls[0])
                    if class_id in Config.VEHICLE_CLASSES:
                        vehicle_count += 1
            
            # Display count
            cv2.putText(annotated_frame, f'Vehicles: {vehicle_count}', 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(annotated_frame, 'Press Q to quit', 
                       (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            
            cv2.imshow('Vehicle Detection System', annotated_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        print("Real-time detection stopped.")

def display_detection_results(image_path, detections):
    """Display detection results with bounding boxes"""
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    plt.figure(figsize=(12, 8))
    plt.imshow(image)
    
    for detection in detections:
        bbox = detection['bbox']
        class_name = detection['class']
        confidence = detection['confidence']
        
        # Draw bounding box
        rect = plt.Rectangle((bbox[0], bbox[1]), bbox[2]-bbox[0], bbox[3]-bbox[1],
                           fill=False, edgecolor='red', linewidth=3)
        plt.gca().add_patch(rect)
        
        # Add label
        plt.text(bbox[0], bbox[1]-10, f'{class_name}: {confidence:.2f}', 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="red", alpha=0.7),
                fontsize=12, color='white', weight='bold')
    
    plt.axis('off')
    plt.title('Vehicle Detection Results', fontsize=16, weight='bold')
    plt.tight_layout()
    plt.show()

def demo_with_sample_image():
    """Create a sample demo if no images available"""
    print("Creating sample demo...")
    
    # Create a simple colored image
    sample_image = np.ones((400, 600, 3), dtype=np.uint8) * 100  # Gray background
    
    # Add some colored rectangles as "vehicles"
    # Car
    cv2.rectangle(sample_image, (100, 150), (300, 250), (0, 0, 255), -1)  # Red car
    cv2.rectangle(sample_image, (350, 200), (500, 300), (255, 0, 0), -1)  # Blue truck
    
    sample_path = os.path.join(Config.DATA_DIR, 'raw_images', 'demo_image.jpg')
    os.makedirs(os.path.dirname(sample_path), exist_ok=True)
    cv2.imwrite(sample_path, sample_image)
    
    print(f"✓ Created demo image: {sample_path}")
    return sample_path

def main():
    """Main application"""
    print("=" * 60)
    print("           VEHICLE DETECTION SYSTEM")
    print("=" * 60)
    
    # Create directories
    Config.create_directories()
    
    # Initialize detector
    detector = VehicleDetector()
    
    while True:
        print("\n" + "=" * 40)
        print("MAIN MENU")
        print("=" * 40)
        print("1. 📷 Detect Vehicles in Image")
        print("2. 🎥 Real-Time Detection (Webcam)")
        print("3. 🚗 Demo with Sample Image")
        print("4. ❌ Exit")
        print("=" * 40)
        
        choice = input("Enter your choice (1-4): ").strip()
        
        if choice == '1':
            # Image detection
            image_path = input("Enter image path (or press Enter for demo): ").strip()
            
            if not image_path:
                image_path = demo_with_sample_image()
            elif not os.path.exists(image_path):
                print("❌ Image not found! Using demo image instead.")
                image_path = demo_with_sample_image()
            
            detections, output_path = detector.detect_vehicles(image_path)
            
            print(f"\n📊 DETECTION RESULTS:")
            print(f"Total vehicles detected: {len(detections)}")
            
            for i, detection in enumerate(detections, 1):
                print(f"  {i}. {detection['class'].upper()} - Confidence: {detection['confidence']:.2f}")
            
            # Display results
            if detections:
                display_detection_results(output_path, detections)
            else:
                print("❌ No vehicles detected!")
        
        elif choice == '2':
            # Real-time detection
            print("\n🎥 Starting webcam...")
            detector.real_time_detection()
        
        elif choice == '3':
            # Demo with sample
            print("\n🚗 Creating demo...")
            image_path = demo_with_sample_image()
            detections, output_path = detector.detect_vehicles(image_path)
            
            print(f"✓ Demo completed! Check: {output_path}")
            
            if detections:
                display_detection_results(output_path, detections)
        
        elif choice == '4':
            print("🙏 Thank you for using Vehicle Detection System!")
            print("Made with ❤️ for College Project")
            break
        
        else:
            print("❌ Invalid choice! Please enter 1, 2, 3, or 4.")

if __name__ == "__main__":
    main()