import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
import os
from .config import Config

class VehicleDetector:
    def __init__(self, model_path=None):
        if model_path is None:
            # Use pre-trained YOLOv8 model
            self.model = YOLO('yolov8n.pt')
        else:
            # Use custom trained model
            self.model = YOLO(model_path)
    
    def detect_vehicles(self, image_path, save_result=True):
        """
        Detect vehicles in a single image
        """
        # Run inference
        results = self.model(image_path, conf=Config.CONFIDENCE_THRESHOLD)
        
        # Filter only vehicle detections
        vehicle_detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                class_id = int(box.cls[0])
                if class_id in Config.VEHICLE_CLASSES:
                    vehicle_detections.append({
                        'class': Config.VEHICLE_CLASSES[class_id],
                        'confidence': float(box.conf[0]),
                        'bbox': box.xyxy[0].tolist()  # [x1, y1, x2, y2]
                    })
        
        # Save result image if requested
        if save_result:
            output_path = os.path.join(Config.RESULTS_DIR, 'detected_images', 
                                     f"detected_{os.path.basename(image_path)}")
            result.save(output_path)
            print(f"Result saved to: {output_path}")
        
        return vehicle_detections, results[0] if results else None
    
    def detect_video(self, video_path, output_path=None):
        """
        Detect vehicles in video
        """
        if output_path is None:
            output_path = os.path.join(Config.RESULTS_DIR, 'output_videos', 
                                     f"output_{os.path.basename(video_path)}")
        
        # Run inference on video
        results = self.model(video_path, save=True, project=Config.RESULTS_DIR, 
                           name='output_videos', conf=Config.CONFIDENCE_THRESHOLD)
        
        print(f"Video processing completed. Output saved.")
        return results
    
    def real_time_detection(self, camera_index=0):
        """
        Real-time vehicle detection from webcam
        """
        cap = cv2.VideoCapture(camera_index)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Run inference
            results = self.model(frame, conf=Config.CONFIDENCE_THRESHOLD)
            
            # Annotate frame with detections
            annotated_frame = results[0].plot()
            
            # Display vehicle count
            vehicle_count = sum(1 for box in results[0].boxes 
                              if int(box.cls[0]) in Config.VEHICLE_CLASSES)
            cv2.putText(annotated_frame, f'Vehicles: {vehicle_count}', 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.imshow('Real-Time Vehicle Detection', annotated_frame)
            
            # Press 'q' to quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()

def plot_results(image_path, detections):
    """
    Plot detection results with bounding boxes
    """
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
                           fill=False, edgecolor='red', linewidth=2)
        plt.gca().add_patch(rect)
        
        # Add label
        plt.text(bbox[0], bbox[1]-10, f'{class_name}: {confidence:.2f}', 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="red", alpha=0.7),
                fontsize=8, color='white')
    
    plt.axis('off')
    plt.tight_layout()
    
    # Save the plot
    output_path = os.path.join(Config.RESULTS_DIR, 'detected_images', 
                             f"plot_{os.path.basename(image_path)}")
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.show()
    
    return output_path