import cv2
from utils.config import Config
from utils.helpers import VehicleDetector

def real_time_detection():
    """
    Real-time vehicle detection using webcam
    """
    print("Initializing Real-Time Vehicle Detection...")
    print("Press 'q' to quit")
    
    detector = VehicleDetector()
    
    # Webcam se video capture karein
    cap = cv2.VideoCapture(0)
    
    # Camera settings set karein
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    print("Webcam started successfully!")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame")
            break
        
        # Vehicle detection run karein
        results = detector.model(frame, conf=Config.CONFIDENCE_THRESHOLD)
        
        # Frame par results draw karein
        annotated_frame = results[0].plot()
        
        # Vehicle count calculate karein
        vehicle_count = 0
        if results[0].boxes is not None:
            for box in results[0].boxes:
                class_id = int(box.cls[0])
                if class_id in Config.VEHICLE_CLASSES:
                    vehicle_count += 1
        
        # Count display karein
        cv2.putText(annotated_frame, f'Vehicles Detected: {vehicle_count}', 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # FPS calculate karein
        cv2.putText(annotated_frame, f'Press Q to quit', 
                   (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        # Output display karein
        cv2.imshow('Real-Time Vehicle Detection System', annotated_frame)
        
        # 'q' press karne par exit karein
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Resources release karein
    cap.release()
    cv2.destroyAllWindows()
    print("Real-time detection stopped.")

def video_file_detection(video_path):
    """
    Video file se vehicle detection
    """
    if not os.path.exists(video_path):
        print(f"Video file not found: {video_path}")
        return
    
    detector = VehicleDetector()
    print(f"Processing video: {video_path}")
    
    # Video detection run karein
    detector.detect_video(video_path)
    print("Video processing completed!")

if __name__ == "__main__":
    Config.create_directories()
    
    print("Real-Time Vehicle Detection")
    print("1. Use Webcam")
    print("2. Process Video File")
    
    choice = input("Enter your choice (1 or 2): ").strip()
    
    if choice == '1':
        real_time_detection()
    elif choice == '2':
        video_path = input("Enter video file path: ").strip()
        video_file_detection(video_path)
    else:
        print("Invalid choice!")