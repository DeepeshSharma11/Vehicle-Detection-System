import os

class Config:
    # Paths
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    MODELS_DIR = os.path.join(BASE_DIR, 'models')
    RESULTS_DIR = os.path.join(BASE_DIR, 'results')
    
    # Model settings
    MODEL_NAME = 'yolov8n.pt'
    IMAGE_SIZE = 640
    CONFIDENCE_THRESHOLD = 0.5
    IOU_THRESHOLD = 0.5
    
    # Training settings
    EPOCHS = 50
    BATCH_SIZE = 16
    LEARNING_RATE = 0.001
    
    # Vehicle classes (COCO dataset classes)
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
            os.path.join(cls.DATA_DIR, 'annotations'),
            cls.MODELS_DIR,
            os.path.join(cls.MODELS_DIR, 'pre_trained'),
            os.path.join(cls.MODELS_DIR, 'custom_trained'),
            cls.RESULTS_DIR,
            os.path.join(cls.RESULTS_DIR, 'detected_images'),
            os.path.join(cls.RESULTS_DIR, 'output_videos'),
            os.path.join(cls.RESULTS_DIR, 'performance_metrics')
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            print(f"Created directory: {directory}")