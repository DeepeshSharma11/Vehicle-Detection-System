import os
from ultralytics import YOLO
from utils.config import Config
import matplotlib.pyplot as plt
import json

def train_vehicle_detector():
    """
    YOLOv8 model ko vehicle detection ke liye train/fine-tune karein
    """
    print("Starting Vehicle Detection Training...")
    
    # Load pre-trained YOLOv8 model
    model = YOLO('yolov8n.pt')
    
    # Training configuration
    training_config = {
        'data': 'coco8.yaml',  # You can create custom YAML for your data
        'epochs': Config.EPOCHS,
        'imgsz': Config.IMAGE_SIZE,
        'batch': Config.BATCH_SIZE,
        'lr0': Config.LEARNING_RATE,
        'device': 'cpu',  # Change to 'cuda' if you have GPU
        'workers': 4,
        'project': Config.MODELS_DIR,
        'name': 'custom_trained',
        'exist_ok': True
    }
    
    # Train the model
    results = model.train(**training_config)
    
    # Save training metrics
    metrics = {
        'map50': results.results_dict.get('metrics/mAP50(B)', 0),
        'map': results.results_dict.get('metrics/mAP50-95(B)', 0),
        'precision': results.results_dict.get('metrics/precision(B)', 0),
        'recall': results.results_dict.get('metrics/recall(B)', 0)
    }
    
    metrics_path = os.path.join(Config.RESULTS_DIR, 'performance_metrics', 'training_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    
    print("Training completed!")
    print(f"Model saved to: {os.path.join(Config.MODELS_DIR, 'custom_trained')}")
    print(f"Training metrics: {metrics}")
    
    return model, metrics

def evaluate_model(model_path):
    """
    Trained model ki evaluation karein
    """
    model = YOLO(model_path)
    
    # Validation metrics
    metrics = model.val()
    print("Model Evaluation Results:")
    print(f"mAP50: {metrics.box.map50}")
    print(f"mAP50-95: {metrics.box.map}")
    print(f"Precision: {metrics.box.mp}")
    print(f"Recall: {metrics.box.mr}")
    
    return metrics

if __name__ == "__main__":
    Config.create_directories()
    
    # Option 1: Use pre-trained model directly (Quick start)
    print("Option 1: Using pre-trained YOLOv8 model (No training needed)")
    
    # Option 2: Train custom model (Uncomment when you have dataset)
    # model, metrics = train_vehicle_detector()
    # evaluate_model(model.path)