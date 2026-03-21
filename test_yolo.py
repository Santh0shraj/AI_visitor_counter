import os
try:
    from ultralytics import YOLO
    print("Ultralytics imported successfully")
    model_path = "yolov8n-face.pt"
    if os.path.exists(model_path):
        print(f"Model file {model_path} exists")
        model = YOLO(model_path)
        print("Model loaded successfully")
    else:
        print(f"Model file {model_path} DOES NOT exist")
except Exception as e:
    print(f"Error: {e}")
