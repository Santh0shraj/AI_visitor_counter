import logging
import os
import urllib.request

try:
    from ultralytics import YOLO
except ImportError:
    logging.warning("ultralytics package not found. Please install it using 'pip install ultralytics'.")

class FaceDetector:
    """
    Detects persons in an image/frame using the standard YOLOv8n object model.
    Passes bounding boxes of 'persons' to InsightFace which subsequently grabs the face.
    """
    def __init__(self, confidence_threshold=0.5):
        """
        Initializes the FaceDetector and loads the generic YOLOv8n model.
        
        Args:
            confidence_threshold (float): Minimum confidence score for a detection to be considered valid.
        """
        self.confidence_threshold = confidence_threshold
        
        try:
            # Load the face-specific YOLOv8 model
            model_path = "yolov8n-face.pt"
            if not os.path.exists(model_path):
                url = "https://huggingface.co/arnabdhar/YOLOv8-Face-Detection/resolve/main/model.pt"
                print("Downloading YOLOv8 face model...")
                urllib.request.urlretrieve(url, model_path)
                print("Download complete.")
            
            self.model = YOLO(model_path)
        except Exception as e:
            logging.error(f"Failed to load YOLO face model: {e}")
            self.model = None

    def detect(self, frame):
        """
        Runs YOLO inference on a single frame to detect persons.
        
        Args:
            frame (numpy.ndarray): The image frame natively from OpenCV or similar.
            
        Returns:
            list: A list of bounding boxes in the format [x1, y1, x2, y2] as integers. 
                  Returns an empty list if no persons are found or if an error occurs.
        """
        if self.model is None:
            logging.error("YOLO model is not loaded. Cannot perform detection.")
            return []

        try:
            # Run inference on the provided frame, suppressing console spam
            # Use imgsz=640 for faster inference
            results = self.model(frame, verbose=False, imgsz=640)
            
            result = results[0]
            boxes = result.boxes
            
            detected_faces = []
            
            # Iterate through all bounding boxes found in the result
            for box in boxes:
                # The confidence score for the current bounding box
                conf = float(box.conf[0])
                
                # Filter by confidence threshold only (since this is already a face detector)
                if conf >= self.confidence_threshold:
                    # Extracts the coordinates [x1, y1, x2, y2]
                    coords = box.xyxy[0].tolist()
                    
                    # Convert the float coordinates to rounded integers as requested
                    face_bbox = [int(coords[0]), int(coords[1]), int(coords[2]), int(coords[3])]
                    
                    bbox_w = face_bbox[2] - face_bbox[0]
                    bbox_h = face_bbox[3] - face_bbox[1]
                    bbox_area = bbox_w * bbox_h
                    if bbox_area < 50 or bbox_area > 100000:
                        continue
                        
                    detected_faces.append(face_bbox)
                    
            return detected_faces
            
        except Exception as e:
            # Gracefully handle unexpected array shapes or prediction layouts
            logging.error(f"Error during detection inference: {e}")
            return []
