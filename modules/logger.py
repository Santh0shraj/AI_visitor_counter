import os
import logging
import cv2
from datetime import datetime

class Logger:
    """
    Handles logging text events and saving cropped face images.
    """
    def __init__(self, log_dir):
        """
        Initializes the Logger, setting up console and file logging.
        
        Args:
            log_dir (str): The root directory where logs and images will be stored.
        """
        self.log_dir = log_dir
        
        # Ensure log_dir exists
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Create a specific logger for FaceTracker
        self.logger = logging.getLogger("FaceTracker")
        self.logger.setLevel(logging.INFO)
        
        # Avoid duplicate handlers if instantiated multiple times
        if not self.logger.handlers:
            log_file_path = os.path.join(self.log_dir, "events.log")
            
            # Format: '2026-03-20 14:23:01 | INFO | message'
            formatter = logging.Formatter(
                fmt="%(asctime)s | %(levelname)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S"
            )
            
            # File handler
            file_handler = logging.FileHandler(log_file_path)
            file_handler.setFormatter(formatter)
            
            # Console handler
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            
            self.logger.addHandler(file_handler)
            self.logger.addHandler(console_handler)

    def log_entry(self, face_id, image_path):
        """
        Logs an ENTRY event to track when a face enters the frame.
        
        Args:
            face_id (str): The identifier for the face.
            image_path (str): The path to the saved cropped image.
        """
        self.logger.info(f"ENTRY | face_id={face_id} | image={image_path}")

    def log_exit(self, face_id, image_path):
        """
        Logs an EXIT event to track when a face leaves the frame.
        
        Args:
            face_id (str): The identifier for the face.
            image_path (str): The path to the saved cropped image.
        """
        self.logger.info(f"EXIT | face_id={face_id} | image={image_path}")

    def log_register(self, face_id):
        """
        Logs a REGISTER event when a new face is first detected.
        
        Args:
            face_id (str): The identifier for the new face.
        """
        self.logger.info(f"REGISTER | face_id={face_id}")

    def log_info(self, message):
        """
        Logs a general INFO message.
        
        Args:
            message (str): The text message to log.
        """
        self.logger.info(message)

    def save_face_image(self, frame, bbox, face_id, event_type):
        """
        Crops a face from the frame using bbox, applies a 20px padding (clipped to borders),
        and saves it to the appropriate dated logs folder based on event type.
        
        Args:
            frame (numpy.ndarray): The full video frame image from OpenCV.
            bbox (list, tuple): The bounding box coordinates as [x1, y1, x2, y2].
            face_id (str): The unique ID of the face.
            event_type (str): The type of event (e.g., 'entry' or 'exit').
            
        Returns:
            str: The relative saved image file path.
        """
        now = datetime.now()
        date_str = now.strftime("%Y-%m-%d")
        time_str = now.strftime("%H-%M-%S")
        
        # Route to entries or exits folder based on event_type string
        subfolder = "entries" if "entry" in event_type.lower() else "exits"
        
        # Ensure the date-specific folder exists
        save_dir = os.path.join(self.log_dir, subfolder, date_str)
        os.makedirs(save_dir, exist_ok=True)
        
        # File name format: {face_id}_{HH-MM-SS}.jpg
        filename = f"{face_id}_{time_str}.jpg"
        save_path = os.path.join(save_dir, filename)
        
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = [int(v) for v in bbox]
        
        # Add 20px padding and clip to frame bounds
        crop_x1 = max(0, x1 - 20)
        crop_y1 = max(0, y1 - 20)
        crop_x2 = min(w, x2 + 20)
        crop_y2 = min(h, y2 + 20)
        
        # Crop the specified dimensions
        cropped_face = frame[crop_y1:crop_y2, crop_x1:crop_x2]
        
        # Save image only if crop is valid
        if cropped_face.size > 0:
            cv2.imwrite(save_path, cropped_face)
            
        return save_path.replace("\\", "/")
