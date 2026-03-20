import logging
import numpy as np
import cv2

try:
    from insightface.app import FaceAnalysis
except ImportError:
    logging.warning("insightface package not found. Please install it using 'pip install insightface'.")

class FaceRecognizer:
    """
    Handles face recognition and embedding extraction using InsightFace.
    """
    def __init__(self):
        """
        Initializes the FaceRecognizer using the ArcFace model strictly on the CPU.
        """
        try:
            # Initialize InsightFace with buffalo_l (more accurate)
            self.app = FaceAnalysis(
                name='buffalo_l',
                allowed_modules=['detection', 'recognition'],
                providers=['CPUExecutionProvider']
            )
            self.app.prepare(ctx_id=0, det_size=(160, 160))
        except Exception as e:
            logging.error(f"Failed to initialize InsightFace model: {e}")
            self.app = None

    def get_embedding(self, frame, bbox):
        """
        Crops a face from the frame using the provided bounding box and
        extracts its structural embedding using InsightFace.
        
        Args:
            frame (numpy.ndarray): The original video frame.
            bbox (list or tuple): The bounding box coordinates (x1, y1, x2, y2).
            
        Returns:
            numpy.ndarray: The 512-d facial embedding array, or None if extraction fails.
        """
        if self.app is None:
            logging.error("InsightFace model is not loaded.")
            return None

        try:
            x1, y1, x2, y2 = [int(v) for v in bbox]
            h, w = frame.shape[:2]
            
            # Slightly expand the crop to give InsightFace context for precise alignment
            pad_x = 10
            pad_y = 10
            x1 = max(0, x1 - pad_x)
            y1 = max(0, y1 - pad_y)
            x2 = min(w, x2 + pad_x)
            y2 = min(h, y2 + pad_y)
            
            cropped_face = frame[y1:y2, x1:x2]
            
            # Ensure crop is mathematically valid
            if cropped_face.size == 0:
                logging.warning("Cropped face region is inherently empty.")
                return None
            
            # Resize the crop to maximum 224x224 for better alignment accuracy
            crop_h, crop_w = cropped_face.shape[:2]
            max_size = 224
            if crop_h > max_size or crop_w > max_size:
                scale = max_size / max(crop_h, crop_w)
                new_w = int(crop_w * scale)
                new_h = int(crop_h * scale)
                cropped_face = cv2.resize(cropped_face, (new_w, new_h))
                
            # Perform inference on the isolated face region
            faces = self.app.get(cropped_face)
            
            if faces:
                # Retrieve the embedding from the highest-confidence match inside the crop
                return faces[0].embedding
            else:
                logging.warning("InsightFace could not capture features in the cropped region.")
                return None
                
        except Exception as e:
            # Trap all unforeseen exceptions to avoid crashing main loop
            logging.error(f"Error analyzing face via InsightFace: {e}")
            return None

    def compare(self, embedding1, embedding2):
        """
        Computes the cosine similarity between two 512-d face embeddings.
        
        Args:
            embedding1 (numpy.ndarray): The first face embedding.
            embedding2 (numpy.ndarray): The second face embedding.
            
        Returns:
            float: A similarity score bounded between 0.0 and 1.0. Returns 0.0 on error.
        """
        try:
            if embedding1 is None or embedding2 is None:
                return 0.0
                
            # Normalize vectors to safely compute cosine similarity using the dot product
            emb1_norm = embedding1 / np.linalg.norm(embedding1)
            emb2_norm = embedding2 / np.linalg.norm(embedding2)
            similarity = np.dot(emb1_norm, emb2_norm)
            
            # Cap values specifically to [0.0, 1.0] manually to account for microscopic float errors
            return float(np.clip(similarity, 0.0, 1.0))
        except Exception as e:
            logging.error(f"Error computing cosine similarity: {e}")
            return 0.0

    def embedding_to_bytes(self, embedding):
        """
        Encodes a given numpy array embedding into standard bytes format 
        ideal for efficient BLOB storage in SQLite database.
        
        Args:
            embedding (numpy.ndarray): The embedding array.
            
        Returns:
            bytes: The serialized byte string of the numeric array, or None on error.
        """
        try:
            if embedding is None:
                return None
            return embedding.astype(np.float32).tobytes()
        except Exception as e:
            logging.error(f"Error encoding embedding into bytes format: {e}")
            return None

    def bytes_to_embedding(self, b):
        """
        Decodes a database-stored byte string back into a functional 
        numpy array with its original data type (float32).
        
        Args:
            b (bytes): The raw byte string from the DB.
            
        Returns:
            numpy.ndarray: The decoded float32 array, or None on error.
        """
        try:
            if b is None:
                return None
            # Resurrect format back to float32 standard utilized by InsightFace
            return np.frombuffer(b, dtype=np.float32).copy()
        except Exception as e:
            logging.error(f"Error decoding DB byte stream to embedding array: {e}")
            return None
