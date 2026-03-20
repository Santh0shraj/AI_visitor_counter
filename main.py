import json
import cv2
import numpy as np
import time
from datetime import datetime
import uuid
import os
import logging

from modules.database import DatabaseManager
from modules.logger import Logger
from modules.detector import FaceDetector
from modules.recognizer import FaceRecognizer
from modules.tracker import FaceTracker
from modules.counter import VisitorCounter

def main():
    try:
        # 1. Load configuration variables from config.json
        with open('config.json', 'r') as f:
            config = json.load(f)

        video_source = config.get("video_source", "sample.mp4")
        frame_skip_interval = config.get("frame_skip_interval", 3)
        similarity_threshold = config.get("similarity_threshold", 0.45)
        exit_timeout_frames = config.get("exit_timeout_frames", 30)
        log_dir = config.get("log_dir", "logs")
        db_path = config.get("db_path", "db/faces.db")
        detection_confidence = config.get("detection_confidence", 0.5)

        # 2. Initialize all modules with values retrieved from your config
        logger = Logger(log_dir)
        logger.log_info("Initializing Face Tracker System...")
        
        db_manager = DatabaseManager(db_path)
        detector = FaceDetector(confidence_threshold=detection_confidence)
        recognizer = FaceRecognizer()
        tracker = FaceTracker()
        counter = VisitorCounter(db_manager)
        
        # 3. Open the target video source using OpenCV VideoCapture
        logger.log_info(f"Opening video source: {video_source}")
        cap = cv2.VideoCapture(video_source)
        
        if not cap.isOpened():
            logger.log_info(f"Failed to open video source: {video_source}. Exiting.")
            return
            
        frame_number = 0
        
        # Load all embeddings into memory once at startup to avoid per-frame DB lookups
        embedding_cache = {}
        for face_id, emb_bytes in db_manager.get_all_embeddings_multi():
            emb = recognizer.bytes_to_embedding(emb_bytes)
            if emb is not None:
                if face_id not in embedding_cache:
                    embedding_cache[face_id] = []
                embedding_cache[face_id].append(emb)
        
        recently_registered = {}  # face_id -> frame_number
        
        # 4. Continuously loop over all frames in the stream
        while True:
            try:
                # a. Read next frame; if None or unable to read, we've reached the end
                ret, frame = cap.read()
                if not ret or frame is None:
                    break
                    
                # b. Increment overall frame counter
                frame_number += 1
                
                # c. Only run expensive detection every subset of frames to preserve CPU/GPU
                if frame_number % frame_skip_interval == 0:
                    t_start = time.time()
                    
                    # Resize frame for detection to reduce pixels processed
                    detection_frame = cv2.resize(frame, (640, 360)) if frame.shape[1] > 640 else frame
                    
                    # Run FaceDetector inference on the scaled-down frame
                    bboxes = detector.detect(detection_frame)
                    
                    # Scale bboxes back to original frame coordinates
                    scale_x = frame.shape[1] / detection_frame.shape[1]
                    scale_y = frame.shape[0] / detection_frame.shape[0]
                    bboxes = [[int(b[0]*scale_x), int(b[1]*scale_y), 
                               int(b[2]*scale_x), int(b[3]*scale_y)] 
                              for b in bboxes]
                    
                    # d. Process each detected boxed region mathematically
                    for bbox in bboxes:
                        try:
                            # Isolate the region and distill it to a 512-d array
                            embedding = recognizer.get_embedding(frame, bbox)
                            
                            # Skip if InsightFace failed on alignment or geometry limitations
                            if embedding is None:
                                continue
                                
                            best_match_id = None
                            best_similarity = 0.0
                            
                            # Scour in-memory cache for closest spatial parallel across ALL angles
                            for stored_face_id, stored_embs in embedding_cache.items():
                                for stored_emb in stored_embs:
                                    # Normalize both embeddings before comparison
                                    e1 = embedding / (np.linalg.norm(embedding) + 1e-10)
                                    e2 = stored_emb / (np.linalg.norm(stored_emb) + 1e-10)
                                    sim = float(np.dot(e1, e2))
                                    sim = max(0.0, min(1.0, sim))
                                    if sim > best_similarity:
                                        best_similarity = sim
                                        best_match_id = stored_face_id
                            
                            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            
                            # If similarity clears threshold -> Confirmed Recognition
                            if best_similarity > similarity_threshold and best_match_id is not None:
                                face_id = best_match_id
                                
                                # If matched but from a different angle, store this 
                                # new embedding to improve future matching
                                if best_similarity < 0.75:
                                    emb_bytes = recognizer.embedding_to_bytes(embedding)
                                    db_manager.insert_embedding(face_id, emb_bytes, current_time)
                                    if face_id not in embedding_cache:
                                        embedding_cache[face_id] = []
                                    embedding_cache[face_id].append(embedding)
                                
                                # Scenario: They were recognized but fell dormant (they re-entered frame)
                                if not tracker.is_active(face_id):
                                    entry_img_path = logger.save_face_image(frame, bbox, face_id, "entry")
                                    logger.log_entry(face_id, entry_img_path)
                                    db_manager.insert_event(face_id, "entry", current_time, entry_img_path)
                                    
                            # Else -> Genuinely New User Detected or Cooldown Check
                            else:
                                # Check if any recently registered face is too close in time
                                is_duplicate = False
                                for recent_id, recent_frame in recently_registered.items():
                                    if frame_number - recent_frame < 90:
                                        if recent_id in embedding_cache:
                                            # Check against ALL angles of the recently registered face
                                            for recent_emb in embedding_cache[recent_id]:
                                                e1 = embedding / (np.linalg.norm(embedding) + 1e-10)
                                                e2 = recent_emb / (np.linalg.norm(recent_emb) + 1e-10)
                                                sim = float(np.dot(e1, e2))
                                                if sim > 0.15:
                                                    is_duplicate = True
                                                    face_id = recent_id
                                                    break
                                        if is_duplicate: break
                                
                                if not is_duplicate:
                                    face_id = str(uuid.uuid4())[:8]
                                    recently_registered[face_id] = frame_number
                                    emb_bytes = recognizer.embedding_to_bytes(embedding)
                                    db_manager.insert_face(face_id, emb_bytes, current_time)
                                    
                                    # Also save initial pose to multi-embedding table
                                    db_manager.insert_embedding(face_id, emb_bytes, current_time)
                                    if face_id not in embedding_cache:
                                        embedding_cache[face_id] = []
                                    embedding_cache[face_id].append(embedding)
                                    
                                    logger.log_register(face_id)
                                    entry_img_path = logger.save_face_image(
                                        frame, bbox, face_id, "entry")
                                    logger.log_entry(face_id, entry_img_path)
                                    db_manager.insert_event(
                                        face_id, "entry", current_time, entry_img_path)
                                    counter.register_new_face(face_id)
                                    tracker.update(face_id, bbox, frame_number)
                                else:
                                    tracker.update(face_id, bbox, frame_number)
                            
                        except Exception as inner_e:
                            logger.log_info(f"Error handling individual detected bounding box: {inner_e}")
                    
                    t_end = time.time()
                    print(f"Frame {frame_number} processed in {t_end - t_start:.2f}s | Faces found: {len(bboxes)}")
                            
                # e. Execute every frame seamlessly to audit timeouts seamlessly
                exited_ids = tracker.get_exited_tracks(frame_number, exit_timeout_frames)
                for exit_id in exited_ids:
                    try:
                        # Extract the final known geometry footprint of the face before it left
                        last_bbox = tracker.active_tracks[exit_id]['bbox']
                        
                        # Generate parting visual evidence for log validation
                        exit_img_path = logger.save_face_image(frame, last_bbox, exit_id, "exit")
                        logger.log_exit(exit_id, exit_img_path)
                        
                        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        db_manager.insert_event(exit_id, "exit", current_time, exit_img_path)
                    except Exception as exit_e:
                        logger.log_info(f"Failed to gracefully close tracker footprint for {exit_id}: {exit_e}")
                        
            except Exception as loop_e:
                logger.log_info(f"Unhandled error disrupted pipeline at frame {frame_number}: {loop_e}")

        # 5. After the primary loop inevitably breaks, execute teardown logic
        cap.release()
        total_unique = counter.get_unique_count()
        logger.log_info(f"Stream ended. Unique visitors: {total_unique}")

    except Exception as e:
        # Ultimate fail-safe so nothing collapses entirely into silent exceptions
        print(f"CRITICAL: Program failed in main loop setup: {e}")

if __name__ == "__main__":
    main()
