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

        # FIX 1 - Video writer setup
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        if fps == 0:
            fps = 30

        output_path = config.get("output_video", "output.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(
            output_path,
            fourcc,
            fps,
            (frame_width, frame_height)
        )
        logger.log_info(f"Output video will be saved to: {output_path}")
        show_display = config.get("show_display", True)

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
        
        # FIX 2 - Main loop with drawing and video writing
        while True:
            # 1. Read frame
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            frame_number += 1
            
            # 2. Detection block - runs every N frames only
            if frame_number % frame_skip_interval == 0:
                t_start = time.time()
                
                detection_frame = cv2.resize(frame, (640, 360)) \
                    if frame.shape[1] > 640 else frame
                bboxes = detector.detect(detection_frame)
                
                scale_x = frame.shape[1] / detection_frame.shape[1]
                scale_y = frame.shape[0] / detection_frame.shape[0]
                bboxes = [[int(b[0]*scale_x), int(b[1]*scale_y),
                           int(b[2]*scale_x), int(b[3]*scale_y)]
                          for b in bboxes]
                
                for bbox in bboxes:
                    try:
                        embedding = recognizer.get_embedding(frame, bbox)
                        if embedding is None:
                            continue
                        
                        best_match_id = None
                        best_similarity = 0.0
                        
                        for stored_face_id, stored_embs in embedding_cache.items():
                            for stored_emb in stored_embs:
                                e1 = embedding / (np.linalg.norm(embedding) + 1e-10)
                                e2 = stored_emb / (np.linalg.norm(stored_emb) + 1e-10)
                                sim = float(np.dot(e1, e2))
                                sim = max(0.0, min(1.0, sim))
                                if sim > best_similarity:
                                    best_similarity = sim
                                    best_match_id = stored_face_id
                        
                        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        
                        if best_similarity > similarity_threshold and best_match_id is not None:
                            face_id = best_match_id
                            
                            # Learn new angle if pose changed
                            if best_similarity < 0.75:
                                emb_bytes = recognizer.embedding_to_bytes(embedding)
                                db_manager.insert_embedding(
                                    face_id, emb_bytes, current_time)
                                if face_id not in embedding_cache:
                                    embedding_cache[face_id] = []
                                embedding_cache[face_id].append(embedding)
                            
                            # Log re-entry if face was not active
                            if not tracker.is_active(face_id):
                                entry_img_path = logger.save_face_image(
                                    frame, bbox, face_id, "entry")
                                logger.log_entry(face_id, entry_img_path)
                                db_manager.insert_event(
                                    face_id, "entry", current_time, entry_img_path)
                        
                        else:
                            # Check cooldown for duplicates
                            is_duplicate = False
                            for recent_id, recent_frame in recently_registered.items():
                                if frame_number - recent_frame < 90:
                                    if recent_id in embedding_cache:
                                        e1 = embedding / (np.linalg.norm(embedding) + 1e-10)
                                        e2 = embedding_cache[recent_id][0] / (np.linalg.norm(embedding_cache[recent_id][0]) + 1e-10)
                                        sim = float(np.dot(e1, e2))
                                        if sim > 0.15:
                                            is_duplicate = True
                                            face_id = recent_id
                                            break
                            
                            if not is_duplicate:
                                face_id = str(uuid.uuid4())[:8]
                                recently_registered[face_id] = frame_number
                                emb_bytes = recognizer.embedding_to_bytes(embedding)
                                db_manager.insert_face(
                                    face_id, emb_bytes, current_time)
                                db_manager.insert_embedding(
                                    face_id, emb_bytes, current_time)
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
                        
                        tracker.update(face_id, bbox, frame_number)
                        
                    except Exception as inner_e:
                        logger.log_info(
                            f"Error processing bbox: {inner_e}")
                
                t_end = time.time()
                print(f"Frame {frame_number} | "
                      f"{t_end - t_start:.2f}s | "
                      f"Faces: {len(bboxes)} | "
                      f"Unique: {counter.get_unique_count()}")
            
            # 3. Exit detection - runs every frame
            exited_ids = tracker.get_exited_tracks(
                frame_number, exit_timeout_frames)
            for exit_id in exited_ids:
                try:
                    last_bbox = tracker.active_tracks[exit_id]['bbox']
                    exit_img_path = logger.save_face_image(
                        frame, last_bbox, exit_id, "exit")
                    logger.log_exit(exit_id, exit_img_path)
                    current_time = datetime.now().strftime(
                        "%Y-%m-%d %H:%M:%S")
                    db_manager.insert_event(
                        exit_id, "exit", current_time, exit_img_path)
                except Exception as exit_e:
                    logger.log_info(
                        f"Exit error for {exit_id}: {exit_e}")
            
            # 4. Draw boxes and write output - runs every frame
            display_frame = frame.copy()
            
            for tracked_id, track_info in tracker.active_tracks.items():
                if track_info['status'] == 'active':
                    bx1, by1, bx2, by2 = track_info['bbox']
                    
                    frames_since_seen = (frame_number - 
                        track_info['last_seen_frame'])
                    
                    # Green = actively detected
                    # Yellow = tracking only between detections
                    if frames_since_seen > frame_skip_interval * 2:
                        box_color = (0, 255, 255)
                        label_color = (0, 255, 255)
                    else:
                        box_color = (0, 255, 0)
                        label_color = (0, 255, 0)
                    
                    cv2.rectangle(
                        display_frame,
                        (bx1, by1),
                        (bx2, by2),
                        box_color,
                        2
                    )
                    
                    label = f"ID: {tracked_id}"
                    label_y = max(by1 - 10, 20)
                    cv2.putText(
                        display_frame,
                        label,
                        (bx1, label_y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        label_color,
                        2
                    )
            
            # Unique visitor count - yellow top left
            cv2.putText(
                display_frame,
                f"Unique Visitors: {counter.get_unique_count()}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 255),
                2
            )
            
            # Active face count - green below
            cv2.putText(
                display_frame,
                f"Active Faces: {tracker.get_active_count()}",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )
            
            # Write every frame to output video
            out.write(display_frame)
            
            # Show live window
            if show_display:
                cv2.imshow('Face Tracker', display_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        # 5. Teardown
        cap.release()
        # FIX 3 - Release video writer and close windows
        out.release()
        logger.log_info(f"Output video saved to: {output_path}")
        cv2.destroyAllWindows()
        total_unique = counter.get_unique_count()
        logger.log_info(f"Stream ended. Unique visitors: {total_unique}")

    except Exception as e:
        print(f"CRITICAL: Program failed in main loop setup: {e}")

if __name__ == "__main__":
    main()
