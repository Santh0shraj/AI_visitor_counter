import logging

class FaceTracker:
    """
    A lightweight custom tracker that maintains the state of detected faces 
    across consecutive video frames without relying on external tracking libraries.
    """
    def __init__(self):
        """
        Initializes the FaceTracker with an empty dictionary.
        """
        # Dictionary keyed by face_id. 
        # Each value is a dictionary storing: face_id, bbox, last_seen_frame, and status.
        self.active_tracks = {}

    def update(self, face_id, bbox, current_frame):
        """
        Updates an existing track or creates a new one for the given face_id.
        
        Args:
            face_id (str): The unique identifier for the face.
            bbox (list, tuple): The current bounding box coordinates [x1, y1, x2, y2].
            current_frame (int): The current frame number in the video stream.
        """
        self.active_tracks[face_id] = {
            'face_id': face_id,
            'bbox': bbox,
            'last_seen_frame': current_frame,
            'status': 'active'
        }

    def get_exited_tracks(self, current_frame, exit_timeout_frames):
        """
        Identifies faces that haven't been seen for a specified number of frames
        and transitions their internal tracking status to 'exited'.
        
        Args:
            current_frame (int): The current frame index in the video loop.
            exit_timeout_frames (int): The maximum number of consecutive frames a face 
                                       can be absent before being considered exited.
                                       
        Returns:
            list: A list of face_ids that were just marked as 'exited'.
        """
        exited_face_ids = []
        
        for face_id, track_info in self.active_tracks.items():
            # Only evaluate tracks that are currently active
            if track_info['status'] == 'active':
                frames_absent = current_frame - track_info['last_seen_frame']
                
                # If they have been absent longer than the timeout threshold
                if frames_absent > exit_timeout_frames:
                    # Update internal status to 'exited' and record the ID
                    track_info['status'] = 'exited'
                    exited_face_ids.append(face_id)
                    
        return exited_face_ids

    def mark_exited(self, face_id):
        """
        Manually forces a specific track's status to 'exited'.
        
        Args:
            face_id (str): The identifier of the face to modify.
        """
        if face_id in self.active_tracks:
            self.active_tracks[face_id]['status'] = 'exited'
        else:
            logging.warning(f"Tracker: Attempted to mark non-existent face_id {face_id} as exited.")

    def is_active(self, face_id):
        """
        Checks whether a specific face is currently actively tracked.
        
        Args:
            face_id (str): The face identifier to check.
            
        Returns:
            bool: True if the face exists in active_tracks AND its status is 'active'.
        """
        if face_id in self.active_tracks:
            return self.active_tracks[face_id]['status'] == 'active'
        return False

    def get_active_count(self):
        """
        Calculates the total number of currently active distinct face tracks.
        
        Returns:
            int: The count of tracks with an 'active' status.
        """
        count = sum(1 for track in self.active_tracks.values() if track['status'] == 'active')
        return count
