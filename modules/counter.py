class VisitorCounter:
    """
    Tracks and counts unique visitors specifically for the current running session,
    while querying the persistent database for overall absolute counts.
    """
    def __init__(self, db_manager):
        """
        Initializes the VisitorCounter with a DatabaseManager instance.
        
        Args:
            db_manager: An initialized DatabaseManager object to query historical data.
        """
        self.db = db_manager
        
        # An in-memory set that tracks which face_ids have been registered 
        # specifically during this active running session.
        self.seen_ids = set()

    def register_new_face(self, face_id):
        """
        Registers a face_id in the session's in-memory set if it's new.
        
        Args:
            face_id (str): The unique identifier for the detected face.
            
        Returns:
            bool: True if the face was genuinely newly added to this session, 
                  False if it had already been encountered and recorded.
        """
        # We perform a direct O(1) membership test on the set
        if face_id not in self.seen_ids:
            # First time seeing this face_id in this active session
            self.seen_ids.add(face_id)
            return True
            
        return False

    def get_unique_count(self):
        """
        Retrieves the total number of unique distinct visitors fully stored 
        in the global persistent database hierarchy.
        
        Returns:
            int: The total count of unique stored face records globally.
        """
        # Delegates the counting operation to the database manager layer
        return self.db.get_unique_visitor_count()

    def is_known(self, face_id):
        """
        Determines if a specific face_id has already been observed and 
        logged during the current session runtime.
        
        Args:
            face_id (str): The identifier to query.
            
        Returns:
            bool: True if it exists in the active seen_ids set.
        """
        # Another O(1) set membership test
        return face_id in self.seen_ids
