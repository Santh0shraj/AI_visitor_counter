import sqlite3
import os

class DatabaseManager:
    """
    Manages the SQLite database operations for the face tracker application.
    """
    def __init__(self, db_path):
        """
        Initializes the DatabaseManager.
        
        Args:
            db_path (str): The file path where the SQLite database will be stored.
        """
        self.db_path = db_path
        
        # Ensure the directory exists
        db_dir = os.path.dirname(os.path.abspath(db_path))
        if db_dir and not os.path.exists(db_dir):
            os.makedirs(db_dir, exist_ok=True)
            
        self._initialize_db()

    def _get_connection(self):
        """
        Creates and returns a new SQLite connection with WAL journal mode.
        """
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA journal_mode=WAL;")
        return conn

    def _initialize_db(self):
        """
        Creates the required 'faces' and 'events' tables if they don't already exist.
        """
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            
            # Create faces table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS faces (
                    face_id TEXT PRIMARY KEY,
                    embedding BLOB,
                    first_seen TEXT
                )
            ''')

            # Create face_embeddings table for pose stability
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS face_embeddings (
                    embedding_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    face_id TEXT,
                    embedding BLOB,
                    timestamp TEXT
                )
            ''')
            
            # Create events table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS events (
                    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    face_id TEXT,
                    event_type TEXT,
                    timestamp TEXT,
                    image_path TEXT
                )
            ''')
            conn.commit()
        except sqlite3.Error as e:
            conn.rollback()
            raise e
        finally:
            conn.close()

    def insert_face(self, face_id, embedding_bytes, timestamp):
        """
        Saves a new face with its embedding and first seen timestamp.
        
        Args:
            face_id (str): A unique identifier for the face.
            embedding_bytes (bytes): The serialized structural face embedding.
            timestamp (str): The time the face was first seen.
        """
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO faces (face_id, embedding, first_seen) VALUES (?, ?, ?)",
                (face_id, embedding_bytes, timestamp)
            )
            conn.commit()
        except sqlite3.Error as e:
            conn.rollback()
            raise e
        finally:
            conn.close()

    def get_all_embeddings(self):
        """
        Retrieves all face embeddings from the database.
        
        Returns:
            list: A list of tuples containing (face_id, embedding_bytes).
        """
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT face_id, embedding FROM faces")
            return cursor.fetchall()
        finally:
            conn.close()

    def insert_event(self, face_id, event_type, timestamp, image_path):
        """
        Logs a specific event associated with a face (e.g., entry or exit).
        
        Args:
            face_id (str): The identifier of the face involved in the event.
            event_type (str): The type of event (e.g., 'entry', 'exit').
            timestamp (str): The time the event occurred.
            image_path (str): The file path to the captured image for the event.
        """
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO events (face_id, event_type, timestamp, image_path) VALUES (?, ?, ?, ?)",
                (face_id, event_type, timestamp, image_path)
            )
            conn.commit()
        except sqlite3.Error as e:
            conn.rollback()
            raise e
        finally:
            conn.close()

    def get_unique_visitor_count(self):
        """
        Gets the total number of unique visitors (faces) stored in the database.
        
        Returns:
            int: The count of unique face records.
        """
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM faces")
            result = cursor.fetchone()
            return result[0] if result else 0
        finally:
            conn.close()

    def face_exists(self, face_id):
        """
        Checks if a specific face ID already exists in the database.
        
        Args:
            face_id (str): The unique identifier to check.
            
        Returns:
            bool: True if the face exists, False otherwise.
        """
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT 1 FROM faces WHERE face_id = ?", (face_id,))
            result = cursor.fetchone()
            return result is not None
        finally:
            conn.close()

    def insert_embedding(self, face_id, embedding_bytes, timestamp):
        """
        Saves a specific pose embedding for an existing face ID.
        """
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO face_embeddings (face_id, embedding, timestamp) VALUES (?, ?, ?)",
                (face_id, embedding_bytes, timestamp)
            )
            conn.commit()
        except sqlite3.Error as e:
            conn.rollback()
            raise e
        finally:
            conn.close()

    def get_all_embeddings_multi(self):
        """
        Retrieves ALL stored embeddings (multi-pose) from the database.
        """
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT face_id, embedding FROM face_embeddings")
            return cursor.fetchall()
        finally:
            conn.close()

    def get_embedding_count_for_face(self, face_id):
        """
        Counts how many unique poses are stored for a specific face.
        """
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM face_embeddings WHERE face_id = ?", (face_id,))
            return cursor.fetchone()[0]
        finally:
            conn.close()
