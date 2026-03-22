# Face Tracker — AI Visitor Counter

## 1. Project Overview
This system is an AI-powered real time face tracking and unique visitor counting application. It processes both pre-recorded video files and live RTSP camera streams to detect, recognize, and track unique individuals using YOLOv8 for face detection and InsightFace ArcFace for recognition. Every entry and exit is logged with a cropped face image, timestamp, and face ID stored in both a structured folder system and SQLite database. A live Flask dashboard provides real time visualization of all visitor events.

## 2. Architecture

### Module Descriptions
- `detector.py`: Uses YOLOv8n-face model specifically trained for face detection to locate faces in each frame and return tight bounding boxes.
- `recognizer.py`: Uses InsightFace buffalo_l ArcFace model to generate 512-dimensional embeddings from face crops. Stores multiple embeddings per person for pose stability across different angles.
- `tracker.py`: Lightweight custom state engine that maintains active face tracks across frames. Detects exits when a face is absent for exit_timeout_frames.
- `database.py`: Manages SQLite database with WAL mode for crash-safe writes. Stores face embeddings, multiple pose embeddings, and full event timeline.
- `logger.py`: Writes formatted events to events.log and saves cropped face images to dated folders.
- `counter.py`: Tracks unique visitors using in-memory set with database fallback for accurate counting.
- `main.py`: Central controller that integrates all modules, manages video capture, coordinates detection, recognition, tracking, and logging.
- `app.py`: Live Flask dashboard showing each visitor as one row with entry photo, entry time, exit time, and status updating every 3 seconds in real time.

### Architecture Flow
```text
Video Source (MP4 file or RTSP camera)
        ↓
Frame Reader — OpenCV VideoCapture
        ↓ every N frames (configurable)
YOLOv8n-face Detector — detector.py
        ↓ tight face bounding boxes
InsightFace Recognizer — recognizer.py
        ↓ 512-d ArcFace embeddings
Similarity Matcher — main.py
        ↓
    ┌───┴────────┐
New face      Known face
    ↓               ↓
Register        Update tracker
Log REGISTER    Log re-entry if exited
Log ENTRY       
    ↓               ↓
    └─────┬──────┘
    FaceTracker — tracker.py
        ↓ when absent > exit_timeout_frames
    Log EXIT — logger.py
        ↓
    DatabaseManager — database.py
    VisitorCounter — counter.py
        ↓
    Flask Dashboard — app.py
    Output Video — output.mp4
```

## 3. Setup Instructions

### Prerequisites
- Python 3.10 or higher
- Windows / Linux / Mac

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/Santh0shraj/AI_visitor_counter.git
   cd AI_visitor_counter
   ```

2. Create and activate virtual environment:
   ```bash
   python -m venv venv
   
   # Windows:
   venv\Scripts\activate.bat
   
   # Linux/Mac:
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Place your video file in the project root:
   - Rename it to sample.mp4
   - OR update video_source in config.json

5. Run the face tracker:
   ```bash
   python main.py
   ```

6. Run the live dashboard (optional, separate terminal):
   ```bash
   python app.py
   # Open http://localhost:5000 in browser
   ```

### Switching to RTSP camera
In `config.json` change:
```json
   "use_rtsp": true,
   "rtsp_url": "rtsp://username:password@ip:554/stream"
```
Then run `python main.py`

## 4. config.json Explained

| Key | Type | Description |
|-----|------|-------------|
| video_source | String | Path to input video file |
| use_rtsp | Boolean | true = RTSP camera, false = video file |
| rtsp_url | String | Full RTSP camera stream URL |
| rtsp_reconnect_attempts | Integer | Retry count if stream drops |
| frame_skip_interval | Integer | Frames to skip between detections |
| similarity_threshold | Float | Minimum cosine similarity to match face |
| exit_timeout_frames | Integer | Absent frames before marking as exited |
| log_dir | String | Root folder for logs and images |
| db_path | String | SQLite database file path |
| detection_confidence | Float | Minimum YOLO confidence to accept detection |
| imgsz | Integer | YOLO inference image resolution |
| output_video | String | Output annotated video file path |
| show_display | Boolean | Show live OpenCV window during processing |

## 5. Sample config.json
```json
{
  "video_source": "sample.mp4",
  "use_rtsp": false,
  "rtsp_url": "rtsp://username:password@camera_ip:554/stream",
  "rtsp_reconnect_attempts": 3,
  "frame_skip_interval": 10,
  "similarity_threshold": 0.35,
  "exit_timeout_frames": 45,
  "log_dir": "logs",
  "db_path": "db/faces.db",
  "detection_confidence": 0.5,
  "imgsz": 640,
  "output_video": "output.mp4",
  "show_display": true
}
```

## 6. Assumptions Made
- YOLOv8n-face model used for face-specific detection
- InsightFace buffalo_l ArcFace model for recognition
- Cosine similarity threshold of 0.35 for face matching
- Multiple embeddings stored per face for pose stability
- System learns new angles automatically during tracking
- SQLite WAL mode ensures crash-safe writes
- Face considered exited after 45 consecutive absent frames
- Frame skip interval of 10 recommended for CPU deployment
- In-memory embedding cache used for fast comparison
- Session tracking ensures exactly one entry and exit per visit
- Output video saved as output.mp4 in file mode only
- RTSP mode disables output video to avoid infinite file growth
- Green bounding box means face actively detected this cycle
- Yellow bounding box means face being tracked between detections
- Same person returning gets same ID and does not increment counter

## 7. Compute Load Estimates

| Module | CPU Usage | GPU Usage | Notes |
|--------|-----------|-----------|-------|
| FaceDetector (YOLOv8n-face) | 30-40% | 20-30% with CUDA | Runs every N frames only |
| FaceRecognizer (InsightFace buffalo_l) | 40-50% | 15-20% with CUDA | Per detected face crop |
| FaceTracker (custom) | 2-3% | None | Dictionary operations only |
| DatabaseManager (SQLite WAL) | 2-3% | None | WAL mode minimizes IO |
| Logger (OpenCV + filesystem) | 1-2% | None | JPEG crop and save |
| Flask Dashboard | 1-2% | None | Lightweight HTTP server |
| Total (CPU only, frame_skip=10) | 70-90% | None | Recommended for CPU |
| Total (GPU CUDA, frame_skip=3) | 20-30% | 40-50% | True real time possible |

## 8. AI Planning Document

### Planning Approach
The system was designed using a modular top-down approach:
1. Identified core pipeline stages: detect, recognize, track, log
2. Selected best models for each stage based on accuracy and CPU performance
3. Planned data flow between all modules before writing any code
4. Designed crash-safe storage using SQLite WAL transactions
5. Estimated compute requirements for CPU deployment
6. Added frame skipping and caching for performance optimization

### Features Implemented
1. Real time face detection using YOLOv8n-face model
2. Face recognition using InsightFace buffalo_l ArcFace 512-d embeddings
3. Multi-embedding pose stability — learns multiple angles per person
4. Unique visitor counting — same person never counted twice
5. Exactly one entry and one exit logged per visit using session tracking
6. Cropped face images saved to dated folder structure
7. SQLite database with WAL mode for crash-safe persistent storage
8. Mandatory events.log tracking all system events
9. Live Flask dashboard with real time row-by-row visitor updates
10. Output video generation with green and yellow bounding boxes
11. RTSP live camera support with buffer flush and auto-reconnect
12. Configurable frame skip interval for CPU performance tuning
13. In-memory embedding cache for fast similarity matching
14. Cooldown mechanism to prevent duplicate registrations
15. Overlap detection to prevent same face processed twice per frame
16. Re-entry detection — same ID assigned when person returns

### Why These Technologies
- YOLOv8n-face: Fastest face-specific YOLO model, optimized for CPU
- InsightFace buffalo_l: State of the art ArcFace accuracy on CPU
- SQLite WAL: Zero-dependency crash-safe database for local deployment
- Flask: Lightweight web framework for live dashboard with no overhead
- OpenCV: Industry standard for video capture and frame processing

## 9. Loom/YouTube Video
[TO BE ADDED AFTER RECORDING]

## 10. Sample Output

### events.log sample
```text
2026-03-21 14:23:01 | INFO | REGISTER | face_id=2a7b189f
2026-03-21 14:23:01 | INFO | ENTRY | face_id=2a7b189f | image=logs/entries/2026-03-21/2a7b189f_14-23-01.jpg
2026-03-21 14:23:45 | INFO | REGISTER | face_id=4cde901a
2026-03-21 14:23:45 | INFO | ENTRY | face_id=4cde901a | image=logs/entries/2026-03-21/4cde901a_14-23-45.jpg
2026-03-21 14:25:44 | INFO | EXIT | face_id=2a7b189f | image=logs/exits/2026-03-21/2a7b189f_14-25-44.jpg
2026-03-21 14:26:12 | INFO | EXIT | face_id=4cde901a | image=logs/exits/2026-03-21/4cde901a_14-26-12.jpg
2026-03-21 14:26:30 | INFO | Stream ended. Unique visitors: 21
```

### Database sample output
```text
Unique faces: 21
Total events: 51
Sample events:
('2a7b189f', 'REGISTER', '2026-03-21 14:23:01')
('2a7b189f', 'ENTRY', '2026-03-21 14:23:01')
('4cde901a', 'REGISTER', '2026-03-21 14:23:45')
('4cde901a', 'ENTRY', '2026-03-21 14:23:45')
('2a7b189f', 'EXIT', '2026-03-21 14:25:44')
```

---
*This project is a part of a hackathon run by [katomaran.com](https://katomaran.com)*
