# Intelligent Face Tracker with Auto-Registration and Visitor Counting

An AI-driven, edge-ready video analytics system that performs real-time face detection, persistent multi-person tracking, ArcFace facial embedding generation, automatic visitor registration, re-identification, virtual-line entry/exit counting, and structured event logging.

---

## 1. Problem Statement Summary

Standard video surveillance and entry management systems often suffer from duplicate counts when individuals linger, turn away, or reappear across multiple frames. 

This project solves this challenge by developing a robust video processing pipeline that:
- Detects faces in real-time using YOLO.
- Tracks individuals across frames and detection skips using ByteTrack.
- Extracts 512-dimensional facial embeddings using InsightFace (ArcFace).
- Automatically registers new faces upon first detection with unique identifiers (`VISITOR_001`, `VISITOR_002`, ...).
- Re-identifies returning visitors across frames and streams without duplicate registrations.
- Detects virtual line crossings to accurately log `ENTRY` and `EXIT` events with timestamped face crops.
- Persists all visitor profiles and events into SQLite and structured `events.log`.

---

## 2. Architecture Diagram & Workflow

```mermaid
flowchart TD
    A[Video File / Live RTSP Camera Stream] --> B[Frame Ingestion & Skipping Controller]
    B -->|Every N frames| C[YOLO Face Detection]
    B -->|Intermediary frames| D[ByteTrack Multi-Object Tracker]
    C --> D
    D --> E[Track Trajectory & Bounding Box Extractor]
    E --> F[Face Crop & Normalization]
    F --> G[InsightFace ArcFace 512-d Embedding Extractor]
    G --> H{Cosine Similarity >= Threshold?}
    H -->|Yes| I[Recognize Existing Visitor ID]
    H -->|No| J[Auto-Register New Visitor ID: VISITOR_XXX]
    I --> K[Update DB last_seen Timestamp]
    J --> L[Store Profile in SQLite DB & Memory Gallery]
    E --> M{Virtual Line Crossed?}
    M -->|Yes| N[Determine Direction: ENTRY / EXIT]
    N --> O[Save Cropped Face: logs/entries or logs/exits]
    N --> P[Record Event in SQLite events Table]
    N --> Q[Write Structured Log to logs/events.log]
    L --> R[Maintain Unique Visitor Count]
    K --> R
    R --> S[Render Real-Time HUD Overlay & Display]
```

---

## 3. Key Features

- **Real-Time YOLO Face Detection**: High-accuracy face localization with configurable detection skipping (`detection_skip_frames`) to optimize FPS.
- **ByteTrack Face Tracking**: Trajectory state tracking that persists track IDs during temporary occlusions and across frame skips.
- **InsightFace ArcFace Auto-Registration**: Automatic gallery registration on the fly; no manual enrollment or training required.
- **Duplicate-Free Unique Visitor Counting**: Distinguishes returning visitors from genuine new visitors using 512-dimensional cosine similarity.
- **Directional Virtual Line Crossing**: Configurable horizontal/vertical line crossing logic with state memory to ensure exactly one event per physical crossing.
- **Automated Evidence Capture**: Automatically crops and saves timestamped face images into structured folders (`logs/entries/YYYY-MM-DD/` and `logs/exits/YYYY-MM-DD/`).
- **Structured SQLite Database & Event Log**: Thread-safe SQLite persistence for visitor embeddings and events alongside a standardized `events.log` file.
- **Configurable Input Sources**: Seamlessly toggle between local MP4/AVI videos, webcams (`0`, `1`), and live RTSP streams via `config.json`.

---

## 4. Project Structure

```
katomaran_face_tracker/
│
├── main.py                     # Main execution and pipeline orchestrator
├── config.json                 # Central system configuration
├── requirements.txt            # Python dependencies
├── generate_sample_video.py    # Synthetic multi-visitor sample video generator
├── README.md                   # Complete system documentation
├── REQUIREMENT_AUDIT.md        # Comprehensive requirement compliance audit
│
├── detection/
│   ├── __init__.py
│   └── face_detector.py        # YOLO-based face detection module
│
├── tracking/
│   ├── __init__.py
│   └── tracker.py              # ByteTrack multi-face tracking module
│
├── recognition/
│   ├── __init__.py
│   └── face_recognizer.py      # InsightFace ArcFace embedding & auto-registration
│
├── database/
│   ├── __init__.py
│   └── database.py             # SQLite thread-safe storage & queries
│
├── logging_system/
│   ├── __init__.py
│   └── event_logger.py         # Formatted events.log and console logger
│
├── utils/
│   ├── __init__.py
│   └── image_utils.py          # Cropping, HUD rendering, and image saving
│
├── data/
│   └── visitors.db             # Auto-created SQLite database
│
├── logs/
│   ├── events.log              # Persistent event log
│   ├── entries/
│   │   └── YYYY-MM-DD/         # Cropped entry event face images
│   └── exits/
│       └── YYYY-MM-DD/         # Cropped exit event face images
│
└── models/                     # Model weights storage
```

---

## 5. Technologies Used

| Module | Technology / Library | Role |
| :--- | :--- | :--- |
| **Language** | Python 3.10+ (Tested on 3.13) | Core runtime |
| **Face Detection** | YOLOv8 (`ultralytics`) | High-speed face / person bounding box detection |
| **Face Recognition** | InsightFace ArcFace (`buffalo_sc`) | 512-d feature embedding extraction |
| **Tracking** | ByteTrack (`supervision`) | Stable multi-object tracking and Kalman filtering |
| **Video Processing** | OpenCV (`opencv-python`) | Frame decoding, HUD overlay, image saving |
| **Database** | SQLite3 (`database.py`) | Persistent visitor embeddings and event storage |
| **Configuration** | JSON (`config.json`) | Centralized runtime parameter management |
| **Logging** | Python `logging` | Structured, chronological `events.log` |

---

## 6. Installation & Setup Instructions

### Prerequisites
- Python 3.10 to 3.13
- Git

### Step 1: Clone or Open the Repository
```bash
cd Katomaran_hackathon
```

### Step 2: Create a Virtual Environment (Optional but Recommended)
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 7. Configuration Guide (`config.json`)

All operational settings are controlled via `config.json` without modifying code:

```json
{
    "source": "sample.mp4",
    "detection_skip_frames": 5,
    "similarity_threshold": 0.45,
    "exit_timeout_seconds": 30,
    "entry_exit_line_position": 0.5,
    "line_orientation": "horizontal",
    "entry_direction": "top_to_bottom",
    "yolo_model_path": "yolov8n.pt",
    "confidence_threshold": 0.45,
    "db_path": "data/visitors.db",
    "logs_dir": "logs",
    "events_log_path": "logs/events.log",
    "entries_dir": "logs/entries",
    "exits_dir": "logs/exits",
    "show_display": true,
    "save_output_video": false,
    "output_video_path": "output_processed.mp4"
}
```

### Configuration Parameters Explained:
- `source`: Path to input video file (e.g. `sample.mp4`), webcam index (e.g. `0`), or RTSP stream URL.
- `detection_skip_frames`: Number of frames to skip between full YOLO inference cycles (tracking runs continuously).
- `similarity_threshold`: Minimum cosine similarity (0.0 - 1.0) to classify an embedding as an existing visitor.
- `entry_exit_line_position`: Normalized position (0.0 to 1.0) of the virtual counting line across the frame.
- `line_orientation`: `"horizontal"` or `"vertical"`.
- `entry_direction`: `"top_to_bottom"` (or `"left_to_right"` for vertical). Opposite direction triggers `EXIT`.
- `show_display`: Set `true` to view the live OpenCV display window, or `false` for headless server execution.

---

## 8. How to Run

### Run with Local Video File
```bash
python main.py --source sample.mp4
```

### Run with Live RTSP Camera Stream
```bash
python main.py --source "rtsp://admin:password@192.168.1.100:554/stream1"
```

### Run with Webcam
```bash
python main.py --source 0
```

### Run in Headless Mode (No GUI)
```bash
python main.py --source sample.mp4 --no-display
```

### Generate a Synthetic Multi-Visitor Test Video
```bash
python generate_sample_video.py
python main.py --source sample.mp4
```

---

## 9. Database Schema

The system initializes a thread-safe SQLite database (`data/visitors.db`) with two indexed tables:

### Table: `visitors`
Stores unique visitor biometric identity profiles and timestamps.
- `id` (INTEGER PRIMARY KEY AUTOINCREMENT)
- `visitor_id` (TEXT UNIQUE NOT NULL, e.g. `VISITOR_001`)
- `embedding` (BLOB NOT NULL - 512 float32 byte array)
- `first_seen` (TEXT NOT NULL - ISO datetime string)
- `last_seen` (TEXT NOT NULL - ISO datetime string)

### Table: `events`
Stores directional entry and exit events with evidence images.
- `id` (INTEGER PRIMARY KEY AUTOINCREMENT)
- `visitor_id` (TEXT NOT NULL)
- `event_type` (TEXT NOT NULL - `ENTRY` or `EXIT`)
- `timestamp` (TEXT NOT NULL - ISO datetime string)
- `image_path` (TEXT - relative path to saved face crop)

---

## 10. Logging Structure & Format

All significant events are logged to `logs/events.log` in standard Katomaran format:

```
YYYY-MM-DD HH:MM:SS | EVENT_TYPE | Details
```

### Sample `events.log` Output:
```log
2026-10-03 14:33:42 | INIT | Loaded 0 registered visitor(s) into recognition gallery.
2026-10-03 14:33:42 | START | Application started with configuration: {"source": "sample.mp4", ...}
2026-10-03 14:33:50 | DETECTION | Detected 1 face(s) in frame 90
2026-10-03 14:33:51 | REGISTER | New visitor auto-registered as VISITOR_001
2026-10-03 14:33:56 | EXIT | VISITOR_001 -> Saved: logs/exits/2026-10-03/VISITOR_001_20261003_143355.jpg
2026-10-03 14:34:01 | RECOGNIZED | VISITOR_001 (Similarity: 0.78)
2026-10-03 14:34:05 | RECOGNIZED | VISITOR_001 (Similarity: 0.70)
2026-10-03 14:34:08 | STOP | Application stopped: Finished 300 frames. Unique visitors: 1
```

---

## 11. Entry / Exit Crossing Logic

1. **Virtual Line Setup**: A virtual line is established at `entry_exit_line_position * height` (e.g. 50% line).
2. **Trajectory State Memory**: Each active track records its historical center positions $(c_x, c_y)$.
3. **Crossing Transition**:
   - Crossing from $y < y_{\text{line}}$ to $y > y_{\text{line}}$ triggers an `ENTRY` event (if `entry_direction` is `"top_to_bottom"`).
   - Crossing from $y > y_{\text{line}}$ to $y < y_{\text{line}}$ triggers an `EXIT` event.
4. **De-duplication Protection**: Track metadata flags (`crossed_entry`, `crossed_exit`) prevent duplicate triggers for persons lingering near the line.

---

## 12. Unique Visitor Counting Logic

- **First Appearance**: When an unregistered individual appears, InsightFace extracts their 512-d ArcFace embedding. Since cosine similarity with all existing gallery profiles is below `similarity_threshold`, the system auto-registers `VISITOR_001` and increments the count to 1.
- **Subsequent Frames & Re-appearance**: When `VISITOR_001` moves across subsequent frames or leaves and returns, their cosine similarity matches ($0.78 \ge 0.45$). The system recognizes the visitor, updates `last_seen`, and **does NOT** increment the unique visitor count.

---

## 13. CPU / GPU Compute Considerations

- **CPU Optimization**:
  - Detection skipping (`detection_skip_frames: 5`) reduces YOLO inference overhead by up to 80%.
  - ByteTrack executes lightweight Kalman filtering on intermediate frames at >100 FPS on modern CPUs.
  - In-memory NumPy cosine similarity matching operates in sub-millisecond time.
- **GPU Acceleration (Optional)**:
  - If CUDA is available, `torch` and `onnxruntime-gpu` will automatically accelerate YOLO and ArcFace inference for high-density multi-camera setups.

---

## 14. Assumptions

1. **Camera Position**: The camera has a clear view of faces entering and exiting the monitored zone.
2. **Line Direction**: Movement from top-to-bottom across the virtual line corresponds to entry into the facility, while bottom-to-top corresponds to exit (fully configurable in `config.json`).
3. **Similarity Threshold**: Default cosine threshold of 0.45 provides optimal balance between high precision and false-positive prevention.

---

## 15. AI-Assisted Development & Planning

This application was architected following modern AI-assisted engineering best practices:
1. **Requirements Extraction**: Deconstructed Katomaran problem statement into 9 distinct phases.
2. **Modular Decoupling**: Separated detection, tracking, recognition, database, and logging into independent, testable modules.
3. **Iterative Verification**: Executed automated unit and integration tests after each phase to validate pipeline correctness.

---

## 16. Video Demonstration Link

- **Video Demo**: *(Add your YouTube or Loom explanatory video link here before submission)*

---

This project is a part of a hackathon run by https://katomaran.com
