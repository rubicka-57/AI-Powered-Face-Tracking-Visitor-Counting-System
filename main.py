"""
Katomaran Intelligent Face Tracker with Auto-Registration and Visitor Counting.
Main application pipeline integrating YOLO detection, ByteTrack tracking,
InsightFace recognition, virtual-line entry/exit counting, SQLite DB, and logging.
"""

import os
import sys
import time
import json
import argparse
from datetime import datetime
import cv2
import numpy as np

from database.database import Database
from logging_system.event_logger import EventLogger
from detection.face_detector import FaceDetector
from tracking.tracker import FaceTracker
from recognition.face_recognizer import FaceRecognizer
from utils.image_utils import crop_face, save_event_image, draw_hud


class KatomaranFaceTrackerApp:
    def __init__(self, config_path: str = "config.json"):
        self.config_path = config_path
        self.config = self._load_config()

        # Initialize logging system
        self.logs_dir = self.config.get("logs_dir", "logs")
        self.events_log_path = self.config.get("events_log_path", "logs/events.log")
        self.logger = EventLogger(log_path=self.events_log_path)

        # Initialize database
        self.db_path = self.config.get("db_path", "data/visitors.db")
        self.db = Database(db_path=self.db_path)

        # Initialize AI Modules
        self.detector = FaceDetector(
            model_path=self.config.get("yolo_model_path", "yolov8n.pt"),
            conf_threshold=self.config.get("confidence_threshold", 0.45),
            detection_skip_frames=self.config.get("detection_skip_frames", 5)
        )

        self.tracker = FaceTracker(
            track_activation_threshold=0.35,
            lost_track_buffer=30,
            minimum_matching_threshold=0.8,
            frame_rate=30
        )

        self.recognizer = FaceRecognizer(
            database=self.db,
            logger=self.logger,
            similarity_threshold=self.config.get("similarity_threshold", 0.45),
            model_name="buffalo_sc"
        )

        # Line configuration
        self.line_position = float(self.config.get("entry_exit_line_position", 0.5))
        self.line_orientation = self.config.get("line_orientation", "horizontal").lower()
        self.entry_direction = self.config.get("entry_direction", "top_to_bottom").lower()
        self.exit_timeout_seconds = float(self.config.get("exit_timeout_seconds", 30))

        # Runtime states
        self.source = self.config.get("source", "sample.mp4")
        self.show_display = self.config.get("show_display", True)
        self.save_output = self.config.get("save_output_video", False)
        self.output_path = self.config.get("output_video_path", "output_processed.mp4")

        self.total_frames_processed = 0
        self.start_time = None

    def _load_config(self) -> dict:
        """Load and validate configuration JSON file."""
        default_config = {
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
            "show_display": True,
            "save_output_video": False,
            "output_video_path": "output_processed.mp4"
        }

        if not os.path.exists(self.config_path):
            print(f"Config file not found at {self.config_path}, creating default config.")
            with open(self.config_path, "w", encoding="utf-8") as f:
                json.dump(default_config, f, indent=4)
            return default_config

        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                loaded_config = json.load(f)
            # Merge with defaults to ensure all keys present
            default_config.update(loaded_config)
            return default_config
        except Exception as e:
            print(f"Error parsing {self.config_path}: {e}. Using defaults.")
            return default_config

    def _determine_side(self, center: tuple, frame_shape: tuple, hysteresis: int = 10) -> int:
        """
        Determine which side of the virtual line a point lies on with hysteresis.
        Returns: -1 for Side A (Top/Left), +1 for Side B (Bottom/Right), 0 if in deadzone.
        """
        h, w = frame_shape[:2]
        cx, cy = center

        if self.line_orientation == "horizontal":
            line_y = h * self.line_position
            if cy < line_y - hysteresis:
                return -1
            elif cy > line_y + hysteresis:
                return 1
            return 0  # In crossing zone
        else:
            line_x = w * self.line_position
            if cx < line_x - hysteresis:
                return -1
            elif cx > line_x + hysteresis:
                return 1
            return 0

    def _check_line_crossing(self, track_id: int, prev_pos: tuple, curr_pos: tuple,
                             frame_shape: tuple, frame: np.ndarray, bbox: list,
                             timestamp_str: str):
        """
        Evaluate trajectory crossing the virtual line with hysteresis and state persistence.
        Guarantees exactly one event per actual crossing and prevents duplicate events.
        """
        meta = self.tracker.get_track_metadata(track_id)
        if meta is None:
            return

        side_prev = self._determine_side(prev_pos, frame_shape)
        side_curr = self._determine_side(curr_pos, frame_shape)

        # Skip if either position is in deadzone or unchanged
        if side_curr == 0:
            return

        # Initialize initial side
        if meta.get("last_stable_side") is None:
            meta["last_stable_side"] = side_curr
            return

        last_side = meta["last_stable_side"]

        # Check if crossed from one stable side to the other
        if side_curr != last_side:
            # Determine if this crossing is ENTRY or EXIT
            if self.line_orientation == "horizontal":
                if self.entry_direction == "top_to_bottom":
                    is_entry = (last_side == -1 and side_curr == 1)
                else:
                    is_entry = (last_side == 1 and side_curr == -1)
            else:
                if self.entry_direction == "left_to_right":
                    is_entry = (last_side == -1 and side_curr == 1)
                else:
                    is_entry = (last_side == 1 and side_curr == -1)

            event_type = "ENTRY" if is_entry else "EXIT"

            # Check if this event type was already generated for this track on this crossing
            already_triggered = meta["crossed_entry"] if is_entry else meta["crossed_exit"]
            if not already_triggered:
                # Ensure visitor ID exists
                visitor_id = meta.get("visitor_id")
                if not visitor_id:
                    face_crop = crop_face(frame, bbox)
                    if face_crop is not None:
                        vid, _, _ = self.recognizer.recognize_or_register(face_crop, timestamp=timestamp_str)
                        if vid:
                            visitor_id = vid
                            self.tracker.set_visitor_id(track_id, vid)

                if visitor_id:
                    # Mark state
                    if is_entry:
                        meta["crossed_entry"] = True
                        meta["crossed_exit"] = False  # Reset opposite side so return journey can be tracked
                    else:
                        meta["crossed_exit"] = True
                        meta["crossed_entry"] = False

                    # Crop and save event face image
                    event_crop = crop_face(frame, bbox)
                    image_path = None
                    if event_crop is not None:
                        image_path = save_event_image(
                            face_img=event_crop,
                            event_type=event_type,
                            visitor_id=visitor_id,
                            timestamp_str=timestamp_str,
                            logs_dir=self.logs_dir
                        )

                    # Record in DB
                    self.db.add_event(
                        visitor_id=visitor_id,
                        event_type=event_type,
                        timestamp=timestamp_str,
                        image_path=image_path
                    )

                    # Log to events.log
                    if is_entry:
                        self.logger.log_entry(visitor_id, image_path)
                    else:
                        self.logger.log_exit(visitor_id, image_path)

            # Update last stable side
            meta["last_stable_side"] = side_curr

    def run(self):
        """Main execution loop for processing video/RTSP stream."""
        self.logger.log_app_start(json.dumps(self.config))
        print("\n=======================================================")
        print("  KATOMARAN INTELLIGENT FACE TRACKER & VISITOR COUNTER")
        print("=======================================================\n")

        # Handle numeric source (webcam index) or file/RTSP
        source_input = self.source
        if isinstance(source_input, str) and source_input.isdigit():
            source_input = int(source_input)

        cap = cv2.VideoCapture(source_input)
        if not cap.isOpened():
            self.logger.error(f"Cannot open video source: {self.source}", event_type="SOURCE_ERROR")
            print(f"Error: Unable to open video source '{self.source}'")
            return

        fps_input = cap.get(cv2.CAP_PROP_FPS)
        if fps_input <= 0 or np.isnan(fps_input):
            fps_input = 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"Source: {self.source} | Resolution: {width}x{height} | FPS: {fps_input:.1f} | Total Frames: {total_frames}")

        # Optional Video Writer
        video_writer = None
        if self.save_output:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            video_writer = cv2.VideoWriter(self.output_path, fourcc, fps_input, (width, height))

        self.start_time = time.time()
        fps_calc = 0.0
        frame_idx = 0

        try:
            while True:
                t0 = time.time()
                ret, frame = cap.read()
                if not ret or frame is None:
                    print("\nEnd of video stream reached.")
                    break

                frame_idx += 1
                self.total_frames_processed += 1
                now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                # Step 1: Face Detection (with frame skipping support)
                detections, is_detection_frame = self.detector.detect(frame)
                if is_detection_frame and len(detections) > 0:
                    self.logger.log_detection(len(detections), frame_idx)

                # Step 2: ByteTrack Multi-Face Tracking
                active_tracks = self.tracker.update(detections)

                # Step 3: Face Recognition & Crossing Analysis
                for trk in active_tracks:
                    track_id = trk["track_id"]
                    bbox = trk["bbox"]
                    meta = trk["metadata"]

                    # Attempt face recognition/auto-registration if not yet identified
                    if meta.get("visitor_id") is None and meta.get("recognition_attempts", 0) < 5:
                        meta["recognition_attempts"] = meta.get("recognition_attempts", 0) + 1
                        face_crop = crop_face(frame, bbox)
                        if face_crop is not None:
                            visitor_id, sim, is_new = self.recognizer.recognize_or_register(
                                face_crop=face_crop,
                                timestamp=now_str
                            )
                            if visitor_id:
                                meta["visitor_id"] = visitor_id
                                trk["visitor_id"] = visitor_id
                                self.tracker.set_visitor_id(track_id, visitor_id)

                    # Check virtual-line crossing
                    history = meta.get("history", [])
                    if len(history) >= 2:
                        prev_pos = history[-2]
                        curr_pos = history[-1]
                        self._check_line_crossing(
                            track_id=track_id,
                            prev_pos=prev_pos,
                            curr_pos=curr_pos,
                            frame_shape=frame.shape,
                            frame=frame,
                            bbox=bbox,
                            timestamp_str=now_str
                        )

                # Unique visitor count from database
                unique_visitors = self.db.get_unique_visitor_count()

                # Calculate smoothed FPS
                dt = time.time() - t0
                if dt > 0:
                    curr_fps = 1.0 / dt
                    fps_calc = 0.9 * fps_calc + 0.1 * curr_fps if fps_calc > 0 else curr_fps

                # Print periodic console status
                if frame_idx % 15 == 0 or is_detection_frame:
                    sys.stdout.write(f"\rFrame: {frame_idx:05d} | Tracked: {len(active_tracks):02d} | Unique Visitors: {unique_visitors:03d} | FPS: {fps_calc:.1f}")
                    sys.stdout.flush()

                # Step 4: Render Annotated Frame
                annotated_frame = draw_hud(
                    frame=frame,
                    tracks=active_tracks,
                    line_position=self.line_position,
                    line_orientation=self.line_orientation,
                    unique_visitor_count=unique_visitors,
                    fps=fps_calc,
                    is_detection_frame=is_detection_frame
                )

                if video_writer is not None:
                    video_writer.write(annotated_frame)

                if self.show_display:
                    cv2.imshow("Katomaran Intelligent Face Tracker", annotated_frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q') or key == 27:  # 'q' or ESC
                        print("\nUser interrupted execution via 'q' key.")
                        break

        except KeyboardInterrupt:
            print("\nReceived keyboard interrupt. Shutting down gracefully...")
        except Exception as e:
            self.logger.error(f"Unexpected runtime error: {e}", event_type="ERROR")
            print(f"\nError occurred: {e}")
        finally:
            cap.release()
            if video_writer is not None:
                video_writer.release()
            if self.show_display:
                cv2.destroyAllWindows()

            # Final summary
            elapsed = time.time() - self.start_time if self.start_time else 0.1
            avg_fps = self.total_frames_processed / elapsed if elapsed > 0 else 0
            final_unique_count = self.db.get_unique_visitor_count()
            event_counts = self.db.get_event_counts()

            print("\n\n=======================================================")
            print("                PROCESSING SUMMARY                     ")
            print("=======================================================")
            print(f"Total Frames Processed : {self.total_frames_processed}")
            print(f"Total Elapsed Time     : {elapsed:.2f} seconds")
            print(f"Average Processing FPS : {avg_fps:.2f} FPS")
            print(f"Total Unique Visitors  : {final_unique_count}")
            print(f"Recorded Events Summary: {event_counts}")
            print("=======================================================\n")

            self.logger.log_app_stop(f"Finished {self.total_frames_processed} frames. Unique visitors: {final_unique_count}")


def main():
    parser = argparse.ArgumentParser(description="Katomaran Intelligent Face Tracker & Auto-Registration System")
    parser.add_argument("--config", type=str, default="config.json", help="Path to config.json")
    parser.add_argument("--source", type=str, default=None, help="Override video source (file, RTSP URL, or webcam index)")
    parser.add_argument("--skip-frames", type=int, default=None, help="Override detection skip frames")
    parser.add_argument("--no-display", action="store_true", help="Disable GUI display window (headless mode)")
    args = parser.parse_args()

    app = KatomaranFaceTrackerApp(config_path=args.config)
    if args.source is not None:
        app.source = args.source
    if args.skip_frames is not None:
        app.detector.detection_skip_frames = args.skip_frames
    if args.no_display:
        app.show_display = False

    app.run()


if __name__ == "__main__":
    main()
