"""
Event Logger Module for Katomaran Intelligent Face Tracker.
Configures Python standard logging to output to both console and persistent events.log.
"""

import os
import logging
from datetime import datetime
from typing import Optional


class CustomEventFormatter(logging.Formatter):
    """Custom log formatter adhering to standard Katomaran hackathon format:
    YYYY-MM-DD HH:MM:SS | LEVEL/TYPE | Message
    """
    def format(self, record: logging.LogRecord) -> str:
        timestamp = datetime.fromtimestamp(record.created).strftime("%Y-%m-%d %H:%M:%S")
        event_type = getattr(record, "event_type", record.levelname)
        return f"{timestamp} | {event_type} | {record.getMessage()}"


class EventLogger:
    def __init__(self, log_path: str = "logs/events.log", log_level: int = logging.INFO):
        self.log_path = log_path
        log_dir = os.path.dirname(self.log_path)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)

        self.logger = logging.getLogger("FaceTracker")
        self.logger.setLevel(log_level)
        self.logger.propagate = False

        # Clear existing handlers if re-initialized
        if self.logger.hasHandlers():
            self.logger.handlers.clear()

        formatter = CustomEventFormatter()

        # File Handler (persistent events.log)
        file_handler = logging.FileHandler(self.log_path, encoding="utf-8")
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

        # Console Handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

    def log_event(self, event_type: str, message: str, level: int = logging.INFO):
        """Log structured event with event type (e.g., ENTRY, EXIT, REGISTER, RECOGNIZED, TRACK, ERROR)."""
        self.logger.log(level, message, extra={"event_type": event_type.upper()})

    def info(self, message: str, event_type: str = "INFO"):
        self.log_event(event_type, message, logging.INFO)

    def warning(self, message: str, event_type: str = "WARNING"):
        self.log_event(event_type, message, logging.WARNING)

    def error(self, message: str, event_type: str = "ERROR"):
        self.log_event(event_type, message, logging.ERROR)

    def debug(self, message: str, event_type: str = "DEBUG"):
        self.log_event(event_type, message, logging.DEBUG)

    def log_detection(self, face_count: int, frame_idx: int):
        self.info(f"Detected {face_count} face(s) in frame {frame_idx}", event_type="DETECTION")

    def log_tracking(self, track_id: int, bbox: tuple):
        self.debug(f"Track ID {track_id} active at {bbox}", event_type="TRACKING")

    def log_embedding(self, track_id: int):
        self.debug(f"Generated 512-d facial embedding for Track ID {track_id}", event_type="EMBEDDING")

    def log_recognition(self, visitor_id: str, similarity: float):
        self.info(f"{visitor_id} (Similarity: {similarity:.2f})", event_type="RECOGNIZED")

    def log_registration(self, visitor_id: str):
        self.info(f"New visitor auto-registered as {visitor_id}", event_type="REGISTER")

    def log_entry(self, visitor_id: str, image_path: Optional[str] = None):
        img_info = f" -> Saved: {image_path}" if image_path else ""
        self.info(f"{visitor_id}{img_info}", event_type="ENTRY")

    def log_exit(self, visitor_id: str, image_path: Optional[str] = None):
        img_info = f" -> Saved: {image_path}" if image_path else ""
        self.info(f"{visitor_id}{img_info}", event_type="EXIT")

    def log_app_start(self, config_summary: str):
        self.info(f"Application started with configuration: {config_summary}", event_type="START")

    def log_app_stop(self, reason: str = "Completed successfully"):
        self.info(f"Application stopped: {reason}", event_type="STOP")
