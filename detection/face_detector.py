"""
Face Detection Module for Katomaran Face Tracker.
Implements YOLO-based face detection with configurable frame skipping,
confidence filtering, and robust fallback mechanisms.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict
from ultralytics import YOLO


class FaceDetector:
    """
    YOLO-based Face Detector.
    Supports configurable frame-skipping, confidence thresholding,
    and returns standard format bounding boxes [x1, y1, x2, y2, score].
    """
    def __init__(self,
                 model_path: str = "yolov8n.pt",
                 conf_threshold: float = 0.45,
                 detection_skip_frames: int = 5):
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.detection_skip_frames = detection_skip_frames
        self.frame_counter = 0
        self.last_detections = []  # List of [x1, y1, x2, y2, conf]

        self.model = None
        self._load_model()

    def _load_model(self):
        """Load YOLO model."""
        try:
            self.model = YOLO(self.model_path)
        except Exception as e:
            # Fallback to standard yolov8n if specific path fails
            print(f"Warning: Failed to load {self.model_path} ({e}), falling back to yolov8n.pt")
            self.model = YOLO("yolov8n.pt")

    def should_detect(self) -> bool:
        """Determines if the current frame should undergo full detection."""
        if self.detection_skip_frames <= 0:
            return True
        return (self.frame_counter % (self.detection_skip_frames + 1)) == 0

    def detect(self, frame: np.ndarray) -> Tuple[List[List[float]], bool]:
        """
        Detect faces in the given frame.

        Returns:
            Tuple: (detections: List of [x1, y1, x2, y2, score], is_new_detection: bool)
        """
        if frame is None or frame.size == 0:
            return [], False

        self.frame_counter += 1
        is_detection_frame = self.should_detect()

        if not is_detection_frame:
            return self.last_detections, False

        # Run inference
        h, w = frame.shape[:2]
        results = self.model.predict(
            source=frame,
            conf=self.conf_threshold,
            verbose=False,
            device="cpu"
        )

        detections = []
        if len(results) > 0:
            boxes = results[0].boxes
            for box in boxes:
                xyxy = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy())
                cls_id = int(box.cls[0].cpu().numpy())

                # If standard COCO model is used, cls 0 is person; we crop upper region for face
                # If face-specific YOLO is used, all detections are faces
                x1, y1, x2, y2 = float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])

                # Check if it's a person detection on standard YOLO
                if cls_id == 0:
                    # Estimate upper 35% as face / head region if person bbox
                    bw = x2 - x1
                    bh = y2 - y1
                    # If aspect ratio is full person (height > width * 1.3)
                    if bh > bw * 1.2:
                        face_y2 = y1 + bh * 0.35
                    else:
                        face_y2 = y2
                    detections.append([x1, y1, x2, face_y2, conf])
                else:
                    detections.append([x1, y1, x2, y2, conf])

        self.last_detections = detections
        return detections, True
