"""
ByteTrack-based Multi-Face Tracking Module for Katomaran Face Tracker.
Maintains persistent track IDs, motion history trajectories, and state management.
"""

import numpy as np
import supervision as sv
from typing import List, Dict, Tuple, Optional


class FaceTracker:
    """
    ByteTrack Wrapper for stable face tracking across frames.
    Maintains trajectories, track continuity during frame skipping, and track states.
    """
    def __init__(self,
                 track_activation_threshold: float = 0.35,
                 lost_track_buffer: int = 30,
                 minimum_matching_threshold: float = 0.8,
                 frame_rate: int = 30):
        self.tracker = sv.ByteTrack(
            track_activation_threshold=track_activation_threshold,
            lost_track_buffer=lost_track_buffer,
            minimum_matching_threshold=minimum_matching_threshold,
            frame_rate=frame_rate
        )
        # Dictionary mapping track_id -> metadata (history, visitor_id, crossed_line, etc.)
        self.track_metadata: Dict[int, Dict] = {}

    def update(self, detections_list: List[List[float]]) -> List[Dict]:
        """
        Update tracker with current bounding boxes.
        detections_list: List of [x1, y1, x2, y2, score]

        Returns:
            List of active track dicts:
            [{
                "track_id": int,
                "bbox": [x1, y1, x2, y2],
                "center": (cx, cy),
                "confidence": float,
                "visitor_id": str (if assigned),
                "history": list of (cx, cy)
            }, ...]
        """
        if len(detections_list) == 0:
            sv_detections = sv.Detections.empty()
        else:
            xyxy = np.array([d[:4] for d in detections_list], dtype=np.float32)
            confidence = np.array([d[4] if len(d) > 4 else 1.0 for d in detections_list], dtype=np.float32)
            class_id = np.zeros(len(detections_list), dtype=int)

            sv_detections = sv.Detections(
                xyxy=xyxy,
                confidence=confidence,
                class_id=class_id
            )

        tracked_detections = self.tracker.update_with_detections(sv_detections)

        active_tracks = []

        if len(tracked_detections) > 0 and tracked_detections.tracker_id is not None:
            for i in range(len(tracked_detections)):
                track_id = int(tracked_detections.tracker_id[i])
                bbox = [int(v) for v in tracked_detections.xyxy[i]]
                conf = float(tracked_detections.confidence[i]) if tracked_detections.confidence is not None else 1.0

                cx = int((bbox[0] + bbox[2]) / 2)
                cy = int((bbox[1] + bbox[3]) / 2)

                # Maintain persistent track metadata
                if track_id not in self.track_metadata:
                    self.track_metadata[track_id] = {
                        "visitor_id": None,
                        "history": [],
                        "crossed_entry": False,
                        "crossed_exit": False,
                        "initial_side": None,
                        "recognition_attempts": 0,
                        "best_crop": None
                    }

                meta = self.track_metadata[track_id]
                meta["history"].append((cx, cy))
                # Keep last 30 positions in history
                if len(meta["history"]) > 30:
                    meta["history"].pop(0)

                active_tracks.append({
                    "track_id": track_id,
                    "bbox": bbox,
                    "center": (cx, cy),
                    "confidence": conf,
                    "visitor_id": meta.get("visitor_id"),
                    "metadata": meta
                })

        return active_tracks

    def get_track_metadata(self, track_id: int) -> Optional[Dict]:
        return self.track_metadata.get(track_id)

    def set_visitor_id(self, track_id: int, visitor_id: str):
        if track_id in self.track_metadata:
            self.track_metadata[track_id]["visitor_id"] = visitor_id
