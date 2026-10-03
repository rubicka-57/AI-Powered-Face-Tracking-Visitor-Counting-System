"""
Image Utilities Module for Katomaran Face Tracker.
Handles face cropping, formatted event image persistence, and visual HUD overlays.
"""

import os
import cv2
import numpy as np
from datetime import datetime
from typing import Tuple, Optional, List, Dict


def crop_face(frame: np.ndarray, bbox: Tuple[int, int, int, int], margin: float = 0.15) -> Optional[np.ndarray]:
    """
    Safely crop face region from frame with an optional contextual margin.
    bbox format: (x1, y1, x2, y2)
    """
    if frame is None or frame.size == 0:
        return None

    h, w = frame.shape[:2]
    x1, y1, x2, y2 = bbox

    bw = x2 - x1
    bh = y2 - y1

    if bw <= 0 or bh <= 0:
        return None

    margin_x = int(bw * margin)
    margin_y = int(bh * margin)

    cx1 = max(0, x1 - margin_x)
    cy1 = max(0, y1 - margin_y)
    cx2 = min(w, x2 + margin_x)
    cy2 = min(h, y2 + margin_y)

    crop = frame[cy1:cy2, cx1:cx2]
    if crop.size == 0:
        return None
    return crop


def save_event_image(face_img: np.ndarray, event_type: str, visitor_id: str,
                     timestamp_str: Optional[str] = None, logs_dir: str = "logs") -> Optional[str]:
    """
    Save event image into organized directory structure:
    logs/entries/YYYY-MM-DD/VISITOR_XXX_HHMMSS.jpg
    logs/exits/YYYY-MM-DD/VISITOR_XXX_HHMMSS.jpg
    """
    if face_img is None or face_img.size == 0:
        return None

    now = datetime.now()
    if timestamp_str:
        try:
            dt = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
        except ValueError:
            dt = now
    else:
        dt = now

    date_folder = dt.strftime("%Y-%m-%d")
    time_stamp = dt.strftime("%Y%m%d_%H%M%S")
    folder_type = "entries" if event_type.upper() == "ENTRY" else "exits"

    target_dir = os.path.join(logs_dir, folder_type, date_folder)
    os.makedirs(target_dir, exist_ok=True)

    filename = f"{visitor_id}_{time_stamp}.jpg"
    file_path = os.path.join(target_dir, filename)

    # Convert to relative path format for consistency
    rel_path = os.path.normpath(file_path).replace("\\", "/")

    success = cv2.imwrite(file_path, face_img)
    if success:
        return rel_path
    return None


def draw_hud(frame: np.ndarray,
             tracks: List[Dict],
             line_position: float,
             line_orientation: str,
             unique_visitor_count: int,
             fps: float = 0.0,
             is_detection_frame: bool = True) -> np.ndarray:
    """
    Render premium bounding boxes, labels, virtual counting line, and HUD overlay.
    """
    annotated = frame.copy()
    h, w = annotated.shape[:2]

    # 1. Draw Virtual Counting Line
    if line_orientation == "horizontal":
        y_line = int(h * line_position)
        # Glow line
        cv2.line(annotated, (0, y_line), (w, y_line), (0, 165, 255), 2, cv2.LINE_AA)
        cv2.putText(annotated, "ENTRY / EXIT LINE", (20, y_line - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 165, 255), 2, cv2.LINE_AA)
    else:
        x_line = int(w * line_position)
        cv2.line(annotated, (x_line, 0), (x_line, h), (0, 165, 255), 2, cv2.LINE_AA)
        cv2.putText(annotated, "ENTRY / EXIT LINE", (x_line + 8, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 165, 255), 2, cv2.LINE_AA)

    # 2. Draw Tracked Faces
    for trk in tracks:
        bbox = trk.get("bbox", [0, 0, 0, 0])
        x1, y1, x2, y2 = [int(v) for v in bbox]
        track_id = trk.get("track_id", "?")
        visitor_id = trk.get("visitor_id") or "Pending..."
        status = trk.get("status", "Active")

        # Select color based on recognition
        box_color = (0, 220, 0) if visitor_id.startswith("VISITOR_") else (240, 180, 0)

        # Draw rounded corner box or clean rect
        cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, 2, cv2.LINE_AA)

        # Center indicator
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)
        cv2.circle(annotated, (cx, cy), 4, (0, 0, 255), -1)

        # Label badge
        label = f"ID:{track_id} | {visitor_id}"
        font_scale = 0.5
        thickness = 1
        (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        cv2.rectangle(annotated, (x1, y1 - lh - 10), (x1 + lw + 10, y1), box_color, -1)
        cv2.putText(annotated, label, (x1 + 5, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)

    # 3. Draw Top Dashboard Card (Glassmorphic style banner)
    overlay = annotated.copy()
    cv2.rectangle(overlay, (15, 15), (360, 135), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.75, annotated, 0.25, 0, annotated)
    cv2.rectangle(annotated, (15, 15), (360, 135), (80, 80, 80), 1, cv2.LINE_AA)

    # Dashboard texts
    cv2.putText(annotated, "KATOMARAN FACE TRACKER", (28, 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2, cv2.LINE_AA)

    cv2.putText(annotated, f"Unique Visitors: {unique_visitor_count}", (28, 65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.putText(annotated, f"Currently Tracked: {len(tracks)}", (28, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    mode_text = "Detection: ON" if is_detection_frame else "Tracking Mode"
    mode_color = (0, 255, 200) if is_detection_frame else (180, 180, 180)
    cv2.putText(annotated, f"FPS: {fps:.1f} | {mode_text}", (28, 115),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, mode_color, 1, cv2.LINE_AA)

    return annotated
