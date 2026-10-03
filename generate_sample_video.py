"""
Sample Video Generator for Katomaran Face Tracker testing.
Generates realistic simulated video with multiple distinct faces crossing the line in both directions.
"""

import cv2
import numpy as np


def draw_stylized_face(canvas, cx, cy, radius, skin_color, eye_color, hair_color):
    """Draw a face with distinct features (eyes, nose, mouth) that can be detected."""
    # Head
    cv2.circle(canvas, (cx, cy), radius, skin_color, -1)
    cv2.circle(canvas, (cx, cy), radius, (50, 50, 50), 2)

    # Hair
    cv2.ellipse(canvas, (cx, cy - int(radius * 0.4)), (radius, int(radius * 0.7)), 0, 180, 360, hair_color, -1)

    # Eyes
    eye_offset_x = int(radius * 0.35)
    eye_offset_y = int(radius * 0.15)
    cv2.circle(canvas, (cx - eye_offset_x, cy - eye_offset_y), int(radius * 0.12), (255, 255, 255), -1)
    cv2.circle(canvas, (cx + eye_offset_x, cy - eye_offset_y), int(radius * 0.12), (255, 255, 255), -1)
    cv2.circle(canvas, (cx - eye_offset_x, cy - eye_offset_y), int(radius * 0.06), eye_color, -1)
    cv2.circle(canvas, (cx + eye_offset_x, cy - eye_offset_y), int(radius * 0.06), eye_color, -1)

    # Nose
    cv2.line(canvas, (cx, cy - int(radius * 0.05)), (cx, cy + int(radius * 0.2)), (80, 80, 80), 2)

    # Mouth / Smile
    cv2.ellipse(canvas, (cx, cy + int(radius * 0.3)), (int(radius * 0.35), int(radius * 0.2)), 0, 0, 180, (50, 50, 200), 2)


def generate_sample_video(output_path="sample.mp4", duration_seconds=10, fps=30):
    width, height = 640, 480
    total_frames = duration_seconds * fps
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Define 3 distinct visitor tracks
    # Person 1: Enters from top (y: 60) moves down past line (y: 400) -> ENTRY
    # Person 2: Enters from bottom (y: 420) moves up past line (y: 80) -> EXIT
    # Person 3: Enters from top after Person 1 (Person 1 reappears -> recognized as VISITOR_001)

    print(f"Generating sample test video: {output_path} ({total_frames} frames)...")

    for f in range(total_frames):
        frame = np.ones((height, width, 3), dtype=np.uint8) * 230  # Light background

        # Person 1 (Frames 0 to 120): Moves top to bottom (ENTRY)
        if 0 <= f < 120:
            progress = f / 120.0
            y1 = int(60 + progress * 360)
            x1 = 200
            # Draw body and face
            cv2.rectangle(frame, (x1 - 35, y1 + 35), (x1 + 35, y1 + 130), (180, 50, 50), -1)
            draw_stylized_face(frame, x1, y1, 35, (190, 210, 240), (150, 75, 0), (20, 20, 20))

        # Person 2 (Frames 80 to 200): Moves bottom to top (EXIT)
        if 80 <= f < 200:
            progress = (f - 80) / 120.0
            y2 = int(420 - progress * 340)
            x2 = 440
            cv2.rectangle(frame, (x2 - 35, y2 + 35), (x2 + 35, y2 + 130), (50, 150, 50), -1)
            draw_stylized_face(frame, x2, y2, 35, (180, 200, 230), (0, 100, 200), (30, 80, 150))

        # Person 1 reappears (Frames 180 to 290): Same Person 1 enters again (Tests Re-identification & No Duplication)
        if 180 <= f < 290:
            progress = (f - 180) / 110.0
            y3 = int(60 + progress * 360)
            x3 = 240
            cv2.rectangle(frame, (x3 - 35, y3 + 35), (x3 + 35, y3 + 130), (180, 50, 50), -1)
            draw_stylized_face(frame, x3, y3, 35, (190, 210, 240), (150, 75, 0), (20, 20, 20))

        out.write(frame)

    out.release()
    print(f"Sample video generated successfully: {output_path}")


if __name__ == "__main__":
    generate_sample_video()
