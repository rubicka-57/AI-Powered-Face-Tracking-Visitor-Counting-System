"""
Face Recognition & Auto-Registration Module for Katomaran Face Tracker.
Uses InsightFace (ArcFace) to generate 512-d facial embeddings, compute cosine similarities,
and automatically register or recognize unique visitors.
"""

import cv2
import numpy as np
from typing import Optional, Tuple, List, Dict
import insightface
from insightface.app import FaceAnalysis
from database.database import Database
from logging_system.event_logger import EventLogger


class FaceRecognizer:
    """
    InsightFace ArcFace Recognition and Auto-Registration engine.
    Maintains an in-memory embedding gallery synchronized with SQLite.
    """
    def __init__(self,
                 database: Database,
                 logger: EventLogger,
                 similarity_threshold: float = 0.45,
                 model_name: str = "buffalo_sc"):
        self.db = database
        self.logger = logger
        self.similarity_threshold = similarity_threshold
        self.model_name = model_name

        # In-memory gallery: list of {"visitor_id": str, "embedding": np.ndarray}
        self.gallery: List[Dict] = []

        # Initialize InsightFace FaceAnalysis app
        self.app = FaceAnalysis(name=self.model_name, providers=['CPUExecutionProvider'])
        self.app.prepare(ctx_id=0, det_size=(320, 320))

        # Load existing registered visitors into memory cache
        self._load_gallery()

    def _load_gallery(self):
        """Populate local gallery cache from database."""
        visitors = self.db.get_all_visitors()
        self.gallery = []
        for v in visitors:
            self.gallery.append({
                "visitor_id": v["visitor_id"],
                "embedding": v["embedding"]
            })
        self.logger.info(f"Loaded {len(self.gallery)} registered visitor(s) into recognition gallery.", event_type="INIT")

    @staticmethod
    def compute_cosine_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Compute cosine similarity between two 512-d embeddings."""
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(np.dot(emb1, emb2) / (norm1 * norm2))

    def extract_embedding(self, face_crop: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract normalized 512-d ArcFace embedding from cropped face image.
        Uses InsightFace recognition model with fallback for challenging crops.
        """
        if face_crop is None or face_crop.size == 0:
            return None

        # 1. Primary: Try InsightFace FaceAnalysis detection + landmark alignment
        faces = self.app.get(face_crop)
        if len(faces) > 0 and faces[0].embedding is not None:
            embedding = faces[0].embedding
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
            return embedding.astype(np.float32)

        # 2. Robust Fallback: Pass resized (112, 112) crop directly to ArcFace ONNX model
        rec_model = self.app.models.get('recognition')
        if rec_model is not None:
            try:
                resized = cv2.resize(face_crop, (112, 112))
                feat = rec_model.get_feat(resized)
                if feat is not None:
                    feat = np.array(feat).flatten()
                    norm = np.linalg.norm(feat)
                    if norm > 0:
                        feat = feat / norm
                    return feat.astype(np.float32)
            except Exception as e:
                self.logger.debug(f"Direct embedding fallback exception: {e}")

        return None

    def recognize_or_register(self, face_crop: np.ndarray, timestamp: Optional[str] = None) -> Tuple[Optional[str], float, bool]:
        """
        Processes a face crop:
        1. Generates embedding.
        2. Compares with all gallery embeddings.
        3. If similarity >= threshold: recognizes existing visitor and updates DB last_seen.
        4. If below threshold: registers new visitor in DB and gallery.

        Returns:
            Tuple: (visitor_id: str, similarity_score: float, is_new_registration: bool)
        """
        embedding = self.extract_embedding(face_crop)
        if embedding is None:
            return None, 0.0, False

        best_match_id = None
        highest_similarity = -1.0

        # Compare with gallery
        for item in self.gallery:
            sim = self.compute_cosine_similarity(embedding, item["embedding"])
            if sim > highest_similarity:
                highest_similarity = sim
                best_match_id = item["visitor_id"]

        # Check against similarity threshold
        if highest_similarity >= self.similarity_threshold and best_match_id is not None:
            # Existing Visitor Recognized
            self.db.update_last_seen(best_match_id, timestamp=timestamp)
            self.logger.log_recognition(best_match_id, highest_similarity)
            return best_match_id, highest_similarity, False
        else:
            # Genuinely New Visitor Detected -> Auto-Register
            new_visitor_id = self.db.get_next_visitor_id()
            success = self.db.register_visitor(new_visitor_id, embedding, timestamp=timestamp)
            if success:
                self.gallery.append({
                    "visitor_id": new_visitor_id,
                    "embedding": embedding
                })
                self.logger.log_registration(new_visitor_id)
                return new_visitor_id, highest_similarity if highest_similarity > 0 else 0.0, True
            else:
                self.logger.error(f"Failed to register new visitor {new_visitor_id} in database", event_type="DB_ERROR")
                return None, 0.0, False
