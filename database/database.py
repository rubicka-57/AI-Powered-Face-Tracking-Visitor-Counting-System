"""
Database Module for Intelligent Face Tracker
Provides thread-safe SQLite database interactions for visitors and event logging.
"""

import os
import sqlite3
import numpy as np
from datetime import datetime
from typing import List, Dict, Optional, Tuple


class Database:
    def __init__(self, db_path: str = "data/visitors.db"):
        self.db_path = db_path
        db_dir = os.path.dirname(self.db_path)
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        """Get a database connection with row factory enabled."""
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        """Initialize database tables if they do not exist."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            # Table: visitors
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS visitors (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    visitor_id TEXT UNIQUE NOT NULL,
                    embedding BLOB NOT NULL,
                    first_seen TEXT NOT NULL,
                    last_seen TEXT NOT NULL
                )
            """)

            # Table: events
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    visitor_id TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    image_path TEXT
                )
            """)

            # Indexes for high performance queries
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_visitor_id ON visitors(visitor_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_events_visitor ON events(visitor_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_events_timestamp ON events(timestamp)")
            conn.commit()

    @staticmethod
    def _serialize_embedding(embedding: np.ndarray) -> bytes:
        """Convert numpy array embedding to byte blob."""
        return embedding.astype(np.float32).tobytes()

    @staticmethod
    def _deserialize_embedding(blob: bytes) -> np.ndarray:
        """Convert byte blob back to numpy array float32."""
        return np.frombuffer(blob, dtype=np.float32)

    def get_next_visitor_id(self) -> str:
        """Generate next sequential visitor ID formatted as VISITOR_001, VISITOR_002, etc."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM visitors")
            count = cursor.fetchone()[0]
            # Verify no ID collision by finding max numeric suffix
            cursor.execute("SELECT visitor_id FROM visitors")
            rows = cursor.fetchall()
            max_num = 0
            for r in rows:
                v_id = r[0]
                if v_id.startswith("VISITOR_"):
                    try:
                        num = int(v_id.split("_")[1])
                        if num > max_num:
                            max_num = num
                    except (ValueError, IndexError):
                        pass
            next_num = max(count + 1, max_num + 1)
            return f"VISITOR_{next_num:03d}"

    def register_visitor(self, visitor_id: str, embedding: np.ndarray, timestamp: Optional[str] = None) -> bool:
        """Register a new unique visitor."""
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        embedding_blob = self._serialize_embedding(embedding)
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO visitors (visitor_id, embedding, first_seen, last_seen)
                    VALUES (?, ?, ?, ?)
                """, (visitor_id, embedding_blob, timestamp, timestamp))
                conn.commit()
                return True
        except sqlite3.IntegrityError:
            return False

    def update_last_seen(self, visitor_id: str, timestamp: Optional[str] = None):
        """Update last seen timestamp for an existing visitor."""
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE visitors
                SET last_seen = ?
                WHERE visitor_id = ?
            """, (timestamp, visitor_id))
            conn.commit()

    def get_all_visitors(self) -> List[Dict]:
        """Fetch all registered visitors with deserialized embeddings."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT visitor_id, embedding, first_seen, last_seen FROM visitors")
            rows = cursor.fetchall()
            visitors = []
            for row in rows:
                visitors.append({
                    "visitor_id": row["visitor_id"],
                    "embedding": self._deserialize_embedding(row["embedding"]),
                    "first_seen": row["first_seen"],
                    "last_seen": row["last_seen"]
                })
            return visitors

    def get_visitor_by_id(self, visitor_id: str) -> Optional[Dict]:
        """Fetch a specific visitor by their visitor_id."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT visitor_id, embedding, first_seen, last_seen FROM visitors WHERE visitor_id = ?", (visitor_id,))
            row = cursor.fetchone()
            if row:
                return {
                    "visitor_id": row["visitor_id"],
                    "embedding": self._deserialize_embedding(row["embedding"]),
                    "first_seen": row["first_seen"],
                    "last_seen": row["last_seen"]
                }
            return None

    def add_event(self, visitor_id: str, event_type: str, timestamp: Optional[str] = None, image_path: Optional[str] = None):
        """Record an event (ENTRY, EXIT, REGISTER, RECOGNIZED) in the database."""
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO events (visitor_id, event_type, timestamp, image_path)
                VALUES (?, ?, ?, ?)
            """, (visitor_id, event_type.upper(), timestamp, image_path))
            conn.commit()

    def get_events(self, visitor_id: Optional[str] = None, limit: int = 100) -> List[Dict]:
        """Retrieve recent events."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            if visitor_id:
                cursor.execute("""
                    SELECT id, visitor_id, event_type, timestamp, image_path
                    FROM events
                    WHERE visitor_id = ?
                    ORDER BY id DESC
                    LIMIT ?
                """, (visitor_id, limit))
            else:
                cursor.execute("""
                    SELECT id, visitor_id, event_type, timestamp, image_path
                    FROM events
                    ORDER BY id DESC
                    LIMIT ?
                """, (limit,))
            rows = cursor.fetchall()
            return [dict(row) for row in rows]

    def get_unique_visitor_count(self) -> int:
        """Get total number of unique registered visitors."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM visitors")
            return cursor.fetchone()[0]

    def get_event_counts(self) -> Dict[str, int]:
        """Get counts for ENTRY, EXIT, etc."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT event_type, COUNT(*) as cnt
                FROM events
                GROUP BY event_type
            """)
            rows = cursor.fetchall()
            return {row["event_type"]: row["cnt"] for row in rows}
