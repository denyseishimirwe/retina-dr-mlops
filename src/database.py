
import os
import sqlite3
from datetime import datetime, timezone

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "data", "retina.db")


def get_connection():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with get_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS uploaded_images (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL,
                label TEXT NOT NULL,
                image_data BLOB NOT NULL,
                stored_path TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending',
                uploaded_at TEXT NOT NULL,
                trained_at TEXT
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_upload_status
            ON uploaded_images(status)
        """)


def save_upload(filename, label, image_data, stored_path):
    with get_connection() as conn:
        conn.execute("""
            INSERT INTO uploaded_images
            (filename, label, image_data, stored_path, status, uploaded_at)
            VALUES (?, ?, ?, ?, 'pending', ?)
        """, (
            filename,
            label,
            sqlite3.Binary(image_data),
            stored_path,
            datetime.now(timezone.utc).isoformat()
        ))


def mark_trained(image_paths):
    if not image_paths:
        return

    with get_connection() as conn:
        conn.executemany("""
            UPDATE uploaded_images
            SET status = 'trained', trained_at = ?
            WHERE stored_path = ? AND status = 'pending'
        """, [
            (datetime.now(timezone.utc).isoformat(), path)
            for path in image_paths
        ])


def count_pending():
    with get_connection() as conn:
        result = conn.execute("""
            SELECT COUNT(*) FROM uploaded_images
            WHERE status = 'pending'
        """).fetchone()
        return result[0]
    
def get_pending_uploads():
    with get_connection() as conn:
        rows = conn.execute("""
            SELECT stored_path, label
            FROM uploaded_images
            WHERE status = 'pending'
        """).fetchall()

        return [
            (row["stored_path"], row["label"])
            for row in rows
        ]