"""
SQLite database persistence for VIZAN Virtual Assistant.
Stores conversation history, user interactions, and algorithm run telemetry.
"""

import json
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

DB_PATH = Path(__file__).resolve().parent / "vizan.db"


def get_connection() -> sqlite3.Connection:
    """Return a connection to the SQLite database."""
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Initializes database tables if they do not exist."""
    conn = get_connection()
    with conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                message TEXT NOT NULL,
                intent TEXT,
                metadata TEXT,
                timestamp REAL NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS algorithm_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                algorithm_name TEXT NOT NULL,
                problem_type TEXT NOT NULL,
                input_params TEXT,
                path TEXT,
                path_cost REAL,
                nodes_expanded INTEGER,
                runtime_ms REAL,
                success INTEGER,
                timestamp REAL NOT NULL
            )
        """)
    conn.close()


def save_message(
    session_id: str,
    role: str,
    message: str,
    intent: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None
):
    """Saves a single conversation turn."""
    init_db()
    conn = get_connection()
    with conn:
        conn.execute(
            """
            INSERT INTO conversations (session_id, role, message, intent, metadata, timestamp)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                session_id,
                role,
                message,
                intent,
                json.dumps(metadata) if metadata else None,
                time.time()
            )
        )
    conn.close()


def get_recent_messages(session_id: str, limit: int = 50) -> List[Dict[str, Any]]:
    """Retrieves recent conversation turns for a session."""
    init_db()
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT role, message, intent, metadata, timestamp
        FROM conversations
        WHERE session_id = ?
        ORDER BY id ASC
        LIMIT ?
        """,
        (session_id, limit)
    )
    rows = cursor.fetchall()
    conn.close()

    history = []
    for row in rows:
        history.append({
            "role": row["role"],
            "content": row["message"],
            "intent": row["intent"],
            "metadata": json.loads(row["metadata"]) if row["metadata"] else {},
            "timestamp": row["timestamp"]
        })
    return history


def clear_session_history(session_id: str):
    """Deletes conversation records for a session."""
    init_db()
    conn = get_connection()
    with conn:
        conn.execute("DELETE FROM conversations WHERE session_id = ?", (session_id,))
    conn.close()

