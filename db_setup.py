"""
db_setup.py
-----------
SQLite database — Users and Videos tables.

Dependencies:
    pip install argon2-cffi
"""

import sqlite3
import os
from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError

DB_PATH = "app.db"
ph = PasswordHasher()


# ── Schema ────────────────────────────────────────────────────────────────────

SCHEMA = """
CREATE TABLE IF NOT EXISTS Users (
    user_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    username  TEXT    NOT NULL UNIQUE,
    email     TEXT    NOT NULL UNIQUE,
    password  TEXT    NOT NULL          -- argon2 hash
);

CREATE TABLE IF NOT EXISTS Videos (
    vid            INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id        INTEGER NOT NULL,
    classification INTEGER NOT NULL DEFAULT 0,  -- 0 = real, 1 = AI
    video          BLOB,                         -- mp4/mp3 stored as raw bytes
    FOREIGN KEY (user_id) REFERENCES Users(user_id)
        ON DELETE CASCADE
);
"""


# ── DB initialisation ─────────────────────────────────────────────────────────

def get_connection(db_path: str = DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: str = DB_PATH) -> None:
    with get_connection(db_path) as conn:
        conn.executescript(SCHEMA)
    print(f"[db_setup] Database ready at: {os.path.abspath(db_path)}")


# ── User helpers ──────────────────────────────────────────────────────────────

def create_user(username: str, email: str, plain_password: str,
                db_path: str = DB_PATH) -> int:
    hashed = ph.hash(plain_password)
    try:
        with get_connection(db_path) as conn:
            cur = conn.execute(
                "INSERT INTO Users (username, email, password) VALUES (?, ?, ?)",
                (username, email, hashed),
            )
            return cur.lastrowid
    except sqlite3.IntegrityError as e:
        if "username" in str(e):
            raise ValueError(f"Username '{username}' is already taken.")
        raise ValueError(f"Email '{email}' is already registered.")


def get_user_by_email(email: str, db_path: str = DB_PATH) -> dict | None:
    with get_connection(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM Users WHERE email = ?", (email,)
        ).fetchone()
    return dict(row) if row else None


def verify_user(email: str, plain_password: str,
                db_path: str = DB_PATH) -> dict | None:
    user = get_user_by_email(email, db_path)
    if user is None:
        return None
    try:
        ph.verify(user["password"], plain_password)
    except VerifyMismatchError:
        return None
    if ph.check_needs_rehash(user["password"]):
        new_hash = ph.hash(plain_password)
        with get_connection(db_path) as conn:
            conn.execute(
                "UPDATE Users SET password = ? WHERE user_id = ?",
                (new_hash, user["user_id"]),
            )
    return user


# ── Video helpers ─────────────────────────────────────────────────────────────

def add_video(user_id: int, classification: bool,
              video_bytes: bytes | None = None,
              db_path: str = DB_PATH) -> int:
    with get_connection(db_path) as conn:
        cur = conn.execute(
            "INSERT INTO Videos (user_id, classification, video) VALUES (?, ?, ?)",
            (user_id, int(classification), video_bytes),
        )
        return cur.lastrowid


def get_videos_for_user(user_id: int, db_path: str = DB_PATH) -> list[dict]:
    with get_connection(db_path) as conn:
        rows = conn.execute(
            "SELECT vid, user_id, classification FROM Videos WHERE user_id = ? ORDER BY vid",
            (user_id,),
        ).fetchall()
    return [dict(r) for r in rows]

if __name__ == "__main__":
    init_db()
    print("Tables: Users, Videos created.")
