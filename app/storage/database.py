"""
database.py — Edu-Insight MLOps
Gestiona la conexión SQLite y la creación de tablas.
Archivo único: app/storage/sessions.db
"""

import sqlite3
import threading
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_DB_PATH = Path(__file__).parent / "sessions.db"
_lock    = threading.Lock()          # protege escrituras concurrentes


def get_connection() -> sqlite3.Connection:
    """Devuelve una conexión lista para usar (row_factory activado)."""
    conn = sqlite3.connect(str(_DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")   # soporta lecturas concurrentes
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db() -> None:
    """Crea todas las tablas si no existen. Se llama una vez en el startup."""
    with _lock:
        conn = get_connection()
        with conn:
            conn.executescript("""
                -- ── Tutores ──────────────────────────────────────────────
                CREATE TABLE IF NOT EXISTS tutors (
                    id            INTEGER PRIMARY KEY AUTOINCREMENT,
                    nombre        TEXT    NOT NULL,
                    username      TEXT    UNIQUE NOT NULL,
                    embedding     BLOB    NOT NULL,
                    registered_at TEXT    NOT NULL DEFAULT (datetime('now')),
                    last_login    TEXT
                );

                -- ── Tokens de sesión ─────────────────────────────────────
                CREATE TABLE IF NOT EXISTS auth_sessions (
                    token      TEXT    PRIMARY KEY,
                    tutor_id   INTEGER NOT NULL,
                    created_at TEXT    NOT NULL DEFAULT (datetime('now')),
                    expires_at TEXT    NOT NULL,
                    FOREIGN KEY (tutor_id) REFERENCES tutors(id) ON DELETE CASCADE
                );

                -- ── Alumnos ──────────────────────────────────────────────
                CREATE TABLE IF NOT EXISTS students (
                    id         INTEGER PRIMARY KEY AUTOINCREMENT,
                    nombre     TEXT    NOT NULL,
                    codigo     TEXT    UNIQUE NOT NULL,
                    tutor_id   INTEGER NOT NULL,
                    created_at TEXT    NOT NULL DEFAULT (datetime('now')),
                    FOREIGN KEY (tutor_id) REFERENCES tutors(id) ON DELETE CASCADE
                );

                -- ── Sesiones de clase ─────────────────────────────────────
                CREATE TABLE IF NOT EXISTS sessions (
                    id         INTEGER PRIMARY KEY AUTOINCREMENT,
                    tutor_id   INTEGER NOT NULL,
                    titulo     TEXT    NOT NULL,
                    materia    TEXT,
                    started_at TEXT    NOT NULL DEFAULT (datetime('now')),
                    ended_at   TEXT,
                    status     TEXT    NOT NULL DEFAULT 'activa',
                    FOREIGN KEY (tutor_id) REFERENCES tutors(id) ON DELETE CASCADE
                );

                -- ── Asignación cara ↔ alumno por sesión ──────────────────
                CREATE TABLE IF NOT EXISTS session_slots (
                    session_id INTEGER NOT NULL,
                    student_id INTEGER NOT NULL,
                    face_slot  INTEGER NOT NULL CHECK (face_slot BETWEEN 0 AND 5),
                    seat_label TEXT,
                    PRIMARY KEY (session_id, face_slot),
                    FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE,
                    FOREIGN KEY (student_id) REFERENCES students(id) ON DELETE CASCADE
                );

                -- ── Telemetría: snapshot cada 5 segundos por alumno ──────
                CREATE TABLE IF NOT EXISTS telemetry_log (
                    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id         INTEGER NOT NULL,
                    student_id         INTEGER NOT NULL,
                    timestamp          TEXT    NOT NULL DEFAULT (datetime('now')),
                    atencion           TEXT,
                    indice_comprension REAL,
                    emocion            TEXT,
                    sentimiento        TEXT,
                    mirada             TEXT,
                    ear                REAL,
                    FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE,
                    FOREIGN KEY (student_id) REFERENCES students(id) ON DELETE CASCADE
                );

                -- índice para queries de reportes (session_id + student_id frecuentes)
                CREATE INDEX IF NOT EXISTS idx_telemetry_session
                    ON telemetry_log (session_id, student_id);
            """)
        conn.close()
    logger.info(f"Base de datos lista: {_DB_PATH}")
