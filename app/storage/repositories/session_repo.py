"""
session_repo.py — CRUD para sesiones de clase y asignacion de slots de cara.
"""
import logging
from typing import Optional
from app.storage.database import get_connection, _lock

logger = logging.getLogger(__name__)


# ── Sesiones ─────────────────────────────────────────────────────────────────

def create_session(tutor_id: int, titulo: str, materia: str) -> int:
    with _lock:
        conn = get_connection()
        with conn:
            cur = conn.execute(
                "INSERT INTO sessions (tutor_id, titulo, materia, status) VALUES (?, ?, ?, 'activa')",
                (tutor_id, titulo, materia),
            )
            sid = cur.lastrowid
        conn.close()
    logger.info(f"Sesion creada: '{titulo}' (id={sid}, tutor={tutor_id})")
    return sid


def close_session(session_id: int) -> None:
    with _lock:
        conn = get_connection()
        with conn:
            conn.execute(
                "UPDATE sessions SET status='completada', ended_at=datetime('now') WHERE id=?",
                (session_id,),
            )
        conn.close()
    logger.info(f"Sesion cerrada: id={session_id}")


def get_session(session_id: int) -> Optional[dict]:
    conn = get_connection()
    row  = conn.execute("SELECT * FROM sessions WHERE id = ?", (session_id,)).fetchone()
    conn.close()
    return dict(row) if row else None


def get_sessions_by_tutor(tutor_id: int) -> list:
    conn = get_connection()
    rows = conn.execute(
        """SELECT s.id, s.titulo, s.materia, s.started_at, s.ended_at, s.status,
                  COUNT(ss.student_id) as num_alumnos
           FROM sessions s
           LEFT JOIN session_slots ss ON ss.session_id = s.id
           WHERE s.tutor_id = ?
           GROUP BY s.id
           ORDER BY s.started_at DESC""",
        (tutor_id,),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_active_session(tutor_id: int) -> Optional[dict]:
    conn = get_connection()
    row  = conn.execute(
        "SELECT * FROM sessions WHERE tutor_id = ? AND status = 'activa' ORDER BY started_at DESC LIMIT 1",
        (tutor_id,),
    ).fetchone()
    conn.close()
    return dict(row) if row else None


# ── Slots (cara ↔ alumno) ─────────────────────────────────────────────────────

def add_slot(session_id: int, student_id: int, face_slot: int, seat_label: str = "") -> None:
    with _lock:
        conn = get_connection()
        with conn:
            conn.execute(
                """INSERT OR REPLACE INTO session_slots
                   (session_id, student_id, face_slot, seat_label) VALUES (?, ?, ?, ?)""",
                (session_id, student_id, face_slot, seat_label),
            )
        conn.close()


def get_slots(session_id: int) -> list:
    """Devuelve lista de slots con info del alumno incluida."""
    conn = get_connection()
    rows = conn.execute(
        """SELECT ss.face_slot, ss.seat_label, ss.student_id,
                  st.nombre, st.codigo
           FROM session_slots ss
           JOIN students st ON st.id = ss.student_id
           WHERE ss.session_id = ?
           ORDER BY ss.face_slot""",
        (session_id,),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]
