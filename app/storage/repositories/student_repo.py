"""
student_repo.py — CRUD para alumnos.
"""
import logging
from typing import Optional
from app.storage.database import get_connection, _lock

logger = logging.getLogger(__name__)


def create_student(nombre: str, codigo: str, tutor_id: int) -> int:
    with _lock:
        conn = get_connection()
        with conn:
            cur = conn.execute(
                "INSERT INTO students (nombre, codigo, tutor_id) VALUES (?, ?, ?)",
                (nombre, codigo, tutor_id),
            )
            sid = cur.lastrowid
        conn.close()
    logger.info(f"Alumno creado: '{nombre}' codigo={codigo} (id={sid})")
    return sid

def get_students_by_tutor(tutor_id: int) -> list:
    conn = get_connection()
    rows = conn.execute(
        "SELECT id, nombre, codigo, created_at FROM students WHERE tutor_id = ? ORDER BY nombre",
        (tutor_id,),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]

def get_student_by_id(student_id: int) -> Optional[dict]:
    conn = get_connection()
    row = conn.execute(
        "SELECT id, nombre, codigo, tutor_id FROM students WHERE id = ?",
        (student_id,),
    ).fetchone()
    conn.close()
    return dict(row) if row else None

def codigo_exists(codigo: str) -> bool:
    conn = get_connection()
    row  = conn.execute("SELECT id FROM students WHERE codigo = ?", (codigo,)).fetchone()
    conn.close()
    return row is not None

def update_student(student_id: int, nombre: str, codigo: str) -> bool:
    """Actualiza nombre y código de un alumno. Devuelve True si existía."""
    with _lock:
        conn = get_connection()
        with conn:
            cur = conn.execute(
                "UPDATE students SET nombre = ?, codigo = ? WHERE id = ?",
                (nombre, codigo, student_id),
            )
        conn.close()
    return cur.rowcount > 0

def delete_student(student_id: int) -> None:
    with _lock:
        conn = get_connection()
        with conn:
            conn.execute("DELETE FROM students WHERE id = ?", (student_id,))
        conn.close()
