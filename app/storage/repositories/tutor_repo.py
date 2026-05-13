"""
tutor_repo.py — CRUD para tutores con embeddings faciales Facenet512.

Almacenamiento:
  - El embedding es un numpy array de 512 floats.
  - Se serializa con pickle a BLOB en SQLite.
  - Nunca se guarda la foto — solo el vector matemático.

Verificación:
  - Similitud coseno entre el embedding capturado y los almacenados.
  - Umbral: 0.58  (Facenet512: arriba = misma persona, abajo = diferente)
"""

import pickle
import uuid
import logging
import numpy as np

from datetime import datetime, timedelta
from typing import Optional

from app.storage.database import get_connection, _lock

logger = logging.getLogger(__name__)

# Umbral de similitud coseno para Facenet512
# Frames con overlays del inference engine + compresion JPEG reducen la similitud.
# 0.58 = equilibrio seguro para condiciones de video en tiempo real con overlays.
# Mismo-persona-mismas-condiciones: tipicamente 0.80-0.95
# Mismo-persona-overlays/angulo: 0.60-0.80
# Personas-distintas: tipicamente < 0.50
SIMILARITY_THRESHOLD = 0.58
# ── Escritura ────────────────────────────────────────────────────────────────
def create_tutor(nombre: str, username: str, embedding: np.ndarray) -> int:
    """Guarda un tutor nuevo. Devuelve su ID."""
    blob = pickle.dumps(embedding.astype(np.float32))
    with _lock:
        conn = get_connection()
        with conn:
            cursor = conn.execute(
                "INSERT INTO tutors (nombre, username, embedding) VALUES (?, ?, ?)",
                (nombre, username, blob),
            )
            tutor_id = cursor.lastrowid
        conn.close()
    logger.info(f"Tutor registrado: '{nombre}' (id={tutor_id})")
    return tutor_id
def update_tutor_nombre(tutor_id: int, nombre: str) -> bool:
    """Actualiza el nombre de un tutor. Devuelve True si existía."""
    with _lock:
        conn = get_connection()
        with conn:
            cur = conn.execute(
                "UPDATE tutors SET nombre = ? WHERE id = ?", (nombre, tutor_id)
            )
        conn.close()
    return cur.rowcount > 0
def delete_tutor(tutor_id: int) -> bool:
    """Elimina un tutor y en cascada sus alumnos, sesiones y tokens. Devuelve True si existía."""
    with _lock:
        conn = get_connection()
        with conn:
            cur = conn.execute("DELETE FROM tutors WHERE id = ?", (tutor_id,))
        conn.close()
    deleted = cur.rowcount > 0
    if deleted:
        logger.info(f"Tutor eliminado: id={tutor_id}")
    return deleted
def username_exists(username: str) -> bool:
    """Verifica si el username ya está registrado."""
    conn = get_connection()
    row = conn.execute(
        "SELECT id FROM tutors WHERE username = ?", (username,)
    ).fetchone()
    conn.close()
    return row is not None
def get_tutor_count() -> int:
    """Devuelve cuántos tutores hay registrados."""
    conn = get_connection()
    count = conn.execute("SELECT COUNT(*) FROM tutors").fetchone()[0]
    conn.close()
    return count
# ── Verificación biométrica ──────────────────────────────────────────────────
def find_matching_tutor(query_embedding: np.ndarray) -> Optional[dict]:
    """
    Compara el embedding con todos los tutores registrados.
    Devuelve el mejor match si supera SIMILARITY_THRESHOLD, o None.

    Retorna dict con: id, nombre, username, similarity
    """
    conn = get_connection()
    rows = conn.execute(
        "SELECT id, nombre, username, embedding FROM tutors"
    ).fetchall()
    conn.close()

    if not rows:
        return None

    best_match = None
    best_sim   = -1.0

    for row in rows:
        stored = pickle.loads(row["embedding"])
        sim    = _cosine_similarity(query_embedding, stored)
        if sim > best_sim:
            best_sim   = sim
            best_match = dict(row)

    if best_match and best_sim >= SIMILARITY_THRESHOLD:
        best_match.pop("embedding", None)
        best_match["similarity"] = round(float(best_sim), 4)
        return best_match

    return None
# ── Tokens de sesión ─────────────────────────────────────────────────────────
def create_session_token(tutor_id: int, hours: int = 8) -> str:
    """Genera un UUID token, lo guarda en DB y actualiza last_login."""
    token   = str(uuid.uuid4())
    expires = (datetime.now() + timedelta(hours=hours)).isoformat()
    with _lock:
        conn = get_connection()
        with conn:
            conn.execute(
                "INSERT INTO auth_sessions (token, tutor_id, expires_at) VALUES (?, ?, ?)",
                (token, tutor_id, expires),
            )
            conn.execute(
                "UPDATE tutors SET last_login = datetime('now') WHERE id = ?",
                (tutor_id,),
            )
        conn.close()
    return token
def verify_token(token: str) -> Optional[dict]:
    """
    Verifica que el token exista y no haya expirado.
    Devuelve info del tutor o None.
    """
    conn = get_connection()
    row  = conn.execute(
        """
        SELECT t.id, t.nombre, t.username, t.last_login
        FROM   auth_sessions s
        JOIN   tutors t ON t.id = s.tutor_id
        WHERE  s.token = ?
          AND  s.expires_at > datetime('now')
        """,
        (token,),
    ).fetchone()
    conn.close()
    return dict(row) if row else None
def delete_token(token: str) -> None:
    """Elimina el token (logout)."""
    with _lock:
        conn = get_connection()
        with conn:
            conn.execute(
                "DELETE FROM auth_sessions WHERE token = ?", (token,)
            )
        conn.close()
# ── Matemática ───────────────────────────────────────────────────────────────
def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Similitud coseno entre dos vectores. Rango: -1 a 1."""
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return float(np.dot(a, b) / (na * nb))
