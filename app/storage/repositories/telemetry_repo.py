"""
telemetry_repo.py — Escritura y lectura de snapshots de telemetria.

Escritura: cada 5 segundos por el TelemetryWriter (una fila por alumno activo).
Lectura:   por reports_api para generar las graficas.
"""
import logging
from typing import Optional
from app.storage.database import get_connection, _lock

logger = logging.getLogger(__name__)


def insert_snapshot(
    session_id:          int,
    student_id:          int,
    atencion:            str,
    indice_comprension:  float,
    emocion:             str,
    sentimiento:         str,
    mirada:              str,
    ear:                 float,
) -> None:
    """Inserta un snapshot. Llamado desde el TelemetryWriter thread."""
    with _lock:
        conn = get_connection()
        with conn:
            conn.execute(
                """INSERT INTO telemetry_log
                   (session_id, student_id, atencion, indice_comprension,
                    emocion, sentimiento, mirada, ear)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (session_id, student_id, atencion, indice_comprension,
                 emocion, sentimiento, mirada, ear),
            )
        conn.close()


def get_snapshots(session_id: int, student_id: Optional[int] = None) -> list:
    """Devuelve todos los snapshots de una sesion (opcionalmente filtrado por alumno)."""
    conn = get_connection()
    if student_id is not None:
        rows = conn.execute(
            """SELECT * FROM telemetry_log
               WHERE session_id = ? AND student_id = ?
               ORDER BY timestamp""",
            (session_id, student_id),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM telemetry_log WHERE session_id = ? ORDER BY timestamp",
            (session_id,),
        ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_student_summary(session_id: int, student_id: int) -> dict:
    """
    Estadisticas agregadas de un alumno en una sesion.
    Devuelve: avg_comprension, atencion_counts, emocion_counts, mirada_counts.
    """
    conn = get_connection()

    # Promedio de comprension
    avg = conn.execute(
        "SELECT AVG(indice_comprension) FROM telemetry_log WHERE session_id=? AND student_id=?",
        (session_id, student_id),
    ).fetchone()[0] or 0.0

    # Distribucion de atencion
    atencion_rows = conn.execute(
        """SELECT atencion, COUNT(*) as cnt FROM telemetry_log
           WHERE session_id=? AND student_id=? GROUP BY atencion""",
        (session_id, student_id),
    ).fetchall()

    # Distribucion de emocion
    emocion_rows = conn.execute(
        """SELECT emocion, COUNT(*) as cnt FROM telemetry_log
           WHERE session_id=? AND student_id=? GROUP BY emocion""",
        (session_id, student_id),
    ).fetchall()

    # Distribucion de mirada
    mirada_rows = conn.execute(
        """SELECT mirada, COUNT(*) as cnt FROM telemetry_log
           WHERE session_id=? AND student_id=? GROUP BY mirada""",
        (session_id, student_id),
    ).fetchall()

    # Total de muestras
    total = conn.execute(
        "SELECT COUNT(*) FROM telemetry_log WHERE session_id=? AND student_id=?",
        (session_id, student_id),
    ).fetchone()[0]

    conn.close()

    return {
        "avg_comprension": round(avg, 1),
        "total_snapshots": total,
        "atencion":  {r["atencion"]: r["cnt"] for r in atencion_rows},
        "emocion":   {r["emocion"]:  r["cnt"] for r in emocion_rows},
        "mirada":    {r["mirada"]:   r["cnt"] for r in mirada_rows},
    }


def get_comprension_timeline(session_id: int, student_id: int, bucket_seconds: int = 60) -> list:
    """
    Agrupa la comprension en buckets de N segundos (por defecto 1 minuto).
    Devuelve lista de {minuto, comprension} para el grafico de linea.
    Usa subquery en lugar de window function para compatibilidad con SQLite < 3.25.
    """
    conn = get_connection()
    rows = conn.execute(
        """SELECT
               CAST((julianday(tl.timestamp) -
                    (SELECT julianday(MIN(t2.timestamp))
                     FROM telemetry_log t2
                     WHERE t2.session_id = ? AND t2.student_id = ?))
                    * 86400 / ? AS INTEGER) AS bucket,
               AVG(tl.indice_comprension) AS avg_comp
           FROM telemetry_log tl
           WHERE tl.session_id = ? AND tl.student_id = ?
           GROUP BY bucket
           ORDER BY bucket""",
        (session_id, student_id, bucket_seconds, session_id, student_id),
    ).fetchall()
    conn.close()
    return [{"minuto": r["bucket"], "comprension": round(r["avg_comp"], 1)} for r in rows]
