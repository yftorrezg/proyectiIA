"""
reports_api.py — Datos agregados para el dashboard de reportes.

  GET /api/reports/{session_id}              Reporte completo de la sesion
  GET /api/reports/{session_id}/csv          Exportar telemetria en CSV
  GET /api/reports/{session_id}/students     Lista de alumnos con resumen
"""
import csv
import io
from typing import Optional
from fastapi import APIRouter, Header, HTTPException
from fastapi.responses import StreamingResponse

router = APIRouter(prefix="/api/reports", tags=["Reports"])


def _verify(authorization: Optional[str]) -> dict:
    from app.storage.repositories.tutor_repo import verify_token
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Token requerido")
    token = authorization.split(" ", 1)[1]
    tutor = verify_token(token)
    if not tutor:
        raise HTTPException(status_code=401, detail="Token invalido o expirado")
    return tutor


@router.get("/{session_id}")
async def full_report(session_id: int,
                      authorization: Optional[str] = Header(None)):
    """
    Reporte completo: info de sesion + resumen por alumno + timeline.
    """
    import logging
    log = logging.getLogger("reports_api")
    _verify(authorization)
    from app.storage.repositories.session_repo   import get_session, get_slots
    from app.storage.repositories.telemetry_repo import (
        get_student_summary, get_comprension_timeline
    )

    session = get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Sesion {session_id} no encontrada")

    slots   = get_slots(session_id)
    students_data = []
    all_avgs = []

    for slot in slots:
        sid = slot["student_id"]
        try:
            summary  = get_student_summary(session_id, sid)
            timeline = get_comprension_timeline(session_id, sid, bucket_seconds=60)
        except Exception as e:
            log.error(f"Error procesando alumno {sid}: {e}", exc_info=True)
            summary  = {"avg_comprension": 0, "total_snapshots": 0,
                        "atencion": {}, "emocion": {}, "mirada": {}}
            timeline = []

        students_data.append({
            "student_id": sid,
            "nombre":     slot["nombre"],
            "codigo":     slot["codigo"],
            "face_slot":  slot["face_slot"],
            "seat_label": slot.get("seat_label", ""),
            "summary":    summary,
            "timeline":   timeline,
        })
        if summary["avg_comprension"] > 0:
            all_avgs.append(summary["avg_comprension"])

    # Duracion de la sesion en minutos
    duracion_min = 0
    if session.get("started_at") and session.get("ended_at"):
        from datetime import datetime
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"):
            try:
                t0 = datetime.strptime(session["started_at"][:19], fmt)
                t1 = datetime.strptime(session["ended_at"][:19],   fmt)
                duracion_min = round((t1 - t0).total_seconds() / 60, 1)
                break
            except Exception:
                continue

    group_avg = round(sum(all_avgs) / len(all_avgs), 1) if all_avgs else 0

    return {
        "session":  dict(session),
        "students": students_data,
        "group": {
            "avg_comprension":  group_avg,
            "total_tiempo_min": duracion_min,
            "num_alumnos":      len(students_data),
        },
    }


@router.get("/{session_id}/csv")
async def export_csv(session_id: int,
                     authorization: Optional[str] = Header(None)):
    """Exporta todos los snapshots de telemetria como CSV descargable."""
    _verify(authorization)
    from app.storage.repositories.telemetry_repo import get_snapshots
    from app.storage.repositories.session_repo   import get_session, get_slots

    session = get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Sesion no encontrada")

    rows    = get_snapshots(session_id)
    slots   = get_slots(session_id)
    id_name = {s["student_id"]: s["nombre"] for s in slots}

    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=[
        "timestamp", "alumno", "student_id",
        "atencion", "indice_comprension", "emocion",
        "sentimiento", "mirada", "ear",
    ])
    writer.writeheader()
    for r in rows:
        writer.writerow({
            "timestamp":          r["timestamp"],
            "alumno":             id_name.get(r["student_id"], "Desconocido"),
            "student_id":         r["student_id"],
            "atencion":           r["atencion"],
            "indice_comprension": r["indice_comprension"],
            "emocion":            r["emocion"],
            "sentimiento":        r["sentimiento"],
            "mirada":             r["mirada"],
            "ear":                r["ear"],
        })

    output.seek(0)
    filename = f"reporte_sesion_{session_id}.csv"
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )
