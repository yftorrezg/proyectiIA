"""
sessions_api.py — Gestion de sesiones de clase.

  GET  /api/sessions              Lista sesiones del tutor
  POST /api/sessions              Crear sesion + iniciar telemetria
  GET  /api/sessions/active       Sesion activa actual
  POST /api/sessions/{id}/close   Cerrar sesion y detener telemetria
  GET  /api/sessions/{id}/slots   Slots (cara↔alumno) de la sesion
"""
from typing import Optional, List
from fastapi import APIRouter, Header, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

router = APIRouter(prefix="/api/sessions", tags=["Sessions"])


def _tutor_id(authorization: Optional[str]) -> int:
    from app.storage.repositories.tutor_repo import verify_token
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Token requerido")
    token  = authorization.split(" ", 1)[1]
    tutor  = verify_token(token)
    if not tutor:
        raise HTTPException(status_code=401, detail="Token invalido o expirado")
    return tutor["id"]


class SlotBody(BaseModel):
    student_id: int
    face_slot:  int          # 0-5
    seat_label: str = ""


class SessionBody(BaseModel):
    titulo:  str
    materia: str = ""
    slots:   List[SlotBody]  # minimo 1 alumno


class UpdateSessionBody(BaseModel):
    titulo:  str
    materia: str = ""


class ReorderBody(BaseModel):
    assignments: List[SlotBody]  # nueva asignacion face_slot ↔ student_id


@router.get("")
async def list_sessions(authorization: Optional[str] = Header(None)):
    tutor_id = _tutor_id(authorization)
    from app.storage.repositories.session_repo import get_sessions_by_tutor
    return get_sessions_by_tutor(tutor_id)


@router.post("")
async def create_session(body: SessionBody,
                         authorization: Optional[str] = Header(None)):
    tutor_id = _tutor_id(authorization)
    from app.storage.repositories.session_repo  import create_session, add_slot, get_active_session
    from app.storage.repositories.student_repo  import get_student_by_id
    from app.core.telemetry_writer               import telemetry_writer
    from app.core.inference_engine               import engine

    titulo  = body.titulo.strip()
    materia = body.materia.strip()

    if not titulo:
        return JSONResponse({"ok": False, "error": "El titulo es obligatorio"}, status_code=400)
    if not body.slots:
        return JSONResponse({"ok": False, "error": "Debes asignar al menos 1 alumno"}, status_code=400)

    # Si habia sesion activa del mismo tutor la cerramos
    active = get_active_session(tutor_id)
    if active:
        from app.storage.repositories.session_repo import close_session
        close_session(active["id"])
        telemetry_writer.stop_session()

    # Crear sesion
    session_id = create_session(tutor_id, titulo, materia)

    # Agregar slots
    slots_info = []
    for s in body.slots:
        student = get_student_by_id(s.student_id)
        if not student:
            continue
        add_slot(session_id, s.student_id, s.face_slot, s.seat_label)
        slots_info.append({
            "face_slot":  s.face_slot,
            "student_id": s.student_id,
            "nombre":     student["nombre"],
        })

    # Ajustar FaceMesh al numero de alumnos
    n_faces = max(1, len(slots_info))
    try:
        engine.set_max_faces(n_faces)
    except Exception:
        pass

    # Iniciar escritura de telemetria
    telemetry_writer.start_session(session_id, slots_info)

    return {
        "ok":        True,
        "session_id": session_id,
        "titulo":    titulo,
        "slots":     slots_info,
    }


@router.get("/active")
async def active_session(authorization: Optional[str] = Header(None)):
    tutor_id = _tutor_id(authorization)
    from app.storage.repositories.session_repo import get_active_session, get_slots
    session = get_active_session(tutor_id)
    if not session:
        return {"active": False}
    slots = get_slots(session["id"])
    return {"active": True, "session": session, "slots": slots}


@router.post("/{session_id}/close")
async def close_session(session_id: int,
                        authorization: Optional[str] = Header(None)):
    _tutor_id(authorization)
    from app.storage.repositories.session_repo import close_session as _close
    from app.core.telemetry_writer              import telemetry_writer
    from app.core.inference_engine              import engine

    _close(session_id)
    telemetry_writer.stop_session()

    # Volver a 1 cara
    try:
        engine.set_max_faces(1)
    except Exception:
        pass

    return {"ok": True, "session_id": session_id}


@router.get("/{session_id}/slots")
async def session_slots(session_id: int,
                        authorization: Optional[str] = Header(None)):
    _tutor_id(authorization)
    from app.storage.repositories.session_repo import get_slots
    return get_slots(session_id)


@router.patch("/{session_id}")
async def update_session(session_id: int,
                         body: UpdateSessionBody,
                         authorization: Optional[str] = Header(None)):
    """Edita título y materia de una sesión."""
    _tutor_id(authorization)
    titulo = body.titulo.strip()
    if not titulo:
        return JSONResponse({"ok": False, "error": "El titulo es obligatorio"}, status_code=400)
    from app.storage.repositories.session_repo import update_session as _update
    _update(session_id, titulo, body.materia.strip())
    return {"ok": True}


@router.delete("/{session_id}")
async def delete_session_endpoint(session_id: int,
                                  authorization: Optional[str] = Header(None)):
    """Elimina una sesión y todos sus slots."""
    _tutor_id(authorization)
    from app.storage.repositories.session_repo import get_session, delete_session as _delete
    from app.core.telemetry_writer import telemetry_writer
    session = get_session(session_id)
    if session and session.get("status") == "activa":
        telemetry_writer.stop_session()
    _delete(session_id)
    return {"ok": True}


@router.patch("/{session_id}/slots/reorder")
async def reorder_session_slots(session_id: int,
                                body: ReorderBody,
                                authorization: Optional[str] = Header(None)):
    """
    Reasigna en caliente que alumno corresponde a cada posicion de camara (face_slot).
    Actualiza la DB y el overlay de video de forma inmediata.
    """
    _tutor_id(authorization)
    from app.storage.repositories.session_repo  import reorder_slots, get_slots
    from app.storage.repositories.student_repo  import get_student_by_id
    from app.core.telemetry_writer               import telemetry_writer

    assignments = [{"face_slot": s.face_slot, "student_id": s.student_id} for s in body.assignments]
    updated_slots = reorder_slots(session_id, assignments)

    # Actualizar TelemetryWriter + slot_map del inference engine
    if telemetry_writer.active and telemetry_writer.session_id == session_id:
        telemetry_writer.update_slots(updated_slots)

    return {"ok": True, "slots": updated_slots}
