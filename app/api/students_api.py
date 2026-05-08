"""
students_api.py — CRUD de alumnos (protegido por token).

  GET  /api/students          Lista alumnos del tutor autenticado
  POST /api/students          Crear alumno nuevo
  DELETE /api/students/{id}   Eliminar alumno
"""
from typing import Optional
from fastapi import APIRouter, Header, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

router = APIRouter(prefix="/api/students", tags=["Students"])


def _tutor_id(authorization: Optional[str]) -> int:
    """Extrae tutor_id del token. Lanza 401 si invalido."""
    from app.storage.repositories.tutor_repo import verify_token
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Token requerido")
    token = authorization.split(" ", 1)[1]
    tutor = verify_token(token)
    if not tutor:
        raise HTTPException(status_code=401, detail="Token invalido o expirado")
    return tutor["id"]


class StudentBody(BaseModel):
    nombre: str
    codigo: str

class UpdateStudentBody(BaseModel):
    nombre: str
    codigo: str


@router.get("")
async def list_students(authorization: Optional[str] = Header(None)):
    tutor_id = _tutor_id(authorization)
    from app.storage.repositories.student_repo import get_students_by_tutor
    return get_students_by_tutor(tutor_id)


@router.post("")
async def create_student(body: StudentBody,
                         authorization: Optional[str] = Header(None)):
    tutor_id = _tutor_id(authorization)
    from app.storage.repositories.student_repo import create_student, codigo_exists

    nombre = body.nombre.strip()
    codigo = body.codigo.strip().upper()

    if not nombre or not codigo:
        return JSONResponse({"ok": False, "error": "Nombre y codigo son obligatorios"}, status_code=400)
    if codigo_exists(codigo):
        return JSONResponse({"ok": False, "error": f"El codigo '{codigo}' ya existe"}, status_code=409)

    sid = create_student(nombre, codigo, tutor_id)
    return {"ok": True, "student_id": sid, "nombre": nombre, "codigo": codigo}


@router.patch("/{student_id}")
async def update_student_endpoint(student_id: int, body: UpdateStudentBody,
                                  authorization: Optional[str] = Header(None)):
    _tutor_id(authorization)
    nombre = body.nombre.strip()
    codigo = body.codigo.strip().upper()
    if not nombre or not codigo:
        raise HTTPException(status_code=400, detail="Nombre y codigo son obligatorios")
    from app.storage.repositories.student_repo import update_student
    updated = update_student(student_id, nombre, codigo)
    if not updated:
        raise HTTPException(status_code=404, detail="Alumno no encontrado")
    return {"ok": True, "student_id": student_id, "nombre": nombre, "codigo": codigo}


@router.delete("/{student_id}")
async def delete_student(student_id: int,
                         authorization: Optional[str] = Header(None)):
    _tutor_id(authorization)
    from app.storage.repositories.student_repo import delete_student
    delete_student(student_id)
    return {"ok": True}
