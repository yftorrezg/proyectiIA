"""
auth_api.py — Autenticacion biometrica facial para tutores.

Endpoints:
  POST /api/auth/enroll   Registrar tutor con embedding facial + liveness
  POST /api/auth/verify   Login facial con liveness → devuelve token
  GET  /api/auth/me       Info del tutor autenticado
  POST /api/auth/logout   Cierra sesion (invalida token)
  GET  /api/auth/status   Estado del motor para la UI de login

Liveness:
  Usa el EAR (Eye Aspect Ratio) calculado por el InferenceEngine en tiempo
  real. Una foto o pantalla no parpadea → EAR constante → liveness FALLIDA.
  Umbral EAR: 0.22  (ya definido en inference_engine.py)
"""

import asyncio
import logging
import time
from typing import Optional

import cv2
import numpy as np
from fastapi import APIRouter, Header, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/auth", tags=["Auth"])

# Umbral EAR — debe coincidir con inference_engine.EAR_UMBRAL
_EAR_CLOSED    = 0.22
_LIVENESS_SECS = 12.0   # segundos maximos esperando un parpadeo


# ── Schemas ──────────────────────────────────────────────────────────────────

class EnrollBody(BaseModel):
    nombre:   str
    username: str

class UpdateTutorBody(BaseModel):
    nombre: str


# ── Helpers internos ─────────────────────────────────────────────────────────

def _get_current_frame() -> Optional[np.ndarray]:
    """Decodifica el JPEG del motor al array BGR de OpenCV."""
    import app.core.inference_engine as ie
    fb = ie.frame_global_bytes
    if fb is None:
        return None
    arr = np.frombuffer(fb, np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


async def _wait_for_blink(timeout: float = _LIVENESS_SECS) -> bool:
    """
    Monitorea estado_api_global['ear'] hasta detectar un parpadeo completo.
    Parpadeo = transicion ojos_cerrados → ojos_abiertos (despues de haber
    estado cerrados al menos un ciclo).

    Devuelve True si detecta parpadeo dentro del timeout, False si no.
    """
    import app.core.inference_engine as ie
    deadline    = time.monotonic() + timeout
    prev_closed = False

    while time.monotonic() < deadline:
        ear = ie.estado_api_global.get("ear", 0.0)

        # EAR = 0.0 → motor no listo o sin cara → esperar
        if ear < 0.01:
            await asyncio.sleep(0.1)
            continue

        curr_closed = ear < _EAR_CLOSED

        # Deteccion del cierre y reapertura
        if prev_closed and not curr_closed:
            return True   # parpadeo completado

        prev_closed = curr_closed
        await asyncio.sleep(0.04)   # polling a 25 Hz

    return False


async def _extract_embedding(frame: np.ndarray) -> Optional[np.ndarray]:
    """
    Extrae el embedding Facenet512 via DeepFace.
    Intenta multiples backends en orden de robustez.
    enforce_detection=False evita que falle en caras grandes/cercanas
    o frames con overlays del inference engine.
    """
    loop = asyncio.get_event_loop()

    def _run() -> Optional[np.ndarray]:
        from deepface import DeepFace
        # retinaface: mas robusto que opencv para caras cercanas/con overlays
        # opencv: rapido, fallback
        # skip: usa imagen completa — funciona cuando MediaPipe ya confirmo la cara
        backends = ["retinaface", "opencv", "skip"]
        for backend in backends:
            try:
                result = DeepFace.represent(
                    frame,
                    model_name        = "Facenet512",
                    enforce_detection = False,   # no fallar si el detector no confía al 100%
                    detector_backend  = backend,
                )
                if result and len(result) > 0:
                    emb = result[0].get("embedding", [])
                    if len(emb) == 512:
                        logger.info(f"Embedding extraido con backend '{backend}'")
                        return np.array(emb, dtype=np.float32)
            except Exception as exc:
                logger.warning(f"Backend '{backend}' fallo: {exc}")
                continue
        logger.error("Todos los backends fallaron para extraer embedding")
        return None

    return await loop.run_in_executor(None, _run)


def _parse_token(authorization: Optional[str]) -> str:
    """Extrae el Bearer token del header Authorization."""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Token requerido")
    return authorization.split(" ", 1)[1]


# ── Endpoints ────────────────────────────────────────────────────────────────

@router.get("/tutors")
async def list_tutors():
    """Lista publica de tutores registrados (solo nombre y username, sin datos biometricos)."""
    from app.storage.database import get_connection
    conn = get_connection()
    rows = conn.execute(
        "SELECT id, nombre, username, registered_at, last_login FROM tutors ORDER BY nombre"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


@router.patch("/tutors/{tutor_id}")
async def update_tutor(tutor_id: int, body: UpdateTutorBody):
    """Actualiza el nombre de un tutor."""
    nombre = body.nombre.strip()
    if not nombre:
        raise HTTPException(status_code=400, detail="Nombre no puede estar vacio")
    from app.storage.repositories.tutor_repo import update_tutor_nombre
    updated = update_tutor_nombre(tutor_id, nombre)
    if not updated:
        raise HTTPException(status_code=404, detail="Tutor no encontrado")
    return {"ok": True, "tutor_id": tutor_id, "nombre": nombre}


@router.delete("/tutors/{tutor_id}")
async def delete_tutor(tutor_id: int):
    """Elimina un tutor y todos sus datos asociados (alumnos, sesiones, tokens) en cascada."""
    from app.storage.repositories.tutor_repo import delete_tutor as _del
    deleted = _del(tutor_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Tutor no encontrado")
    return {"ok": True, "deleted_id": tutor_id}


@router.get("/debug")
async def auth_debug():
    """
    Diagnostico: extrae el embedding del frame actual y muestra la similitud
    con todos los tutores registrados. Util para afinar el umbral.
    GET /api/auth/debug
    """
    from app.storage.repositories.tutor_repo import get_connection
    import pickle

    frame = _get_current_frame()
    if frame is None:
        return {"error": "Sin frame de camara"}

    embedding = await _extract_embedding(frame)
    if embedding is None:
        return {"error": "No se pudo extraer embedding del frame actual"}

    conn = get_connection()
    rows = conn.execute("SELECT id, nombre, username, embedding FROM tutors").fetchall()
    conn.close()

    if not rows:
        return {"error": "No hay tutores registrados"}

    results = []
    for row in rows:
        stored = pickle.loads(row["embedding"])
        from numpy.linalg import norm
        import numpy as np
        na = norm(embedding); nb = norm(stored)
        sim = float(np.dot(embedding, stored) / (na * nb)) if na > 0 and nb > 0 else 0.0
        results.append({
            "tutor_id":   row["id"],
            "nombre":     row["nombre"],
            "username":   row["username"],
            "similarity": round(sim, 4),
            "would_match": sim >= 0.58,
        })

    results.sort(key=lambda x: x["similarity"], reverse=True)
    return {
        "embedding_dims": len(embedding),
        "threshold":      0.58,
        "matches":        results,
    }


@router.get("/status")
async def auth_status():
    """
    Devuelve el estado del motor para que la UI de login sepa
    si la camara esta activa y si hay cara detectada.
    """
    import app.core.inference_engine as ie
    ear = float(ie.estado_api_global.get("ear", 0.0))
    return {
        "engine_ready":   bool(ie.frame_global_bytes is not None),
        "face_detected":  bool(ear > 0.01),
        "ear":            round(ear, 3),
        "eyes_closed":    bool(ear < _EAR_CLOSED and ear > 0.01),
    }


@router.post("/enroll")
async def enroll(body: EnrollBody):
    """
    Registra un tutor nuevo:
      1. Verifica que el username no exista
      2. Espera un parpadeo (liveness)
      3. Captura el frame actual
      4. Extrae embedding Facenet512
      5. Guarda en DB
    """
    from app.storage.repositories.tutor_repo import create_tutor, username_exists

    nombre   = body.nombre.strip()
    username = body.username.strip().lower()

    if not nombre or not username:
        return JSONResponse(
            {"ok": False, "error": "Nombre y username son obligatorios"},
            status_code=400,
        )

    if username_exists(username):
        return JSONResponse(
            {"ok": False, "error": "username_taken",
             "message": f"El usuario '{username}' ya esta registrado."},
            status_code=409,
        )

    # 1. Liveness
    blinked = await _wait_for_blink()
    if not blinked:
        return JSONResponse(
            {"ok": False, "error": "liveness_failed",
             "message": "No se detecto parpadeo. Mirá a la camara y parpadeá naturalmente."},
            status_code=400,
        )

    # 2. Frame
    frame = _get_current_frame()
    if frame is None:
        return JSONResponse(
            {"ok": False, "error": "no_frame",
             "message": "Sin señal de camara. Verifica que el motor este corriendo."},
            status_code=503,
        )

    # 3. Embedding
    embedding = await _extract_embedding(frame)
    if embedding is None:
        return JSONResponse(
            {"ok": False, "error": "no_face",
             "message": "No se detecto un rostro claro. Mejorá la iluminacion y centra tu cara."},
            status_code=400,
        )

    # 4. Guardar
    tutor_id = create_tutor(nombre, username, embedding)

    return JSONResponse({
        "ok":       True,
        "tutor_id": tutor_id,
        "nombre":   nombre,
        "message":  f"Registro exitoso. Bienvenido, {nombre}.",
    })


@router.post("/verify")
async def verify():
    """
    Login biometrico:
      1. Espera parpadeo (liveness — rechaza fotos/pantallas)
      2. Captura frame y extrae embedding
      3. Busca el mejor match en DB
      4. Si similitud >= 0.68 → genera token y devuelve info
    """
    from app.storage.repositories.tutor_repo import (
        find_matching_tutor, create_session_token, get_tutor_count
    )

    if get_tutor_count() == 0:
        return JSONResponse(
            {"ok": False, "error": "no_tutors",
             "message": "No hay tutores registrados. Ve a /register para crear tu cuenta."},
            status_code=404,
        )

    # 1. Liveness
    blinked = await _wait_for_blink()
    if not blinked:
        return JSONResponse(
            {"ok": False, "error": "liveness_failed",
             "message": "Liveness no detectada. Una foto no puede iniciar sesion — parpadeá."},
            status_code=401,
        )

    # 2. Frame
    frame = _get_current_frame()
    if frame is None:
        return JSONResponse(
            {"ok": False, "error": "no_frame",
             "message": "Sin señal de camara."},
            status_code=503,
        )

    # 3. Embedding
    embedding = await _extract_embedding(frame)
    if embedding is None:
        return JSONResponse(
            {"ok": False, "error": "no_face",
             "message": "No se detecto un rostro claro. Mejorá la iluminacion e intentá de nuevo."},
            status_code=400,
        )

    # 4. Match
    match = find_matching_tutor(embedding)
    if not match:
        return JSONResponse(
            {"ok": False, "error": "no_match",
             "message": "Rostro no reconocido. Si sos nuevo, registrate en /register."},
            status_code=401,
        )

    token = create_session_token(match["id"])

    return JSONResponse({
        "ok":        True,
        "token":     token,
        "tutor_id":  match["id"],
        "nombre":    match["nombre"],
        "username":  match["username"],
        "similarity": match["similarity"],
        "message":   f"Bienvenido, {match['nombre']}",
    })


@router.get("/me")
async def me(authorization: Optional[str] = Header(None)):
    """Devuelve info del tutor autenticado segun el Bearer token."""
    from app.storage.repositories.tutor_repo import verify_token
    token = _parse_token(authorization)
    tutor = verify_token(token)
    if not tutor:
        raise HTTPException(status_code=401, detail="Token invalido o expirado")
    return tutor


@router.post("/logout")
async def logout(authorization: Optional[str] = Header(None)):
    """Invalida el token de sesion."""
    from app.storage.repositories.tutor_repo import delete_token
    token = _parse_token(authorization)
    delete_token(token)
    return {"ok": True}
