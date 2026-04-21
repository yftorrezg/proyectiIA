"""
telemetry.py — Edu-Insight MLOps
Endpoints de telemetría en tiempo real (idénticos a pruebas.py):

  GET  /api/video_feed   → MJPEG stream de la cámara
  WS   /ws               → JSON con estado cognitivo a 10 Hz
  POST /api/modo         → Cambiar modo heurístico ↔ ML
  GET  /api/ml_metricas  → JSON de métricas del modelo activo
  POST /api/models/reload → Recarga el modelo de atención (post hot-swap)
"""

import asyncio
import json
import logging
import time
from pathlib import Path

from fastapi import APIRouter, Request, WebSocket
from fastapi.responses import JSONResponse, StreamingResponse

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Telemetry"])


# ── MJPEG video feed ──────────────────────────────────────────────────────────

def _generador_video():
    from app.core.inference_engine import frame_global_bytes
    while True:
        # Acceder al global actualizado en tiempo real
        import app.core.inference_engine as ie
        fb = ie.frame_global_bytes
        if fb is not None:
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + fb + b"\r\n"
        time.sleep(0.033)   # ~30 fps


@router.get("/api/video_feed", summary="MJPEG stream de la cámara")
def video_feed():
    return StreamingResponse(
        _generador_video(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


# ── WebSocket /ws — telemetría a 10 Hz ────────────────────────────────────────

@router.websocket("/ws")
async def websocket_telemetry(websocket: WebSocket):
    """
    Emite el estado cognitivo completo a 10 Hz.
    Payload JSON:
      emocion, texto, sentimiento, atencion, mirada, ear,
      indice_comprension, fps, modo_atencion, ml_confidence,
      ml_disponible, modelo_activo
    """
    await websocket.accept()
    try:
        while True:
            from app.core.inference_engine import estado_api_global
            await websocket.send_json(estado_api_global)
            await asyncio.sleep(0.1)
    except Exception:
        pass


# ── Cambio de modo ─────────────────────────────────────────────────────────────

@router.post("/api/modo", summary="Cambiar modo de atención: heuristico | ml")
async def cambiar_modo(request: Request):
    try:
        body = await request.json()
        modo = body.get("modo", "heuristico")
        if modo not in ("heuristico", "ml"):
            return JSONResponse({"ok": False, "error": "Modo inválido"}, status_code=400)

        from app.core.inference_engine import estado_api_global
        estado_api_global["modo_atencion"] = modo
        return JSONResponse({"ok": True, "modo": modo})
    except Exception as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=400)


# ── Métricas del modelo activo ─────────────────────────────────────────────────

@router.get("/api/ml_metricas", summary="JSON de métricas del modelo de atención activo")
def ml_metricas():
    """
    Prioridad:
      1. Métricas del modelo entrenado activo (app/storage/metrics/)
      2. Métricas del modelo original (reports/metricas.json)
    """
    # Buscar métricas del modelo entrenado activo
    try:
        from app.core.model_registry import registry, ModelCategory
        cfg = registry.active_configs[ModelCategory.ATENCION]
        trained_path = cfg.metadata.get("trained_model_path", "")
        if trained_path:
            # Buscar el .json correspondiente al .joblib
            metrics_path = Path(trained_path).with_suffix(".json")
            alt_path = Path("app/storage/metrics") / (Path(trained_path).stem + ".json")
            for p in [metrics_path, alt_path]:
                if p.exists():
                    with open(p, encoding="utf-8") as f:
                        return JSONResponse(json.load(f))
    except Exception:
        pass

    # Fallback: metricas originales
    fallback = Path("reports/metricas.json")
    if fallback.exists():
        with open(fallback, encoding="utf-8") as f:
            return JSONResponse(json.load(f))

    return JSONResponse({"error": "No hay métricas disponibles"}, status_code=404)


# ── Reload modelo tras hot-swap ────────────────────────────────────────────────

@router.post("/api/models/reload", summary="Recargar el modelo de atención tras hot-swap")
async def reload_model():
    """
    Llamado automáticamente después de POST /api/models/active
    para que el motor de inferencia cargue el nuevo modelo sin reiniciar.
    """
    try:
        from app.core.inference_engine import engine
        engine.reload_attention_model()
        from app.core.inference_engine import estado_api_global
        return JSONResponse({
            "ok": True,
            "modelo_activo": estado_api_global.get("modelo_activo", "desconocido"),
            "ml_disponible": estado_api_global.get("ml_disponible", False),
        })
    except Exception as exc:
        logger.exception("Error en reload_model")
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=500)
