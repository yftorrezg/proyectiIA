"""
training_api.py — Edu-Insight MLOps Fase 2
Endpoints para el Training Lab:

  POST /api/train/start          → Inicia entrenamiento (BackgroundTask)
  GET  /api/train/status         → Estado del job activo
  GET  /api/train/jobs           → Historial de todos los jobs
  GET  /api/train/metrics        → Lista los .json de métricas guardados en disco
  GET  /api/train/metrics/{file} → Devuelve un .json de métricas específico
  WS   /ws/training              → Stream de progreso en tiempo real
"""

import asyncio
import json
import logging
from pathlib import Path
from typing import Any, Dict

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from app.core.model_registry import MODEL_CATALOG, ModelCategory
from app.core.trainer import training_manager

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Training Lab"])

METRICS_DIR = Path("app/storage/metrics")


# ─────────────────────────────────────────────
#  SCHEMAS
# ─────────────────────────────────────────────

class TrainRequest(BaseModel):
    category: str                  # "atencion" | "emocion"
    model_id: str
    hyperparams: Dict[str, Any] = {}


# ─────────────────────────────────────────────
#  REST ENDPOINTS
# ─────────────────────────────────────────────

@router.post("/api/train/start", summary="Iniciar entrenamiento de un modelo")
async def start_training(request: TrainRequest):
    """
    Inicia un entrenamiento asíncrono en segundo plano.
    Solo puede haber un entrenamiento activo a la vez.

    Devuelve el job_id para correlacionar con los eventos del WebSocket.
    """
    # Validar categoría
    try:
        cat = ModelCategory(request.category)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Categoría inválida: '{request.category}'. Usa 'atencion' o 'emocion'.",
        )

    # Validar que el modelo existe y es entrenable
    catalog = MODEL_CATALOG.get(cat, [])
    config  = next((m for m in catalog if m.id == request.model_id), None)
    if config is None:
        raise HTTPException(
            status_code=404,
            detail=f"Modelo '{request.model_id}' no encontrado en categoría '{cat.value}'.",
        )
    if not config.trainable:
        raise HTTPException(
            status_code=400,
            detail=f"'{config.name}' no es entrenable. Solo se puede intercambiar.",
        )

    # Iniciar
    try:
        job = training_manager.start_training(
            category   = request.category,
            model_id   = request.model_id,
            hyperparams= request.hyperparams,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc))

    return {
        "ok":         True,
        "job_id":     job.job_id,
        "model_name": config.name,
        "category":   request.category,
        "message":    f"Entrenamiento de '{config.name}' iniciado. Conéctate a /ws/training para el progreso.",
    }


@router.get("/api/train/status", summary="Estado del entrenamiento activo")
async def get_training_status():
    """Snapshot del job actualmente en ejecución."""
    job = training_manager.current_job
    if job and training_manager.is_busy:
        return {
            "is_busy":      True,
            "job_id":       job.job_id,
            "model_id":     job.model_id,
            "category":     job.category,
            "status":       job.status,
            "progress":     job.progress,
            "current_epoch":job.current_epoch,
            "total_epochs": job.total_epochs,
            "message":      job.message,
        }
    return {"is_busy": False, "job": None}


@router.get("/api/train/jobs", summary="Historial de entrenamientos de la sesión")
async def get_all_jobs():
    """
    Devuelve todos los jobs iniciados en esta sesión del servidor
    (se pierde al reiniciar; los datos permanentes están en /api/train/metrics).
    """
    jobs = [
        {
            "job_id":       j.job_id,
            "category":     j.category,
            "model_id":     j.model_id,
            "status":       j.status,
            "progress":     j.progress,
            "message":      j.message,
            "accuracy":     j.metrics.get("accuracy") if j.metrics else None,
            "f1_macro":     j.metrics.get("f1_macro") if j.metrics else None,
            "model_path":   j.model_path,
            "metrics_path": j.metrics_path,
            "created_at":   j.created_at,
            "completed_at": j.completed_at,
            "error":        j.error,
        }
        for j in sorted(
            training_manager.all_jobs.values(),
            key=lambda x: x.created_at,
            reverse=True,
        )
    ]
    return {"total": len(jobs), "jobs": jobs}


@router.get("/api/train/metrics", summary="Listar métricas guardadas en disco")
async def list_metrics():
    """
    Lee todos los archivos .json de métricas guardados en app/storage/metrics/
    y los devuelve ordenados por fecha (más reciente primero).
    """
    if not METRICS_DIR.exists():
        return {"total": 0, "metrics": []}

    files   = sorted(METRICS_DIR.glob("*.json"), key=lambda f: f.stat().st_mtime, reverse=True)
    results = []
    for f in files:
        try:
            with open(f, encoding="utf-8") as fp:
                data = json.load(fp)
            data["filename"] = f.name
            results.append(data)
        except Exception as exc:
            logger.warning(f"No se pudo leer {f.name}: {exc}")

    return {"total": len(results), "metrics": results}


@router.get("/api/train/metrics/{filename}", summary="Métricas de un entrenamiento específico")
async def get_metric_file(filename: str):
    """Devuelve el JSON de métricas de un entrenamiento guardado."""
    safe_name = Path(filename).name
    path = METRICS_DIR / safe_name

    if not path.exists() or path.suffix != ".json":
        raise HTTPException(status_code=404, detail=f"Archivo no encontrado: {safe_name}")

    with open(path, encoding="utf-8") as f:
        return json.load(f)


@router.delete("/api/train/models/{stem}", summary="Eliminar un modelo entrenado y sus métricas")
async def delete_trained_model(stem: str):
    """
    Elimina el archivo de modelo (.joblib o .pt) Y su .json de métricas
    usando el nombre base del archivo (sin extensión).

    Ejemplo: DELETE /api/train/models/xgboost_20260420_231447
    """
    safe_stem = Path(stem).stem   # evita path traversal

    # Verificar que el modelo no está activo en producción
    try:
        from app.core.model_registry import registry, ModelCategory
        for cat in [ModelCategory.ATENCION, ModelCategory.EMOCION]:
            cfg = registry.active_configs[cat]
            active_path = cfg.metadata.get("trained_model_path", "")
            if active_path and Path(active_path).stem == safe_stem:
                raise HTTPException(
                    status_code=409,
                    detail=f"No puedes eliminar '{safe_stem}': está activo en producción. Activa otro modelo primero.",
                )
    except HTTPException:
        raise
    except Exception:
        pass

    deleted = []
    not_found = []

    # Eliminar .json de métricas
    json_path = METRICS_DIR / f"{safe_stem}.json"
    if json_path.exists():
        json_path.unlink()
        deleted.append(str(json_path))
    else:
        not_found.append(str(json_path))

    # Eliminar modelo (.joblib o .pt)
    STORAGE_DIR = Path("app/storage/trained_models")
    for ext in [".joblib", ".pt"]:
        model_path = STORAGE_DIR / f"{safe_stem}{ext}"
        if model_path.exists():
            model_path.unlink()
            deleted.append(str(model_path))
            break
    else:
        not_found.append(f"{safe_stem}.joblib/.pt")

    if not deleted:
        raise HTTPException(status_code=404, detail=f"No se encontraron archivos para '{safe_stem}'.")

    return {
        "ok": True,
        "deleted": deleted,
        "not_found": not_found,
        "message": f"Modelo '{safe_stem}' eliminado correctamente.",
    }


@router.get("/api/train/active-info", summary="Info del modelo de atención activo en producción")
async def get_active_model_info():
    """
    Devuelve qué modelo de atención está activo, si tiene métricas disponibles,
    y si el motor de inferencia lo tiene cargado.
    """
    from app.core.model_registry import registry, ModelCategory
    from app.core.inference_engine import estado_api_global

    cfg          = registry.active_configs[ModelCategory.ATENCION]
    trained_path = cfg.metadata.get("trained_model_path", "")

    # Buscar métricas del modelo activo
    metrics = None
    if trained_path:
        stem = Path(trained_path).stem
        for p in [
            Path("app/storage/metrics") / f"{stem}.json",
            Path(trained_path).with_suffix(".json"),
        ]:
            if p.exists():
                with open(p, encoding="utf-8") as f:
                    metrics = json.load(f)
                break

    # Fallback a métricas del modelo original
    if not metrics:
        fallback = Path("reports/metricas.json")
        if fallback.exists():
            with open(fallback, encoding="utf-8") as f:
                metrics = json.load(f)

    return {
        "model_id":       cfg.id,
        "model_name":     cfg.name,
        "trained_path":   trained_path or None,
        "is_custom":      bool(trained_path),
        "ml_disponible":  estado_api_global.get("ml_disponible", False),
        "modo_atencion":  estado_api_global.get("modo_atencion", "heuristico"),
        "metrics_summary": {
            "accuracy":  metrics.get("accuracy")  if metrics else None,
            "f1_macro":  metrics.get("f1_macro")  if metrics else None,
            "trained_at": metrics.get("trained_at") if metrics else None,
        } if metrics else None,
    }


# ─────────────────────────────────────────────
#  WEBSOCKET /ws/training
# ─────────────────────────────────────────────

@router.websocket("/ws/training")
async def training_websocket(websocket: WebSocket):
    """
    WebSocket de progreso en tiempo real.

    Eventos que emite el servidor → cliente:

      {"type": "start",    "job_id": "...", "message": "..."}
      {"type": "progress", "job_id": "...", "progress": 0-100,
       "current_epoch": N, "total_epochs": N, "message": "...",
       "train_loss": X, "val_loss": X, "val_acc": X}   ← campos opcionales
      {"type": "complete", "job_id": "...", "progress": 100,
       "metrics": {...}, "model_path": "..."}
      {"type": "error",    "job_id": "...", "error": "..."}
      {"type": "idle",     "is_busy": false}            ← heartbeat cuando no hay job
    """
    await websocket.accept()
    logger.info("[WS/training] Cliente conectado.")

    try:
        while True:
            try:
                # Leer del queue sin bloquear
                event = training_manager.progress_queue.get_nowait()
                await websocket.send_json(event)

            except Exception:
                # Queue vacío — esperar antes de reintentar
                await asyncio.sleep(0.15)

                # Heartbeat cuando el sistema está en reposo
                if not training_manager.is_busy:
                    try:
                        await websocket.send_json({"type": "idle", "is_busy": False})
                    except Exception:
                        break           # cliente desconectado
                    await asyncio.sleep(3.0)   # heartbeat cada 3s en idle

    except WebSocketDisconnect:
        logger.info("[WS/training] Cliente desconectado normalmente.")
    except Exception as exc:
        logger.exception(f"[WS/training] Error inesperado: {exc}")
        try:
            await websocket.close(code=1011)
        except Exception:
            pass
