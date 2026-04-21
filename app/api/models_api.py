"""
models_api.py — Edu-Insight MLOps
Endpoints REST para gestionar el ModelRegistry:
  GET  /api/models              → catálogo completo + estado
  GET  /api/models/status       → VRAM + modelos activos
  GET  /api/models/catalog      → solo el catálogo
  POST /api/models/active       → cambiar modelo activo (hot-swap)
  GET  /api/models/hyperparams  → guía de hiperparámetros de un modelo entrenable
"""

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.core.model_registry import ModelCategory, registry

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/models", tags=["Model Registry"])


# ─────────────────────────────────────────────
#  SCHEMAS
# ─────────────────────────────────────────────

class ActivateModelRequest(BaseModel):
    category: str       # "atencion" | "audio" | "emocion" | "semantica"
    model_id: str       # id del modelo del catálogo
    trained_model_path: Optional[str] = None  # ruta a un .joblib/.pt entrenado


# ─────────────────────────────────────────────
#  ENDPOINTS
# ─────────────────────────────────────────────

@router.get("/", summary="Catálogo completo + estado del registry")
async def get_models():
    """
    Retorna el catálogo de modelos disponibles por categoría
    junto con el estado actual del sistema (VRAM, GPU, modelos activos).
    """
    return {
        "catalog": registry.get_catalog(),
        "status":  registry.get_status(),
    }


@router.get("/status", summary="Estado de VRAM y modelos activos")
async def get_status():
    """
    Snapshot rápido del estado del sistema:
    - GPU disponible y nombre
    - VRAM libre / usada / total
    - Modelo activo por categoría
    """
    return registry.get_status()


@router.get("/catalog", summary="Solo el catálogo de modelos")
async def get_catalog():
    """
    Devuelve únicamente el catálogo de modelos disponibles,
    marcando cuál está activo en cada categoría.
    """
    return registry.get_catalog()


@router.post("/active", summary="Activar un modelo (hot-swap)")
async def set_active_model(request: ActivateModelRequest):
    """
    Cambia el modelo activo de una categoría.

    - Verifica VRAM disponible antes de cargar el nuevo modelo.
    - Descarga la instancia anterior liberando VRAM.
    - Acepta opcionalmente la ruta a un modelo entrenado propio.

    Categorías válidas: atencion | audio | emocion | semantica
    """
    # Validar categoría
    try:
        category = ModelCategory(request.category)
    except ValueError:
        valid = [c.value for c in ModelCategory]
        raise HTTPException(
            status_code=400,
            detail=f"Categoría inválida: '{request.category}'. Válidas: {valid}",
        )

    # Intentar hot-swap
    try:
        config = registry.set_active_model(
            category=category,
            model_id=request.model_id,
            trained_model_path=request.trained_model_path,
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=507, detail=str(e))
    except Exception as e:
        logger.exception("Error inesperado en hot-swap")
        raise HTTPException(status_code=500, detail=f"Error interno: {e}")

    # ── Notificar al motor de inferencia directamente ──────────────────────────
    engine_reloaded = False
    engine_error    = None
    try:
        from app.core.inference_engine import engine, estado_api_global
        if category == ModelCategory.ATENCION:
            engine.reload_attention_model()
            engine_reloaded = True
        elif category == ModelCategory.EMOCION:
            if request.trained_model_path:
                engine.load_emocion_cnn(request.trained_model_path, request.model_id)
            else:
                # Volver a DeepFace (modelo base)
                engine._emocion_cnn = None
                estado_api_global["emocion_model"] = request.model_id
            engine_reloaded = True
        elif category == ModelCategory.AUDIO:
            estado_api_global["audio_model"] = request.model_id
            engine_reloaded = True
        elif category == ModelCategory.SEMANTICA:
            estado_api_global["semantica_model"] = request.model_id
            engine_reloaded = True
    except Exception as e:
        engine_error = str(e)
        logger.warning(f"Motor de inferencia no pudo recargar: {e}")

    try:
        from app.core.inference_engine import estado_api_global as eg
    except Exception:
        eg = {}
    return {
        "ok": True,
        "message": f"Modelo '{config.name}' activado en '{category.value}'.",
        "active_model": {
            "id":        config.id,
            "name":      config.name,
            "vram_mb":   config.vram_mb,
            "trainable": config.trainable,
            "tags":      config.tags,
        },
        "vram":            registry.get_vram_info(),
        "engine_reloaded": engine_reloaded,
        "engine_error":    engine_error,
        "ml_disponible":   eg.get("ml_disponible", False),
        "modelo_activo":   eg.get("modelo_activo", "—"),
        "audio_model":     eg.get("audio_model", "—"),
        "emocion_model":   eg.get("emocion_model", "—"),
        "semantica_model": eg.get("semantica_model", "—"),
    }


@router.get("/hyperparams", summary="Guía de hiperparámetros de un modelo entrenable")
async def get_hyperparams(category: str, model_id: str):
    """
    Devuelve los hiperparámetros ajustables de un modelo entrenable
    junto con la guía de recomendaciones para el Training Lab.

    Parámetros query:
      - category: categoría del modelo (atencion | emocion)
      - model_id: id del modelo
    """
    try:
        cat = ModelCategory(category)
    except ValueError:
        valid = [c.value for c in ModelCategory]
        raise HTTPException(
            status_code=400,
            detail=f"Categoría inválida: '{category}'. Válidas: {valid}",
        )

    try:
        guide = registry.get_hyperparams_guide(cat, model_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    return guide


@router.get("/vram", summary="Estado actual de VRAM en MB")
async def get_vram():
    """Endpoint ligero para polling de VRAM desde el frontend."""
    return registry.get_vram_info()


class FacesRequest(BaseModel):
    n: int   # número de rostros (1-6)


@router.post("/faces", summary="Cambiar número máximo de rostros detectados")
async def set_max_faces(request: FacesRequest):
    """
    Reinicializa MediaPipe FaceMesh con max_num_faces = n (entre 1 y 6).
    Permite analizar landmark y emociones de múltiples personas.
    """
    from app.core.inference_engine import engine, estado_api_global
    n = max(1, min(request.n, 6))
    engine.set_max_faces(n)
    return {
        "ok": True,
        "max_num_faces": estado_api_global.get("max_num_faces", n),
        "message": f"FaceMesh configurado para {n} rostro(s).",
    }
