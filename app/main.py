"""
main.py — Edu-Insight MLOps v2  (punto de entrada único)
Reemplaza pruebas.py. Integra en un solo servidor:
  - Motor de inferencia (cámara + audio + IA)
  - Model Registry + hot-swap
  - Training Lab (entrenamiento asíncrono + /ws/training)
  - Dashboard y Admin Panel

Ejecutar:
    uvicorn app.main:app --host 0.0.0.0 --port 8080

Para desarrollo con recarga automática (sin --reload por los hilos de cámara):
    python -m app.main
"""

# ══════════════════════════════════════════════════════════════
#  0. ENTORNO Y CUDA — debe ir ANTES de cualquier import de IA
# ══════════════════════════════════════════════════════════════
import os
import sys
import site
import socket
import subprocess
import getpass
import logging

getpass.getuser = lambda: "Usuario_EduInsight"
os.environ["USERNAME"]                = "Usuario_EduInsight"
os.environ["USER"]                    = "Usuario_EduInsight"
os.environ["LOGNAME"]                 = "Usuario_EduInsight"
os.environ["TORCHINDUCTOR_CACHE_DIR"] = os.path.join(os.getcwd(), ".torch_cache")
os.environ["CUDA_VISIBLE_DEVICES"]    = "0"
os.environ["TF_ENABLE_ONEDNN_OPTS"]   = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"]    = "2"

# Inyector de DLLs NVIDIA para Windows (CUDA 11 + 12)
if os.name == "nt":
    for sp in site.getsitepackages():
        for lib in ["cublas", "cudnn"]:
            ruta = os.path.join(sp, "nvidia", lib, "bin")
            if os.path.exists(ruta):
                os.environ["PATH"] = ruta + os.pathsep + os.environ.get("PATH", "")
                if hasattr(os, "add_dll_directory"):
                    try:
                        os.add_dll_directory(ruta)
                    except Exception:
                        pass

# ══════════════════════════════════════════════════════════════
#  1. LOGGING
# ══════════════════════════════════════════════════════════════
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("edu_insight")

# ══════════════════════════════════════════════════════════════
#  2. FASTAPI
# ══════════════════════════════════════════════════════════════
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

app = FastAPI(
    title="Edu-Insight MLOps",
    description="Telemetría educativa en tiempo real con Model Registry y Training Lab.",
    version="2.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ══════════════════════════════════════════════════════════════
#  3. ROUTERS
# ══════════════════════════════════════════════════════════════
from app.api.models_api   import router as models_router
from app.api.training_api import router as training_router
from app.api.telemetry    import router as telemetry_router

app.include_router(models_router)
app.include_router(training_router)
app.include_router(telemetry_router)

# ══════════════════════════════════════════════════════════════
#  4. PÁGINAS HTML
# ══════════════════════════════════════════════════════════════
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@app.get("/", include_in_schema=False)
async def dashboard():
    # Busca primero en frontend/, luego en la raíz (compatibilidad con pruebas.py)
    for candidate in [
        os.path.join(_ROOT, "frontend", "dashboard.html"),
        os.path.join(_ROOT, "dashboard.html"),
    ]:
        if os.path.exists(candidate):
            return FileResponse(candidate)
    return JSONResponse({"error": "dashboard.html no encontrado"}, status_code=404)


@app.get("/admin", include_in_schema=False)
async def admin_panel():
    path = os.path.join(_ROOT, "frontend", "admin.html")
    if os.path.exists(path):
        return FileResponse(path)
    return JSONResponse({"error": "admin.html no encontrado"}, status_code=404)


# ══════════════════════════════════════════════════════════════
#  6. HEALTH CHECK
# ══════════════════════════════════════════════════════════════
@app.get("/api/health", tags=["System"])
async def health():
    from app.core.model_registry import registry
    from app.core.inference_engine import estado_api_global
    return {
        "status":  "ok",
        "version": "2.0.0",
        "registry": registry.get_status(),
        "inference": {
            "modo":        estado_api_global["modo_atencion"],
            "ml_disponible": estado_api_global["ml_disponible"],
            "modelo_activo": estado_api_global["modelo_activo"],
            "fps":         estado_api_global["fps"],
        },
    }


# ══════════════════════════════════════════════════════════════
#  7. STARTUP / SHUTDOWN
# ══════════════════════════════════════════════════════════════
@app.on_event("startup")
async def on_startup():
    logger.info("=" * 62)
    logger.info("  Edu-Insight MLOps v2.0 — Iniciando")
    logger.info("  Dashboard   : http://localhost:8080/")
    logger.info("  Admin Lab   : http://localhost:8080/admin")
    logger.info("  API Docs    : http://localhost:8080/docs")
    logger.info("  WS Telemetry: ws://localhost:8080/ws")
    logger.info("  WS Training : ws://localhost:8080/ws/training")
    logger.info("=" * 62)

    # Estado de GPU
    from app.core.model_registry import registry
    st  = registry.get_status()
    gpu = st["gpu"]
    if gpu["available"]:
        v = gpu["vram"]
        logger.info(f"  GPU : {gpu['name']}")
        logger.info(f"  VRAM: {v['free_mb']} MB libres / {v['total_mb']} MB total")
    else:
        logger.warning("  GPU no disponible — modo CPU.")

    # Iniciar motor de inferencia en segundo plano
    import asyncio
    loop = asyncio.get_event_loop()
    loop.run_in_executor(None, _start_engine)


def _start_engine():
    """Inicia el motor de inferencia en un executor (no bloquea el event loop)."""
    try:
        from app.core.inference_engine import engine
        engine.inicializar()
    except Exception as exc:
        logger.exception(f"Error iniciando motor de inferencia: {exc}")


@app.on_event("shutdown")
async def on_shutdown():
    logger.info("Cerrando Edu-Insight MLOps...")

    # Detener motor
    try:
        from app.core.inference_engine import engine
        engine.detener()
    except Exception:
        pass

    # Liberar modelos del registry
    import gc
    from app.core.model_registry import registry
    for cat in list(registry.loaded_instances.keys()):
        registry.unload_instance(cat)
    gc.collect()

    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

    logger.info("Recursos liberados. Hasta luego.")


# ══════════════════════════════════════════════════════════════
#  8. ENTRY POINT DIRECTO (python -m app.main)
# ══════════════════════════════════════════════════════════════
def _liberar_puerto(puerto: int) -> None:
    """Mata el proceso que ocupa el puerto en Windows."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        if s.connect_ex(("127.0.0.1", puerto)) != 0:
            return
    try:
        result = subprocess.check_output(
            f'netstat -ano | findstr ":{puerto} "', shell=True
        ).decode()
        for line in result.splitlines():
            parts = line.split()
            if parts and parts[1].endswith(f":{puerto}") and parts[3] == "LISTENING":
                subprocess.call(f"taskkill /F /PID {parts[-1]}", shell=True)
                logger.warning(f"Puerto {puerto} liberado (PID {parts[-1]}).")
    except Exception:
        pass


if __name__ == "__main__":
    import uvicorn
    _liberar_puerto(8080)
    print("\n" + "=" * 62)
    print("  Edu-Insight MLOps v2 — Iniciando servidor")
    print("  Dashboard : http://localhost:8080/")
    print("  Admin Lab : http://localhost:8080/admin")
    print("  API Docs  : http://localhost:8080/docs")
    print("=" * 62 + "\n")
    uvicorn.run("app.main:app", host="0.0.0.0", port=8080, reload=False)
