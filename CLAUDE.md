# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Edu-Insight PRO** is a real-time multimodal cognitive monitoring system for educational settings. It simultaneously analyzes video (face mesh, emotion, gaze), audio (speech transcription), and NLP (sentiment) from students and fuses them into a comprehension index (0–100).

## Commands

### Run the server
```bash
# Development (auto-kills port 8080 on Windows)
python -m app.main

# Production
uvicorn app.main:app --host 0.0.0.0 --port 8080
```

> Do NOT use `--reload` with uvicorn — camera threads are incompatible with it.

### Install dependencies
```bash
pip install -r requirements.txt
```

The project uses a `.venv` virtual environment. PyTorch is pinned to `2.7.1+cu118` (CUDA 11.8). There are no tests.

### Key URLs after startup
- Dashboard: `http://localhost:8080/`
- Admin/MLOps Lab: `http://localhost:8080/admin`
- API docs: `http://localhost:8080/docs`
- WebSocket telemetry: `ws://localhost:8080/ws`
- WebSocket training progress: `ws://localhost:8080/ws/training`

## Architecture

### Entry point: `app/main.py`
Configures CUDA DLL injection (Windows), sets up FastAPI, registers all routers, serves HTML files from `frontend/`, and on startup calls `init_db()` then launches `InferenceEngine.inicializar()` in a thread executor.

### Layer 1 — API (`app/api/`)
Each file is a FastAPI `APIRouter`:
- `telemetry.py` — MJPEG video feed (`GET /api/video_feed`), WebSocket state at 10 Hz (`/ws`), mode switching heuristic↔ML
- `auth_api.py` — Tutor register/login using face embeddings stored as BLOBs
- `students_api.py` — CRUD for students (linked to a tutor)
- `sessions_api.py` — Session lifecycle (create/start/stop), assigns students to face slots (0–5)
- `reports_api.py` — Telemetry queries and aggregations per session/student
- `models_api.py` — Model catalog, hot-swap active model per category, VRAM status
- `training_api.py` — Start/stop async training jobs, stream progress via `/ws/training`

### Layer 2 — Core (`app/core/`)
- **`inference_engine.py`** — Central AI engine, singleton `engine`. Runs three daemon threads:
  - `cam` thread: MediaPipe FaceMesh → head pose 3D (pitch/yaw), EAR, iris gaze → attention classification per face. Inline ONNX emotion inference. MJPEG encode to `frame_global_bytes`.
  - `vision` thread: Fallback DeepFace/CNN emotion analysis when ONNX unavailable.
  - `audio-prod` / `audio-cons` threads: SpeechRecognition → Faster-Whisper → RoBERTa sentiment.
  - Publishes to `estado_api_global` dict (read by WebSocket) and `estado_api_global["alumnos"]` for per-face state.
  - Comprehension index formula: `50 + emotion_score + sentiment_score ± attention_penalty`, EMA smoothed α=0.3.
- **`model_registry.py`** — Singleton `registry`. Manages 4 model categories (`ATENCION`, `AUDIO`, `EMOCION`, `SEMANTICA`). Handles VRAM checking before hot-swap and `torch.cuda.empty_cache()` on unload.
- **`trainer.py`** — Async training manager. Attention models (XGBoost/RF/SVM/LogisticRegression via sklearn) trained on `datasets/atencion/raw_data.csv`. Emotion CNNs (MobileNetV2/ResNet18/ResNet50/EfficientNet-B0) trained on FER-2013 at `datasets/emotion/`. Trained models saved to `app/storage/trained_models/`, metrics to `app/storage/metrics/`.
- **`telemetry_writer.py`** — Background daemon thread writing cognitive state snapshots to SQLite every 5 seconds. Reads from `estado_api_global["alumnos"]` for multi-student sessions.

### Layer 3 — Storage (`app/storage/`)
- **`database.py`** — SQLite at `app/storage/sessions.db`, WAL mode, foreign keys ON. Creates tables on startup via `init_db()`.
- **`repositories/`** — One file per domain entity: `tutor_repo.py`, `student_repo.py`, `session_repo.py`, `telemetry_repo.py`. Each uses `get_connection()` directly (no ORM).

### DB Schema (key tables)
| Table | Purpose |
|---|---|
| `tutors` | Tutor accounts with face embedding BLOB |
| `auth_sessions` | Token-based auth (expires_at) |
| `students` | Students linked to a tutor |
| `sessions` | Class sessions (activa/finalizada) |
| `session_slots` | Maps `face_slot` (0–5) → `student_id` per session |
| `telemetry_log` | Per-student snapshots: atencion, indice_comprension, emocion, sentimiento, mirada, EAR |

### Frontend (`frontend/`)
Plain HTML files served directly by FastAPI `FileResponse`. No build step. Pages: `dashboard.html`, `admin.html`, `login.html`, `register.html`, `session_setup.html`, `reports.html`.

### Model files
- `models/attention_model.joblib` — Default XGBoost model (ships with repo)
- `models/emotion_model.onnx` — Auto-generated from DeepFace's Keras model on first run if missing
- `app/storage/trained_models/` — User-trained models saved by `trainer.py`

## Key Patterns

### Global shared state
`estado_api_global` in `inference_engine.py` is the central shared dict read by the WebSocket. It is modified from multiple threads (camera, audio, NLP). No lock is used — reads/writes are considered atomic enough for this use case.

### Multi-student support
Up to 6 faces tracked simultaneously. Each `face_slot` (0–5) maps to a student via the `session_slots` table. `estado_api_global["slot_map"]` holds the `{face_idx: nombre}` mapping used for overlay labels.

### Attention classification modes
- `"heuristico"`: pitch/yaw thresholds + EAR somnolence counter
- `"ml"`: sklearn model loaded via `joblib.load()`, predicts from `[ear, pitch, yaw, ratio_h, ratio_v]`

Mode is toggled via `POST /api/modo` and reflected in `estado_api_global["modo_atencion"]`.

### Model hot-swap flow
`POST /api/models/{category}/activate` → `registry.set_active_model()` (checks VRAM, unloads old) → `engine.reload_attention_model()` or `engine.load_emocion_cnn()` → new model active without restart.
