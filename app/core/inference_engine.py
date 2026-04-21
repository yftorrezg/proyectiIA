"""
inference_engine.py — Edu-Insight MLOps
Motor de inferencia extraído de pruebas.py.

Gestiona:
  - MediaPipe Face Mesh (Head Pose 3D + EAR + Iris Gaze)
  - DeepFace (emoción) — o CNN entrenada propia si está activada
  - Faster-Whisper (transcripción GPU)
  - RoBERTa NLP (sentimiento)
  - Modelo de atención (XGBoost u otro sklearn) vía ModelRegistry
  - Fusión cognitiva multimodal → indice_comprension (0-100)

Estado compartido:
  estado_api_global  — dict leído por el WebSocket /ws
  frame_global_bytes — bytes MJPEG leídos por /api/video_feed
"""

import os
import sys
import time
import queue
import threading
import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import joblib

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
#  CONFIGURACIÓN (editable en caliente vía /api/config)
# ─────────────────────────────────────────────
CONFIG = {
    "resolucion":              (960, 540),
    "fps_objetivo":            30,
    "analisis_emocion_cada":   15,
    "transcripcion_activa":    True,
    "emocion_activa":          True,
    "whisper_compute":         "float16",
    "forzar_gpu_whisper":      True,
}

# ─────────────────────────────────────────────
#  ESTADO GLOBAL COMPARTIDO
# ─────────────────────────────────────────────
estado_api_global = {
    "emocion":          "Neutral",
    "texto":            "Esperando voz...",
    "sentimiento":      "NEU",
    "atencion":         "Enfocado",
    "mirada":           "Centro",
    "ear":              0.0,
    "indice_comprension": 50,
    "fps":              0,
    "modo_atencion":    "heuristico",   # "heuristico" | "ml"
    "ml_confidence":    0.0,
    "ml_disponible":    False,
    "modelo_activo":    "xgboost",      # id del modelo de atención activo
    # ── Modelos activos por categoría ──────────────────────────
    "audio_model":      "whisper_base",
    "emocion_model":    "deepface_vggface",
    "semantica_model":  "robertuito",
    "max_num_faces":    1,
}

frame_global_bytes: Optional[bytes] = None   # bytes MJPEG del frame actual

# ─────────────────────────────────────────────
#  CONSTANTES MEDIAPIPE
# ─────────────────────────────────────────────
face_3d_model = np.array([
    [0.0,    0.0,    0.0],
    [0.0,    330.0, -65.0],
    [-225.0, -170.0, -135.0],
    [225.0,  -170.0, -135.0],
    [-150.0,  150.0, -125.0],
    [150.0,   150.0, -125.0],
], dtype=np.float64)

OJO_IZQ_EAR     = [33, 160, 158, 133, 153, 144]
OJO_DER_EAR     = [362, 385, 387, 263, 373, 380]
EAR_UMBRAL      = 0.22
EAR_FRAMES_SOMNO = 20

IRIS_IZQ = 468;  IRIS_DER  = 473
OJO_IZQ_EXT = 33;   OJO_IZQ_INT = 133
OJO_DER_INT = 362;  OJO_DER_EXT = 263
OJO_IZQ_TOP = 159;  OJO_IZQ_BOT = 145
OJO_DER_TOP = 386;  OJO_DER_BOT = 374

TRADUCCION_EMOCION = {
    "angry": "Enojo", "disgust": "Disgusto", "fear": "Miedo",
    "happy": "Felicidad", "sad": "Tristeza",
    "surprise": "Sorpresa", "neutral": "Neutral",
}


def _calcular_ear(landmarks, indices, w, h) -> float:
    p = [np.array([landmarks[i].x * w, landmarks[i].y * h]) for i in indices]
    return (np.linalg.norm(p[1] - p[5]) + np.linalg.norm(p[2] - p[4])) / \
           (2.0 * np.linalg.norm(p[0] - p[3]) + 1e-6)


# ─────────────────────────────────────────────
#  MOTOR DE INFERENCIA
# ─────────────────────────────────────────────

class InferenceEngine:
    """
    Encapsula todos los modelos de IA y los hilos de procesamiento.
    Se inicializa una sola vez en el startup de FastAPI.
    """

    def __init__(self) -> None:
        self.running        = False
        self.device         = "cuda" if torch.cuda.is_available() else "cpu"
        self.audio_queue: queue.Queue = queue.Queue(maxsize=10)
        self.frame_queue:  queue.Queue = queue.Queue(maxsize=3)
        self.somnolencia_counter = 0
        self._mediapipe_crop: Optional[np.ndarray] = None

        # Modelos
        self.face_mesh      = None
        self.whisper        = None
        self.analizador_nlp = None
        self.face_cascade   = None

        # Modelo de atención (sklearn joblib)
        self._ml_package    = None   # dict: model, label_encoder, features, …
        self._ml_cache_key  = None   # (model_id, trained_path) — evita recargas innecesarias
        self._ml_lock       = threading.Lock()

        # Modelo de emoción CNN entrenado (PyTorch) — None = usar DeepFace
        self._emocion_cnn      = None
        self._emocion_classes  = None
        self._emocion_img_size = 224   # se actualiza al cargar el checkpoint

    # ── Inicialización ────────────────────────────────────────────────────────

    def inicializar(self) -> None:
        """Carga todos los modelos e inicia los hilos. Llamado en startup."""
        logger.info("Iniciando motor de inferencia...")
        self.running = True

        self._init_mediapipe()
        self._init_whisper_nlp()
        self._load_attention_model_from_registry()

        # Haar cascade para recorte rápido (fallback si MediaPipe falla)
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

        # Hilo de cámara (siempre activo)
        threading.Thread(target=self._capturar_camara, daemon=True, name="cam").start()

        # Hilo de DeepFace/CNN (si habilitado)
        if CONFIG["emocion_activa"]:
            threading.Thread(target=self._worker_vision, daemon=True, name="vision").start()

        logger.info(f"Motor listo — device: {self.device.upper()}")

    def detener(self) -> None:
        self.running = False

    # ── MediaPipe ─────────────────────────────────────────────────────────────

    def _init_mediapipe(self) -> None:
        try:
            import mediapipe as mp
            if hasattr(mp, "solutions"):
                mp_fm = mp.solutions.face_mesh
            else:
                import mediapipe.python.solutions.face_mesh as mp_fm

            self.face_mesh = mp_fm.FaceMesh(
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
                refine_landmarks=True,
                max_num_faces=1,
            )
            logger.info("MediaPipe Face Mesh inicializado (iris + 3D pose).")
        except Exception as exc:
            logger.warning(f"MediaPipe no disponible: {exc}")
            self.face_mesh = None

    # ── Whisper + RoBERTa ─────────────────────────────────────────────────────

    def _init_whisper_nlp(self) -> None:
        # Whisper
        if CONFIG["transcripcion_activa"]:
            try:
                from faster_whisper import WhisperModel
                from app.core.model_registry import registry, ModelCategory
                whisper_id   = registry.active_configs[ModelCategory.AUDIO].id
                whisper_size = whisper_id.replace("whisper_", "")   # "base", "small", …

                estado_api_global["audio_model"] = whisper_id
                try:
                    self.whisper = WhisperModel(whisper_size, device="cuda",
                                                compute_type=CONFIG["whisper_compute"])
                    logger.info(f"Whisper '{whisper_size}' cargado en CUDA.")
                except Exception as exc:
                    logger.warning(f"Whisper CUDA falló: {exc}")
                    if CONFIG["forzar_gpu_whisper"]:
                        self.whisper = None
                        logger.error("Whisper no disponible (GPU forzado).")
                    else:
                        self.whisper = WhisperModel(whisper_size, device="cpu",
                                                    compute_type="int8")
                        logger.info(f"Whisper '{whisper_size}' cargado en CPU (fallback).")

                # Iniciar hilos de audio si whisper cargó
                if self.whisper:
                    try:
                        import speech_recognition as sr
                        threading.Thread(target=self._productor_audio, daemon=True, name="audio-prod").start()
                        threading.Thread(target=self._consumidor_audio, daemon=True, name="audio-cons").start()
                        logger.info("Hilos de audio iniciados.")
                    except ImportError:
                        logger.warning("SpeechRecognition / PyAudio no disponibles.")
            except ImportError:
                logger.warning("faster-whisper no instalado.")

        # RoBERTa NLP
        try:
            from transformers import pipeline
            from app.core.model_registry import registry, ModelCategory
            model_cfg    = registry.active_configs[ModelCategory.SEMANTICA]
            hf_model     = model_cfg.metadata.get("hf_model",
                           "pysentimiento/robertuito-sentiment-analysis")
            device_nlp   = 0 if self.device == "cuda" else -1
            self.analizador_nlp = pipeline("sentiment-analysis",
                                           model=hf_model, device=device_nlp)
            estado_api_global["semantica_model"] = model_cfg.id
            logger.info(f"NLP '{hf_model}' cargado.")
        except Exception as exc:
            logger.warning(f"RoBERTa NLP no disponible: {exc}")

    # ── Modelo de atención ────────────────────────────────────────────────────

    def _load_attention_model_from_registry(self) -> None:
        """
        Carga el modelo de atención indicado por el ModelRegistry.
        Prioridad:
          1. trained_model_path en metadata (modelo entrenado por el usuario)
          2. models/attention_model.joblib (modelo original XGBoost)

        La clave de caché es (model_id, trained_path), por lo que reentrena
        el mismo tipo de modelo con un path diferente → sí recarga.
        """
        try:
            from app.core.model_registry import registry, ModelCategory
            cfg          = registry.active_configs[ModelCategory.ATENCION]
            model_id     = cfg.id
            trained_path = cfg.metadata.get("trained_model_path", "")
            cache_key    = (model_id, trained_path)

            # Mismo modelo Y mismo path → ya está cargado
            with self._ml_lock:
                if self._ml_cache_key == cache_key and self._ml_package is not None:
                    return

            # ── Prioridad 1: modelo entrenado por el usuario ──────────────
            if trained_path and Path(trained_path).exists():
                pkg = joblib.load(trained_path)
                with self._ml_lock:
                    self._ml_package   = pkg
                    self._ml_cache_key = cache_key
                estado_api_global["ml_disponible"] = True
                estado_api_global["modelo_activo"] = model_id
                logger.info(f"Modelo atención cargado (entrenado): {Path(trained_path).name}")
                return

            # ── Prioridad 2: modelo original en models/ ───────────────────
            default_path = Path("models/attention_model.joblib")
            if default_path.exists():
                pkg = joblib.load(default_path)
                with self._ml_lock:
                    self._ml_package   = pkg
                    self._ml_cache_key = cache_key
                estado_api_global["ml_disponible"] = True
                estado_api_global["modelo_activo"] = model_id
                logger.info(f"Modelo atención cargado (default): {default_path}")
            else:
                with self._ml_lock:
                    self._ml_package   = None
                    self._ml_cache_key = cache_key
                estado_api_global["ml_disponible"] = False
                logger.info("Modelo de atención no encontrado — modo heurístico activo.")

        except Exception as exc:
            logger.warning(f"Error cargando modelo de atención: {exc}")
            with self._ml_lock:
                self._ml_package = None
            estado_api_global["ml_disponible"] = False

    def reload_attention_model(self) -> None:
        """Fuerza recarga completa ignorando el caché. Llamado desde hot-swap."""
        with self._ml_lock:
            self._ml_cache_key = None   # invalida caché
        self._load_attention_model_from_registry()

    def set_max_faces(self, n: int) -> None:
        """Reinicializa MediaPipe FaceMesh con un nuevo número máximo de rostros."""
        n = max(1, min(n, 6))
        try:
            import mediapipe as mp
            if hasattr(mp, "solutions"):
                mp_fm = mp.solutions.face_mesh
            else:
                import mediapipe.python.solutions.face_mesh as mp_fm

            old = self.face_mesh
            self.face_mesh = mp_fm.FaceMesh(
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
                refine_landmarks=True,
                max_num_faces=n,
            )
            if old is not None:
                try:
                    old.close()
                except Exception:
                    pass
            estado_api_global["max_num_faces"] = n
            logger.info(f"FaceMesh reinicializado: max_num_faces={n}")
        except Exception as exc:
            logger.error(f"Error reinicializando FaceMesh: {exc}")

    # ── Fusión cognitiva ──────────────────────────────────────────────────────

    def _calcular_indice(self) -> None:
        score_facial = {"Felicidad": 25, "Neutral": 15, "Sorpresa": 15,
                        "Tristeza": -15, "Miedo": -15, "Enojo": -25, "Disgusto": -25
                        }.get(estado_api_global["emocion"], 0)

        score_texto  = {"POS": 25, "NEU": 5, "NEG": -25}.get(
            estado_api_global["sentimiento"], 0)

        crudo = 50 + score_facial + score_texto
        if estado_api_global["atencion"] == "Distraído":   crudo -= 30
        elif estado_api_global["atencion"] == "Somnoliento": crudo -= 40
        crudo = max(0, min(100, crudo))

        # Suavizado exponencial α=0.3
        estado_api_global["indice_comprension"] = int(
            estado_api_global["indice_comprension"] * 0.7 + crudo * 0.3
        )

    # ── Audio ──────────────────────────────────────────────────────────────────

    def _productor_audio(self) -> None:
        import speech_recognition as sr
        recognizer = sr.Recognizer()
        recognizer.energy_threshold        = 300
        recognizer.dynamic_energy_threshold = True
        recognizer.pause_threshold         = 0.8
        try:
            mic = sr.Microphone()
            with mic as src:
                logger.info("Calibrando ruido ambiental...")
                recognizer.adjust_for_ambient_noise(src, duration=2)
            logger.info("Micrófono abierto.")
            while self.running:
                try:
                    with mic as src:
                        audio = recognizer.listen(src, timeout=3, phrase_time_limit=5)
                        if not self.audio_queue.full():
                            self.audio_queue.put(audio)
                except sr.WaitTimeoutError:
                    continue
        except Exception as exc:
            logger.error(f"Error de micrófono: {exc}")

    def _consumidor_audio(self) -> None:
        while self.running:
            try:
                audio = self.audio_queue.get(timeout=1)
            except queue.Empty:
                continue
            if not self.whisper:
                continue
            try:
                with open("temp_audio.wav", "wb") as f:
                    f.write(audio.get_wav_data())
                segments, _ = self.whisper.transcribe(
                    "temp_audio.wav", language="es", beam_size=5,
                    vad_filter=True,
                    vad_parameters=dict(min_silence_duration_ms=500),
                    condition_on_previous_text=False,
                )
                texto = " ".join(seg.text for seg in segments).strip()
                if texto:
                    estado_api_global["texto"] = texto
                    logger.debug(f"Whisper: {texto!r}")
                    if self.analizador_nlp:
                        try:
                            res = self.analizador_nlp(texto[:512])
                            lbl = res[0]["label"] if isinstance(res, list) else res["label"]
                            estado_api_global["sentimiento"] = lbl
                        except Exception:
                            estado_api_global["sentimiento"] = "NEU"
                    self._calcular_indice()
            except Exception as exc:
                logger.debug(f"Error en Whisper: {exc}")
            finally:
                self.audio_queue.task_done()

    # ── DeepFace / CNN ────────────────────────────────────────────────────────

    def _worker_vision(self) -> None:
        """
        Procesa frames de rostro en segundo plano.
        Usa CNN propia si hay un modelo de emoción entrenado activado,
        de lo contrario usa DeepFace.
        """
        while self.running:
            try:
                crop = self.frame_queue.get(timeout=1)
            except queue.Empty:
                continue
            try:
                self._analizar_emocion(crop)
                self._calcular_indice()
            except Exception as exc:
                logger.debug(f"Error en worker_vision: {exc}")
            finally:
                self.frame_queue.task_done()

    def _analizar_emocion(self, crop: np.ndarray) -> None:
        """Delega a DeepFace o CNN entrenada según el modelo activo del registry."""
        from app.core.model_registry import registry, ModelCategory
        cfg_emocion = registry.active_configs[ModelCategory.EMOCION]

        # Si hay CNN propia cargada Y el modelo activo es diferente a VGG-Face → usar CNN
        if cfg_emocion.id != "deepface_vggface" and self._emocion_cnn is not None:
            self._analizar_con_cnn(crop)
        else:
            self._analizar_con_deepface(crop)

    def _analizar_con_deepface(self, crop: np.ndarray) -> None:
        try:
            from deepface import DeepFace
            cv2.imwrite("temp_rostro.jpg", crop)
            resultado = DeepFace.analyze(
                img_path="temp_rostro.jpg",
                actions=["emotion"],
                enforce_detection=False,
                detector_backend="skip",
                silent=True,
            )
            res = resultado[0] if isinstance(resultado, list) else resultado
            emocion = res.get("dominant_emotion", "neutral")
            estado_api_global["emocion"] = TRADUCCION_EMOCION.get(emocion, emocion)
        except Exception as exc:
            logger.debug(f"DeepFace error: {exc}")

    def _analizar_con_cnn(self, crop: np.ndarray) -> None:
        """Inferencia con CNN torchvision entrenada en FER-2013 (7 clases)."""
        try:
            import torch
            from torchvision import transforms
            sz = self._emocion_img_size   # usa el tamaño con el que fue entrenado el modelo
            tf = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Grayscale(num_output_channels=3),
                transforms.Resize((sz, sz)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
            tensor = tf(crop).unsqueeze(0).to(self.device)
            with torch.no_grad():
                logits = self._emocion_cnn(tensor)
                pred   = torch.argmax(logits, dim=1).item()
            emocion_raw = self._emocion_classes[pred]
            estado_api_global["emocion"] = TRADUCCION_EMOCION.get(emocion_raw, emocion_raw)
        except Exception as exc:
            logger.debug(f"CNN emoción error: {exc}")

    def load_emocion_cnn(self, model_path: str, model_id: str) -> None:
        """Carga un checkpoint .pt de CNN de emoción (llamado desde hot-swap)."""
        try:
            import torch
            checkpoint = torch.load(model_path, map_location=self.device)
            num_classes = checkpoint.get("num_classes", 7)
            classes     = checkpoint.get("classes", ["angry","disgust","fear","happy","neutral","sad","surprise"])

            from app.core.trainer import training_manager
            cnn = training_manager._build_cnn(model_id, num_classes, freeze=False)
            cnn.load_state_dict(checkpoint["model_state_dict"])
            cnn.eval()
            cnn.to(self.device)

            self._emocion_cnn     = cnn
            self._emocion_classes = classes
            # Guardar img_size del checkpoint para usar el mismo tamaño en inferencia
            hp = checkpoint.get("hyperparams", {})
            self._emocion_img_size = int(hp.get("img_size", 224))
            estado_api_global["emocion_model"] = model_id
            logger.info(f"CNN de emoción cargada: {model_id} ({num_classes} clases, {self._emocion_img_size}px)")
        except Exception as exc:
            logger.error(f"Error cargando CNN de emoción: {exc}")

    # ── Cámara principal ──────────────────────────────────────────────────────

    def _capturar_camara(self) -> None:
        global frame_global_bytes
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CONFIG["resolucion"][0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CONFIG["resolucion"][1])
        cap.set(cv2.CAP_PROP_FPS,          CONFIG["fps_objetivo"])

        frame_counter = 0
        fps_time  = time.time()
        fps_count = 0

        while self.running:
            ret, frame = cap.read()
            if not ret:
                continue

            frame = cv2.flip(frame, 1)
            frame_limpio = frame.copy()
            h, w, _ = frame.shape

            frame_counter += 1
            fps_count     += 1

            if time.time() - fps_time >= 1.0:
                estado_api_global["fps"] = fps_count
                fps_count = 0
                fps_time  = time.time()

            x_min, y_min, x_max, y_max = w, h, 0, 0

            # ── MediaPipe ───────────────────────────────────────────────────
            if self.face_mesh is not None:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.face_mesh.process(rgb)

                if results.multi_face_landmarks:
                    for face_lm in results.multi_face_landmarks:
                        lm = face_lm.landmark

                        # Bounding box
                        xs = [int(lm[i].x * w) for i in range(468)]
                        ys = [int(lm[i].y * h) for i in range(468)]
                        x_min, x_max = min(xs), max(xs)
                        y_min, y_max = min(ys), max(ys)

                        # Crop para DeepFace / CNN
                        mg  = int((x_max - x_min) * 0.25)
                        cx1 = max(0, x_min - mg);  cx2 = min(w, x_max + mg)
                        cy1 = max(0, y_min - mg);  cy2 = min(h, y_max + mg)
                        crop = frame_limpio[cy1:cy2, cx1:cx2]
                        if crop.size > 0:
                            self._mediapipe_crop = crop.copy()

                        # EAR
                        ear_izq = _calcular_ear(lm, OJO_IZQ_EAR, w, h)
                        ear_der = _calcular_ear(lm, OJO_DER_EAR, w, h)
                        ear     = (ear_izq + ear_der) / 2.0
                        estado_api_global["ear"] = round(ear, 3)

                        if ear < EAR_UMBRAL:
                            self.somnolencia_counter += 1
                        else:
                            self.somnolencia_counter = max(0, self.somnolencia_counter - 1)
                        es_somnoliento = self.somnolencia_counter >= EAR_FRAMES_SOMNO

                        # Iris gaze
                        iris_izq = lm[IRIS_IZQ];  iris_der = lm[IRIS_DER]
                        rng_h_izq = (lm[OJO_IZQ_INT].x - lm[OJO_IZQ_EXT].x) + 1e-6
                        rng_h_der = (lm[OJO_DER_EXT].x - lm[OJO_DER_INT].x) + 1e-6
                        ratio_h = ((iris_izq.x - lm[OJO_IZQ_EXT].x) / rng_h_izq +
                                   (iris_der.x - lm[OJO_DER_INT].x) / rng_h_der) / 2.0

                        rng_v_izq = (lm[OJO_IZQ_BOT].y - lm[OJO_IZQ_TOP].y) + 1e-6
                        rng_v_der = (lm[OJO_DER_BOT].y - lm[OJO_DER_TOP].y) + 1e-6
                        ratio_v = ((iris_izq.y - lm[OJO_IZQ_TOP].y) / rng_v_izq +
                                   (iris_der.y - lm[OJO_DER_TOP].y) / rng_v_der) / 2.0

                        mir_h = "Izquierda" if ratio_h < 0.35 else ("Derecha" if ratio_h > 0.65 else "Centro")
                        mir_v = "Abajo"     if ratio_v > 0.65 else ("Arriba"  if ratio_v < 0.35 else "Centro")
                        partes = [p for p in [mir_v, mir_h] if p != "Centro"]
                        estado_api_global["mirada"] = "-".join(partes) if partes else "Centro"

                        # Head Pose 3D
                        face_2d = np.array([
                            [int(lm[1].x*w),   int(lm[1].y*h)],
                            [int(lm[152].x*w), int(lm[152].y*h)],
                            [int(lm[33].x*w),  int(lm[33].y*h)],
                            [int(lm[263].x*w), int(lm[263].y*h)],
                            [int(lm[61].x*w),  int(lm[61].y*h)],
                            [int(lm[291].x*w), int(lm[291].y*h)],
                        ], dtype=np.float64)

                        fl = 1 * w
                        cam_m = np.array([[fl,0,w/2],[0,fl,h/2],[0,0,1]], dtype=np.float64)
                        dist_m = np.zeros((4,1), dtype=np.float64)
                        ok, rot_vec, _ = cv2.solvePnP(face_3d_model, face_2d, cam_m, dist_m)

                        if ok:
                            rmat, _ = cv2.Rodrigues(rot_vec)
                            angles, *_ = cv2.RQDecomp3x3(rmat)
                            pitch, yaw = angles[0], angles[1]

                            # ── Atención ────────────────────────────────────
                            if (estado_api_global["modo_atencion"] == "ml"
                                    and estado_api_global["ml_disponible"]):
                                # Solo recarga si el registry cambió (cache_key mismatch)
                                self._load_attention_model_from_registry()
                                with self._ml_lock:
                                    pkg = self._ml_package
                                if pkg:
                                    feats = np.array([[ear, pitch, yaw, ratio_h]])
                                    pred_enc  = pkg["model"].predict(feats)[0]
                                    pred_prob = pkg["model"].predict_proba(feats)[0]
                                    pred_lbl  = pkg["label_encoder"].inverse_transform([pred_enc])[0]
                                    conf      = float(np.max(pred_prob))
                                    MAPA_ML   = {"Enfocado":"Enfocado","Distraido":"Distraído","Somnoliento":"Somnoliento"}
                                    estado_api_global["atencion"]     = MAPA_ML.get(pred_lbl, pred_lbl)
                                    estado_api_global["ml_confidence"] = round(conf, 3)
                                    color_ejes   = {"Enfocado":(0,255,0),"Distraído":(0,0,255),"Somnoliento":(0,165,255)}.get(estado_api_global["atencion"],(0,255,0))
                                    texto_atencion = f"ML:{pred_lbl.upper()} {conf*100:.0f}%"
                                else:
                                    estado_api_global["modo_atencion"] = "heuristico"
                                    texto_atencion = "HEURISTICO"
                                    color_ejes = (0,255,0)
                            else:
                                estado_api_global["ml_confidence"] = 0.0
                                if es_somnoliento:
                                    estado_api_global["atencion"] = "Somnoliento"
                                    color_ejes   = (0, 165, 255)
                                    texto_atencion = "SOMNOLIENTO"
                                elif pitch < -30 or pitch > 35 or yaw < -35 or yaw > 35:
                                    estado_api_global["atencion"] = "Distraído"
                                    color_ejes   = (0, 0, 255)
                                    texto_atencion = "DISTRAIDO"
                                else:
                                    estado_api_global["atencion"] = "Enfocado"
                                    color_ejes   = (0, 255, 0)
                                    texto_atencion = "ATENTO"

                            self._calcular_indice()

                            # Overlay
                            cv2.rectangle(frame, (x_min,y_min),(x_max,y_max), color_ejes, 2)
                            emo_vis = estado_api_global["emocion"].upper()
                            if emo_vis == "FELICIDAD": emo_vis = "FELIZ"
                            by = max(0, y_min - 65)
                            cv2.rectangle(frame,(x_min,by),(x_max+80,y_min),(15,23,42),-1)
                            cv2.putText(frame, f"EMOCION: {emo_vis}",
                                (x_min+5, by+18), cv2.FONT_HERSHEY_SIMPLEX, 0.48,(0,255,255),2)
                            cv2.putText(frame, texto_atencion,
                                (x_min+5, by+38), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color_ejes, 2)
                            cv2.putText(frame, f"MIRADA: {estado_api_global['mirada'].upper()}",
                                (x_min+5, by+58), cv2.FONT_HERSHEY_SIMPLEX, 0.44,(255,200,0),1)

                            # Puntos iris
                            cv2.circle(frame,(int(iris_izq.x*w),int(iris_izq.y*h)),3,(0,255,255),-1)
                            cv2.circle(frame,(int(iris_der.x*w),int(iris_der.y*h)),3,(0,255,255),-1)
                else:
                    self.somnolencia_counter = 0

            # Enviar crop a DeepFace/CNN
            if (CONFIG["emocion_activa"]
                    and frame_counter % CONFIG["analisis_emocion_cada"] == 0
                    and self._mediapipe_crop is not None
                    and not self.frame_queue.full()):
                self.frame_queue.put(self._mediapipe_crop.copy())

            # MJPEG encode
            ret_enc, buf = cv2.imencode(".jpg", frame)
            if ret_enc:
                frame_global_bytes = buf.tobytes()

        cap.release()


# ── Instancia global del motor ─────────────────────────────────────────────────
engine = InferenceEngine()
