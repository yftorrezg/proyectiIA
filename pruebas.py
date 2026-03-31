# ==============================================================================
# PROYECTO: Edu-Insight - BACKEND WEB STREAMING (RTX 3060 + WEBSOCKETS + POSE 3D)
# ==============================================================================
import os
import sys
import time
import getpass 
import site

print("\n" + "█"*70)
print("⏳ INICIANDO MOTORES CUDA Y COMPILANDO KERNELS 3D...")
print("⚠️ ATENCIÓN: ESTE PROCESO NO ES UN BUCLE.")
print("⚠️ LA PRIMERA VEZ PUEDE TARDAR ENTRE 40 Y 60 SEGUNDOS EN PENSAR.")
print("⚠️ POR FAVOR, NO PRESIONES 'CTRL+C'. ESPERA PACIENTEMENTE.")
print("█"*70 + "\n")

# ==============================================================================
# 0. ⚠️ CONFIGURACIÓN MAESTRA DE RENDIMIENTO (RTX 3060)
# ==============================================================================
CONFIG = {
    "resolucion": (960, 540),      # Balance perfecto entre calidad y rendimiento
    "fps_objetivo": 30,            # Tasa de refresco de la cámara
    "analisis_emocion_cada": 15,   # Analiza rostro 2 veces por segundo (a 30fps)
    "transcripcion_activa": True,  # Habilita el módulo Whisper
    "emocion_activa": True,        # Habilita el módulo DeepFace
    "whisper_modelo": "base",      # 'base' da precisión excelente sin alucinaciones
    "whisper_compute": "float16",  # float16 es hiper-rápido en GPU
    "forzar_gpu_whisper": True     # 🔴 NUEVO: Obliga al sistema a usar la RTX 3060 para audio
}

# ==============================================================================
# 1. MONKEY PATCHING Y AUTO-INYECTOR DE DLLS DE NVIDIA (SOPORTE CUDA 12)
# ==============================================================================
getpass.getuser = lambda: "Usuario_EduInsight"

os.environ["USERNAME"] = "Usuario_EduInsight"
os.environ["USER"] = "Usuario_EduInsight"
os.environ["LOGNAME"] = "Usuario_EduInsight"
os.environ["TORCHINDUCTOR_CACHE_DIR"] = os.path.join(os.getcwd(), ".torch_cache")

# 🔴 INYECTOR AVANZADO: Busca DLLs de CUDA 11 y CUDA 12 simultáneamente
if os.name == 'nt':
    paquetes_instalados = site.getsitepackages()
    for sp in paquetes_instalados:
        for lib in ["cublas", "cudnn"]:
            ruta_bin = os.path.join(sp, "nvidia", lib, "bin")
            if os.path.exists(ruta_bin):
                os.environ["PATH"] = ruta_bin + os.pathsep + os.environ.get("PATH", "")
                if hasattr(os, 'add_dll_directory'):
                    try: os.add_dll_directory(ruta_bin)
                    except: pass

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# ==============================================================================
# 2. IMPORTACIÓN CRÍTICA: MEDIAPIPE (El Radar 3D)
# ==============================================================================
import mediapipe as mp
import numpy as np

try:
    if hasattr(mp, 'solutions'):
        mp_face_mesh = mp.solutions.face_mesh
    else:
        import mediapipe.python.solutions.face_mesh as mp_face_mesh
        
    face_mesh = mp_face_mesh.FaceMesh(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        refine_landmarks=True,   # CRÍTICO: habilita iris landmarks (468-477)
        max_num_faces=1
    )
    MEDIAPIPE_DISPONIBLE = True
    print("✅ Radar 3D + Iris (MediaPipe) inicializado correctamente en memoria.")
except Exception as e:
    MEDIAPIPE_DISPONIBLE = False
    print(f"⚠️ Error cargando MediaPipe: {e}")

# ORDEN CRÍTICO ANATÓMICO 3D (Ejes Y corregidos para OpenCV)
face_3d_model = np.array([
    [0.0, 0.0, 0.0],            # 1. Nariz
    [0.0, 330.0, -65.0],        # 2. Barbilla 
    [-225.0, -170.0, -135.0],   # 3. Ojo izquierdo 
    [225.0, -170.0, -135.0],    # 4. Ojo derecho
    [-150.0, 150.0, -125.0],    # 5. Boca izquierda
    [150.0, 150.0, -125.0]      # 6. Boca derecha
], dtype=np.float64)

# ------------------------------------------------------------------------------
# LANDMARKS PARA EYE ASPECT RATIO (EAR) — Detección de somnolencia
# Orden: [p1_ext, p2_sup_ext, p3_sup_int, p4_int, p5_inf_int, p6_inf_ext]
# ------------------------------------------------------------------------------
OJO_IZQ_EAR = [33, 160, 158, 133, 153, 144]
OJO_DER_EAR = [362, 385, 387, 263, 373, 380]
EAR_UMBRAL        = 0.22   # Por debajo = ojo casi cerrado
EAR_FRAMES_SOMNO  = 20    # ~0.67 s a 30 fps = somnolencia confirmada

# Iris centers (solo con refine_landmarks=True)
IRIS_IZQ = 468
IRIS_DER = 473
# Esquinas horizontales de cada ojo
OJO_IZQ_EXT, OJO_IZQ_INT = 33, 133
OJO_DER_INT, OJO_DER_EXT = 362, 263
# Extremos verticales de cada ojo
OJO_IZQ_TOP, OJO_IZQ_BOT = 159, 145
OJO_DER_TOP, OJO_DER_BOT = 386, 374


def calcular_ear(landmarks, indices, w, h):
    """Eye Aspect Ratio — valor bajo = ojo cerrado."""
    p = [np.array([landmarks[i].x * w, landmarks[i].y * h]) for i in indices]
    ear = (np.linalg.norm(p[1] - p[5]) + np.linalg.norm(p[2] - p[4])) / \
          (2.0 * np.linalg.norm(p[0] - p[3]) + 1e-6)
    return ear


# ==============================================================================
# 3. RESTO DE IMPORTACIONES Y LIBRERÍAS
# ==============================================================================
import cv2
import threading
import queue
import torch
import asyncio
from fastapi import FastAPI, WebSocket
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

try:
    from faster_whisper import WhisperModel
    WHISPER_DISPONIBLE = True
except: WHISPER_DISPONIBLE = False

try:
    from deepface import DeepFace
    DEEPFACE_DISPONIBLE = True
except: DEEPFACE_DISPONIBLE = False

try:
    import speech_recognition as sr
    SR_DISPONIBLE = True
except: 
    SR_DISPONIBLE = False
    print("\n❌ LIBRERÍA FALTANTE: No se detectó 'SpeechRecognition' o 'pyaudio'.")

from transformers import pipeline

# ==============================================================================
# INICIALIZACIÓN DE FASTAPI
# ==============================================================================
app = FastAPI(title="Edu-Insight Streaming API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

frame_global_bytes = None
estado_api_global = {
    "emocion": "Neutral",
    "texto": "Esperando voz...",
    "sentimiento": "Neutral",
    "atencion": "Enfocado",    # "Enfocado" | "Distraído" | "Somnoliento"
    "mirada": "Centro",        # "Centro" | "Izquierda" | "Derecha" | "Arriba" | "Abajo" | combinados
    "ear": 0.0,                # Eye Aspect Ratio en tiempo real (útil para debug)
    "indice_comprension": 50,
    "fps": 0
}

# ==============================================================================
# CLASE PRINCIPAL DE IA (Background)
# ==============================================================================
class EduInsightWebWorker:
    def __init__(self):
        self.audio_queue = queue.Queue(maxsize=10)
        self.frame_queue = queue.Queue(maxsize=3)
        self.running = True
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.somnolencia_counter = 0   # frames consecutivos con EAR bajo
        self._mediapipe_crop = None    # último crop de rostro detectado por MediaPipe
        
        # Detector facial ultrarrápido por CPU (Haar Cascades)
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        self.inicializar_modelos()

    def inicializar_modelos(self):
        print(f"🔥 Hardware detectado: {self.device.upper()}")
        
        if WHISPER_DISPONIBLE and CONFIG["transcripcion_activa"]:
            print(f"🎙️ Cargando Whisper (Modelo '{CONFIG['whisper_modelo'].upper()}' en GPU RTX 3060)...")
            try:
                # 🚀 INTENTO PRIMARIO: Ejecución estricta en CUDA 
                self.whisper = WhisperModel(
                    CONFIG["whisper_modelo"], 
                    device="cuda", 
                    compute_type=CONFIG["whisper_compute"]
                )
                print("   ✅ Whisper cargado exitosamente en CUDA (Hardware Gráfico).")
            except Exception as e: 
                print(f"   ⚠️ Error de compatibilidad CUDA en Whisper: {e}")
                if CONFIG["forzar_gpu_whisper"]:
                    print("   ❌ Fallo crítico. No se pudo iniciar Whisper en la GPU.")
                    self.whisper = None
                else:
                    print("   🔄 Intentando fallback a CPU...")
                    self.whisper = WhisperModel(CONFIG["whisper_modelo"], device="cpu", compute_type="int8")
                    print("   ✅ Whisper cargado en CPU.")
            
            if self.whisper and SR_DISPONIBLE:
                threading.Thread(target=self.productor_audio, daemon=True).start()
                threading.Thread(target=self.consumidor_audio, daemon=True).start()

        print("Cargando RoBERTa NLP (Procesamiento de Lenguaje Natural)...")
        dispositivo_nlp = 0 if self.device == "cuda" else -1
        self.analizador_nlp = pipeline("sentiment-analysis", model="pysentimiento/robertuito-sentiment-analysis", device=dispositivo_nlp)

        if DEEPFACE_DISPONIBLE and CONFIG["emocion_activa"]:
            print("✅ DeepFace cargado y en espera para procesamiento asíncrono.")
            threading.Thread(target=self.worker_vision, daemon=True).start()

    def calcular_indice(self):
        """Módulo Matemático de Fusión Cognitiva con Suavizado Exponencial"""
        score_facial, score_texto = 0, 0
        emocion = estado_api_global["emocion"]
        
        if emocion == "Felicidad": score_facial = 25
        elif emocion in ["Neutral", "Sorpresa"]: score_facial = 15
        elif emocion in ["Tristeza", "Miedo"]: score_facial = -15
        elif emocion in ["Enojo", "Disgusto"]: score_facial = -25
            
        sentimiento = estado_api_global["sentimiento"]
        if sentimiento == "POS": score_texto = 25
        elif sentimiento == "NEU": score_texto = 5
        elif sentimiento == "NEG": score_texto = -25
            
        nuevo_indice_crudo = 50 + score_facial + score_texto
        
        if estado_api_global["atencion"] == "Distraído":
            nuevo_indice_crudo -= 30
        elif estado_api_global["atencion"] == "Somnoliento":
            nuevo_indice_crudo -= 40  # Somnoliento penaliza más que distracción
            
        nuevo_indice_crudo = max(0, min(100, nuevo_indice_crudo))
        
        indice_actual = estado_api_global["indice_comprension"]
        indice_suavizado = int((indice_actual * 0.7) + (nuevo_indice_crudo * 0.3))
        
        estado_api_global["indice_comprension"] = indice_suavizado

    def productor_audio(self):
        recognizer = sr.Recognizer()
        recognizer.energy_threshold = 300  
        recognizer.dynamic_energy_threshold = True 
        recognizer.pause_threshold = 0.8  
        
        try:
            mic = sr.Microphone()
            with mic as source: 
                print("\n🎙️ [AUDIO] Calibrando ruido ambiental...")
                recognizer.adjust_for_ambient_noise(source, duration=2)
                
            print("✅ [AUDIO] Micrófono ABIERTO.")
            while self.running:
                try:
                    with mic as source:
                        audio = recognizer.listen(source, timeout=3, phrase_time_limit=5)
                        if not self.audio_queue.full(): 
                            self.audio_queue.put(audio)
                except sr.WaitTimeoutError: 
                    continue 
        except Exception as e: 
            print(f"\n❌ [ERROR GRAVE AUDIO] No se puede acceder al micrófono: {e}\n")

    def consumidor_audio(self):
        while self.running:
            try: audio = self.audio_queue.get(timeout=1)
            except queue.Empty: continue
            
            if not self.whisper: continue
            try:
                with open("temp_audio.wav", "wb") as f: f.write(audio.get_wav_data())
                
                # 🚀 Transcripción optimizada en GPU con VAD Filter (Destruye alucinaciones)
                segments, _ = self.whisper.transcribe(
                    "temp_audio.wav", 
                    language="es", 
                    beam_size=5, 
                    vad_filter=True, 
                    vad_parameters=dict(min_silence_duration_ms=500),
                    condition_on_previous_text=False
                )
                texto = " ".join([seg.text for seg in segments]).strip()
                
                if texto:
                    estado_api_global["texto"] = texto
                    print(f"🗣️ [WHISPER GPU] Transcrito: '{texto}'")
                    try:
                        res_nlp = self.analizador_nlp(texto[:512])
                        estado_api_global["sentimiento"] = res_nlp[0]['label'] if isinstance(res_nlp, list) else res_nlp['label']
                    except: estado_api_global["sentimiento"] = "NEU"
                    self.calcular_indice()
            except Exception as e: 
                print(f"❌ Error interno en Whisper: {e}")
            finally: self.audio_queue.task_done()

    def worker_vision(self):
        """DeepFace asíncrono sobre el rostro ya detectado por Haar Cascades"""
        while self.running:
            try: 
                rostro_recortado = self.frame_queue.get(timeout=1)
            except queue.Empty: 
                continue
                
            try:
                # Guardamos la imagen física para corregir el formato BGR a RGB automáticamente
                exito = cv2.imwrite("temp_rostro.jpg", rostro_recortado)
                if not exito:
                    continue
                
                # detector_backend='skip': el crop ya ES el rostro, no buscar de nuevo
                resultado = DeepFace.analyze(
                    img_path="temp_rostro.jpg",
                    actions=['emotion'],
                    enforce_detection=False,
                    detector_backend='skip',
                    silent=True
                )
                res_dict = resultado[0] if isinstance(resultado, list) else resultado
                emocion = res_dict.get('dominant_emotion', 'neutral')
                
                print(f"🎭 [DEEPFACE GPU] Emoción Detectada: {emocion.upper()}")
                
                traduccion = { "angry": "Enojo", "disgust": "Disgusto", "fear": "Miedo", "happy": "Felicidad", "sad": "Tristeza", "surprise": "Sorpresa", "neutral": "Neutral" }
                estado_api_global["emocion"] = traduccion.get(emocion, emocion)
                self.calcular_indice()
            except Exception as e:
                # Silenciamos el log de error para no saturar la consola si hay un frame borroso
                pass
            finally: 
                self.frame_queue.task_done()

    def capturar_camara(self):
        global frame_global_bytes
        cap = cv2.VideoCapture(0)
        
        # Aplicamos la configuración centralizada (CONFIG)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CONFIG["resolucion"][0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CONFIG["resolucion"][1])
        cap.set(cv2.CAP_PROP_FPS, CONFIG["fps_objetivo"])
        
        frame_counter = 0
        fps_time = time.time()
        fps_count = 0
        
        while self.running:
            ret, frame = cap.read()
            if not ret: continue
            
            frame = cv2.flip(frame, 1)
            
            # Fotograma inmaculado para análisis visual
            frame_limpio = frame.copy() 
            h, w, _ = frame.shape
            
            frame_counter += 1
            fps_count += 1
            
            if time.time() - fps_time >= 1:
                estado_api_global["fps"] = fps_count
                fps_count = 0
                fps_time = time.time()

            x_min, y_min, x_max, y_max = w, h, 0, 0

            # ------------------------------------------------------------------
            # 1. DETECCIÓN DE ATENCIÓN (Head Pose 3D + Iris Gaze + EAR)
            # ------------------------------------------------------------------
            if MEDIAPIPE_DISPONIBLE:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = face_mesh.process(rgb_frame)

                if results.multi_face_landmarks:
                    for face_landmarks in results.multi_face_landmarks:
                        lm = face_landmarks.landmark

                        # ── A. Bounding box del rostro ──────────────────────
                        x_coords = [int(lm[i].x * w) for i in range(468)]
                        y_coords = [int(lm[i].y * h) for i in range(468)]
                        x_min, x_max = min(x_coords), max(x_coords)
                        y_min, y_max = min(y_coords), max(y_coords)

                        # Guardar crop para DeepFace (con margen generoso)
                        mg = int((x_max - x_min) * 0.25)
                        cx1 = max(0, x_min - mg);  cx2 = min(w, x_max + mg)
                        cy1 = max(0, y_min - mg);  cy2 = min(h, y_max + mg)
                        crop = frame_limpio[cy1:cy2, cx1:cx2]
                        if crop.size > 0:
                            self._mediapipe_crop = crop.copy()

                        # ── B. EAR — Detección de somnolencia ───────────────
                        ear_izq = calcular_ear(lm, OJO_IZQ_EAR, w, h)
                        ear_der = calcular_ear(lm, OJO_DER_EAR, w, h)
                        ear_promedio = (ear_izq + ear_der) / 2.0
                        estado_api_global["ear"] = round(ear_promedio, 3)

                        if ear_promedio < EAR_UMBRAL:
                            self.somnolencia_counter += 1
                        else:
                            self.somnolencia_counter = max(0, self.somnolencia_counter - 1)

                        es_somnoliento = self.somnolencia_counter >= EAR_FRAMES_SOMNO

                        # ── C. Iris Gaze — Dirección de la mirada ───────────
                        iris_izq_lm = lm[IRIS_IZQ]
                        iris_der_lm = lm[IRIS_DER]

                        # Ratio horizontal: 0=extremo ext, 1=extremo int
                        rng_h_izq = (lm[OJO_IZQ_INT].x - lm[OJO_IZQ_EXT].x) + 1e-6
                        rng_h_der = (lm[OJO_DER_EXT].x - lm[OJO_DER_INT].x) + 1e-6
                        ratio_h_izq = (iris_izq_lm.x - lm[OJO_IZQ_EXT].x) / rng_h_izq
                        ratio_h_der = (iris_der_lm.x - lm[OJO_DER_INT].x) / rng_h_der
                        ratio_h = (ratio_h_izq + ratio_h_der) / 2.0

                        # Ratio vertical: 0=arriba, 1=abajo
                        rng_v_izq = (lm[OJO_IZQ_BOT].y - lm[OJO_IZQ_TOP].y) + 1e-6
                        rng_v_der = (lm[OJO_DER_BOT].y - lm[OJO_DER_TOP].y) + 1e-6
                        ratio_v_izq = (iris_izq_lm.y - lm[OJO_IZQ_TOP].y) / rng_v_izq
                        ratio_v_der = (iris_der_lm.y - lm[OJO_DER_TOP].y) / rng_v_der
                        ratio_v = (ratio_v_izq + ratio_v_der) / 2.0

                        mirada_h = "Izquierda" if ratio_h < 0.35 else ("Derecha" if ratio_h > 0.65 else "Centro")
                        # ratio_v alto = iris abajo en imagen = mirando abajo
                        mirada_v = "Abajo"  if ratio_v > 0.65 else ("Arriba" if ratio_v < 0.35 else "Centro")

                        if mirada_h == "Centro" and mirada_v == "Centro":
                            mirada = "Centro"
                        else:
                            partes = [p for p in [mirada_v, mirada_h] if p != "Centro"]
                            mirada = "-".join(partes)
                        estado_api_global["mirada"] = mirada

                        # ── D. Head Pose 3D (pitch / yaw) ───────────────────
                        face_2d = np.array([
                            [int(lm[1].x * w),   int(lm[1].y * h)],
                            [int(lm[152].x * w), int(lm[152].y * h)],
                            [int(lm[33].x * w),  int(lm[33].y * h)],
                            [int(lm[263].x * w), int(lm[263].y * h)],
                            [int(lm[61].x * w),  int(lm[61].y * h)],
                            [int(lm[291].x * w), int(lm[291].y * h)]
                        ], dtype=np.float64)

                        focal_length = 1 * w
                        # BUG FIX: cx=w/2, cy=h/2 (antes estaban invertidos)
                        cam_matrix  = np.array([[focal_length, 0, w/2],
                                                [0, focal_length, h/2],
                                                [0, 0, 1]], dtype=np.float64)
                        dist_matrix = np.zeros((4, 1), dtype=np.float64)

                        success, rot_vec, _ = cv2.solvePnP(
                            face_3d_model, face_2d, cam_matrix, dist_matrix)

                        if success:
                            rmat, _ = cv2.Rodrigues(rot_vec)
                            angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
                            pitch, yaw = angles[0], angles[1]

                            # ── E. Estado de atención final ─────────────────
                            # Somnoliento tiene prioridad sobre Distraído
                            if es_somnoliento:
                                estado_api_global["atencion"] = "Somnoliento"
                                color_ejes   = (0, 165, 255)  # Naranja
                                texto_atencion = "SOMNOLIENTO"
                            elif pitch < -30 or pitch > 35 or yaw < -35 or yaw > 35:
                                estado_api_global["atencion"] = "Distraído"
                                color_ejes   = (0, 0, 255)    # Rojo
                                texto_atencion = "DISTRAIDO"
                            else:
                                estado_api_global["atencion"] = "Enfocado"
                                color_ejes   = (0, 255, 0)    # Verde
                                texto_atencion = "ATENTO"

                            self.calcular_indice()

                            # ── F. Overlay visual ───────────────────────────
                            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color_ejes, 2)

                            emocion_visual = estado_api_global["emocion"].upper()
                            if emocion_visual == "FELICIDAD": emocion_visual = "FELIZ"

                            # Caja de texto superior (3 líneas: emoción, atención, mirada)
                            banner_y = max(0, y_min - 65)
                            cv2.rectangle(frame,
                                          (x_min, banner_y),
                                          (x_max + 80, y_min),
                                          (15, 23, 42), -1)
                            cv2.putText(frame, f"EMOCION: {emocion_visual}",
                                        (x_min + 5, banner_y + 18),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 255), 2)
                            cv2.putText(frame, texto_atencion,
                                        (x_min + 5, banner_y + 38),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, color_ejes, 2)
                            cv2.putText(frame, f"MIRADA: {mirada.upper()}",
                                        (x_min + 5, banner_y + 58),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.44, (255, 200, 0), 1)

                            # Punto iris izquierdo
                            ix_l = int(iris_izq_lm.x * w)
                            iy_l = int(iris_izq_lm.y * h)
                            cv2.circle(frame, (ix_l, iy_l), 3, (0, 255, 255), -1)
                            # Punto iris derecho
                            ix_r = int(iris_der_lm.x * w)
                            iy_r = int(iris_der_lm.y * h)
                            cv2.circle(frame, (ix_r, iy_r), 3, (0, 255, 255), -1)

                else:
                    # Sin rostro detectado: reset somnolencia
                    self.somnolencia_counter = 0

            # ------------------------------------------------------------------
            # 2. ENVÍO A DEEPFACE usando crop de MediaPipe (más confiable)
            # ------------------------------------------------------------------
            if CONFIG["emocion_activa"] and frame_counter % CONFIG["analisis_emocion_cada"] == 0:
                if self._mediapipe_crop is not None and not self.frame_queue.full():
                    self.frame_queue.put(self._mediapipe_crop.copy())
            
            # Codificación MJPEG
            ret_encode, buffer = cv2.imencode('.jpg', frame)
            if ret_encode:
                frame_global_bytes = buffer.tobytes()
                
        cap.release()

motor_ia = EduInsightWebWorker()
threading.Thread(target=motor_ia.capturar_camara, daemon=True).start()

# ==============================================================================
# RUTAS FASTAPI (WEBSOCKETS)
# ==============================================================================
def generador_video():
    while True:
        if frame_global_bytes is not None:
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame_global_bytes + b'\r\n')
        time.sleep(0.03)

@app.get("/api/video_feed")
def video_feed():
    return StreamingResponse(generador_video(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            await websocket.send_json(estado_api_global)
            await asyncio.sleep(0.1)
    except Exception:
        pass

if __name__ == "__main__":
    import socket, subprocess

    def liberar_puerto(puerto: int):
        """Mata el proceso que ocupa el puerto si existe (Windows)."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", puerto)) != 0:
                return  # puerto libre, nada que hacer
        try:
            result = subprocess.check_output(
                f'netstat -ano | findstr ":{puerto} "', shell=True
            ).decode()
            for line in result.splitlines():
                parts = line.split()
                if parts and parts[1].endswith(f":{puerto}") and parts[3] == "LISTENING":
                    pid = parts[-1]
                    subprocess.call(f"taskkill /F /PID {pid}", shell=True)
                    print(f"⚠️  Puerto {puerto} ocupado por PID {pid} — proceso liberado.")
                    break
        except Exception:
            pass

    liberar_puerto(8000)
    print("🚀 Servidor Web Activo. Levantando en http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)