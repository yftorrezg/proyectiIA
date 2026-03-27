# ==============================================================================
# PROYECTO: Edu-Insight - BACKEND WEB STREAMING (RTX 3060 + WEBSOCKETS + POSE 3D)
# ==============================================================================
import os
import sys
import time
import getpass # Librería nativa que causaba el conflicto

print("\n" + "█"*70)
print("⏳ INICIANDO MOTORES CUDA Y COMPILANDO KERNELS 3D...")
print("⚠️ ATENCIÓN: ESTE PROCESO NO ES UN BUCLE.")
print("⚠️ LA PRIMERA VEZ PUEDE TARDAR ENTRE 40 Y 60 SEGUNDOS EN PENSAR.")
print("⚠️ POR FAVOR, NO PRESIONES 'CTRL+C'. ESPERA PACIENTEMENTE.")
print("█"*70 + "\n")

# ==============================================================================
# 0. ⚠️ MONKEY PATCHING (TÉCNICA AVANZADA DE INTERCEPCIÓN)
# ==============================================================================
# Reescribimos la función interna de Python en caliente. 
# Si PyTorch pregunta por el usuario, le devolvemos un texto fijo sin buscar en el SO.
getpass.getuser = lambda: "Usuario_EduInsight"

# Respaldos adicionales de variables
os.environ["USERNAME"] = "Usuario_EduInsight"
os.environ["USER"] = "Usuario_EduInsight"
os.environ["LOGNAME"] = "Usuario_EduInsight"

# Obligamos a PyTorch a guardar su caché aquí
os.environ["TORCHINDUCTOR_CACHE_DIR"] = os.path.join(os.getcwd(), ".torch_cache")


# ==============================================================================
# 1. ⚠️ AUTO-INYECTOR DE DLLS DE NVIDIA (Para RTX 3060)
# ==============================================================================
if os.name == 'nt':
    for path_dir in sys.path:
        for lib in ["cublas", "cudnn"]:
            ruta_bin = os.path.join(path_dir, "nvidia", lib, "bin")
            if os.path.exists(ruta_bin):
                os.environ["PATH"] = ruta_bin + os.pathsep + os.environ.get("PATH", "")
                if hasattr(os, 'add_dll_directory'):
                    try: os.add_dll_directory(ruta_bin)
                    except: pass

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# ==============================================================================
# 2. ⚠️ IMPORTACIÓN CRÍTICA: MEDIAPIPE (Auto-Reparación Quirúrgica)
# ==============================================================================
import mediapipe as mp
import numpy as np
import importlib.metadata

try:
    # Importación limpia y directa (Evita el Lazy Loading problemático)
    import mediapipe.python.solutions.face_mesh as mp_face_mesh
    face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    MEDIAPIPE_DISPONIBLE = True
    print("✅ Radar 3D (MediaPipe) inicializado correctamente en memoria.")
except Exception as e:
    error_str = str(e)
    try:
        mp_version = importlib.metadata.version("mediapipe")
    except:
        mp_version = "0.0.0"
        
    # Inteligencia MLOps: Si detecta el bug y NO estamos en la versión dorada (0.10.5)
    if ("solutions" in error_str or "mediapipe.python" in error_str or "attribute" in error_str) and mp_version != "0.10.5":
        print(f"\n🚨 ALERTA MLOps: MediaPipe v{mp_version} es incompatible con nuestro Protobuf 3.x.")
        print("🤖 Aplicando 'Downgrade Quirúrgico' a la versión Dorada (0.10.5)...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "mediapipe==0.10.5"])
        print("✅ Reparación completada. Reiniciando motores...\n")
        os.execv(sys.executable, ['python'] + sys.argv)
    else:
        MEDIAPIPE_DISPONIBLE = False
        print(f"⚠️ Error definitivo cargando MediaPipe: {e}")

# Modelo 3D genérico de un rostro humano (para calcular Geometría Epipolar)
face_3d_model = np.array([
    [0.0, 0.0, 0.0],            # Nariz
    [0.0, -330.0, -65.0],       # Barbilla
    [-225.0, 170.0, -135.0],    # Esquina ojo izquierdo
    [225.0, 170.0, -135.0],     # Esquina ojo derecho
    [-150.0, -150.0, -125.0],   # Esquina boca izquierda
    [150.0, -150.0, -125.0]     # Esquina boca derecha
], dtype=np.float64)


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
except: SR_DISPONIBLE = False

from transformers import pipeline

# ==============================================================================
# INICIALIZACIÓN DE FASTAPI
# ==============================================================================
app = FastAPI(title="Edu-Insight Streaming API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# Variables Globales
frame_global_bytes = None
estado_api_global = {
    "emocion": "Neutral",
    "texto": "Esperando voz...",
    "sentimiento": "Neutral",
    "atencion": "Enfocado", 
    "indice_comprension": 50,
    "fps": 0
}

# ==============================================================================
# CLASE PRINCIPAL DE IA (Background)
# ==============================================================================
class EduInsightWebWorker:
    def __init__(self):
        self.audio_queue = queue.Queue(maxsize=5)
        self.frame_queue = queue.Queue(maxsize=3)
        self.running = True
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        self.inicializar_modelos()

    def inicializar_modelos(self):
        print(f"🔥 Hardware detectado: {self.device.upper()}")
        if MEDIAPIPE_DISPONIBLE:
            print("✅ ¡Geometría Epipolar activa lista para rastrear a 60FPS!")
        else:
            print("⚠️ Radar 3D falló. Usando Radar 2D de respaldo.")
        
        if WHISPER_DISPONIBLE:
            try:
                compute = "float16" if self.device == "cuda" else "int8"
                self.whisper = WhisperModel("small", device=self.device, compute_type=compute)
                if SR_DISPONIBLE:
                    threading.Thread(target=self.productor_audio, daemon=True).start()
                    threading.Thread(target=self.consumidor_audio, daemon=True).start()
            except Exception: self.whisper = None

        print("Cargando RoBERTa NLP (Procesamiento de Lenguaje Natural)...")
        dispositivo_nlp = 0 if self.device == "cuda" else -1
        self.analizador_nlp = pipeline("sentiment-analysis", model="pysentimiento/robertuito-sentiment-analysis", device=dispositivo_nlp)

        if DEEPFACE_DISPONIBLE:
            threading.Thread(target=self.worker_vision, daemon=True).start()

    def calcular_indice(self):
        # 1. Penalización extrema por distracción (Head Pose 3D)
        if estado_api_global["atencion"] == "Distraído":
            estado_api_global["indice_comprension"] = max(0, estado_api_global["indice_comprension"] - 40)
            return

        # 2. Cálculo normal multimodal
        score_facial, score_texto = 0, 0
        emocion = estado_api_global["emocion"]
        if emocion == "Felicidad": score_facial = 25
        elif emocion in ["Neutral", "Sorpresa"]: score_facial = 10
        elif emocion in ["Tristeza", "Miedo"]: score_facial = -15
        elif emocion in ["Enojo", "Disgusto"]: score_facial = -25
            
        sentimiento = estado_api_global["sentimiento"]
        if sentimiento == "POS": score_texto = 25
        elif sentimiento == "NEU": score_texto = 5
        elif sentimiento == "NEG": score_texto = -25
            
        nuevo_indice = 50 + score_facial + score_texto
        estado_api_global["indice_comprension"] = max(0, min(100, nuevo_indice))

    def productor_audio(self):
        recognizer = sr.Recognizer()
        recognizer.energy_threshold = 300
        recognizer.dynamic_energy_threshold = True
        recognizer.pause_threshold = 0.8
        try:
            mic = sr.Microphone()
            with mic as source: recognizer.adjust_for_ambient_noise(source, duration=2)
            while self.running:
                try:
                    with mic as source:
                        audio = recognizer.listen(source, timeout=3, phrase_time_limit=10)
                        if not self.audio_queue.full(): self.audio_queue.put(audio)
                except sr.WaitTimeoutError: continue
        except Exception: pass

    def consumidor_audio(self):
        while self.running:
            try: audio = self.audio_queue.get(timeout=1)
            except queue.Empty: continue
            if not self.whisper: continue
            try:
                with open("temp_audio.wav", "wb") as f: f.write(audio.get_wav_data())
                estado_api_global["texto"] = "Procesando..."
                segments, _ = self.whisper.transcribe("temp_audio.wav", language="es", beam_size=5)
                texto = " ".join([seg.text for seg in segments]).strip()
                if texto:
                    estado_api_global["texto"] = texto
                    try:
                        res_nlp = self.analizador_nlp(texto[:512])
                        estado_api_global["sentimiento"] = res_nlp[0]['label'] if isinstance(res_nlp, list) else res_nlp['label']
                    except: estado_api_global["sentimiento"] = "NEU"
                    self.calcular_indice()
            except Exception: pass
            finally: self.audio_queue.task_done()

    def worker_vision(self):
        while self.running:
            try: rostro_img = self.frame_queue.get(timeout=1)
            except queue.Empty: continue
            try:
                cv2.imwrite("temp_rostro.jpg", rostro_img)
                resultado = DeepFace.analyze("temp_rostro.jpg", actions=['emotion'], enforce_detection=False, silent=True)
                res_dict = resultado[0] if isinstance(resultado, list) else resultado
                emocion = res_dict.get('dominant_emotion', 'neutral')
                
                traduccion = { "angry": "Enojo", "disgust": "Disgusto", "fear": "Miedo", "happy": "Felicidad", "sad": "Tristeza", "surprise": "Sorpresa", "neutral": "Neutral" }
                estado_api_global["emocion"] = traduccion.get(emocion, emocion)
                self.calcular_indice()
            except: pass
            finally: self.frame_queue.task_done()

    def capturar_camara(self):
        global frame_global_bytes
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)
        
        frame_counter = 0
        fps_time = time.time()
        fps_count = 0
        
        while self.running:
            ret, frame = cap.read()
            if not ret: continue
            
            frame = cv2.flip(frame, 1)
            frame_counter += 1
            fps_count += 1
            h, w, _ = frame.shape
            
            if time.time() - fps_time >= 1:
                estado_api_global["fps"] = fps_count
                fps_count = 0
                fps_time = time.time()

            x_min, y_min, x_max, y_max = w, h, 0, 0

            # ------------------------------------------------------------------
            # LA MAGIA DEL RADAR 3D (GEOMETRÍA EPIPOLAR CON MEDIAPIPE)
            # ------------------------------------------------------------------
            if MEDIAPIPE_DISPONIBLE:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = face_mesh.process(rgb_frame)
                
                if results.multi_face_landmarks:
                    for face_landmarks in results.multi_face_landmarks:
                        face_2d = []
                        for idx, lm in enumerate(face_landmarks.landmark):
                            if idx in [1, 152, 33, 263, 61, 291]:
                                x, y = int(lm.x * w), int(lm.y * h)
                                face_2d.append([x, y])
                            
                            x, y = int(lm.x * w), int(lm.y * h)
                            if x < x_min: x_min = x
                            if y < y_min: y_min = y
                            if x > x_max: x_max = x
                            if y > y_max: y_max = y

                        face_2d = np.array(face_2d, dtype=np.float64)
                        focal_length = 1 * w
                        cam_matrix = np.array([[focal_length, 0, h/2], [0, focal_length, w/2], [0, 0, 1]])
                        dist_matrix = np.zeros((4, 1), dtype=np.float64)
                        
                        # ALGORITMO SOLVE PNP: Proyecta la cara plana 2D a un espacio 3D
                        success, rot_vec, trans_vec = cv2.solvePnP(face_3d_model, face_2d, cam_matrix, dist_matrix)
                        
                        if success:
                            rmat, _ = cv2.Rodrigues(rot_vec)
                            angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
                            
                            pitch = angles[0] * 360 # Inclinación (arriba/abajo)
                            yaw = angles[1] * 360   # Giro (izquierda/derecha)
                            
                            # EVALUACIÓN DE ATENCIÓN
                            if pitch < -10 or yaw < -15 or yaw > 15:
                                estado_api_global["atencion"] = "Distraído"
                                color_ejes = (0, 0, 255) # Rojo
                            else:
                                estado_api_global["atencion"] = "Enfocado"
                                color_ejes = (0, 255, 0) # Verde
                                
                            self.calcular_indice()

                            # Dibujar el Recuadro y los Ejes 3D (X, Y, Z) saliendo de la nariz
                            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color_ejes, 2)
                            cv2.drawFrameAxes(frame, cam_matrix, dist_matrix, rot_vec, trans_vec, 150, 3)
            else:
                # Fallback 2D si MediaPipe falla
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                rostros = self.face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(100, 100))
                if len(rostros) > 0:
                    rx, ry, rw, rh = max(rostros, key=lambda rect: rect[2] * rect[3])
                    x_min, y_min = rx, ry
                    x_max, y_max = rx + rw, ry + rh
                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (16, 185, 129), 2)
                    estado_api_global["atencion"] = "Enfocado"
                    self.calcular_indice()

            # Envío a DeepFace
            if frame_counter % 15 == 0 and x_max > x_min:
                rostro_recortado = frame[max(0, y_min-20):y_max+20, max(0, x_min-20):x_max+20]
                if not self.frame_queue.full() and rostro_recortado.size > 0:
                    self.frame_queue.put(rostro_recortado.copy())
            
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
    print("🚀 Servidor Web Activo. Levantando en http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
     