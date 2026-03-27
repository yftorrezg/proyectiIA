# ==============================================================================
# PROYECTO: Edu-Insight - BACKEND WEB STREAMING (RTX 3060 + WEBSOCKETS + POSE 3D)
# ==============================================================================
import os
import sys
import time
import getpass 

print("\n" + "█"*70)
print("⏳ INICIANDO MOTORES CUDA Y COMPILANDO KERNELS 3D...")
print("⚠️ ATENCIÓN: ESTE PROCESO NO ES UN BUCLE.")
print("⚠️ LA PRIMERA VEZ PUEDE TARDAR ENTRE 40 Y 60 SEGUNDOS EN PENSAR.")
print("⚠️ POR FAVOR, NO PRESIONES 'CTRL+C'. ESPERA PACIENTEMENTE.")
print("█"*70 + "\n")

# ==============================================================================
# 0. ⚠️ MONKEY PATCHING (TÉCNICA AVANZADA DE INTERCEPCIÓN)
# ==============================================================================
getpass.getuser = lambda: "Usuario_EduInsight"

os.environ["USERNAME"] = "Usuario_EduInsight"
os.environ["USER"] = "Usuario_EduInsight"
os.environ["LOGNAME"] = "Usuario_EduInsight"
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
# 2. ⚠️ IMPORTACIÓN CRÍTICA: MEDIAPIPE (El Radar 3D)
# ==============================================================================
import mediapipe as mp
import numpy as np

try:
    if hasattr(mp, 'solutions'):
        mp_face_mesh = mp.solutions.face_mesh
    else:
        import mediapipe.python.solutions.face_mesh as mp_face_mesh
        
    face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    MEDIAPIPE_DISPONIBLE = True
    print("✅ Radar 3D (MediaPipe) inicializado correctamente en memoria.")
except Exception as e:
    MEDIAPIPE_DISPONIBLE = False
    print(f"⚠️ Error cargando MediaPipe: {e}")

# 🔴 CORRECCIÓN MATEMÁTICA CRÍTICA: Ejes Y corregidos para OpenCV
face_3d_model = np.array([
    [0.0, 0.0, 0.0],            # 1. Nariz
    [0.0, 330.0, -65.0],        # 2. Barbilla (Y positivo = Abajo en la imagen)
    [-225.0, -170.0, -135.0],   # 3. Ojo izquierdo (Y negativo = Arriba en la imagen)
    [225.0, -170.0, -135.0],    # 4. Ojo derecho
    [-150.0, 150.0, -125.0],    # 5. Boca izquierda
    [150.0, 150.0, -125.0]      # 6. Boca derecha
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
except: 
    SR_DISPONIBLE = False
    print("\n❌ LIBRERÍA FALTANTE: No se detectó 'SpeechRecognition' o 'pyaudio'.")
    print("❌ El micrófono estará APAGADO. Ejecuta: pip install SpeechRecognition pyaudio\n")

from transformers import pipeline

# ==============================================================================
# INICIALIZACIÓN DE FASTAPI
# ==============================================================================
app = FastAPI(title="Edu-Insight Streaming API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

frame_global_bytes = None
estado_api_global = {
    "emocion": "Neutral",
    "texto": "Esperando voz... (Habla para comenzar)",
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
        self.audio_queue = queue.Queue(maxsize=10)
        self.frame_queue = queue.Queue(maxsize=3)
        self.running = True
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        self.inicializar_modelos()

    def inicializar_modelos(self):
        print(f"🔥 Hardware detectado: {self.device.upper()}")
        
        if WHISPER_DISPONIBLE:
            try:
                print("🎙️ Cargando Whisper (Forzando CPU para evitar colisiones CUDA)...")
                self.whisper = WhisperModel("small", device="cpu", compute_type="int8")
                if SR_DISPONIBLE:
                    threading.Thread(target=self.productor_audio, daemon=True).start()
                    threading.Thread(target=self.consumidor_audio, daemon=True).start()
                else:
                    print("⚠️ Whisper está listo, pero el hardware de audio está apagado.")
            except Exception as e: 
                print(f"❌ Error cargando Whisper: {e}")
                self.whisper = None

        print("Cargando RoBERTa NLP (Procesamiento de Lenguaje Natural)...")
        dispositivo_nlp = 0 if self.device == "cuda" else -1
        self.analizador_nlp = pipeline("sentiment-analysis", model="pysentimiento/robertuito-sentiment-analysis", device=dispositivo_nlp)

        if DEEPFACE_DISPONIBLE:
            threading.Thread(target=self.worker_vision, daemon=True).start()

    def calcular_indice(self):
        """Módulo Matemático de Fusión Cognitiva con Suavizado Exponencial"""
        score_facial, score_texto = 0, 0
        emocion = estado_api_global["emocion"]
        
        # Ponderación facial
        if emocion == "Felicidad": score_facial = 25
        elif emocion in ["Neutral", "Sorpresa"]: score_facial = 15
        elif emocion in ["Tristeza", "Miedo"]: score_facial = -15
        elif emocion in ["Enojo", "Disgusto"]: score_facial = -25
            
        # Ponderación semántica (Voz)
        sentimiento = estado_api_global["sentimiento"]
        if sentimiento == "POS": score_texto = 25
        elif sentimiento == "NEU": score_texto = 5
        elif sentimiento == "NEG": score_texto = -25
            
        nuevo_indice_crudo = 50 + score_facial + score_texto
        
        if estado_api_global["atencion"] == "Distraído":
            nuevo_indice_crudo -= 30 
            
        nuevo_indice_crudo = max(0, min(100, nuevo_indice_crudo))
        
        indice_actual = estado_api_global["indice_comprension"]
        indice_suavizado = int((indice_actual * 0.7) + (nuevo_indice_crudo * 0.3))
        
        estado_api_global["indice_comprension"] = indice_suavizado

    def productor_audio(self):
        """Micrófono auditando decibelios para evitar fallos de hardware"""
        recognizer = sr.Recognizer()
        
        recognizer.energy_threshold = 300  
        recognizer.dynamic_energy_threshold = True 
        recognizer.pause_threshold = 0.8  
        
        try:
            mic = sr.Microphone()
            with mic as source: 
                print("\n🎙️ [AUDIO] Calibrando ruido de fondo de tu habitación (2 seg)...")
                recognizer.adjust_for_ambient_noise(source, duration=2)
                print(f"🎙️ [AUDIO] Calibración completa. Umbral base fijado en: {recognizer.energy_threshold}")
                
            print("✅ [AUDIO] Micrófono ABIERTO y escuchando activamente.")
            while self.running:
                try:
                    with mic as source:
                        audio = recognizer.listen(source, timeout=3, phrase_time_limit=15)
                        if not self.audio_queue.full(): 
                            self.audio_queue.put(audio)
                            print(f"📡 [AUDIO] Voz detectada. Enviando paquete a Whisper...")
                except sr.WaitTimeoutError: 
                    continue 
        except Exception as e: 
            print(f"\n❌ [ERROR GRAVE AUDIO] No se puede acceder al micrófono: {e}")
            print("❌ Revisa la configuración de privacidad de Windows o tus dispositivos de audio.\n")

    def consumidor_audio(self):
        """Procesa el audio con Whisper y RoBERTa NLP"""
        while self.running:
            try: audio = self.audio_queue.get(timeout=1)
            except queue.Empty: continue
            
            if not self.whisper: continue
            try:
                with open("temp_audio.wav", "wb") as f: f.write(audio.get_wav_data())
                
                # 🔴 CORRECCIÓN AUDIO: Eliminamos el texto "IA Transcribiendo..." para que no parpadee
                segments, _ = self.whisper.transcribe("temp_audio.wav", language="es", beam_size=5, condition_on_previous_text=False)
                texto = " ".join([seg.text for seg in segments]).strip()
                
                if texto:
                    # Solo actualiza el texto si realmente transcribió algo
                    estado_api_global["texto"] = texto
                    print(f"🗣️ [WHISPER] Transcripción exitosa: '{texto}'")
                    try:
                        res_nlp = self.analizador_nlp(texto[:512])
                        estado_api_global["sentimiento"] = res_nlp[0]['label'] if isinstance(res_nlp, list) else res_nlp['label']
                    except: estado_api_global["sentimiento"] = "NEU"
                    self.calcular_indice()
            except Exception as e: 
                print(f"❌ Error interno en Whisper: {e}")
            finally: self.audio_queue.task_done()

    def worker_vision(self):
        """CNN DeepFace Directo en Memoria RAM"""
        while self.running:
            try: 
                rostro_numpy = self.frame_queue.get(timeout=1)
            except queue.Empty: 
                continue
                
            try:
                # 🔴 CORRECCIÓN VISIÓN: Pasamos el numpy array directo. Sin 'skip'. 
                # DeepFace usará su detector propio dentro del recorte generoso.
                resultado = DeepFace.analyze(img_path=rostro_numpy, actions=['emotion'], enforce_detection=False, silent=True)
                res_dict = resultado[0] if isinstance(resultado, list) else resultado
                emocion = res_dict.get('dominant_emotion', 'neutral')
                
                traduccion = { "angry": "Enojo", "disgust": "Disgusto", "fear": "Miedo", "happy": "Felicidad", "sad": "Tristeza", "surprise": "Sorpresa", "neutral": "Neutral" }
                estado_api_global["emocion"] = traduccion.get(emocion, emocion)
                self.calcular_indice()
            except Exception as e:
                pass
            finally: 
                self.frame_queue.task_done()

    def capturar_camara(self):
        """Motor Gráfico Principal de OpenCV con Telemetría Visual Inyectada"""
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
            
            frame_limpio = frame.copy() 
            
            frame_counter += 1
            fps_count += 1
            h, w, _ = frame.shape
            
            if time.time() - fps_time >= 1:
                estado_api_global["fps"] = fps_count
                fps_count = 0
                fps_time = time.time()

            x_min, y_min, x_max, y_max = w, h, 0, 0

            # ------------------------------------------------------------------
            # LA MAGIA DEL RADAR 3D (GEOMETRÍA EPIPOLAR CORREGIDA)
            # ------------------------------------------------------------------
            if MEDIAPIPE_DISPONIBLE:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = face_mesh.process(rgb_frame)
                
                if results.multi_face_landmarks:
                    for face_landmarks in results.multi_face_landmarks:
                        
                        p_nariz = face_landmarks.landmark[1]
                        p_barbilla = face_landmarks.landmark[152]
                        p_ojo_izq = face_landmarks.landmark[33]
                        p_ojo_der = face_landmarks.landmark[263]
                        p_boca_izq = face_landmarks.landmark[61]
                        p_boca_der = face_landmarks.landmark[291]
                        
                        face_2d = np.array([
                            [int(p_nariz.x * w), int(p_nariz.y * h)],
                            [int(p_barbilla.x * w), int(p_barbilla.y * h)],
                            [int(p_ojo_izq.x * w), int(p_ojo_izq.y * h)],
                            [int(p_ojo_der.x * w), int(p_ojo_der.y * h)],
                            [int(p_boca_izq.x * w), int(p_boca_izq.y * h)],
                            [int(p_boca_der.x * w), int(p_boca_der.y * h)]
                        ], dtype=np.float64)
                        
                        # Extraer bounding box
                        x_coords = [int(lm.x * w) for lm in face_landmarks.landmark]
                        y_coords = [int(lm.y * h) for lm in face_landmarks.landmark]
                        x_min, x_max = min(x_coords), max(x_coords)
                        y_min, y_max = min(y_coords), max(y_coords)
                        
                        focal_length = 1 * w
                        cam_matrix = np.array([[focal_length, 0, h/2], [0, focal_length, w/2], [0, 0, 1]])
                        dist_matrix = np.zeros((4, 1), dtype=np.float64)
                        
                        success, rot_vec, trans_vec = cv2.solvePnP(face_3d_model, face_2d, cam_matrix, dist_matrix)
                        
                        if success:
                            rmat, _ = cv2.Rodrigues(rot_vec)
                            angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
                            
                            pitch = angles[0] 
                            yaw = angles[1]   
                            
                            # 🔴 CORRECCIÓN DISTRACCIÓN: Umbrales super permisivos.
                            # Solo se marca distraído si miras MUY abajo, MUY arriba, o totalmente de lado.
                            if pitch < -25 or pitch > 30 or yaw < -35 or yaw > 35:
                                estado_api_global["atencion"] = "Distraído"
                                color_ejes = (0, 0, 255) # Rojo
                                texto_atencion = f"DISTRAIDO (Yaw:{int(yaw)} Pitch:{int(pitch)})"
                            else:
                                estado_api_global["atencion"] = "Enfocado"
                                color_ejes = (0, 255, 0) # Verde
                                texto_atencion = f"ATENTO (Yaw:{int(yaw)} Pitch:{int(pitch)})"
                                
                            self.calcular_indice()

                            # Dibujar el Recuadro 
                            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color_ejes, 2)
                            
                            # 🔴 SE ELIMINÓ cv2.drawFrameAxes PARA NO DIBUJAR LAS LÍNEAS ROJO, AZUL, VERDE
                            
                            # Etiqueta de Emoción y Atención superior
                            emocion_visual = estado_api_global["emocion"].upper()
                            if emocion_visual == "FELICIDAD": emocion_visual = "FELIZ"
                            
                            cv2.rectangle(frame, (x_min, max(0, y_min - 45)), (x_max + 60, y_min), (15, 23, 42), -1)
                            cv2.putText(frame, f"EMOCION: {emocion_visual}", (x_min + 5, max(15, y_min - 25)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                            cv2.putText(frame, texto_atencion, (x_min + 5, max(35, y_min - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_ejes, 2)

            # SMART CROP MEJORADO: Usamos el frame limpio para DeepFace
            if frame_counter % 10 == 0 and x_max > x_min:
                margen_x = int((x_max - x_min) * 0.3)
                margen_y_top = int((y_max - y_min) * 0.4) # Mucha frente y cejas
                margen_y_bot = int((y_max - y_min) * 0.2) 
                
                y1_crop = max(0, y_min - margen_y_top)
                y2_crop = min(h, y_max + margen_y_bot)
                x1_crop = max(0, x_min - margen_x)
                x2_crop = min(w, x_max + margen_x)
                
                rostro_smart_crop = frame_limpio[y1_crop:y2_crop, x1_crop:x2_crop] # Usamos el clon LIMPIO
                
                if not self.frame_queue.full() and rostro_smart_crop.size > 0:
                    self.frame_queue.put(rostro_smart_crop.copy())
            
            # Codificación MJPEG del frame con dibujos para la Web
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