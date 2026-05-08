# 🧠 Edu-Insight PRO — Documentación Técnica Completa

> **Sistema de Monitoreo Cognitivo Multimodal en Tiempo Real**
> Análisis simultáneo de visión, audio y lenguaje natural mediante IA de última generación.

---

## 📋 Índice

1. [¿Qué es Edu-Insight?](#1-qué-es-edu-insight)
2. [Stack Tecnológico](#2-stack-tecnológico)
3. [Arquitecturas de IA Utilizadas](#3-arquitecturas-de-ia-utilizadas)
   - [3.1 MediaPipe Face Mesh — Geometría 3D](#31-mediapipe-face-mesh--geometría-3d)
   - [3.2 DeepFace — Detección de Emociones (CNN)](#32-deepface--detección-de-emociones-cnn)
   - [3.3 Faster-Whisper — Transcripción de Voz (Transformer)](#33-faster-whisper--transcripción-de-voz-transformer)
   - [3.4 RoBERTa — Análisis de Sentimiento (NLP)](#34-roberta--análisis-de-sentimiento-nlp)
4. [Pipeline de Datos](#4-pipeline-de-datos)
   - [4.1 Pipeline Visual](#41-pipeline-visual-30-fps)
   - [4.2 Pipeline Acústico](#42-pipeline-acústico-continuo)
   - [4.3 Fusión Multimodal](#43-fusión-multimodal--índice-de-comprensión)
5. [Documentación del Código Bloque a Bloque](#5-documentación-del-código-bloque-a-bloque)
6. [Guía de Usuario — Paso a Paso](#6-guía-de-usuario--paso-a-paso)
7. [Solución de Problemas](#7-solución-de-problemas)
8. [Escalabilidad y Mejoras Futuras](#8-escalabilidad-y-mejoras-futuras)
9. [Clasificador de Atención — MLOps Training Lab](#9-clasificador-de-atención--mlops-training-lab)

---

## 1. ¿Qué es Edu-Insight?

**Edu-Insight PRO** es un sistema de **monitoreo cognitivo en tiempo real** diseñado para el contexto educativo. Analiza simultáneamente tres canales de información del usuario y los combina en una única métrica de comprensión.

### Canales de Análisis

| Canal | ¿Qué captura? | Tecnología principal |
|---|---|---|
| 👁️ **Orientacino visual** | Rostro, cabeza, estado de ojos, dirección de mirada | MediaPipe + DeepFace |
| 🎙️ **Acústico** | Voz y palabras dichas en español | Faster-Whisper (OpenAI) |
| 💬 **Semántico** | Significado emocional del discurso | RoBERTa NLP (pysentimiento) |

### Output Final

Estos tres canales se **fusionan matemáticamente** en un único número:

```
┌─────────────────────────────────────────────┐
│         ÍNDICE DE COMPRENSIÓN               │
│                                             │
│   🟢  70 – 100  →  Cognitivamente activo    │
│   🟡  40 – 69   →  Estado promedio          │
│   🔴   0 – 39   →  Distracción/frustración  │
└─────────────────────────────────────────────┘
```

---

## 2. Stack Tecnológico

### Backend

| Librería | Versión | Rol en el proyecto |
|---|---|---|
| `FastAPI` | 0.135.2 | Servidor web y API REST |
| `Uvicorn` | 0.42.0 | Servidor ASGI asíncrono |
| `WebSockets` | — | Canal full-duplex con el dashboard |
| `OpenCV` | 4.9.0 | Captura de cámara, overlay visual, codificación MJPEG |
| `MediaPipe` | 0.10.5 | Face Mesh 3D + Iris tracking |
| `DeepFace` | 0.0.99 | Clasificación de emociones faciales |
| `Faster-Whisper` | 1.2.1 | Transcripción de voz en GPU |
| `Transformers` | 5.4.0 | Pipeline NLP (RoBERTa) |
| `pysentimiento` | 0.7.3 | Modelo robertuito fine-tuneado en español |
| `PyTorch` | 2.7.1+cu118 | Backend de deep learning con CUDA |
| `TensorFlow` | 2.15.0 | Backend de DeepFace |
| `NumPy` | 1.26.4 | Álgebra lineal y operaciones matriciales |

### Frontend

| Tecnología | Rol |
|---|---|
| HTML5 + JavaScript vanilla | Dashboard interactivo |
| Tailwind CSS (CDN) | Diseño UI/UX responsivo |
| Lucide Icons (CDN) | Iconografía vectorial |
| SVG nativo | Gráfica de comprensión en tiempo real |
| WebSocket API del navegador | Recepción de datos del backend |

### Hardware Objetivo

| Componente | Especificación |
|---|---|
| CPU | Intel Core i7-12700H (14 cores, 20 threads) |
| GPU | NVIDIA RTX 3060 Laptop (6 GB VRAM, CUDA 11.8) |
| RAM | 32 GB DDR5 |
| OS | Windows 11 |

---

## 3. Arquitecturas de IA Utilizadas

El proyecto implementa **cuatro arquitecturas de IA distintas** trabajando en paralelo, cada una especializada en un tipo de dato diferente.

---

### 3.1 MediaPipe Face Mesh — Geometría 3D

**Tipo de arquitectura:** Graph Neural Network (GNN) + Regresión

**¿Qué hace?** Detecta **478 puntos (landmarks) tridimensionales** en el rostro en tiempo real, a 30 fps, directamente en CPU.

#### Arquitectura interna

```
Imagen RGB (960 × 540 px)
         │
         ▼
┌─────────────────────────────┐
│  BlazeFace (CNN liviana)    │  ← Detector de rostro
│  ~100 ms primera detección  │    Corre en CPU eficientemente
│  ~2 ms en tracking          │
└─────────────────────────────┘
         │
         ▼
┌─────────────────────────────┐
│  Face Mesh (GNN)            │  ← Predictor de landmarks
│  Entrada: 192 × 192 crop    │    Aprende relaciones entre
│  Salida: 478 puntos (x,y,z) │    puntos vecinos del rostro
└─────────────────────────────┘
         │
         ▼
   478 landmarks normalizados
   (0.0 a 1.0 en cada eje)
         │
    ┌────┴────┐
    │         │
    ▼         ▼
  EAR        solvePnP
(ojos)    (pose 3D)
```

**Analogía para principiantes:** Imagina que MediaPipe pone 478 stickers GPS en tu cara, cada uno con coordenadas exactas en el espacio 3D. Luego usa esas coordenadas para calcular ángulos con geometría pura, sin más IA adicional.

#### Eye Aspect Ratio (EAR) — Detección de somnolencia

El EAR mide qué tan abierto está el ojo usando 6 landmarks:

```
         p2 ─── p3
        /           \
      p1             p4    EAR = (|p2-p6| + |p3-p5|) / (2 × |p1-p4|)
        \           /
         p6 ─── p5
```

| Valor EAR | Significado |
|---|---|
| ~0.30 | Ojo completamente abierto |
| ~0.22 | Ojo casi cerrado (umbral de alerta) |
| EAR < 0.22 por 20 frames (~0.67 seg) | → Estado `"Somnoliento"` |

#### Head Pose 3D (solvePnP) — Detección de distracción

```python
# Puntos 3D reales del rostro humano promedio (en milímetros)
face_3d_model = [nariz, barbilla, ojo_izq, ojo_der, boca_izq, boca_der]

# Los mismos 6 puntos en píxeles de la imagen actual
face_2d = landmarks_detectados_en_frame

# solvePnP resuelve: ¿qué rotación explica esta proyección 2D?
success, rot_vec, _ = cv2.solvePnP(face_3d_model, face_2d, cam_matrix, ...)

# Convertir vector de rotación a ángulos de Euler
angles  = cv2.RQDecomp3x3(Rodrigues(rot_vec))
pitch   = angles[0]   # Inclinación Arriba / Abajo
yaw     = angles[1]   # Giro Izquierda / Derecha
```

| Condición | Estado detectado |
|---|---|
| `pitch < -30°` o `pitch > 35°` | `Distraído` |
| `yaw < -35°` o `yaw > 35°` | `Distraído` |
| Dentro del rango | `Enfocado` |

---

### 3.2 DeepFace — Detección de Emociones (CNN)

**Tipo de arquitectura:** Red Neuronal Convolucional (CNN)

**¿Qué hace?** Clasifica la emoción dominante del rostro en **7 categorías**.

#### ¿Cómo funciona una CNN?

```
Imagen del Rostro (crop recortado)
         │
         ▼
┌──────────────────────────────────┐
│  Bloque Conv 1 + ReLU + Pool     │  Detecta: bordes y líneas simples
└──────────────────────────────────┘
         │
         ▼
┌──────────────────────────────────┐
│  Bloque Conv 2 + ReLU + Pool     │  Detecta: ojos, nariz, boca
└──────────────────────────────────┘
         │
         ▼
┌──────────────────────────────────┐
│  Bloque Conv N + ReLU + Pool     │  Detecta: patrones de emoción
└──────────────────────────────────┘
         │
         ▼
┌──────────────────────────────────┐
│  Flatten + Dense (Fully Conn.)   │  Combina todos los rasgos
└──────────────────────────────────┘
         │
         ▼
┌──────────────────────────────────┐
│  Softmax — 7 salidas             │
│  angry:    0.03                  │
│  disgust:  0.01                  │
│  fear:     0.05                  │
│  happy:    0.72  ← ganador       │
│  neutral:  0.12                  │
│  sad:      0.04                  │
│  surprise: 0.03                  │
└──────────────────────────────────┘
         │
         ▼
   Salida: "Felicidad"
```

**Analogía:** Una CNN es como un sistema de lupas apiladas. Cada lupa detecta patrones más complejos: la primera ve bordes simples, la siguiente ve formas (ojos, cejas), la última reconoce "esto es una cara feliz".

#### Optimización aplicada

```python
# Analizar solo cada 15 frames = 2 veces/seg a 30 fps
# Las emociones duran 0.5–4 segundos; no tiene sentido analizarlas más seguido
if frame_counter % CONFIG["analisis_emocion_cada"] == 0:
    frame_queue.put(crop_rostro)

# 'skip': MediaPipe ya encontró la cara → DeepFace no necesita buscarla de nuevo
resultado = DeepFace.analyze(
    img_path="temp_rostro.jpg",
    detector_backend='skip',    # ← optimización clave
    enforce_detection=False
)
```

#### Mapa de emociones e impacto

| DeepFace (inglés) | Dashboard (español) | Impacto en índice |
|---|---|---|
| `happy` | Felicidad | +25 |
| `neutral` | Neutral | +15 |
| `surprise` | Sorpresa | +15 |
| `sad` | Tristeza | -15 |
| `fear` | Miedo | -15 |
| `angry` | Enojo | -25 |
| `disgust` | Disgusto | -25 |

---

### 3.3 Faster-Whisper — Transcripción de Voz (Transformer)

**Tipo de arquitectura:** Transformer Encoder-Decoder (Seq2Seq)

**¿Qué hace?** Convierte audio hablado en español a texto escrito, en tiempo real, en la GPU.

#### ¿Cómo funciona un Transformer de audio?

```
Audio crudo (WAV, 16 kHz, mono)
         │
         ▼
┌──────────────────────────────────┐
│  Espectrograma Mel (80 bandas)   │  Convierte el sonido en una
│  Eje X: tiempo                   │  "imagen de frecuencias"
│  Eje Y: frecuencia               │  (como una partitura musical)
└──────────────────────────────────┘
         │
         ▼
┌──────────────────────────────────┐
│  ENCODER Transformer             │
│  12 capas de Self-Attention      │  Aprende qué partes del audio
│                                  │  son fonemas importantes
└──────────────────────────────────┘
         │
         ▼
┌──────────────────────────────────┐
│  DECODER Transformer             │
│  Cross-Attention con encoder     │  Genera texto token a token:
│                                  │  "hola" → "mi" → "nombre" → ...
└──────────────────────────────────┘
         │
         ▼
   "Hola, mi nombre es Yafer"
```

#### ¿Por qué "Faster"?

| Whisper estándar (OpenAI) | Faster-Whisper (Systran) |
|---|---|
| PyTorch puro (float32) | CTranslate2 (float16 / int8) |
| ~3–5 seg por fragmento de audio | ~0.5–1 seg en GPU RTX 3060 |
| Sin VAD integrado | VAD filter incluido |
| Mayor uso de VRAM | Menor uso de VRAM |

**CTranslate2** convierte los pesos del modelo a formatos numéricos más eficientes para el hardware, logrando 2–4× más velocidad sin pérdida significativa de calidad.

#### Hiperparámetros y su significado

```python
segments, _ = self.whisper.transcribe(
    "temp_audio.wav",

    language="es",
    # Fuerza español → evita confusión con portugués, italiano, etc.
    # Sin esto, Whisper detecta el idioma en cada fragmento (más lento)

    beam_size=5,
    # Evalúa 5 hipótesis de texto simultáneamente, elige la más probable.
    # Analogía: un ajedrecista que piensa 5 jugadas antes de mover.
    # beam_size=1 es más rápido pero menos preciso (greedy decoding)

    vad_filter=True,
    # Voice Activity Detection: descarta silencios puros.
    # Evita "alucinaciones" (el modelo inventa texto donde no hay voz)

    vad_parameters=dict(
        min_silence_duration_ms=500  # 0.5 seg de silencio = fin de frase
    ),

    condition_on_previous_text=False,
    # Cada fragmento se procesa de forma independiente.
    # Evita que un error en un fragmento "contagie" al siguiente
)
```

---

### 3.4 RoBERTa — Análisis de Sentimiento (NLP)

**Tipo de arquitectura:** Transformer Encoder bidireccional (BERT-like)

**¿Qué hace?** Lee el texto transcrito y determina la polaridad emocional del discurso.

#### ¿Por qué `robertuito` y no ChatGPT?

`pysentimiento/robertuito-sentiment-analysis` es RoBERTa **entrenado en Twitter latinoamericano** y **fine-tuneado para sentimiento en español de América Latina**. Ventajas:

- Entiende jerga, modismos y lenguaje coloquial regional
- Es un modelo liviano de clasificación (milisegundos vs. segundos de GPT)
- No necesita prompt engineering ni API externa de pago
- Corre localmente en GPU sin costo por llamada

#### ¿Cómo funciona el análisis de sentimiento?

```
Texto: "No entiendo nada, es muy difícil para mí"
         │
         ▼
┌──────────────────────────────────────────────┐
│  Tokenizador RoBERTa                         │
│  [CLS] "No" "entiendo" "nada" "es"           │
│        "muy" "difícil" "para" "mí" [SEP]     │
│                                              │
│  [CLS] = token especial que resume           │
│           el significado de toda la frase    │
└──────────────────────────────────────────────┘
         │
         ▼
┌──────────────────────────────────────────────┐
│  12 Capas de Atención Bidireccional          │
│                                              │
│  A diferencia de Whisper (izq → der),        │
│  RoBERTa lee la frase COMPLETA en ambas      │
│  direcciones → entiende el contexto total    │
│                                              │
│  "no" + "difícil" juntos → negatividad fuerte│
└──────────────────────────────────────────────┘
         │
         ▼
┌──────────────────────────────────────────────┐
│  Clasificador lineal sobre vector [CLS]      │
│  POS: 0.04                                   │
│  NEU: 0.13                                   │
│  NEG: 0.83  ← ganador                        │
└──────────────────────────────────────────────┘
         │
         ▼
   "NEG" → penaliza el índice de comprensión en -25
```

#### Impacto en el Índice de Comprensión

| Label | Significado | Impacto |
|---|---|---|
| `POS` | Discurso positivo, confiado, entusiasta | +25 |
| `NEU` | Discurso neutral, informativo, descriptivo | +5 |
| `NEG` | Frustración, confusión, negatividad | -25 |

---

## 4. Pipeline de Datos

### 4.1 Pipeline Visual (30 fps)

```
╔══════════════════════════════════════════════════════════════════╗
║                      PIPELINE VISUAL                            ║
╚══════════════════════════════════════════════════════════════════╝

INPUT ──► Cámara Web (960×540 px @ 30 fps)
              │
              ▼
  ┌───────────────────────────────────────────┐
  │  TRANSFORM 1: Captura y espejo            │
  │  frame = cv2.flip(frame, 1)               │
  │  → Imagen no invertida (natural)          │
  └───────────────────────────────────────────┘
              │
              ▼
  ┌───────────────────────────────────────────┐
  │  TRANSFORM 2: MediaPipe FaceMesh          │
  │  BGR → RGB → 478 landmarks (x, y, z)     │
  └───────────────────────────────────────────┘
              │
    ┌─────────┼──────────┬──────────────┐
    ▼         ▼          ▼              ▼
  EAR       Iris      Head Pose      Bounding
 (ojos)    Gaze         3D             Box
    │         │       (solvePnP)      (crop)
    ▼         ▼          ▼              │
Somnol.    mirada   Distraído/         │
 count    (h + v)   Enfocado           ▼
                               frame_queue
                               (→ DeepFace)
              ▼
  ┌───────────────────────────────────────────┐
  │  TRANSFORM 3: DeepFace (cada 15 frames)   │
  │  crop_rostro → CNN → emoción dominante   │
  └───────────────────────────────────────────┘
              │
              ▼
OUTPUT ──► estado_api_global
           { atencion, mirada, ear, emocion, indice }
              │
              ▼
       cv2.imencode('.jpg') → MJPEG → navegador
```

### 4.2 Pipeline Acústico (continuo)

```
╔══════════════════════════════════════════════════════════════════╗
║                     PIPELINE ACÚSTICO                           ║
╚══════════════════════════════════════════════════════════════════╝

INPUT ──► Micrófono (16 kHz, mono)
              │
              ▼
  ┌───────────────────────────────────────────┐
  │  HILO PRODUCTOR: SpeechRecognition        │
  │  1. Calibración de ruido (2 seg inicio)   │
  │  2. Escucha bloques de máx. 5 seg         │
  │  3. audio_queue.put(audio_chunk)           │
  └───────────────────────────────────────────┘
              │
              │  Cola thread-safe (máx. 10 items)
              ▼
  ┌───────────────────────────────────────────┐
  │  HILO CONSUMIDOR: Faster-Whisper (GPU)    │
  │  audio_chunk → temp_audio.wav             │
  │  → Espectrograma Mel (80 bandas)          │
  │  → Transformer Encoder → Decoder          │
  │  → texto_transcrito (str)                 │
  └───────────────────────────────────────────┘
              │
              ▼
  ┌───────────────────────────────────────────┐
  │  RoBERTa NLP (GPU)                       │
  │  texto[:512] → tokenización              │
  │  → 12 capas bidireccionales              │
  │  → { "POS" | "NEU" | "NEG" }             │
  └───────────────────────────────────────────┘
              │
              ▼
OUTPUT ──► estado_api_global["texto"] y ["sentimiento"]
           → calcular_indice() → índice fusionado actualizado
```

### 4.3 Fusión Multimodal — Índice de Comprensión

```
╔══════════════════════════════════════════════════════════════════╗
║                    MÓDULO DE FUSIÓN                             ║
╚══════════════════════════════════════════════════════════════════╝

INPUTS:
  ┌──────────────┐  ┌──────────────┐  ┌───────────────────┐
  │   Emoción    │  │  Sentimiento │  │    Atención        │
  │  (DeepFace)  │  │  (RoBERTa)   │  │   (MediaPipe)      │
  │              │  │              │  │                   │
  │ Felicidad +25│  │  POS    +25  │  │ Distraído    -30  │
  │ Neutral  +15 │  │  NEU    +5   │  │ Somnoliento  -40  │
  │ Sorpresa +15 │  │  NEG    -25  │  │ Enfocado     ±0   │
  │ Tristeza -15 │  │              │  │                   │
  │ Miedo    -15 │  └──────────────┘  └───────────────────┘
  │ Enojo    -25 │
  └──────────────┘
        │                │                   │
        └────────────────┴───────────────────┘
                         │
                         ▼
               BASE = 50 (punto neutro)
               + score_facial
               + score_texto
               + penalización_atención
                         │
                         ▼
               clamp(valor, 0, 100)
                         │
                         ▼
      ┌──────────────────────────────────────┐
      │      SUAVIZADO EXPONENCIAL (EMA)     │
      │                                      │
      │  α = 0.3  →  peso del valor nuevo    │
      │  1-α = 0.7 →  peso del historial     │
      │                                      │
      │  índice = (anterior × 0.7) +         │
      │           (nuevo_crudo × 0.3)        │
      │                                      │
      │  Resultado: gráfica fluida y estable │
      └──────────────────────────────────────┘
                         │
                         ▼
OUTPUT ──► indice_comprension (0–100)
           → WebSocket → Dashboard → Gráfica SVG
```

---

## 5. Documentación del Código Bloque a Bloque

### Bloque 0 — Configuración Maestra (`CONFIG`)

```python
CONFIG = {
    "resolucion": (960, 540),
    # Balance entre calidad de imagen y rendimiento.
    # 1080p consumiría ~40% más GPU sin mejorar la detección de landmarks.

    "fps_objetivo": 30,
    # 30 fps es fluido para el ojo humano y suficiente para detectar
    # cambios cognitivos (que ocurren en escala de segundos, no ms).

    "analisis_emocion_cada": 15,
    # DeepFace es computacionalmente costoso.
    # Analizarlo cada 15 frames (= 2 veces/seg a 30fps) es óptimo:
    # las emociones faciales duran entre 0.5 y 4 segundos.

    "whisper_modelo": "base",
    # Tamaños disponibles ordenados por VRAM necesaria:
    #   tiny   (~390 MB VRAM) → rápido, menos preciso
    #   base   (~550 MB VRAM) → balance óptimo para RTX 3060  ← usamos este
    #   small  (~970 MB VRAM) → más preciso, algo más lento
    #   medium (~3 GB VRAM)   → muy preciso
    #   large  (~6 GB VRAM)   → máxima precisión, ocupa toda la VRAM

    "whisper_compute": "float16",
    # float32: precisión completa (lento)
    # float16: mitad de precisión numérica, 2-4× más rápido (usa Tensor Cores)
    # int8:    cuantización agresiva, más rápido, menor calidad

    "forzar_gpu_whisper": True,
    # True: si CUDA falla, el sistema lanza error en lugar de caer a CPU.
    # Garantiza rendimiento en tiempo real durante presentaciones.
}
```

---

### Bloque 1 — Inyector de DLLs NVIDIA (Windows)

```python
# PROBLEMA: En Windows, Python no encuentra automáticamente las DLLs de CUDA.
# Estas son archivos .dll que la GPU necesita para operaciones matemáticas.
# Sin ellas: "CUDA not available" aunque la GPU exista físicamente.

if os.name == 'nt':  # Solo en Windows ('nt' = Windows NT kernel)
    for sp in site.getsitepackages():
        for lib in ["cublas", "cudnn"]:
            # cublas: álgebra lineal de NVIDIA (base de toda red neuronal)
            # cudnn:  operaciones de deep learning optimizadas en hardware GPU
            ruta_bin = os.path.join(sp, "nvidia", lib, "bin")
            if os.path.exists(ruta_bin):
                os.environ["PATH"] = ruta_bin + os.pathsep + os.environ.get("PATH", "")
                try:
                    os.add_dll_directory(ruta_bin)  # API Python 3.8+
                except:
                    pass
```

---

### Bloque 2 — Inicialización de MediaPipe

```python
face_mesh = mp_face_mesh.FaceMesh(
    min_detection_confidence=0.5,
    # Umbral mínimo de confianza para aceptar una detección como válida.
    # 0.5 = equilibrio entre sensibilidad y falsos positivos.

    min_tracking_confidence=0.5,
    # Si el tracking baja de este umbral, re-detecta el rostro desde cero.

    refine_landmarks=True,
    # CRÍTICO: activa los 10 landmarks adicionales del iris (468–477).
    # Sin esto: 468 puntos, sin iris tracking.
    # Con esto: 478 puntos, con gaze tracking completo.

    max_num_faces=1
    # Solo analiza 1 cara. Analizar más incrementa el costo computacional.
)
```

---

### Bloque 3 — Función `calcular_ear()`

```python
def calcular_ear(landmarks, indices, w, h):
    """
    Eye Aspect Ratio — razón de apertura del ojo.
    Referencia: Soukupová & Čech (2016), CVWW.

    Geometría de los 6 puntos:
         p2 ─── p3
        /           \
      p1             p4
        \           /
         p6 ─── p5

    Fórmula:
    EAR = (|p2-p6| + |p3-p5|) / (2 × |p1-p4|)
           ← apertura vertical →   ← ancho →

    El denominador normaliza por el ancho del ojo, haciendo el ratio
    INVARIANTE al tamaño: un ojo pequeño y uno grande dan el mismo
    EAR si están igual de abiertos.
    """
    # Convertir coordenadas normalizadas (0.0-1.0) a píxeles reales
    p = [np.array([landmarks[i].x * w, landmarks[i].y * h]) for i in indices]

    # np.linalg.norm = distancia euclidiana √((x2-x1)² + (y2-y1)²)
    ear = (np.linalg.norm(p[1] - p[5]) + np.linalg.norm(p[2] - p[4])) / \
          (2.0 * np.linalg.norm(p[0] - p[3]) + 1e-6)
    #                                              ↑
    #                                   Epsilon: evita división por cero
    #                                   si los landmarks salen del frame
    return ear
```

---

### Bloque 4 — Arquitectura de Concurrencia (Hilos)

El sistema usa **4 hilos simultáneos** para que la cámara nunca se bloquee esperando que Whisper o DeepFace terminen:

```
HILO PRINCIPAL — capturar_camara()
│  Corre a 30 fps sin interrupciones
│  Ejecuta: MediaPipe, EAR, Gaze, Head Pose, overlay visual
│  Deposita crops de rostro en frame_queue
│
├── HILO PRODUCTOR — productor_audio()
│   │  Escucha el micrófono continuamente
│   │  Cuando detecta voz → deposita en audio_queue
│   │  timeout=3s para no bloquearse en silencio
│   │
│   └──► audio_queue (Queue thread-safe, máx 10 items)
│
├── HILO CONSUMIDOR — consumidor_audio()
│   │  Saca de audio_queue
│   │  Ejecuta Whisper → RoBERTa
│   │  Actualiza estado_api_global
│
└── HILO VISIÓN — worker_vision()
        Saca de frame_queue
        Ejecuta DeepFace CNN
        Actualiza estado_api_global

¿Por qué Queue y no variables compartidas directas?
→ Queue es thread-safe: múltiples hilos leen/escriben sin race conditions.
→ maxsize descarta items viejos si el consumidor es lento.
```

---

### Bloque 5 — Head Pose con `solvePnP`

```python
# PRINCIPIO: Geometría Epipolar (Perspective-n-Point problem)
# "Si conozco cómo se ve un objeto 3D conocido proyectado en 2D,
#  puedo calcular exactamente su orientación en el espacio."

# Mismo principio que usan drones y robots para estimar
# su posición mirando puntos de referencia conocidos.

# Modelo 3D del rostro humano promedio (milímetros, origen en la nariz)
face_3d_model = np.array([
    [0.0, 0.0, 0.0],            # Nariz (origen del sistema de coordenadas)
    [0.0, 330.0, -65.0],        # Barbilla
    [-225.0, -170.0, -135.0],   # Esquina exterior ojo izquierdo
    [225.0, -170.0, -135.0],    # Esquina exterior ojo derecho
    [-150.0, 150.0, -125.0],    # Comisura boca izquierda
    [150.0, 150.0, -125.0]      # Comisura boca derecha
])

# Matriz intrínseca de la cámara (aproximación para webcam estándar)
focal_length = 1 * w
cam_matrix = np.array([
    [focal_length, 0,            w/2],  # fx, 0, cx
    [0,            focal_length, h/2],  # 0, fy, cy
    [0,            0,            1  ]   # Plano normalizado
])

# Resolver la pose: rot_vec = vector de rotación en 3D
success, rot_vec, _ = cv2.solvePnP(face_3d_model, face_2d, cam_matrix, dist_matrix)

# Rodrigues: convierte vector de rotación a matriz de rotación 3×3
rmat, _ = cv2.Rodrigues(rot_vec)

# RQDecomp3x3: descompone la matriz en ángulos de Euler (pitch, yaw, roll)
angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
pitch = angles[0]   # Inclinación arriba/abajo
yaw   = angles[1]   # Giro izquierda/derecha
```

---

### Bloque 6 — Suavizado Exponencial (EMA)

```python
def calcular_indice(self):
    """
    Exponential Moving Average (EMA) — suavizado del Índice de Comprensión.

    PROBLEMA sin suavizado:
      Frame N:    Felicidad + POS + Enfocado   → índice = 100
      Frame N+1:  Enojo + NEG + Distraído      → índice = 0
      Resultado: la gráfica salta de 100 a 0 → inutilizable

    SOLUCIÓN con EMA (α = 0.3):
      El índice actual tiene 70% del historial + 30% del nuevo valor.
      Los cambios son graduales → gráfica fluida y legible.

    Fórmula: EMA_t = α × x_t + (1-α) × EMA_{t-1}

    Efecto de α:
      α = 0.1 → muy estable, reacciona lento a cambios reales
      α = 0.3 → balance óptimo (usado aquí)
      α = 0.9 → muy reactivo, casi sin suavizado
    """
    nuevo_indice_crudo = max(0, min(100,
        50 + score_facial + score_texto - penalizacion_atencion
    ))

    indice_actual = estado_api_global["indice_comprension"]
    indice_suavizado = int((indice_actual * 0.7) + (nuevo_indice_crudo * 0.3))

    estado_api_global["indice_comprension"] = indice_suavizado
```

---

### Bloque 7 — API FastAPI: Rutas y WebSocket

```python
# RUTA 1: Dashboard HTML servido desde el mismo origen
# Clave: al estar en localhost:8080, el navegador permite
# conexiones WebSocket y MJPEG sin bloqueos de seguridad CORS.
@app.get("/", response_class=HTMLResponse)
def root():
    with open("dashboard.html", encoding="utf-8") as f:
        return f.read()

# RUTA 2: Stream de video MJPEG
# MJPEG = Motion JPEG: secuencia de imágenes JPEG en un solo HTTP response.
# El tipo MIME especial indica al navegador: "cuando llegue '--frame',
# reemplaza la imagen anterior por la nueva."
@app.get("/api/video_feed")
def video_feed():
    return StreamingResponse(
        generador_video(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

# RUTA 3: WebSocket de telemetría
# WebSocket = conexión TCP persistente. El backend empuja datos
# proactivamente cada 100 ms sin que el cliente pregunte.
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()   # Completa el handshake WebSocket
    while True:
        await websocket.send_json(estado_api_global)  # dict → JSON
        await asyncio.sleep(0.1)  # 10 actualizaciones/seg
```

**¿Por qué WebSocket y no HTTP polling?**

| Técnica | Latencia | Overhead | Uso ideal |
|---|---|---|---|
| HTTP Polling cada 1s | ~1000 ms | Alto | Datos que cambian raramente |
| HTTP Long Polling | ~100 ms | Medio | Notificaciones ocasionales |
| **WebSocket** | **< 10 ms** | **Mínimo** | **Telemetría en tiempo real** |

---

## 6. Guía de Usuario — Paso a Paso

### Requisitos Previos

- ✅ Python 3.10 instalado
- ✅ GPU NVIDIA con drivers actualizados (o CPU — más lento)
- ✅ Micrófono y cámara web funcionales y libres de otras apps
- ✅ Navegador moderno: Chrome o Edge recomendado

---

### Paso 1 — Abrir el proyecto

Abre **Git Bash** en VS Code:

```bash
cd ~/Desktop/proyectos/ProyectoIA
```

---

### Paso 2 — Activar el entorno virtual

```bash
source .venv/Scripts/activate
```

✅ Éxito: la línea del terminal muestra `(.venv)` al inicio.

> **¿Qué es un entorno virtual?** Una carpeta aislada (`.venv/`) con las librerías del proyecto. Evita conflictos con el Python global del sistema.

---

### Paso 3 — Verificar la instalación (opcional)

```bash
python -c "
import torch, cv2, mediapipe, fastapi
print('Librerías OK')
print('CUDA disponible:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))
"
```

Salida esperada:

```
Librerías OK
CUDA disponible: True
GPU: NVIDIA GeForce RTX 3060 Laptop GPU
```

---

### Paso 4 — Ejecutar el servidor

```bash
python -m app.main
```

> También puedes usar el comando uvicorn directamente:
> ```bash
> uvicorn app.main:app --host 0.0.0.0 --port 8080
> ```
> **No uses `--reload`** — los hilos de cámara y audio no son compatibles con el modo de recarga automática.

Espera estos mensajes en orden (la primera vez: 40–60 seg mientras se cargan los modelos de IA):

```
==============================================================
  Edu-Insight MLOps v2 — Iniciando servidor
  Dashboard : http://localhost:8080/
  Admin Lab : http://localhost:8080/admin
  API Docs  : http://localhost:8080/docs
==============================================================

INFO  edu_insight: GPU : NVIDIA GeForce RTX 3060 Laptop GPU
INFO  edu_insight: VRAM: 5130 MB libres / 6143 MB total
INFO  edu_insight: MediaPipe Face Mesh inicializado.
INFO  edu_insight: Whisper cargado en CUDA.
INFO  edu_insight: DeepFace listo para procesamiento asíncrono.
INFO  edu_insight: Micrófono abierto.
INFO  uvicorn: Application startup complete.
```

---

### Paso 5 — Abrir las páginas

El sistema ahora tiene **dos páginas**:

| URL | Página | Para qué sirve |
|---|---|---|
| `http://localhost:8080/` | **Dashboard** | Monitoreo en tiempo real (cámara, índice, transcripción) |
| `http://localhost:8080/admin` | **Admin Panel** | Training Lab: entrenar y cambiar modelos |
| `http://localhost:8080/docs` | **API Docs** | Documentación interactiva (Swagger) |

> ⚠️ **Nunca** abras los `.html` directamente desde el explorador de archivos.
> El protocolo `file://` bloquea las conexiones WebSocket y de video.
> Siempre usa las URLs `http://localhost:8080/...`.

---

### Paso 6 — Interpretar el Dashboard (`/`)

```
┌──────────────────────────────────────────────────┐
│  GEOMETRÍA ESPACIAL 3D (MediaPipe Head Pose)     │
│  🟢 ALUMNO ENFOCADO    → Cabeza centrada          │
│  🔴 PÉRDIDA CONTACTO   → Cabeza girada > 35°     │
│  🟠 SOMNOLIENTO        → Ojos cerrados > 0.67s   │
├──────────────────────────────────────────────────┤
│  EMOCIÓN DOMINANTE (DeepFace CNN)                │
│  Felicidad / Neutral / Sorpresa /                │
│  Tristeza / Miedo / Enojo / Disgusto             │
├──────────────────────────────────────────────────┤
│  ÍNDICE DE COMPRENSIÓN (Fusión Multimodal)       │
│  🟢 70–100%  → Cognitivamente activo             │
│  🟡 40–69%   → Estado promedio                  │
│  🔴  0–39%   → Distracción o frustración        │
├──────────────────────────────────────────────────┤
│  SEMÁNTICA NLP (RoBERTa)                         │
│  Polaridad Positiva / Neutral / Negativa         │
├──────────────────────────────────────────────────┤
│  TRANSCRIPCIÓN (Faster-Whisper)                  │
│  Texto de lo que estás hablando en tiempo real   │
└──────────────────────────────────────────────────┘
```

### Paso 6b — Usar el Admin Panel (`/admin`)

Desde el Admin Panel puedes:

1. **Seleccionar un modelo** de la categoría Atención (XGBoost, Random Forest, SVM, Regresión Logística)
2. **Ajustar hiperparámetros** con los controles interactivos y sus guías de recomendación
3. **Lanzar el entrenamiento** y ver la barra de progreso en tiempo real vía WebSocket
4. **Ver las métricas** al finalizar: Accuracy, F1, Recall, Precision y Matriz de Confusión
5. **Activar el modelo entrenado** con un clic — hot-swap sin reiniciar el servidor

---

### Paso 7 — Para ensayar tu presentación

| Acción | Lo que detecta | Efecto en índice |
|---|---|---|
| Hablar con confianza y claridad | RoBERTa: polaridad positiva | +25 |
| Hablar con dudas o frustración | RoBERTa: polaridad negativa | -25 |
| Mirar al frente / a la cámara | Head Pose: "Enfocado" | ±0 (neutro) |
| Girar la cabeza > 35° | Head Pose: "Distraído" | -30 |
| Cerrar los ojos > 0.67 seg | EAR: "Somnoliento" | -40 |
| Sonreír genuinamente | DeepFace: "Felicidad" | +25 |

---

### Paso 8 — Detener el sistema

```
Ctrl + C
```

El servidor se apaga de forma limpia y libera la cámara y el micrófono.

---

## 7. Solución de Problemas

### Video no aparece

| Causa | Solución |
|---|---|
| Abriste el `.html` directamente | Usar `http://localhost:8080` |
| Puerto 8080 ocupado | Reiniciar el servidor (mata el proceso viejo) |
| Servidor no terminó de cargar | Esperar el mensaje `🚀 Servidor Web Activo` |
| Cámara usada por otra app | Cerrar Zoom, Teams, Discord, etc. |

### Transcripción nunca aparece

```bash
# Verificar PyAudio
python -c "import pyaudio; print('PyAudio OK')"

# Listar dispositivos de audio disponibles
python -c "
import pyaudio
p = pyaudio.PyAudio()
for i in range(p.get_device_count()):
    d = p.get_device_info_by_index(i)
    if d['maxInputChannels'] > 0:
        print(f'[{i}] {d[\"name\"]}')
"
```

Consejos adicionales:
- Habla claramente y pausa ~1 seg al terminar cada frase (el VAD necesita silencio para procesar)
- Verifica que el micrófono correcto esté seleccionado en Configuración de Windows

### Emoción siempre "Neutral"

- Mejora la iluminación frontal (evita contraluz y sombras en el rostro)
- Acércate a la cámara (el rostro debe ocupar al menos 1/4 del frame)
- El análisis ocurre 2 veces/segundo → puede haber ~0.5 seg de retraso visible

### CUDA no disponible / Sistema lento

```python
import torch
print(torch.cuda.is_available())        # ¿GPU detectada?
print(torch.cuda.get_device_name(0))    # Nombre de la GPU
print(torch.version.cuda)              # Versión CUDA de PyTorch
```

Si `False`: actualiza los drivers de NVIDIA a la versión más reciente.

---

## 8. Escalabilidad y Mejoras Futuras

### Arquitectura Actual vs. Futura

```
ACTUAL — v2.0 (Monolítico local)        FUTURO — v4.0 (Microservicios)
┌──────────────────────────┐            ┌──────────┐  ┌──────────┐
│       pruebas.py         │            │ vision   │  │ audio    │
│  Cámara + MediaPipe      │            │ :8001    │  │ :8002    │
│  DeepFace + Whisper      │     →      │          │  │          │
│  RoBERTa + FastAPI       │            └────┬─────┘  └────┬─────┘
│  Todo en un proceso      │                 └──────┬───────┘
└──────────────────────────┘                        ▼
                                              ┌──────────┐
                                              │ nlp      │
                                              │ :8003    │
                                              └────┬─────┘
                                                   ▼
                                              ┌──────────┐
                                              │ gateway  │
                                              │ :8080    │
                                              └──────────┘
```

---

### Mejora 1 — Docker (Portabilidad total)

```dockerfile
FROM nvidia/cuda:11.8-cudnn8-runtime-ubuntu22.04
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8080
CMD ["python", "pruebas.py"]
```

```bash
# Correr en cualquier PC con Docker + GPU NVIDIA:
docker run --gpus all -p 8080:8080 edu-insight:latest
```

**Beneficio:** Un solo comando para levantar el sistema en cualquier máquina.

---

### Mejora 2 — Persistencia de Sesiones (SQLite)

```sql
CREATE TABLE sesiones (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp   DATETIME DEFAULT CURRENT_TIMESTAMP,
    atencion    TEXT,
    emocion     TEXT,
    sentimiento TEXT,
    indice      INTEGER,
    texto       TEXT
);
```

**Beneficio:** Historial completo, reportes de progreso, análisis post-sesión.

---

### Mejora 3 — Whisper Large-v3 Turbo

```python
# En CONFIG:
"whisper_modelo": "large-v3-turbo",
# Modelo 2024 de OpenAI:
# ✅ Más preciso en vocabulario técnico que base/small
# ✅ Más rápido que large-v3 estándar (~3× speedup)
# ✅ Cabe en los 6 GB VRAM de la RTX 3060
```

---

### Mejora 4 — Fine-Tuning del Modelo de Emoción

```python
# Transfer Learning sobre MobileNetV3 con datos del contexto educativo
from tensorflow.keras.applications import MobileNetV3Small

base_model = MobileNetV3Small(weights='imagenet', include_top=False)
base_model.trainable = False  # Capas base ya aprendidas: congelar

model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(7, activation='softmax')  # 7 emociones
])

model.fit(dataset_educativo, epochs=10, validation_split=0.2)
```

---

### Mejora 5 — MLOps con MLflow

```python
import mlflow

with mlflow.start_run(run_name="edu-insight-sesion"):
    mlflow.log_params({
        "whisper_model": CONFIG["whisper_modelo"],
        "ear_threshold": EAR_UMBRAL,
    })
    mlflow.log_metrics({
        "promedio_indice": np.mean(historial),
        "tiempo_enfocado_pct": t_enfocado / t_total * 100,
    })
```

**Beneficio:** Historial de rendimiento del modelo, detección de degradación en el tiempo, comparación de versiones.

---

### Roadmap de Versiones

```
v2.0  ✅  Sistema funcional local, GPU, tiempo real
      │
v2.1  →  Persistencia SQLite + exportación de reportes PDF
      │
v2.2  →  Docker + soporte multiplataforma (Linux/macOS)
      │
v2.3  →  Whisper large-v3-turbo + vocabulario técnico mejorado
      │
v3.0  →  Fine-tuning DeepFace con datos propios del contexto educativo
      │
v3.1  →  Dashboard multi-usuario (varios alumnos simultáneos)
      │
v3.2  →  MLflow tracking + alertas automáticas al docente
      │
v4.0  →  Deploy cloud (AWS/GCP) + Kubernetes + CI/CD pipeline
```

---

---

## 9. Clasificador de Atención — MLOps Training Lab

El **Clasificador de Atención** es el primer componente **entrenable** del sistema Edu-Insight. A diferencia de MediaPipe (reglas geométricas puras), este módulo aprende patrones de atención a partir de datos reales capturados con la propia cámara, y puede reentrenarse con nuevos datos directamente desde el Admin Panel.

---

### 9.1 Problema: ¿Qué clasifica?

El clasificador recibe **4 medidas numéricas** extraídas por MediaPipe en cada frame y predice uno de **3 estados cognitivos**:

```
INPUT (4 features por frame)          OUTPUT (1 clase)
┌──────────────────────────┐          ┌──────────────────┐
│  ear     = 0.328         │          │                  │
│  pitch   = -11.9°        │  ──────► │  "Enfocado"      │
│  yaw     = 2.1°          │          │                  │
│  ratio_h = 0.503         │          └──────────────────┘
└──────────────────────────┘
```

| Clase | Descripción |
|---|---|
| `Enfocado` | Cabeza centrada, ojos abiertos, mirada al frente |
| `Distraído` | Cabeza girada > 35° en cualquier dirección |
| `Somnoliento` | Ojos parcialmente cerrados (EAR bajo) |

---

### 9.2 Dataset: `raw_data.csv`

**Ruta:** `datasets/atencion/raw_data.csv`
**Tamaño:** 3,571 muestras · 7 columnas

#### Estructura del CSV

```
timestamp,              ear,    pitch,   yaw,    ratio_h, ratio_v, label
2026-04-15T10:25:15,   0.322,  -12.26,  3.64,   0.484,   0.407,   Enfocado
2026-04-15T10:22:12,   0.388,  -1.64,   18.82,  0.198,   0.664,   Somnoliento
```

| Columna | Tipo | Descripción |
|---|---|---|
| `timestamp` | ISO datetime | Marca temporal de captura (descartada en entrenamiento) |
| `ear` | float [0.0–0.5] | Eye Aspect Ratio promedio (ambos ojos) |
| `pitch` | float [°] | Inclinación vertical de la cabeza (solvePnP) |
| `yaw` | float [°] | Giro horizontal de la cabeza (solvePnP) |
| `ratio_h` | float [-1.0–1.0] | Posición horizontal normalizada del iris |
| `ratio_v` | float [-1.0–1.0] | Posición vertical normalizada del iris *(no usada en entrenamiento)* |
| `label` | str | Clase objetivo: `Enfocado` / `Distraído` / `Somnoliento` |

#### ¿Por qué solo 4 features y no 6?

`ratio_v` (mirada vertical) se descarta porque en un contexto de conferencia/clase el alumno puede mirar hacia abajo para tomar notas sin estar distraído. El `pitch` ya captura esa inclinación a nivel de cabeza completa, que es más estable y menos ruidoso.

```python
# trainer.py — features seleccionadas
feature_cols = ["ear", "pitch", "yaw", "ratio_h"]
```

---

### 9.3 Significado Físico de las Features

#### `ear` — Eye Aspect Ratio

Mide la apertura vertical del ojo relativa a su ancho. Calculado con 6 landmarks por ojo (Soukupová & Čech, 2016):

```
EAR = (|p2-p6| + |p3-p5|) / (2 × |p1-p4|)

  Rango típico:
  EAR ≈ 0.30  →  ojo completamente abierto
  EAR ≈ 0.22  →  ojo casi cerrado
  EAR < 0.22  →  somnolencia
```

**Por qué es útil:** Invariante al tamaño del rostro en cámara. Un EAR de 0.25 indica el mismo nivel de cierre independientemente de si el alumno está cerca o lejos.

#### `pitch` — Inclinación Vertical (solvePnP)

Ángulo de rotación de la cabeza en el eje X:

```
  pitch > +35°  →  cabeza inclinada hacia atrás
   0° a ±10°   →  postura normal frente a pantalla
  pitch < -30°  →  cabeza inclinada hacia abajo (→ Distraído)
```

#### `yaw` — Giro Horizontal (solvePnP)

Ángulo de rotación de la cabeza en el eje Y:

```
  yaw > +35°   →  cabeza girada a la derecha (→ Distraído)
   0° a ±10°   →  mirando al frente
  yaw < -35°   →  cabeza girada a la izquierda (→ Distraído)
```

#### `ratio_h` — Posición Horizontal del Iris

Posición normalizada del centro del iris dentro de la apertura del ojo:

```
  ratio_h ≈  0.5   →  iris centrado (mirada al frente)
  ratio_h ≈  0.1   →  iris desplazado a la izquierda
  ratio_h ≈  0.9   →  iris desplazado a la derecha
```

---

### 9.4 Los 4 Modelos Disponibles

Todos los modelos de atención corren en **CPU** (sklearn), sin conflicto con la GPU usada por Whisper y DeepFace en inferencia.

#### Comparativa Rápida

| Modelo | Complejidad | Tiempo estimado | Cuándo usarlo |
|---|---|---|---|
| **Regresión Logística** | Baja | ~10 seg | Baseline inicial, interpretabilidad máxima |
| **Random Forest** | Media | ~20 seg | Robusto al overfitting, buen punto de partida |
| **XGBoost** | Media-alta | ~30 seg | Mayor precisión en datos tabulares (**recomendado**) |
| **SVM (Kernel RBF)** | Alta | ~45 seg | Máxima precisión en datasets pequeños y bien normalizados |

---

#### XGBoost — Gradient Boosting

**Tipo:** Ensemble secuencial de árboles de decisión

```
Árbol 1 → predice parcialmente
    │
    ▼  (aprende de los errores del anterior)
Árbol 2 → corrige los errores del Árbol 1
    │
    ▼
Árbol N → suma ponderada → predicción final
```

**Analogía:** Cada árbol es un "experto que aprende de los errores del anterior". El último árbol ya no comete los mismos errores que el primero.

| Hiperparámetro | Default | Rango | Efecto |
|---|---|---|---|
| `n_estimators` | 100 | 50–500 | Número de árboles. Más = más preciso pero más lento |
| `max_depth` | 5 | 3–10 | Profundidad máxima. >6 puede causar overfitting con 3,571 muestras |
| `learning_rate` | 0.1 | 0.01–0.3 | Peso de cada árbol nuevo. LR bajo + más árboles = mejor generalización |
| `subsample` | 0.8 | 0.5–1.0 | Fracción de muestras por árbol. 0.8 actúa como regularización |

---

#### Random Forest — Ensemble Paralelo

**Tipo:** Ensemble paralelo de árboles independientes

```
         Dataset completo (bootstrap)
        /          |          \
   Árbol 1     Árbol 2     Árbol N     ← cada uno ve un subconjunto aleatorio
      │            │           │
      ▼            ▼           ▼
  pred_1        pred_2      pred_N
        \           |        /
         └──── VOTACIÓN ────┘
               → clase ganadora
```

**Diferencia con XGBoost:** Los árboles se entrenan en **paralelo** (independientes entre sí) en lugar de secuencialmente. Esto lo hace más robusto al overfitting pero potencialmente menos preciso.

| Hiperparámetro | Default | Rango | Efecto |
|---|---|---|---|
| `n_estimators` | 100 | 50–500 | Número de árboles paralelos |
| `max_depth` | 10 | 3–20 | `None` = crecer ilimitado. Usar 10–15 para evitar overfitting |
| `min_samples_split` | 2 | 2–20 | Muestras mínimas para dividir un nodo. Aumentar a 5 reduce overfitting |
| `max_features` | `sqrt` | `sqrt`/`log2`/`none` | Features a evaluar por división. `sqrt` es el estándar para clasificación |

---

#### SVM con Kernel RBF — Máxima Precisión

**Tipo:** Clasificador de margen máximo con kernel radial

```
      Feature space (EAR, pitch, yaw, ratio_h)

      Enfocado ●  ●  ●                    ● Distraído
              ● ●                      ● ●
                    ────────────────
                    ← MARGEN MÁXIMO →      ← hiperplano de separación
                    ────────────────
                 ● ● ●Somnoliento●
```

**Kernel RBF:** Proyecta los datos a un espacio de dimensión superior donde son linealmente separables, sin calcular explícitamente esa proyección (kernel trick).

**Importante:** Requiere normalización obligatoria (`StandardScaler`). Se aplica automáticamente en el pipeline de entrenamiento.

| Hiperparámetro | Default | Rango | Efecto |
|---|---|---|---|
| `C` | 1.0 | 0.01–100 | Penalización por errores de clasificación. Alto = menos regularización |
| `gamma` | `scale` | `scale`/`auto` | Influencia de cada muestra. `scale` = 1/(n_features × var(X)) |
| `kernel` | `rbf` | `rbf`/`linear`/`poly` | Tipo de separación. `rbf` para separación no lineal |

---

#### Regresión Logística — Baseline Interpretable

**Tipo:** Modelo lineal probabilístico multiclase

```
z = w1·ear + w2·pitch + w3·yaw + w4·ratio_h + b

P(Enfocado)    = softmax(z1)   →  0.85
P(Distraído)   = softmax(z2)   →  0.10
P(Somnoliento) = softmax(z3)   →  0.05
                                   ───
                                   1.00  → clase: "Enfocado"
```

**Por qué usarlo:** Los pesos `w1..w4` son interpretables directamente — si `w1 (ear)` tiene el mayor peso absoluto, el modelo aprendió que EAR es la feature más discriminativa.

**Requiere normalización** (`StandardScaler`) aplicada automáticamente.

| Hiperparámetro | Default | Rango | Efecto |
|---|---|---|---|
| `C` | 1.0 | 0.001–100 | Inverso de regularización. C pequeño = más regularización |
| `max_iter` | 1000 | 100–5000 | Máximo de iteraciones para convergencia. Aumentar si hay `ConvergenceWarning` |
| `solver` | `lbfgs` | `lbfgs`/`saga`/`liblinear` | Algoritmo de optimización. `lbfgs` es eficiente para multiclase |

---

### 9.5 Pipeline de Entrenamiento

```
╔══════════════════════════════════════════════════════════════╗
║              PIPELINE CLASIFICADOR DE ATENCIÓN               ║
╚══════════════════════════════════════════════════════════════╝

INPUT ──► datasets/atencion/raw_data.csv  (3,571 filas)
              │
              ▼
  ┌──────────────────────────────────────────┐
  │  PASO 1: Limpieza                        │
  │  • dropna() sobre las 4 features + label │
  │  • Outlier removal: IQR × 2.5 por columna│
  │    q1 - 2.5×IQR ≤ valor ≤ q3 + 2.5×IQR  │
  └──────────────────────────────────────────┘
              │
              ▼
  ┌──────────────────────────────────────────┐
  │  PASO 2: Codificación                    │
  │  LabelEncoder:                           │
  │    "Distraído"   → 0                     │
  │    "Enfocado"    → 1                     │
  │    "Somnoliento" → 2                     │
  └──────────────────────────────────────────┘
              │
              ▼
  ┌──────────────────────────────────────────┐
  │  PASO 3: Train / Test Split              │
  │  test_size = 0.30  →  70% train / 30% test│
  │  stratify = y       →  distribución        │
  │  random_state = 42  →  reproducible        │
  └──────────────────────────────────────────┘
              │
              ▼ (solo SVM y LogReg)
  ┌──────────────────────────────────────────┐
  │  PASO 4: Normalización (condicional)     │
  │  StandardScaler: media=0, std=1          │
  │  fit() en train → transform() en test   │
  │  (se guarda junto al modelo en .joblib)  │
  └──────────────────────────────────────────┘
              │
              ▼
  ┌──────────────────────────────────────────┐
  │  PASO 5: Entrenamiento con progreso      │
  │  XGBoost     → incremental por chunks   │
  │  RandomForest → warm_start por chunks   │
  │  SVM / LogReg → fit() directo           │
  │  Emite eventos a progress_queue         │
  └──────────────────────────────────────────┘
              │
              ▼
  ┌──────────────────────────────────────────┐
  │  PASO 6: Evaluación (sobre X_test)       │
  │  • Accuracy, F1-macro, Recall, Precision │
  │  • Matriz de Confusión 3×3               │
  │  • VP / VN / FP / FN por clase           │
  └──────────────────────────────────────────┘
              │
              ▼
  ┌──────────────────────────────────────────┐
  │  PASO 7: Persistencia                    │
  │  modelo.joblib → app/storage/trained_models/│
  │  metricas.json → app/storage/metrics/   │
  │  Incluye: modelo, scaler, LabelEncoder,  │
  │           features, hiperparámetros, ts  │
  └──────────────────────────────────────────┘
              │
              ▼
OUTPUT ──► job.metrics + job.model_path
           → WebSocket /ws/training → Admin Panel
```

#### Emisión de progreso en tiempo real

XGBoost y Random Forest emiten eventos cada `chunk` de árboles entrenados, sin bloquear el hilo principal:

```python
# Entrenamiento por chunks con warm_start (Random Forest)
trained = 0
while trained < n_total:
    batch   = min(chunk, n_total - trained)   # chunk ≈ 5% del total
    trained += batch
    model.set_params(n_estimators=trained)
    model.fit(X_train, y_train)               # warm_start: solo agrega 'batch' árboles

    job.progress = 20 + int((trained / n_total) * 62)
    job.message  = f"RandomForest: árbol {trained}/{n_total}"
    self._emit(job, "progress")               # → progress_queue → WebSocket → barra UI
```

---

### 9.6 Métricas Generadas

Cada entrenamiento produce un archivo `.json` con las siguientes métricas científicas:

```json
{
  "accuracy":  0.9412,
  "f1_macro":  0.9389,
  "recall":    0.9401,
  "precision": 0.9378,
  "n_train":   2499,
  "n_test":    1072,
  "confusion_matrix": [[342, 5, 2], [3, 498, 1], [0, 4, 217]],
  "class_names": ["Distraído", "Enfocado", "Somnoliento"],
  "per_class": {
    "Enfocado": {
      "VP": 498, "VN": 561, "FP": 9, "FN": 4,
      "precision": 0.9823, "recall": 0.9921, "f1": 0.9872
    }
  }
}
```

#### Interpretación de VP / VN / FP / FN

Para cada clase se usa esquema **One-vs-Rest**:

| Sigla | Significado en contexto |
|---|---|
| **VP** (Verdadero Positivo) | El alumno estaba Enfocado y el modelo dijo Enfocado |
| **VN** (Verdadero Negativo) | El alumno no estaba Enfocado y el modelo dijo que no |
| **FP** (Falso Positivo) | El alumno NO estaba Enfocado pero el modelo dijo que sí |
| **FN** (Falso Negativo) | El alumno estaba Enfocado pero el modelo lo clasificó mal |

**F1-macro** es la métrica principal porque el dataset puede estar desbalanceado (más muestras de `Enfocado` que de `Somnoliento`). Promedia el F1 de cada clase con el mismo peso, independientemente de la frecuencia.

---

### 9.7 Arquitectura MLOps

#### ModelRegistry — Catálogo y Hot-Swap

Singleton que gestiona el ciclo de vida de todos los modelos del sistema:

```python
# Estado activo por defecto (configurable en el Admin Panel)
active_configs = {
    ModelCategory.ATENCION:  XGBoost,          # CPU, 0 MB VRAM
    ModelCategory.AUDIO:     Whisper Base,      # GPU, 800 MB VRAM
    ModelCategory.EMOCION:   VGG-Face,          # GPU, 500 MB VRAM
    ModelCategory.SEMANTICA: RoBERTuito,        # GPU, 700 MB VRAM
}

# Hot-swap: cambiar modelo sin reiniciar el servidor
registry.set_active_model(
    category="atencion",
    model_id="random_forest",
    trained_model_path="app/storage/trained_models/random_forest_20260430.joblib"
)
# → descarga instancia anterior → gc.collect() → carga nueva
```

**VRAM check automático:** Antes de cargar un modelo de GPU, el Registry verifica que haya VRAM suficiente (VRAM libre + VRAM que liberará el modelo anterior). Nunca falla silenciosamente por falta de memoria.

#### TrainingManager — Cola de Entrenamiento Thread-Safe

Singleton con `Queue(maxsize=500)` que desacopla el entrenamiento de la API:

```
POST /api/train/start
         │
         ▼
TrainingManager.start_training()
         │
         ├── Thread daemon "trainer-{job_id}"   ← corre en segundo plano
         │        │
         │        ├── _emit(job, "progress") ──► progress_queue
         │        │                                    │
         │        └── _emit(job, "complete")           ▼
         │                                     GET /ws/training
         │                                     → polling progress_queue
         │                                     → send_json al cliente
         │
         └── Retorna TrainingJob(job_id, status="running")
             → Frontend recibe job_id para trackear progreso
```

**Regla de una tarea a la vez:** `TrainingManager.is_busy` previene lanzar dos entrenamientos simultáneos. Esto evita conflictos de GPU (entre entrenamiento de CNN y inferencia de Whisper).

---

### 9.8 Endpoints de la Training API

| Método | Ruta | Descripción |
|---|---|---|
| `POST` | `/api/train/start` | Lanza un nuevo job de entrenamiento |
| `GET` | `/api/train/status` | Estado del job activo |
| `GET` | `/api/train/job/{job_id}` | Historial de un job específico |
| `GET` | `/api/train/jobs` | Lista de todos los jobs |
| `WS` | `/ws/training` | Stream de progreso en tiempo real |

#### Ejemplo: lanzar un entrenamiento

```python
# POST /api/train/start
{
    "category": "atencion",
    "model_id": "xgboost",
    "hyperparams": {
        "n_estimators": 200,
        "max_depth": 5,
        "learning_rate": 0.08,
        "subsample": 0.8
    }
}

# Respuesta inmediata (no bloquea):
{
    "job_id": "a3f7b1c2",
    "status": "running",
    "message": "Iniciando entrenamiento..."
}
```

#### Eventos WebSocket durante entrenamiento

```json
{ "type": "progress", "job_id": "a3f7b1c2", "progress": 45,
  "current_epoch": 90, "total_epochs": 200,
  "message": "XGBoost: árbol 90/200", "status": "running" }

{ "type": "complete", "job_id": "a3f7b1c2", "progress": 100,
  "metrics": { "accuracy": 0.9412, "f1_macro": 0.9389 },
  "model_path": "app/storage/trained_models/xgboost_20260430_143022.joblib" }
```

---

### 9.9 Archivos Persistidos

```
app/storage/
├── trained_models/
│   ├── xgboost_20260430_143022.joblib     ← modelo + scaler + LabelEncoder
│   ├── random_forest_20260430_150011.joblib
│   └── ...
└── metrics/
    ├── xgboost_20260430_143022.json       ← accuracy, F1, matriz confusión, etc.
    ├── random_forest_20260430_150011.json
    └── ...
```

**Contenido del `.joblib`:**

```python
{
    "model":         <XGBClassifier fitted>,
    "scaler":        None,        # StandardScaler solo para SVM/LogReg
    "label_encoder": <LabelEncoder: [Distraído, Enfocado, Somnoliento]>,
    "features":      ["ear", "pitch", "yaw", "ratio_h"],
    "hyperparams":   {"n_estimators": 200, "max_depth": 5, ...},
    "job_id":        "a3f7b1c2",
    "trained_at":    "20260430_143022",
}
```

Este formato garantiza que el modelo cargado en inferencia use exactamente el mismo orden de features y el mismo codificador de etiquetas que durante el entrenamiento, sin necesidad de sincronización manual.

---

## 📚 Referencias

- Soukupová, T. & Čech, J. (2016). *Real-Time Eye Blink Detection using Facial Landmarks*. CVWW.
- Vaswani, A. et al. (2017). *Attention Is All You Need*. NeurIPS.
- Radford, A. et al. (2022). *Robust Speech Recognition via Large-Scale Weak Supervision*. OpenAI.
- Lugaresi, C. et al. (2019). *MediaPipe: A Framework for Building Perception Pipelines*. Google Research.
- Serengil, S.I. & Ozpinar, A. (2020). *LightFace: A Hybrid Deep Face Recognition Framework*. ASYU.
- Perez, J.M. et al. (2021). *pysentimiento: A Python Toolkit for Sentiment Analysis and SocialNLP tasks*.

---

> **Proyecto:** Edu-Insight PRO &nbsp;|&nbsp; **Autor:** Yafer Torrez &nbsp;|&nbsp; **Versión:** 2.0 — rama `gemini`
> **Stack:** Python 3.10 · FastAPI · MediaPipe · DeepFace · Faster-Whisper · RoBERTa
> **Hardware:** NVIDIA RTX 3060 Laptop + Intel i7-12700H + 32 GB DDR5
