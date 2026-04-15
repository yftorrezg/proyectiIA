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

---

## 1. ¿Qué es Edu-Insight?

**Edu-Insight PRO** es un sistema de **monitoreo cognitivo en tiempo real** diseñado para el contexto educativo. Analiza simultáneamente tres canales de información del usuario y los combina en una única métrica de comprensión.

### Canales de Análisis

| Canal | ¿Qué captura? | Tecnología principal |
|---|---|---|
| 👁️ **Virientación de casual** | Rostro, obeza, estado de ojos, dirección de mirada | MediaPipe + DeepFace |
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
python pruebas.py
```

Espera estos mensajes en orden (la primera vez: 40–60 seg):

```
✅ Radar 3D + Iris (MediaPipe) inicializado correctamente en memoria.
🔥 Hardware detectado: CUDA
✅ Whisper cargado exitosamente en CUDA (Hardware Gráfico).
✅ DeepFace cargado y en espera para procesamiento asíncrono.
✅ [AUDIO] Micrófono ABIERTO.
🚀 Servidor Web Activo. Levantando en http://localhost:8080
```

---

### Paso 5 — Abrir el Dashboard

En el navegador:

```
http://localhost:8080
```

> ⚠️ **Nunca** abras `dashboard.html` directamente desde el explorador de archivos.
> El protocolo `file://` bloquea las conexiones al backend por seguridad del navegador.
> Siempre usa la URL `http://localhost:8080`.

---

### Paso 6 — Interpretar el Dashboard

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
