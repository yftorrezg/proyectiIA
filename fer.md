# Edu-Insight — Documentacion Completa del Sistema

> **Para quien es este documento:** Para cualquier persona, desarrollador o no, que quiera entender como funciona el sistema de principio a fin. Se usan analogias simples para explicar cada concepto tecnico.

---

## Indice

1. [Que es Edu-Insight en pocas palabras](#1-que-es-edu-insight-en-pocas-palabras)
2. [La base de datos — el archivo que guarda todo](#2-la-base-de-datos)
3. [Registro de tutores — crear una cuenta con tu cara](#3-registro-de-tutores)
4. [Login biometrico — entrar con tu cara](#4-login-biometrico)
5. [Gestion de alumnos — el libro de clases digital](#5-gestion-de-alumnos)
6. [Sesiones de clase — asignar alumnos a camaras](#6-sesiones-de-clase)
7. [Telemetria — el vigilante silencioso](#7-telemetria)
8. [Reportes y graficas — el informe final](#8-reportes-y-graficas)
9. [Flujo completo de una clase](#9-flujo-completo-de-una-clase)
10. [Estructura de archivos](#10-estructura-de-archivos)

---

## 1. Que es Edu-Insight en pocas palabras

Imagina que eres un tutor dando clase. Quisieras saber en tiempo real si tus alumnos estan atentos, si parecen confundidos, o si estan somnolientos — pero no puedes mirar a todos al mismo tiempo.

**Edu-Insight** es un sistema de camaras con inteligencia artificial que hace eso por ti. Analiza los rostros de los alumnos y genera reportes de atencion y emocion para cada uno.

El sistema tiene estas partes:

```
[Tutor se registra con su cara]
        |
        v
[Tutor hace login con parpadeo]
        |
        v
[Tutor crea una sesion y asigna alumnos a posiciones de camara]
        |
        v
[Durante la clase: la IA graba atencion, emocion y mirada cada 5 segundos]
        |
        v
[Al terminar: reporte con graficas por alumno]
```

---

## 2. La base de datos

**Analogia:** La base de datos es como un archivero con 6 cajones. Cada cajito guarda un tipo de informacion diferente.

**Archivo:** `app/storage/database.py`

```python
# database.py — el archivero central del sistema

import sqlite3       # biblioteca para manejar bases de datos simples en un archivo
import threading     # para manejar acceso desde multiples hilos a la vez
import logging       # para imprimir mensajes de estado en la consola
from pathlib import Path  # para manejar rutas de archivos de forma segura

# La base de datos vive en este archivo dentro del proyecto
_DB_PATH = Path(__file__).parent / "sessions.db"

# Candado para que solo un hilo escriba a la vez
# (como una fila en el banco: solo uno puede hablar con el cajero a la vez)
_lock = threading.Lock()


def get_connection() -> sqlite3.Connection:
    """
    Abre una conexion a la base de datos.
    row_factory=sqlite3.Row permite acceder a columnas por nombre (ej: row["nombre"])
    en lugar de por numero (ej: row[0]).
    """
    conn = sqlite3.connect(str(_DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row          # acceso por nombre de columna
    conn.execute("PRAGMA journal_mode=WAL") # permite leer mientras se escribe
    conn.execute("PRAGMA foreign_keys=ON")  # activa relaciones entre tablas
    return conn


def init_db() -> None:
    """
    Crea las 6 tablas si no existen todavia.
    Se ejecuta una sola vez cuando arranca el servidor.
    """
    with _lock:
        conn = get_connection()
        with conn:
            conn.executescript("""

                -- CAJON 1: Tutores (los profesores registrados)
                CREATE TABLE IF NOT EXISTS tutors (
                    id            INTEGER PRIMARY KEY AUTOINCREMENT, -- numero unico
                    nombre        TEXT    NOT NULL,                   -- "Prof. Ana Garcia"
                    username      TEXT    UNIQUE NOT NULL,            -- "ana_garcia" (unico)
                    embedding     BLOB    NOT NULL,     -- huella facial matematica (512 numeros)
                    registered_at TEXT    NOT NULL DEFAULT (datetime('now')), -- cuando se registro
                    last_login    TEXT                  -- ultima vez que entro
                );

                -- CAJON 2: Tokens de sesion (como llaves temporales de 8 horas)
                CREATE TABLE IF NOT EXISTS auth_sessions (
                    token      TEXT    PRIMARY KEY,     -- codigo largo y unico (UUID)
                    tutor_id   INTEGER NOT NULL,        -- a que tutor pertenece
                    created_at TEXT    NOT NULL DEFAULT (datetime('now')),
                    expires_at TEXT    NOT NULL,        -- cuando vence la llave
                    FOREIGN KEY (tutor_id) REFERENCES tutors(id) ON DELETE CASCADE
                    -- ON DELETE CASCADE: si se borra el tutor, se borra su llave tambien
                );

                -- CAJON 3: Alumnos (los estudiantes de cada tutor)
                CREATE TABLE IF NOT EXISTS students (
                    id         INTEGER PRIMARY KEY AUTOINCREMENT,
                    nombre     TEXT    NOT NULL,         -- "Juan Perez"
                    codigo     TEXT    UNIQUE NOT NULL,  -- "A001" (unico en todo el sistema)
                    tutor_id   INTEGER NOT NULL,         -- a que tutor pertenece
                    created_at TEXT    NOT NULL DEFAULT (datetime('now')),
                    FOREIGN KEY (tutor_id) REFERENCES tutors(id) ON DELETE CASCADE
                );

                -- CAJON 4: Sesiones de clase
                CREATE TABLE IF NOT EXISTS sessions (
                    id         INTEGER PRIMARY KEY AUTOINCREMENT,
                    tutor_id   INTEGER NOT NULL,
                    titulo     TEXT    NOT NULL,  -- "Clase de Matematicas - Unidad 3"
                    materia    TEXT,              -- "Matematicas" (opcional)
                    started_at TEXT    NOT NULL DEFAULT (datetime('now')), -- inicio
                    ended_at   TEXT,              -- fin (vacio si todavia esta activa)
                    status     TEXT    NOT NULL DEFAULT 'activa', -- 'activa' o 'completada'
                    FOREIGN KEY (tutor_id) REFERENCES tutors(id) ON DELETE CASCADE
                );

                -- CAJON 5: Asignacion cara <-> alumno por sesion
                -- (que alumno corresponde a cual posicion en la camara)
                CREATE TABLE IF NOT EXISTS session_slots (
                    session_id INTEGER NOT NULL,
                    student_id INTEGER NOT NULL,
                    face_slot  INTEGER NOT NULL CHECK (face_slot BETWEEN 0 AND 5), -- posicion 0-5
                    seat_label TEXT,   -- etiqueta como "Izquierda", "Centro", etc.
                    PRIMARY KEY (session_id, face_slot), -- cada posicion es unica por sesion
                    FOREIGN KEY (session_id) REFERENCES sessions(id)  ON DELETE CASCADE,
                    FOREIGN KEY (student_id) REFERENCES students(id)  ON DELETE CASCADE
                );

                -- CAJON 6: Telemetria (el corazon del sistema)
                -- Una fila nueva cada 5 segundos por cada alumno activo
                CREATE TABLE IF NOT EXISTS telemetry_log (
                    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id         INTEGER NOT NULL,
                    student_id         INTEGER NOT NULL,
                    timestamp          TEXT    NOT NULL DEFAULT (datetime('now')),
                    atencion           TEXT,   -- "Atento", "Distraido", "Somnoliento"
                    indice_comprension REAL,   -- numero de 0 a 100
                    emocion            TEXT,   -- "Feliz", "Triste", "Neutral", etc.
                    sentimiento        TEXT,   -- "POS", "NEG", "NEU"
                    mirada             TEXT,   -- "Centro", "Izquierda", "Derecha", "Arriba"
                    ear                REAL,   -- Eye Aspect Ratio (apertura de ojos, 0.0-0.4)
                    FOREIGN KEY (session_id) REFERENCES sessions(id)  ON DELETE CASCADE,
                    FOREIGN KEY (student_id) REFERENCES students(id)  ON DELETE CASCADE
                );

                -- Indice para que las consultas de reportes sean rapidas
                -- (como el indice de un libro: saltar directo a la pagina correcta)
                CREATE INDEX IF NOT EXISTS idx_telemetry_session
                    ON telemetry_log (session_id, student_id);
            """)
        conn.close()
```

---

## 3. Registro de tutores

**Analogia:** Piensa en el registro como cuando un banco te saca una foto para tu tarjeta. Pero en lugar de guardar la foto, el sistema convierte tu cara en 512 numeros matematicos (un "embedding"). La proxima vez que llegues al banco, compara esos numeros con tu cara actual.

### La pagina de registro (`frontend/register.html`)

El flujo visual es:

```
[Usuario llena nombre y username]
         |
         v
[Hace clic en "Registrarme con mi Rostro"]
         |
         v
[El sistema espera que parpadee — prueba de vida]
         |    (si no parpadeas en 12 segundos -> error)
         v
[Captura el frame de la camara]
         |
         v
[Extrae 512 numeros de la cara con Facenet512]
         |
         v
[Guarda nombre + username + los 512 numeros en la base de datos]
         |
         v
[Redirige al login]
```

El JavaScript del formulario:

```javascript
// register.html — parte del script

async function startEnroll() {
    // Leer los campos del formulario
    const nombre   = document.getElementById('inp-nombre').value.trim();
    const username = document.getElementById('inp-user').value.trim().toLowerCase();

    // Validacion basica antes de hacer el pedido al servidor
    if (!nombre || !username) {
        showError('Completa el nombre y el usuario antes de continuar.');
        return;
    }

    // Enviar datos al servidor — el servidor hara el resto
    const res  = await fetch('/api/auth/enroll', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body:    JSON.stringify({ nombre, username }), // envia los datos como JSON
    });
    const data = await res.json(); // leer la respuesta del servidor

    if (res.ok && data.ok) {
        // Mostrar pantalla de exito y redirigir al login
        sucBox.style.display = 'block';
        setTimeout(() => { location.href = '/login'; }, 3000);
    } else {
        // Mostrar mensaje de error segun el tipo
        const MSGS = {
            username_taken:  'El usuario ya existe. Elegi otro.',
            liveness_failed: 'No se detecto parpadeo.',
            no_face:         'No se detecto un rostro claro.',
        };
        showError(MSGS[data.error] ?? data.message);
    }
}
```

### El servidor — endpoint de registro (`app/api/auth_api.py`)

```python
# auth_api.py — endpoint POST /api/auth/enroll

@router.post("/enroll")
async def enroll(body: EnrollBody):
    """
    Registra un tutor nuevo siguiendo 4 pasos:
    1. Verifica que el username no exista ya
    2. Espera un parpadeo real (anti-foto)
    3. Captura el frame de la camara
    4. Extrae la huella facial y guarda en DB
    """
    from app.storage.repositories.tutor_repo import create_tutor, username_exists

    nombre   = body.nombre.strip()
    username = body.username.strip().lower()

    # Paso 0: validar que los campos no esten vacios
    if not nombre or not username:
        return JSONResponse({"ok": False, "error": "campos_vacios"}, status_code=400)

    # Paso 0b: verificar que el username no este tomado
    if username_exists(username):
        return JSONResponse(
            {"ok": False, "error": "username_taken"},
            status_code=409,  # 409 = Conflict (ya existe)
        )

    # Paso 1: esperar que la persona parpadee (prueba de vida)
    # Si no parpadea en 12 segundos, rechaza el registro
    blinked = await _wait_for_blink()
    if not blinked:
        return JSONResponse({"ok": False, "error": "liveness_failed"}, status_code=400)

    # Paso 2: obtener el frame actual de la camara como imagen
    frame = _get_current_frame()
    if frame is None:
        return JSONResponse({"ok": False, "error": "no_frame"}, status_code=503)

    # Paso 3: extraer los 512 numeros de la cara (embedding)
    embedding = await _extract_embedding(frame)
    if embedding is None:
        return JSONResponse({"ok": False, "error": "no_face"}, status_code=400)

    # Paso 4: guardar en la base de datos
    tutor_id = create_tutor(nombre, username, embedding)

    return JSONResponse({
        "ok":       True,
        "tutor_id": tutor_id,
        "nombre":   nombre,
        "message":  f"Registro exitoso. Bienvenido, {nombre}.",
    })
```

### Como funciona el "parpadeo como prueba de vida"

**Analogia:** Una foto impresa nunca puede parpadear. Si el sistema exige que parpadees antes de registrarte, una persona mal intencionada no puede usar tu foto para hacerse pasar por ti.

```python
# auth_api.py

# EAR = Eye Aspect Ratio = que tan abiertos estan los ojos
# Valor alto (0.3+) = ojos abiertos
# Valor bajo (< 0.22) = ojos cerrados (parpadeo)
_EAR_CLOSED    = 0.22
_LIVENESS_SECS = 12.0   # segundos maximos para detectar el parpadeo


async def _wait_for_blink(timeout: float = _LIVENESS_SECS) -> bool:
    """
    Observa el valor de EAR que calcula el motor de IA en tiempo real.
    Detecta cuando los ojos pasan de CERRADOS a ABIERTOS (= un parpadeo completo).
    """
    import app.core.inference_engine as ie

    deadline    = time.monotonic() + timeout  # marca el tiempo limite
    prev_closed = False                        # estado anterior de los ojos

    while time.monotonic() < deadline:        # repetir hasta que se acabe el tiempo
        ear = ie.estado_api_global.get("ear", 0.0)  # leer EAR actual de la IA

        if ear < 0.01:              # EAR muy bajo = sin cara en camara, esperar
            await asyncio.sleep(0.1)
            continue

        curr_closed = ear < _EAR_CLOSED  # True si los ojos estan cerrados ahora

        # Deteccion del parpadeo: los ojos estaban cerrados y ahora se abrieron
        if prev_closed and not curr_closed:
            return True   # parpadeo detectado!

        prev_closed = curr_closed   # guardar estado para la proxima iteracion
        await asyncio.sleep(0.04)   # revisar 25 veces por segundo

    return False  # se acabo el tiempo sin parpadeo
```

### Como se extrae la "huella facial" (`app/storage/repositories/tutor_repo.py`)

```python
# tutor_repo.py

import pickle    # para convertir arrays numpy a bytes (para guardar en SQLite)
import numpy as np

# Umbral de similitud: si dos caras tienen similitud >= 0.58, son la misma persona
# (1.0 = exactamente igual, 0.0 = completamente diferente)
SIMILARITY_THRESHOLD = 0.58


def create_tutor(nombre: str, username: str, embedding: np.ndarray) -> int:
    """
    Guarda el tutor en la base de datos.
    El embedding (512 numeros de tipo float32) se convierte a bytes
    con pickle antes de guardarlo como BLOB en SQLite.
    NUNCA se guarda la foto, solo los numeros matematicos.
    """
    blob = pickle.dumps(embedding.astype(np.float32))  # array -> bytes
    with _lock:             # tomar el candado para escritura segura
        conn = get_connection()
        with conn:          # transaccion: si algo falla, no guarda nada
            cursor = conn.execute(
                "INSERT INTO tutors (nombre, username, embedding) VALUES (?, ?, ?)",
                (nombre, username, blob),
            )
            tutor_id = cursor.lastrowid  # ID asignado automaticamente
        conn.close()
    return tutor_id


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Calcula que tan parecidos son dos vectores (embeddings).

    Analogia: imagina dos flechas en el espacio. Si apuntan en la misma
    direccion, la similitud es 1.0. Si apuntan en direcciones opuestas, es -1.0.
    Dos fotos de la misma persona generan vectores que apuntan en direcciones
    muy similares.

    La formula es: (A . B) / (|A| * |B|)
    """
    na = np.linalg.norm(a)  # longitud del vector A
    nb = np.linalg.norm(b)  # longitud del vector B
    if na < 1e-9 or nb < 1e-9:  # si alguno es casi cero, evitar division por cero
        return 0.0
    return float(np.dot(a, b) / (na * nb))  # similitud coseno
```

---

## 4. Login biometrico

**Analogia:** El login es como entrar a un edificio con reconocimiento facial. Te paras frente al lector, parpadeas para demostrar que eres humano, y el sistema compara tu cara con todas las caras registradas. Si hay coincidencia, te abre la puerta.

### La pagina de login (`frontend/login.html`)

Caracteristicas especiales:
- **Lista de tutores:** muestra todos los tutores registrados con su avatar (inicial del nombre) y ultimo acceso.
- **Seleccion de tutor:** el tutor hace clic en su nombre para "pre-seleccionarse".
- **Deteccion de mismatch:** si el sistema reconoce a OTRA persona (no la seleccionada), muestra un modal de aviso con texto de voz.
- **EAR badge:** barra en tiempo real que muestra si los ojos estan abiertos/cerrados.

```javascript
// login.html — funcion principal de verificacion

async function startVerify() {
    if (verifying || !selectedTutor) return;  // no hacer doble clic ni sin tutor

    verifying = true;
    // Actualizar la UI para mostrar que estamos procesando
    document.getElementById('btn-text').textContent = 'Esperando parpadeo...';

    // Llamar al servidor — el servidor espera el parpadeo y verifica la cara
    const res  = await fetch('/api/auth/verify', { method: 'POST' });
    const data = await res.json();

    if (data.ok) {
        // El servidor reconocio a alguien — pero puede no ser el tutor seleccionado
        if (data.tutor_id !== selectedTutor.id) {
            // El rostro corresponde a OTRA persona registrada
            showMismatch(data);  // mostrar modal de aviso
            return;
        }

        // Exito: guardar el token en el navegador (como una llave temporal)
        sessionStorage.setItem('edu_token',    data.token);    // token de autenticacion
        sessionStorage.setItem('edu_tutor',    data.nombre);   // nombre del tutor
        sessionStorage.setItem('edu_tutor_id', data.tutor_id); // ID del tutor

        // Mostrar pantalla de bienvenida y redirigir
        document.getElementById('suc-name').textContent = `Bienvenido, ${data.nombre}!`;
        document.getElementById('suc-sim').textContent  =
            `Similitud: ${Math.round(data.similarity * 100)}%`;

        setTimeout(() => { location.href = '/session-setup'; }, 2800);
    }
}


// Cuando el sistema reconoce a otro tutor (no el seleccionado):
function showMismatch(data) {
    // Mostrar modal con la foto/avatar del tutor reconocido
    document.getElementById('mismatch-nombre').textContent = data.nombre;

    // Anunciar por voz (Text-to-Speech del navegador)
    if (window.speechSynthesis) {
        const msg = `No eres ${selectedTutor.nombre}. Eres ${data.nombre}.`;
        const u = new SpeechSynthesisUtterance(msg);
        u.lang = 'es-ES';
        speechSynthesis.speak(u);
    }

    document.getElementById('mismatch-modal').classList.add('open');
}
```

### El servidor — endpoint de login (`app/api/auth_api.py`)

```python
# auth_api.py — endpoint POST /api/auth/verify

@router.post("/verify")
async def verify():
    """
    Login biometrico en 4 pasos:
    1. Esperar parpadeo (liveness)
    2. Capturar frame
    3. Extraer embedding
    4. Buscar quien es en la base de datos
    """
    from app.storage.repositories.tutor_repo import (
        find_matching_tutor, create_session_token, get_tutor_count
    )

    # Verificar que haya tutores registrados
    if get_tutor_count() == 0:
        return JSONResponse({"ok": False, "error": "no_tutors"}, status_code=404)

    # Paso 1: liveness (anti-foto)
    blinked = await _wait_for_blink()
    if not blinked:
        return JSONResponse({"ok": False, "error": "liveness_failed"}, status_code=401)

    # Paso 2 y 3: obtener embedding de la cara actual
    frame     = _get_current_frame()
    embedding = await _extract_embedding(frame)
    if embedding is None:
        return JSONResponse({"ok": False, "error": "no_face"}, status_code=400)

    # Paso 4: buscar quien es comparando con todos los tutores guardados
    match = find_matching_tutor(embedding)
    if not match:
        return JSONResponse({"ok": False, "error": "no_match"}, status_code=401)

    # Generar token (llave temporal de 8 horas)
    token = create_session_token(match["id"])

    # Devolver datos del tutor reconocido (puede ser diferente al seleccionado)
    return JSONResponse({
        "ok":        True,
        "token":     token,          # la llave temporal
        "tutor_id":  match["id"],
        "nombre":    match["nombre"],
        "similarity": match["similarity"],  # que tan seguro esta el sistema (0-1)
    })
```

### Como el sistema busca quien es (`app/storage/repositories/tutor_repo.py`)

```python
# tutor_repo.py

def find_matching_tutor(query_embedding: np.ndarray) -> Optional[dict]:
    """
    Compara la cara actual contra TODOS los tutores registrados.

    Analogia: es como mirar un album de fotos y preguntar
    "a cual de estas personas se parece mas esta cara?"
    Si la similitud con el mejor candidato supera 0.58, lo reconoce.
    """
    conn = get_connection()
    # Traer todos los tutores con sus embeddings guardados
    rows = conn.execute("SELECT id, nombre, username, embedding FROM tutors").fetchall()
    conn.close()

    best_match = None
    best_sim   = -1.0  # iniciar con la peor similitud posible

    for row in rows:
        stored = pickle.loads(row["embedding"])  # bytes -> array numpy
        sim    = _cosine_similarity(query_embedding, stored)  # comparar
        if sim > best_sim:          # si es mejor que el actual campeon
            best_sim   = sim
            best_match = dict(row)  # guardar como nuevo mejor candidato

    # Solo reconocer si supera el umbral de confianza
    if best_match and best_sim >= SIMILARITY_THRESHOLD:
        best_match.pop("embedding", None)        # no devolver el blob de bytes
        best_match["similarity"] = round(float(best_sim), 4)
        return best_match

    return None  # nadie supero el umbral -> cara no reconocida
```

### Los tokens — llaves temporales de 8 horas

```python
# tutor_repo.py

def create_session_token(tutor_id: int, hours: int = 8) -> str:
    """
    Genera un token UUID (cadena unica de 36 caracteres como:
    "550e8400-e29b-41d4-a716-446655440000").
    Lo guarda en la tabla auth_sessions con fecha de expiracion.

    Analogia: es como una pulsera de evento que caduca a medianoche.
    Mientras la tengas puesta (y no haya vencido), puedes entrar sin
    volver a mostrar tu cara.
    """
    token   = str(uuid.uuid4())    # generar ID unico aleatorio
    expires = (datetime.now() + timedelta(hours=hours)).isoformat()

    conn = get_connection()
    with conn:
        conn.execute(
            "INSERT INTO auth_sessions (token, tutor_id, expires_at) VALUES (?, ?, ?)",
            (token, tutor_id, expires),
        )
        # Actualizar la fecha de ultimo login del tutor
        conn.execute(
            "UPDATE tutors SET last_login = datetime('now') WHERE id = ?",
            (tutor_id,),
        )
    conn.close()
    return token
```

---

## 5. Gestion de alumnos

**Analogia:** Es el libro de clases digital del tutor. Puede agregar alumnos con su nombre y codigo, editarlos o eliminarlos. Cada tutor solo ve SUS alumnos.

**Archivo:** `app/api/students_api.py`

```python
# students_api.py — CRUD de alumnos (Create, Read, Update, Delete)

# Todos los endpoints estan protegidos — requieren el token del tutor
# Si no envias el token, devuelve error 401 (No autorizado)

def _tutor_id(authorization: Optional[str]) -> int:
    """
    Extrae el ID del tutor del token en el header.
    El frontend envia: Authorization: Bearer <token>
    """
    from app.storage.repositories.tutor_repo import verify_token
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Token requerido")
    token = authorization.split(" ", 1)[1]  # separar "Bearer " del token
    tutor = verify_token(token)             # verificar que el token sea valido
    if not tutor:
        raise HTTPException(status_code=401, detail="Token invalido o expirado")
    return tutor["id"]


@router.get("")
async def list_students(authorization: Optional[str] = Header(None)):
    """
    GET /api/students
    Devuelve la lista de alumnos del tutor autenticado.
    Cada tutor solo puede ver sus propios alumnos.
    """
    tutor_id = _tutor_id(authorization)  # primero verificar quien es
    from app.storage.repositories.student_repo import get_students_by_tutor
    return get_students_by_tutor(tutor_id)  # solo alumnos de este tutor


@router.post("")
async def create_student(body: StudentBody, authorization: Optional[str] = Header(None)):
    """
    POST /api/students
    Crea un alumno nuevo asociado al tutor autenticado.
    body tiene: nombre (ej: "Juan Perez") y codigo (ej: "A001")
    """
    tutor_id = _tutor_id(authorization)
    from app.storage.repositories.student_repo import create_student, codigo_exists

    nombre = body.nombre.strip()
    codigo = body.codigo.strip().upper()  # convertir a mayusculas: "a001" -> "A001"

    if not nombre or not codigo:
        return JSONResponse({"ok": False, "error": "Nombre y codigo son obligatorios"}, status_code=400)

    # Verificar que el codigo no este ya en uso (es unico en todo el sistema)
    if codigo_exists(codigo):
        return JSONResponse({"ok": False, "error": f"El codigo '{codigo}' ya existe"}, status_code=409)

    sid = create_student(nombre, codigo, tutor_id)  # guardar en DB
    return {"ok": True, "student_id": sid, "nombre": nombre, "codigo": codigo}


@router.patch("/{student_id}")
async def update_student_endpoint(student_id: int, body: UpdateStudentBody,
                                  authorization: Optional[str] = Header(None)):
    """
    PATCH /api/students/{id}
    Edita nombre y/o codigo de un alumno existente.
    """
    _tutor_id(authorization)  # solo verificar que el token sea valido
    nombre = body.nombre.strip()
    codigo = body.codigo.strip().upper()
    from app.storage.repositories.student_repo import update_student
    updated = update_student(student_id, nombre, codigo)
    if not updated:
        raise HTTPException(status_code=404, detail="Alumno no encontrado")
    return {"ok": True, "student_id": student_id, "nombre": nombre, "codigo": codigo}


@router.delete("/{student_id}")
async def delete_student(student_id: int, authorization: Optional[str] = Header(None)):
    """
    DELETE /api/students/{id}
    Elimina un alumno y en cascada toda su telemetria.
    """
    _tutor_id(authorization)
    from app.storage.repositories.student_repo import delete_student
    delete_student(student_id)
    return {"ok": True}
```

---

## 6. Sesiones de clase

**Analogia:** Una sesion es como una fotografia de una clase. Antes de empezar, el tutor "etiqueta" quienes estan sentados y en que posicion de la camara. Eso le permite al sistema saber que cara en el video corresponde a que alumno.

**Archivo:** `app/api/sessions_api.py`

### Estructura de un "slot" (posicion de camara)

```
Camara detecta hasta 6 caras:
+----------------------------------+
|  Cara 0  |  Cara 1  |  Cara 2   |
|  Juan    |  Maria   |  Pedro    |
+----------+----------+-----------+
|  Cara 3  |  Cara 4  |  Cara 5   |
|  Ana     |  Luis    |  (vacio)  |
+----------+----------+-----------+

face_slot = numero de posicion (0-5)
student_id = ID del alumno asignado a esa posicion
```

### Crear una sesion

```python
# sessions_api.py

@router.post("")
async def create_session(body: SessionBody, authorization: Optional[str] = Header(None)):
    """
    POST /api/sessions
    Crea una sesion nueva y comienza a grabar telemetria.

    body contiene:
    - titulo: "Clase de Matematicas"
    - materia: "Matematicas" (opcional)
    - slots: lista de {student_id, face_slot, seat_label}
             (que alumno va en que posicion de camara)
    """
    tutor_id = _tutor_id(authorization)

    titulo  = body.titulo.strip()
    materia = body.materia.strip()

    if not titulo:
        return JSONResponse({"ok": False, "error": "El titulo es obligatorio"}, status_code=400)
    if not body.slots:
        return JSONResponse({"ok": False, "error": "Debes asignar al menos 1 alumno"}, status_code=400)

    # Si habia una sesion activa, cerrarla primero (un tutor, una sesion a la vez)
    active = get_active_session(tutor_id)
    if active:
        from app.storage.repositories.session_repo import close_session
        close_session(active["id"])
        telemetry_writer.stop_session()  # detener la grabacion de la sesion anterior

    # Crear la sesion en la base de datos (status = 'activa')
    session_id = create_session(tutor_id, titulo, materia)

    # Guardar que alumno corresponde a cada posicion de camara
    slots_info = []
    for s in body.slots:
        student = get_student_by_id(s.student_id)
        if not student:
            continue  # saltar si el alumno no existe
        add_slot(session_id, s.student_id, s.face_slot, s.seat_label)
        slots_info.append({
            "face_slot":  s.face_slot,
            "student_id": s.student_id,
            "nombre":     student["nombre"],
        })

    # Ajustar el motor de IA para detectar exactamente N caras
    # (si hay 4 alumnos, buscar exactamente 4 caras — mas eficiente)
    n_faces = max(1, len(slots_info))
    try:
        engine.set_max_faces(n_faces)
    except Exception:
        pass

    # ARRANCAR LA GRABACION DE TELEMETRIA
    # A partir de aqui, cada 5 segundos se guarda el estado de cada alumno
    telemetry_writer.start_session(session_id, slots_info)

    return {
        "ok":        True,
        "session_id": session_id,
        "titulo":    titulo,
        "slots":     slots_info,
    }
```

### Reasignar alumnos en caliente

**Analogia:** Durante la clase, si un alumno se mueve de lugar, el tutor puede re-asignar que cara corresponde a que alumno SIN detener la sesion.

```python
# sessions_api.py

@router.patch("/{session_id}/slots/reorder")
async def reorder_session_slots(session_id: int, body: ReorderBody,
                                authorization: Optional[str] = Header(None)):
    """
    PATCH /api/sessions/{id}/slots/reorder
    Reasigna en caliente que alumno corresponde a cada posicion de camara.
    El overlay de video se actualiza inmediatamente.
    """
    _tutor_id(authorization)

    # Convertir el body a lista de dicts simples
    assignments = [{"face_slot": s.face_slot, "student_id": s.student_id}
                   for s in body.assignments]

    # Actualizar en la base de datos
    updated_slots = reorder_slots(session_id, assignments)

    # Actualizar el TelemetryWriter para que grabe con la nueva asignacion
    if telemetry_writer.active and telemetry_writer.session_id == session_id:
        telemetry_writer.update_slots(updated_slots)
        # Esto tambien actualiza el overlay del video en tiempo real

    return {"ok": True, "slots": updated_slots}
```

### Cerrar una sesion

```python
# sessions_api.py

@router.post("/{session_id}/close")
async def close_session(session_id: int, authorization: Optional[str] = Header(None)):
    """
    POST /api/sessions/{id}/close
    Cierra la sesion y detiene la grabacion de telemetria.
    Actualiza status = 'completada' y guarda ended_at en la DB.
    """
    _tutor_id(authorization)

    _close(session_id)              # marcar como completada en DB
    telemetry_writer.stop_session() # detener la grabacion automatica

    # Volver el motor a detectar solo 1 cara (modo espera)
    try:
        engine.set_max_faces(1)
    except Exception:
        pass

    return {"ok": True, "session_id": session_id}
```

---

## 7. Telemetria

**Analogia:** El TelemetryWriter es como un asistente que cada 5 segundos mira a cada alumno y anota en un cuaderno: "Juan — Atento — Emocion Neutral — Mirada al centro — 72/100 de comprension". Al terminar la clase, ese cuaderno tiene cientos de anotaciones que se pueden graficar.

**Archivo:** `app/core/telemetry_writer.py`

```python
# telemetry_writer.py

import threading
import time
import logging

INTERVAL = 5   # segundos entre cada "foto" de estado de los alumnos


class TelemetryWriter:
    """
    Hilo de fondo (background thread) que automaticamente guarda
    el estado cognitivo de cada alumno cada INTERVAL segundos.

    Analogia: es como un camarógrafo que toma una foto cada 5 segundos
    de cada alumno durante toda la clase.
    """

    def __init__(self):
        self._thread     = None   # el hilo de fondo
        self._running    = False  # bandera: esta corriendo?
        self._session_id = None   # ID de la sesion activa
        self._slots      = []     # [{face_slot, student_id, nombre}]
        self._lock = threading.Lock()  # candado para modificaciones seguras

    def start_session(self, session_id: int, slots: list) -> None:
        """
        Inicia la grabacion para una sesion.
        Crea el hilo de fondo si no existe todavia.
        """
        with self._lock:
            self._session_id = session_id
            self._slots      = slots
            self._running    = True

        # Actualizar el mapa de slots en el motor de IA
        # (para que el overlay del video muestre los nombres correctos)
        self._sync_slot_map(slots)

        # Crear y arrancar el hilo solo si no habia uno ya corriendo
        if self._thread is None or not self._thread.is_alive():
            self._thread = threading.Thread(
                target = self._loop,
                daemon = True,       # daemon=True: el hilo muere si el servidor muere
                name   = "telemetry-writer",
            )
            self._thread.start()

    def stop_session(self) -> None:
        """Detiene la grabacion."""
        with self._lock:
            self._running    = False
            self._session_id = None
            self._slots      = []
        self._sync_slot_map([])  # limpiar nombres del overlay

    def update_slots(self, new_slots: list) -> None:
        """
        Actualiza la asignacion de alumnos SIN reiniciar la sesion.
        Se llama cuando el tutor reasigna posiciones en caliente.
        """
        with self._lock:
            self._slots = new_slots
        self._sync_slot_map(new_slots)

    @staticmethod
    def _sync_slot_map(slots: list) -> None:
        """
        Actualiza el diccionario face_slot -> nombre en el motor de IA.
        Esto hace que el overlay del video muestre el nombre correcto
        sobre cada cara detectada.
        """
        from app.core.inference_engine import estado_api_global
        estado_api_global["slot_map"] = {
            s["face_slot"]: s["nombre"] for s in slots
            # Resultado ejemplo: {0: "Juan", 1: "Maria", 2: "Pedro"}
        }

    def _loop(self) -> None:
        """
        El loop principal del hilo.
        Cada INTERVAL segundos, lee el estado de cada alumno del motor de IA
        y lo guarda en la base de datos.
        """
        from app.core.inference_engine import estado_api_global
        from app.storage.repositories.telemetry_repo import insert_snapshot

        while True:
            time.sleep(INTERVAL)  # esperar 5 segundos

            with self._lock:
                if not self._running or not self._session_id:
                    continue  # si se detuvo, no hacer nada
                session_id = self._session_id
                slots      = list(self._slots)  # copia para evitar race conditions

            try:
                # El motor puede exponer estado por cara individual
                alumnos_state = estado_api_global.get("alumnos")

                for slot in slots:
                    student_id = slot["student_id"]
                    face_slot  = slot["face_slot"]

                    if alumnos_state and face_slot in alumnos_state:
                        # Modo multi-alumno: cada cara tiene su propio estado
                        s = alumnos_state[face_slot]
                    else:
                        # Modo 1 alumno: usar el estado global
                        s = estado_api_global

                    # Guardar snapshot en la base de datos
                    insert_snapshot(
                        session_id         = session_id,
                        student_id         = student_id,
                        atencion           = str(s.get("atencion",           "Desconocido")),
                        indice_comprension = float(s.get("indice_comprension", 50)),
                        emocion            = str(s.get("emocion",            "Neutral")),
                        sentimiento        = str(s.get("sentimiento",        "NEU")),
                        mirada             = str(s.get("mirada",             "Centro")),
                        ear                = float(s.get("ear",              0.0)),
                    )

            except Exception as exc:
                # Nunca dejar que un error detenga el hilo
                logger.error(f"TelemetryWriter error: {exc}")


# Instancia global — un solo TelemetryWriter para todo el sistema
telemetry_writer = TelemetryWriter()
```

### Como se guardan los snapshots (`app/storage/repositories/telemetry_repo.py`)

```python
# telemetry_repo.py

def insert_snapshot(session_id, student_id, atencion, indice_comprension,
                    emocion, sentimiento, mirada, ear):
    """
    Inserta una fila en telemetry_log.
    Esta funcion se llama 12 veces por minuto (una vez cada 5 segundos)
    por cada alumno activo en la sesion.

    Una clase de 1 hora con 4 alumnos genera:
    60 min * 12 snapshots/min * 4 alumnos = 2880 filas
    """
    with _lock:
        conn = get_connection()
        with conn:
            conn.execute(
                """INSERT INTO telemetry_log
                   (session_id, student_id, atencion, indice_comprension,
                    emocion, sentimiento, mirada, ear)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (session_id, student_id, atencion, indice_comprension,
                 emocion, sentimiento, mirada, ear),
            )
        conn.close()
```

---

## 8. Reportes y graficas

**Analogia:** Al terminar la clase, el sistema convierte todas esas anotaciones del cuaderno (los 2880 snapshots de una clase de 1 hora con 4 alumnos) en graficas visuales: linea de comprension en el tiempo, torta de emociones, barra de atencion.

**Archivo:** `app/api/reports_api.py`

```python
# reports_api.py

@router.get("/{session_id}")
async def full_report(session_id: int, authorization: Optional[str] = Header(None)):
    """
    GET /api/reports/{session_id}
    Genera el reporte completo de una sesion.

    Devuelve:
    - Info de la sesion (titulo, materia, duracion)
    - Por cada alumno: resumen estadistico + timeline de comprension
    - Promedio grupal
    """
    _verify(authorization)

    session = get_session(session_id)
    slots   = get_slots(session_id)    # lista de alumnos con sus posiciones

    students_data = []
    all_avgs = []

    for slot in slots:
        sid = slot["student_id"]

        # Calcular estadisticas del alumno para esta sesion
        summary  = get_student_summary(session_id, sid)
        # Calcular la linea de comprension agrupada por minuto
        timeline = get_comprension_timeline(session_id, sid, bucket_seconds=60)

        students_data.append({
            "student_id": sid,
            "nombre":     slot["nombre"],
            "codigo":     slot["codigo"],
            "face_slot":  slot["face_slot"],
            "summary":    summary,   # contiene avg_comprension, atencion, emocion, mirada
            "timeline":   timeline,  # lista de {minuto: 0, comprension: 72.3}
        })
        if summary["avg_comprension"] > 0:
            all_avgs.append(summary["avg_comprension"])

    # Calcular duracion de la sesion en minutos
    duracion_min = 0
    if session.get("started_at") and session.get("ended_at"):
        from datetime import datetime
        t0 = datetime.strptime(session["started_at"][:19], "%Y-%m-%d %H:%M:%S")
        t1 = datetime.strptime(session["ended_at"][:19],   "%Y-%m-%d %H:%M:%S")
        duracion_min = round((t1 - t0).total_seconds() / 60, 1)

    # Promedio grupal de comprension
    group_avg = round(sum(all_avgs) / len(all_avgs), 1) if all_avgs else 0

    return {
        "session":  dict(session),
        "students": students_data,
        "group": {
            "avg_comprension":  group_avg,       # ej: 68.4
            "total_tiempo_min": duracion_min,    # ej: 47.2
            "num_alumnos":      len(students_data),
        },
    }
```

### Como se calculan las estadisticas (`app/storage/repositories/telemetry_repo.py`)

```python
# telemetry_repo.py

def get_student_summary(session_id: int, student_id: int) -> dict:
    """
    Calcula estadisticas agregadas de un alumno en una sesion.

    Ejemplo de lo que devuelve:
    {
        "avg_comprension": 73.5,
        "total_snapshots": 144,   # 144 fotos = 12 min de clase
        "atencion": {
            "Atento": 98,
            "Distraido": 30,
            "Somnoliento": 16
        },
        "emocion": {
            "Neutral": 80,
            "Feliz": 45,
            "Sorprendido": 19
        },
        "mirada": {
            "Centro": 110,
            "Derecha": 20,
            "Izquierda": 14
        }
    }
    """
    conn = get_connection()

    # Promedio del indice de comprension de todos los snapshots
    avg = conn.execute(
        "SELECT AVG(indice_comprension) FROM telemetry_log WHERE session_id=? AND student_id=?",
        (session_id, student_id),
    ).fetchone()[0] or 0.0

    # Contar cuantos snapshots hubo de cada tipo de atencion
    atencion_rows = conn.execute(
        """SELECT atencion, COUNT(*) as cnt FROM telemetry_log
           WHERE session_id=? AND student_id=? GROUP BY atencion""",
        (session_id, student_id),
    ).fetchall()

    # Idem para emocion y mirada...
    # (codigo resumido por brevedad)

    return {
        "avg_comprension": round(avg, 1),
        "total_snapshots": total,
        "atencion":  {r["atencion"]: r["cnt"] for r in atencion_rows},
        "emocion":   {r["emocion"]:  r["cnt"] for r in emocion_rows},
        "mirada":    {r["mirada"]:   r["cnt"] for r in mirada_rows},
    }


def get_comprension_timeline(session_id: int, student_id: int, bucket_seconds: int = 60) -> list:
    """
    Agrupa los snapshots por minutos para el grafico de linea.

    Analogia: en lugar de mostrar 2880 puntos individuales,
    calcula el promedio de cada minuto. Resultado:
    [
        {"minuto": 0, "comprension": 75.2},
        {"minuto": 1, "comprension": 71.8},
        {"minuto": 2, "comprension": 68.5},
        ...
    ]

    Usa SQL puro con SQLite para ser eficiente incluso con miles de filas.
    """
    conn = get_connection()
    rows = conn.execute(
        """SELECT
               -- Calcular en que "balde" de N segundos cae este snapshot
               CAST((julianday(tl.timestamp) -
                    (SELECT julianday(MIN(t2.timestamp))    -- tiempo desde el inicio
                     FROM telemetry_log t2
                     WHERE t2.session_id = ? AND t2.student_id = ?))
                    * 86400 / ? AS INTEGER) AS bucket,    -- 86400 = segundos en un dia
               AVG(tl.indice_comprension) AS avg_comp     -- promedio del balde
           FROM telemetry_log tl
           WHERE tl.session_id = ? AND tl.student_id = ?
           GROUP BY bucket
           ORDER BY bucket""",
        (session_id, student_id, bucket_seconds, session_id, student_id),
    ).fetchall()
    conn.close()
    return [{"minuto": r["bucket"], "comprension": round(r["avg_comp"], 1)} for r in rows]
```

### Exportar a CSV (`app/api/reports_api.py`)

```python
# reports_api.py

@router.get("/{session_id}/csv")
async def export_csv(session_id: int, authorization: Optional[str] = Header(None)):
    """
    GET /api/reports/{session_id}/csv
    Exporta TODOS los snapshots de la sesion como un archivo CSV descargable.

    El archivo tendra columnas:
    timestamp, alumno, student_id, atencion, indice_comprension,
    emocion, sentimiento, mirada, ear

    Util para analisis externo en Excel, Python, R, etc.
    """
    _verify(authorization)

    rows  = get_snapshots(session_id)          # todos los snapshots sin filtrar
    slots = get_slots(session_id)
    id_name = {s["student_id"]: s["nombre"] for s in slots}  # mapa ID -> nombre

    output = io.StringIO()  # buffer en memoria (no necesita guardar en disco)
    writer = csv.DictWriter(output, fieldnames=[
        "timestamp", "alumno", "student_id",
        "atencion", "indice_comprension", "emocion",
        "sentimiento", "mirada", "ear",
    ])
    writer.writeheader()    # primera fila = nombres de columnas

    for r in rows:
        writer.writerow({
            "timestamp":          r["timestamp"],
            "alumno":             id_name.get(r["student_id"], "Desconocido"),  # nombre del alumno
            "student_id":         r["student_id"],
            "atencion":           r["atencion"],
            "indice_comprension": r["indice_comprension"],
            "emocion":            r["emocion"],
            "sentimiento":        r["sentimiento"],
            "mirada":             r["mirada"],
            "ear":                r["ear"],
        })

    output.seek(0)  # volver al inicio del buffer

    # Devolver como respuesta de descarga (el navegador abre "Guardar como...")
    filename = f"reporte_sesion_{session_id}.csv"
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type = "text/csv",
        headers    = {"Content-Disposition": f"attachment; filename={filename}"},
    )
```

---

## 9. Flujo completo de una clase

Aqui se muestra el camino completo desde que el tutor arranca el sistema hasta que ve los reportes.

```
PASO 1 — PRIMER USO: REGISTRO
==============================
Tutor va a: http://localhost:8080/register
  |
  |-- Llena nombre ("Prof. Ana Garcia") y username ("ana_garcia")
  |-- Hace clic en "Registrarme con mi Rostro"
  |-- El servidor espera que parpadee (max 12 segundos)
  |-- La IA extrae 512 numeros de su cara (Facenet512)
  |-- Se guarda en tutors (nombre, username, embedding)
  |
  v
Redirige a /login


PASO 2 — LOGIN DIARIO
======================
Tutor va a: http://localhost:8080/login
  |
  |-- Ve la lista de tutores registrados, hace clic en su nombre
  |-- Hace clic en "Verificar"
  |-- El servidor espera parpadeo (anti-foto)
  |-- Extrae embedding de la cara actual
  |-- Compara con TODOS los embeddings en la DB (similitud coseno)
  |-- Si similitud >= 0.58 con alguno -> match
  |-- Si el match es diferente al seleccionado -> modal de aviso + voz
  |-- Genera token UUID valido por 8 horas
  |-- Guarda token en sessionStorage del navegador
  |
  v
Redirige a /session-setup


PASO 3 — CONFIGURAR LA CLASE
==============================
Tutor va a: http://localhost:8080/session-setup
  |
  |-- Ve su lista de alumnos (los que registro previamente)
  |-- Puede agregar alumnos nuevos (nombre + codigo)
  |-- Asigna alumnos a posiciones de camara (slot 0, 1, 2...)
  |-- Escribe titulo y materia de la sesion
  |-- Hace clic en "Iniciar Clase"
  |
  |-- El servidor:
  |     1. Crea la sesion en DB (status='activa')
  |     2. Guarda los slots (cara <-> alumno)
  |     3. Ajusta el motor de IA para N caras
  |     4. Arranca TelemetryWriter (graba cada 5 segundos)
  |
  v
Redirige al dashboard de la clase


PASO 4 — DURANTE LA CLASE
===========================
TelemetryWriter corre en fondo:
  Cada 5 segundos por cada alumno:
    |-- Lee estado del motor de IA (atencion, emocion, mirada, comprension, EAR)
    |-- Inserta fila en telemetry_log
    |
    v
    telemetry_log crece en tiempo real

El tutor puede en cualquier momento:
  - Ver el dashboard en vivo
  - Reasignar posiciones de alumnos (si se movieron)
  - Cerrar la sesion


PASO 5 — CERRAR LA CLASE
==========================
Tutor hace clic en "Cerrar Sesion"
  |
  |-- session.status = 'completada'
  |-- session.ended_at = ahora
  |-- TelemetryWriter.stop_session()
  |-- Motor de IA vuelve a detectar 1 cara
  |
  v
Sesion disponible para reporte


PASO 6 — VER EL REPORTE
=========================
Tutor va a la lista de sesiones, hace clic en una
  |
  |-- GET /api/reports/{session_id}
  |-- El servidor calcula:
  |     - Promedio de comprension por alumno
  |     - Distribucion de atencion (torta)
  |     - Distribucion de emocion (torta)
  |     - Timeline de comprension (linea por minuto)
  |     - Promedio grupal
  |
  v
Graficas interactivas con Chart.js en el navegador
Opcion de descargar CSV completo
```

---

## 10. Estructura de archivos

```
ProyectoIA/
|
|-- app/                           <- Backend Python (FastAPI)
|   |-- main.py                    <- Punto de entrada, registra todos los routers
|   |
|   |-- api/                       <- Endpoints REST
|   |   |-- auth_api.py            <- /api/auth/* (login, registro, tokens)
|   |   |-- students_api.py        <- /api/students/* (CRUD de alumnos)
|   |   |-- sessions_api.py        <- /api/sessions/* (sesiones de clase)
|   |   |-- reports_api.py         <- /api/reports/* (reportes y CSV)
|   |   |-- telemetry.py           <- /api/telemetry (estado en tiempo real)
|   |   |-- models_api.py          <- /api/models/* (catalogo de modelos IA)
|   |   `-- training_api.py        <- /api/training/* (entrenamiento de modelos)
|   |
|   |-- core/                      <- Logica central de IA
|   |   |-- inference_engine.py    <- Motor de deteccion facial (MediaPipe + modelos)
|   |   |-- telemetry_writer.py    <- Hilo de grabacion de telemetria (cada 5 seg)
|   |   |-- model_registry.py      <- Catalogo de 16 modelos disponibles
|   |   `-- trainer.py             <- Entrenamiento de modelos con WebSocket
|   |
|   `-- storage/                   <- Persistencia de datos
|       |-- database.py            <- SQLite: conexion + creacion de tablas
|       `-- repositories/          <- Funciones CRUD por entidad
|           |-- tutor_repo.py      <- Tutores: crear, buscar, tokens
|           |-- student_repo.py    <- Alumnos: CRUD basico
|           |-- session_repo.py    <- Sesiones + slots (cara <-> alumno)
|           |-- telemetry_repo.py  <- Snapshots: insertar + consultas de reportes
|           `-- sessions.db        <- Archivo SQLite (toda la data)
|
`-- frontend/                      <- Paginas HTML del usuario
    |-- login.html                 <- Pagina de login biometrico
    |-- register.html              <- Pagina de registro con cara
    |-- session_setup.html         <- Configurar clase + alumnos
    |-- reports.html               <- Graficas y reportes de sesiones
    `-- admin.html                 <- Panel de administracion MLOps
```

### Endpoints disponibles (resumen)

| Metodo | Ruta | Descripcion |
|--------|------|-------------|
| POST | `/api/auth/enroll` | Registrar tutor con cara |
| POST | `/api/auth/verify` | Login biometrico -> token |
| GET | `/api/auth/me` | Info del tutor por token |
| POST | `/api/auth/logout` | Invalidar token |
| GET | `/api/auth/tutors` | Lista de tutores |
| PATCH | `/api/auth/tutors/{id}` | Editar nombre de tutor |
| DELETE | `/api/auth/tutors/{id}` | Eliminar tutor (cascada) |
| GET | `/api/students` | Listar alumnos del tutor |
| POST | `/api/students` | Crear alumno |
| PATCH | `/api/students/{id}` | Editar alumno |
| DELETE | `/api/students/{id}` | Eliminar alumno |
| GET | `/api/sessions` | Listar sesiones del tutor |
| POST | `/api/sessions` | Crear sesion + arrancar telemetria |
| GET | `/api/sessions/active` | Sesion activa actual |
| POST | `/api/sessions/{id}/close` | Cerrar sesion |
| GET | `/api/sessions/{id}/slots` | Alumnos de una sesion |
| PATCH | `/api/sessions/{id}/slots/reorder` | Reasignar caras en caliente |
| GET | `/api/reports/{id}` | Reporte completo de sesion |
| GET | `/api/reports/{id}/csv` | Descargar telemetria en CSV |

---

> **Resumen tecnico en una frase:** Edu-Insight combina reconocimiento facial biometrico (Facenet512), deteccion de rostros en tiempo real (MediaPipe), base de datos relacional (SQLite), una API REST (FastAPI) y graficas web (Chart.js) para transformar una clase presencial en datos cuantificables de atencion y emocion por alumno.
