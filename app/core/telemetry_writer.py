"""
telemetry_writer.py — Hilo de escritura de telemetria en SQLite.

Corre en background y cada INTERVAL segundos escribe el estado actual
de cada alumno activo de la sesion en telemetry_log.

Uso:
    writer = TelemetryWriter()
    writer.start_session(session_id=1, slots=[{face_slot:0, student_id:1, ...}])
    writer.stop_session()
"""

import threading
import time
import logging

logger = logging.getLogger(__name__)

INTERVAL = 5   # segundos entre snapshots


class TelemetryWriter:
    """
    Hilo daemon que persiste el estado cognitivo periodicamente.

    - Para 1 alumno: lee directamente estado_api_global.
    - Para N alumnos: lee estado_api_global['alumnos'][face_slot]
      (disponible cuando el engine corra con max_num_faces > 1).
    """

    def __init__(self):
        self._thread:     threading.Thread = None
        self._running:    bool             = False
        self._session_id: int              = None
        self._slots:      list             = []   # [{face_slot, student_id, nombre}]
        self._lock = threading.Lock()

    # ── Interfaz publica ──────────────────────────────────────────────────────

    def start_session(self, session_id: int, slots: list) -> None:
        """
        Inicia la escritura para una sesion.
        slots: lista de dicts con {face_slot, student_id, nombre}.
        """
        with self._lock:
            self._session_id = session_id
            self._slots      = slots
            self._running    = True

        if self._thread is None or not self._thread.is_alive():
            self._thread = threading.Thread(
                target  = self._loop,
                daemon  = True,
                name    = "telemetry-writer",
            )
            self._thread.start()
            logger.info(f"TelemetryWriter iniciado — sesion={session_id}, "
                        f"alumnos={[s['student_id'] for s in slots]}")

    def stop_session(self) -> None:
        """Detiene la escritura de la sesion actual."""
        with self._lock:
            self._running    = False
            self._session_id = None
            self._slots      = []
        logger.info("TelemetryWriter detenido.")

    @property
    def active(self) -> bool:
        return self._running and self._session_id is not None

    @property
    def session_id(self) -> int:
        return self._session_id

    # ── Loop interno ──────────────────────────────────────────────────────────

    def _loop(self) -> None:
        from app.core.inference_engine import estado_api_global
        from app.storage.repositories.telemetry_repo import insert_snapshot

        while True:
            time.sleep(INTERVAL)

            with self._lock:
                if not self._running or not self._session_id:
                    continue
                session_id = self._session_id
                slots      = list(self._slots)

            try:
                # Modo multi-alumno: el engine puede exponer estado por face_slot
                alumnos_state = estado_api_global.get("alumnos")

                for slot in slots:
                    student_id = slot["student_id"]
                    face_slot  = slot["face_slot"]

                    if alumnos_state and face_slot in alumnos_state:
                        # Estado individual por cara
                        s = alumnos_state[face_slot]
                    else:
                        # Modo 1 alumno: usa el estado global directamente
                        s = estado_api_global

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

                logger.debug(f"Telemetria guardada — sesion={session_id}, alumnos={len(slots)}")

            except Exception as exc:
                logger.error(f"TelemetryWriter error: {exc}")


# Singleton global
telemetry_writer = TelemetryWriter()
