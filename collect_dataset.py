# ==============================================================================
# EDU-INSIGHT — RECOLECTOR DE DATASET v2.0  (UI MEJORADA, MÁS CONTROLES, FEATURES EN VIVO)
# Script INDEPENDIENTE — no toca pruebas.py.
#
# CONTROLES:
#   E          → Activar/Desactivar "Enfocado"
#   D          → Activar/Desactivar "Distraido"
#   S          → Activar/Desactivar "Somnoliento"
#   ESPACIO    → PAUSA si grabando | REANUDA si pausado (toggle)
#   Q / ESC    → Guardar y salir
# ==============================================================================
import os, csv, datetime, math

# Silenciar advertencias de TensorFlow/Keras antes de importar MediaPipe
os.environ["TF_CPP_MIN_LOG_LEVEL"]   = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"]  = "0"
os.environ["GLOG_minloglevel"]       = "3"

import cv2
import mediapipe as mp
import numpy as np

# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN GENERAL
# ──────────────────────────────────────────────────────────────────────────────
RUTA_CSV      = os.path.join("datasets", "atencion", "raw_data.csv")
META_MINIMA   = 300           # muestras por clase para considerar objetivo alcanzado
GUARDAR_CADA  = 5             # 1 muestra cada N frames ≈ 6 muestras/seg a 30fps
CAM_W, CAM_H  = 960, 720      # resolución del frame de cámara (la ventana es más ancha)
PANEL_W       = 400           # ancho del panel derecho
CANVAS_W      = CAM_W + PANEL_W   # 1360
CANVAS_H      = CAM_H             # 720
WIN_NAME      = "EDU-INSIGHT | Dataset Collector v2.0"

# ──────────────────────────────────────────────────────────────────────────────
# PALETA DE COLORES (BGR)
# ──────────────────────────────────────────────────────────────────────────────
# Fondos
BG_PANEL      = ( 20,  27,  42)   # #0F172A  slate-900
BG_CARD       = ( 38,  52,  72)   # #1E293B  slate-800
BG_CARD_ACT   = ( 55,  72,  98)   # slate-700 iluminado
BG_OVERLAY    = ( 10,  14,  22)   # casi negro para separadores

# Textos
T_WHITE       = (245, 248, 255)
T_GRAY        = (160, 165, 180)
T_MUTED       = ( 90,  95, 110)

# Acento del sistema
INDIGO        = (200, 110,  80)   # #6366f1 (BGR invertido)
INDIGO_DIM    = (120,  65,  45)

# Colores de clase (BGR)
CLR = {
    "Enfocado":    ( 60, 200,  80),   # verde esmeralda
    "Distraido":   ( 55,  60, 225),   # rojo vivo
    "Somnoliento": ( 20, 150, 255),   # naranja cálido
}
CLR_DIM = {
    "Enfocado":    ( 25,  90,  35),
    "Distraido":   ( 22,  25,  95),
    "Somnoliento": (  8,  60, 105),
}

# ──────────────────────────────────────────────────────────────────────────────
# LANDMARKS (idénticos a pruebas.py)
# ──────────────────────────────────────────────────────────────────────────────
OJO_IZQ_EAR  = [33, 160, 158, 133, 153, 144]
OJO_DER_EAR  = [362, 385, 387, 263, 373, 380]
IRIS_IZQ, IRIS_DER               = 468, 473
OJO_IZQ_EXT, OJO_IZQ_INT         = 33, 133
OJO_DER_INT, OJO_DER_EXT         = 362, 263
OJO_IZQ_TOP, OJO_IZQ_BOT         = 159, 145
OJO_DER_TOP, OJO_DER_BOT         = 386, 374

FACE_3D = np.array([
    [  0.0,   0.0,    0.0],
    [  0.0, 330.0,  -65.0],
    [-225.0,-170.0, -135.0],
    [ 225.0,-170.0, -135.0],
    [-150.0, 150.0, -125.0],
    [ 150.0, 150.0, -125.0],
], dtype=np.float64)

CLASES_KEYS = {ord('e'): "Enfocado", ord('d'): "Distraido", ord('s'): "Somnoliento"}


# ──────────────────────────────────────────────────────────────────────────────
# HELPERS DE DIBUJO
# ──────────────────────────────────────────────────────────────────────────────
def rrect(img, pt1, pt2, color, radius=8, filled=True, thickness=2):
    """Rectángulo con esquinas redondeadas."""
    x1, y1 = pt1; x2, y2 = pt2
    r = min(radius, (x2-x1)//2, (y2-y1)//2)
    if filled:
        cv2.rectangle(img, (x1+r, y1), (x2-r, y2), color, -1)
        cv2.rectangle(img, (x1, y1+r), (x2, y2-r), color, -1)
        for cx, cy in [(x1+r, y1+r), (x2-r, y1+r), (x1+r, y2-r), (x2-r, y2-r)]:
            cv2.circle(img, (cx, cy), r, color, -1)
    else:
        cv2.rectangle(img, (x1+r, y1), (x2-r, y1), color, thickness)
        cv2.rectangle(img, (x1+r, y2), (x2-r, y2), color, thickness)
        cv2.rectangle(img, (x1, y1+r), (x1, y2-r), color, thickness)
        cv2.rectangle(img, (x2, y1+r), (x2, y2-r), color, thickness)
        cv2.ellipse(img, (x1+r, y1+r), (r,r), 180, 0, 90, color, thickness)
        cv2.ellipse(img, (x2-r, y1+r), (r,r), 270, 0, 90, color, thickness)
        cv2.ellipse(img, (x1+r, y2-r), (r,r),  90, 0, 90, color, thickness)
        cv2.ellipse(img, (x2-r, y2-r), (r,r),   0, 0, 90, color, thickness)


def txt(img, text, pos, scale, color, bold=False, anchor="tl"):
    font = cv2.FONT_HERSHEY_DUPLEX if bold else cv2.FONT_HERSHEY_SIMPLEX
    thick = 2 if bold else 1
    (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
    x, y = pos
    if anchor == "tc": x -= tw // 2
    elif anchor == "tr": x -= tw
    cv2.putText(img, text, (x, y + th), font, scale, color, thick, cv2.LINE_AA)
    return tw, th


def progress_bar(img, x, y, w, h, value, total, color, bg=(50,55,70), radius=4):
    """Barra de progreso con esquinas redondeadas."""
    pct = min(value / max(total, 1), 1.0)
    rrect(img, (x, y), (x+w, y+h), bg, radius=radius)
    fill = max(int(w * pct), 0)
    if fill > 0:
        rrect(img, (x, y), (x+fill, y+h), color, radius=radius)


def calcular_ear(lm, idx, w, h):
    p = [np.array([lm[i].x*w, lm[i].y*h]) for i in idx]
    return (np.linalg.norm(p[1]-p[5]) + np.linalg.norm(p[2]-p[4])) / \
           (2.0 * np.linalg.norm(p[0]-p[3]) + 1e-6)


# ──────────────────────────────────────────────────────────────────────────────
# RENDERIZADO DEL PANEL DERECHO
# ──────────────────────────────────────────────────────────────────────────────
def render_panel(canvas, contadores, label_activo, features, t_inicio, frame_n):
    px = CAM_W   # x de inicio del panel
    pw = PANEL_W
    ph = CANVAS_H

    # Fondo del panel
    canvas[0:ph, px:px+pw] = BG_PANEL

    # Línea separadora vertical izquierda del panel
    cv2.line(canvas, (px, 0), (px, ph), INDIGO_DIM, 2)

    # ── HEADER ────────────────────────────────────────────────────────────────
    rrect(canvas, (px+14, 12), (px+pw-14, 78), BG_CARD, radius=10)
    txt(canvas, "EDU-INSIGHT", (px+24, 18), 0.78, T_WHITE, bold=True)
    txt(canvas, "Dataset Collector  v2.0", (px+24, 44), 0.40, T_GRAY)

    # Reloj en tiempo real
    ahora  = datetime.datetime.now().strftime("%H:%M:%S")
    elapsed = int(time.time() - t_inicio)
    mm, ss = elapsed // 60, elapsed % 60
    txt(canvas, ahora, (px+pw-22, 18), 0.50, INDIGO, anchor="tr")
    txt(canvas, f"Session  {mm:02d}:{ss:02d}", (px+pw-22, 44), 0.38, T_GRAY, anchor="tr")

    # ── CARDS DE CLASE ─────────────────────────────────────────────────────────
    clases_orden = ["Enfocado", "Distraido", "Somnoliento"]
    labels_visual = {"Enfocado": "ENFOCADO", "Distraido": "DISTRAIDO", "Somnoliento": "SOMNOLIENTO"}

    card_y = 92
    for clase in clases_orden:
        n      = contadores.get(clase, 0)
        activo = label_activo == clase
        clr    = CLR[clase]
        clr_d  = CLR_DIM[clase]
        bg     = BG_CARD_ACT if activo else BG_CARD
        cx1, cx2 = px+14, px+pw-14
        cy1, cy2 = card_y, card_y+100

        # Fondo de la card
        rrect(canvas, (cx1, cy1), (cx2, cy2), bg, radius=10)

        # Franja de color izquierda
        rrect(canvas, (cx1, cy1), (cx1+8, cy2), clr if activo else clr_d, radius=6)

        # Nombre de la clase
        txt(canvas, labels_visual[clase], (cx1+20, cy1+8), 0.58,
            clr if activo else T_GRAY, bold=True)

        # Contador y porcentaje
        pct = min(n / META_MINIMA * 100, 100)
        if n >= META_MINIMA:
            badge_txt = "COMPLETADO"
            badge_clr = clr
        else:
            badge_txt = f"{pct:.0f}%"
            badge_clr = clr if activo else T_MUTED

        txt(canvas, f"{n} / {META_MINIMA}", (cx1+20, cy1+33), 0.46, T_WHITE if activo else T_GRAY)
        txt(canvas, badge_txt, (cx2-12, cy1+33), 0.44, badge_clr, bold=(n>=META_MINIMA), anchor="tr")

        # Barra de progreso
        progress_bar(canvas, cx1+12, cy1+60, (cx2-cx1)-24, 12,
                     n, META_MINIMA, clr if activo else clr_d)

        # Badge "● REC" pulsante si está activo
        if activo:
            pulse = (math.sin(frame_n * 0.15) + 1) / 2   # 0.0 – 1.0
            alpha = int(180 + 75 * pulse)
            # Punto rojo de grabación
            dot_x, dot_y = cx2-28, cy1+14
            r_clr = tuple(int(c * (0.7 + 0.3*pulse)) for c in clr)
            cv2.circle(canvas, (dot_x, dot_y), 6, r_clr, -1)
            txt(canvas, "REC", (dot_x+10, cy1+8), 0.40, r_clr, bold=True)

        card_y += 108

    # ── CONTROLES ─────────────────────────────────────────────────────────────
    sep_y = card_y + 2
    cv2.line(canvas, (px+14, sep_y), (px+pw-14, sep_y), (50, 55, 75), 1)

    txt(canvas, "CONTROLES", (px+14, sep_y+8), 0.38, T_MUTED)

    ctrl_data = [
        ("E", "Enfocado",    CLR["Enfocado"],    False),
        ("D", "Distraido",   CLR["Distraido"],   False),
        ("S", "Somnoliento", CLR["Somnoliento"], False),
        ("SPACE", "Pausa / Reanudar", T_GRAY,    False),
        ("Q",  "Guardar y Salir",     (80,80,200), False),
    ]
    ky = sep_y + 28
    for i, (key, desc, clr_k, _) in enumerate(ctrl_data):
        kx = px + 18
        # Badge de tecla
        kw = max(len(key) * 8 + 10, 28)
        rrect(canvas, (kx, ky-2), (kx+kw, ky+16), (45, 52, 68), radius=4)
        txt(canvas, key, (kx + kw//2, ky-3), 0.36, clr_k, bold=True, anchor="tc")
        txt(canvas, desc, (kx+kw+8, ky-3), 0.38, T_GRAY)
        ky += 24

    # ── FEATURES EN VIVO ──────────────────────────────────────────────────────
    ear, pitch, yaw, rh, rv = features
    feat_y = ky + 8
    cv2.line(canvas, (px+14, feat_y), (px+pw-14, feat_y), (50, 55, 75), 1)
    txt(canvas, "FEATURES  (live)", (px+14, feat_y+8), 0.38, T_MUTED)

    feat_items = [
        ("EAR",      f"{ear:.3f}",   ear / 0.5,                 (80,200,100) if ear > 0.22 else (60,60,220)),
        ("pitch",    f"{pitch:+.1f}", (abs(pitch)+45) / 90,      INDIGO),
        ("yaw",      f"{yaw:+.1f}",  (abs(yaw)+45) / 90,        INDIGO),
        ("ratio_h",  f"{rh:.3f}",    rh,                        (180,150,80)),
        ("ratio_v",  f"{rv:.3f}",    rv,                        (180,150,80)),
    ]
    fy = feat_y + 28
    bar_x  = px + 110
    bar_w  = pw - 130
    for fname, fval, fnorm, fclr in feat_items:
        txt(canvas, fname, (px+18, fy-2), 0.40, T_GRAY)
        txt(canvas, fval, (bar_x-6, fy-2), 0.40, T_WHITE, anchor="tr")
        progress_bar(canvas, bar_x, fy-10, bar_w-14, 8,
                     min(max(fnorm, 0), 1), 1.0, fclr, radius=3)
        fy += 22

    # ── TOTAL EN LA PARTE INFERIOR ────────────────────────────────────────────
    total = sum(contadores.values())
    complete = sum(1 for v in contadores.values() if v >= META_MINIMA)
    rrect(canvas, (px+14, CANVAS_H-46), (px+pw-14, CANVAS_H-10), BG_CARD, radius=8)
    txt(canvas, f"Total muestras: {total}", (px+24, CANVAS_H-40), 0.42, T_WHITE)
    pct_done = f"{complete}/3 clases completas"
    clr_done = CLR["Enfocado"] if complete == 3 else T_GRAY
    txt(canvas, pct_done, (px+pw-20, CANVAS_H-40), 0.40, clr_done, anchor="tr")


# ──────────────────────────────────────────────────────────────────────────────
# RENDERIZADO DEL ÁREA DE CÁMARA
# ──────────────────────────────────────────────────────────────────────────────
def render_camera_area(canvas, cam_frame, label_activo, cara_ok, frame_n):
    """Escala el frame de cámara a CAM_W x CAM_H con letterbox y lo pega en el canvas."""
    fh, fw = cam_frame.shape[:2]
    scale  = min(CAM_W / fw, CAM_H / fh)
    nw     = int(fw * scale)
    nh     = int(fh * scale)
    resized = cv2.resize(cam_frame, (nw, nh), interpolation=cv2.INTER_LINEAR)

    # Fondo negro detrás del frame (letterbox)
    canvas[0:CAM_H, 0:CAM_W] = (8, 10, 16)
    y_off = (CAM_H - nh) // 2
    x_off = (CAM_W - nw) // 2
    canvas[y_off:y_off+nh, x_off:x_off+nw] = resized

    # Borde de color según clase activa (con efecto pulsante)
    if label_activo is not None:
        clr    = CLR[label_activo]
        pulse  = (math.sin(frame_n * 0.12) + 1) / 2
        thick  = int(6 + 4 * pulse)
        cv2.rectangle(canvas, (x_off, y_off), (x_off+nw-1, y_off+nh-1), clr, thick)

        # Etiqueta activa sobre el video
        label_txt = f"  {label_activo.upper()}  "
        (lw, lh), _ = cv2.getTextSize(label_txt, cv2.FONT_HERSHEY_DUPLEX, 0.75, 2)
        lx = x_off + nw//2 - lw//2
        ly = y_off + nh - 40
        rrect(canvas, (lx-6, ly-4), (lx+lw+6, ly+lh+6), clr, radius=6)
        cv2.putText(canvas, label_txt, (lx, ly+lh),
                    cv2.FONT_HERSHEY_DUPLEX, 0.75, BG_OVERLAY, 2, cv2.LINE_AA)
    else:
        # Sin clase activa: borde gris tenue
        cv2.rectangle(canvas, (x_off, y_off), (x_off+nw-1, y_off+nh-1), (50,55,70), 2)

    # Aviso "PAUSA"
    if label_activo is None:
        pw_txt = "  PAUSA  "
        (pw_w, pw_h), _ = cv2.getTextSize(pw_txt, cv2.FONT_HERSHEY_DUPLEX, 0.65, 2)
        px2 = x_off + nw//2 - pw_w//2
        py2 = y_off + nh - 40
        rrect(canvas, (px2-6, py2-4), (px2+pw_w+6, py2+pw_h+6), (45,52,68), radius=6)
        cv2.putText(canvas, pw_txt, (px2, py2+pw_h),
                    cv2.FONT_HERSHEY_DUPLEX, 0.65, T_GRAY, 2, cv2.LINE_AA)

    # Aviso sin cara
    if not cara_ok:
        pulse = (math.sin(frame_n * 0.20) + 1) / 2
        alpha = 0.4 + 0.3 * pulse
        ovl   = canvas.copy()
        cv2.rectangle(ovl, (x_off, y_off), (x_off+nw, y_off+nh), (0, 0, 180), -1)
        cv2.addWeighted(ovl, alpha * 0.35, canvas, 1 - alpha * 0.35, 0, canvas)
        msg = "SIN CARA DETECTADA"
        (mw, mh), _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_DUPLEX, 0.9, 2)
        cv2.putText(canvas, msg,
                    (x_off + nw//2 - mw//2, y_off + nh//2 + mh//2),
                    cv2.FONT_HERSHEY_DUPLEX, 0.9, T_WHITE, 2, cv2.LINE_AA)


# ──────────────────────────────────────────────────────────────────────────────
# BUCLE PRINCIPAL
# ──────────────────────────────────────────────────────────────────────────────
import time   # importar aquí para que esté disponible en render_panel

def main():
    os.makedirs(os.path.dirname(RUTA_CSV), exist_ok=True)
    archivo_nuevo = not os.path.exists(RUTA_CSV)

    csv_file   = open(RUTA_CSV, "a", newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_file)
    if archivo_nuevo:
        csv_writer.writerow(["timestamp", "ear", "pitch", "yaw",
                              "ratio_h", "ratio_v", "label"])
        csv_file.flush()

    contadores = {"Enfocado": 0, "Distraido": 0, "Somnoliento": 0}
    if not archivo_nuevo:
        with open(RUTA_CSV, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                lbl = row.get("label", "")
                if lbl in contadores:
                    contadores[lbl] += 1

    # MediaPipe
    mp_fm = mp.solutions.face_mesh
    fm    = mp_fm.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5,
                            refine_landmarks=True, max_num_faces=1)

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN_NAME, CANVAS_W, CANVAS_H)

    label_activo  = None
    ultimo_label  = None   # recuerda el último label para SPACE toggle
    frame_n       = 0
    t_inicio      = time.time()

    ear_v, pitch_v, yaw_v, rh_v, rv_v = 0.3, 0.0, 0.0, 0.5, 0.5

    print("\n" + "="*60)
    print("  EDU-INSIGHT — Dataset Collector v2.0")
    print("="*60)
    print(f"  Archivo : {os.path.abspath(RUTA_CSV)}")
    print(f"  Existentes: {sum(contadores.values())} muestras")
    print(f"  Meta/clase: {META_MINIMA}")
    print("  E=Enfocado  D=Distraido  S=Somnoliento")
    print("  SPACE=Pausa/Reanudar  Q=Salir")
    print("="*60 + "\n")

    canvas = np.zeros((CANVAS_H, CANVAS_W, 3), dtype=np.uint8)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame  = cv2.flip(frame, 1)
        fh, fw = frame.shape[:2]
        frame_n += 1

        # ── MEDIAPIPE ──────────────────────────────────────────────────────────
        rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results  = fm.process(rgb)
        cara_ok  = False

        if results.multi_face_landmarks:
            for fl in results.multi_face_landmarks:
                lm = fl.landmark
                cara_ok = True

                ear_izq = calcular_ear(lm, OJO_IZQ_EAR, fw, fh)
                ear_der = calcular_ear(lm, OJO_DER_EAR, fw, fh)
                ear_v   = round((ear_izq + ear_der) / 2.0, 4)

                rng_hi  = (lm[OJO_IZQ_INT].x - lm[OJO_IZQ_EXT].x) + 1e-6
                rng_hd  = (lm[OJO_DER_EXT].x - lm[OJO_DER_INT].x) + 1e-6
                rh_v    = round(((lm[IRIS_IZQ].x - lm[OJO_IZQ_EXT].x)/rng_hi +
                                  (lm[IRIS_DER].x - lm[OJO_DER_INT].x)/rng_hd) / 2.0, 4)

                rng_vi  = (lm[OJO_IZQ_BOT].y - lm[OJO_IZQ_TOP].y) + 1e-6
                rng_vd  = (lm[OJO_DER_BOT].y - lm[OJO_DER_TOP].y) + 1e-6
                rv_v    = round(((lm[IRIS_IZQ].y - lm[OJO_IZQ_TOP].y)/rng_vi +
                                  (lm[IRIS_DER].y - lm[OJO_DER_TOP].y)/rng_vd) / 2.0, 4)

                face_2d = np.array([
                    [int(lm[1].x*fw),   int(lm[1].y*fh)],
                    [int(lm[152].x*fw), int(lm[152].y*fh)],
                    [int(lm[33].x*fw),  int(lm[33].y*fh)],
                    [int(lm[263].x*fw), int(lm[263].y*fh)],
                    [int(lm[61].x*fw),  int(lm[61].y*fh)],
                    [int(lm[291].x*fw), int(lm[291].y*fh)],
                ], dtype=np.float64)

                focal  = 1 * fw
                cam_m  = np.array([[focal,0,fw/2],[0,focal,fh/2],[0,0,1]], dtype=np.float64)
                ok, rvec, _ = cv2.solvePnP(FACE_3D, face_2d, cam_m,
                                            np.zeros((4,1),np.float64))
                if ok:
                    rmat, _ = cv2.Rodrigues(rvec)
                    ang, *_ = cv2.RQDecomp3x3(rmat)
                    pitch_v, yaw_v = round(ang[0], 2), round(ang[1], 2)

                # Puntos de iris sobre el frame de cámara
                for iris_idx in [IRIS_IZQ, IRIS_DER]:
                    ix = int(lm[iris_idx].x * fw)
                    iy = int(lm[iris_idx].y * fh)
                    cv2.circle(frame, (ix, iy), 3, (0, 255, 255), -1)
                    cv2.circle(frame, (ix, iy), 5, (0, 200, 180), 1)

        # ── GUARDAR MUESTRA ────────────────────────────────────────────────────
        if cara_ok and label_activo and frame_n % GUARDAR_CADA == 0:
            ts = datetime.datetime.now().isoformat(timespec="milliseconds")
            csv_writer.writerow([ts, ear_v, pitch_v, yaw_v, rh_v, rv_v, label_activo])
            csv_file.flush()
            contadores[label_activo] = contadores.get(label_activo, 0) + 1
            if contadores[label_activo] == META_MINIMA:
                print(f"  [OK] Meta alcanzada: '{label_activo}' — {META_MINIMA} muestras")

        # ── RENDER ────────────────────────────────────────────────────────────
        render_camera_area(canvas, frame, label_activo, cara_ok, frame_n)
        render_panel(canvas, contadores, label_activo,
                     (ear_v, pitch_v, yaw_v, rh_v, rv_v), t_inicio, frame_n)

        cv2.imshow(WIN_NAME, canvas)

        # ── TECLADO ────────────────────────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF

        if key in CLASES_KEYS:
            nuevo = CLASES_KEYS[key]
            if label_activo == nuevo:
                # Toggle: desactiva si ya está activa
                label_activo = None
                print(f"  [PAUSA] Modo desactivado.")
            else:
                label_activo  = nuevo
                ultimo_label  = nuevo
                print(f"  [REC]   Clase activa: {nuevo}")

        elif key == ord(' '):
            # SPACE = toggle pausa / reanudar
            if label_activo is not None:
                print(f"  [PAUSA] Grabación pausada.")
                label_activo = None
            else:
                if ultimo_label:
                    label_activo = ultimo_label
                    print(f"  [REC]   Reanudando: {ultimo_label}")
                else:
                    print("  [INFO]  Presiona E / D / S para elegir una clase primero.")

        elif key in (ord('q'), 27):  # Q o ESC
            break

    # ── CIERRE ────────────────────────────────────────────────────────────────
    csv_file.close()
    cap.release()
    fm.close()
    cv2.destroyAllWindows()

    total = sum(contadores.values())
    print("\n" + "="*60)
    print("  RESUMEN FINAL")
    print("="*60)
    for clase, n in contadores.items():
        ok    = "OK " if n >= META_MINIMA else "---"
        barra = "█" * min(n // 10, 30)
        print(f"  [{ok}] {clase:<12}: {n:>4}  {barra}")
    print(f"  Total          : {total}")
    print(f"  Guardado en    : {os.path.abspath(RUTA_CSV)}")
    print("="*60)
    if all(v >= META_MINIMA for v in contadores.values()):
        print("\n  Dataset LISTO. Ejecuta: python train_model.py")
    else:
        print("\n  Vuelve a ejecutar collect_dataset.py para continuar.")


if __name__ == "__main__":
    main()
