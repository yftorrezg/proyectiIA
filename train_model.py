# ==============================================================================
# EDU-INSIGHT — FASE 2: ENTRENAMIENTO DEL CLASIFICADOR DE ATENCIÓN
# Genera: modelo, métricas JSON, imágenes de reporte para el dashboard.
#
# Uso: python train_model.py
# Salidas:
#   models/attention_model.joblib      ← modelo listo para producción
#   reports/metricas.json              ← métricas para el dashboard
#   reports/confusion_matrix.png       ← heatmap para presentación
#   reports/feature_importance.png     ← importancia de variables
#   reports/reporte_clasificacion.txt  ← reporte de texto completo
# ==============================================================================
import os, json, datetime, sys
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
# Forzar UTF-8 en la terminal de Windows
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy  as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # sin ventana gráfica
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import joblib

from sklearn.model_selection  import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing    import LabelEncoder
from sklearn.metrics          import (accuracy_score, classification_report,
                                      confusion_matrix, f1_score)
from sklearn.ensemble         import RandomForestClassifier
import xgboost as xgb

# ──────────────────────────────────────────────────────────────────────────────
# RUTAS
# ──────────────────────────────────────────────────────────────────────────────
RUTA_CSV    = os.path.join("datasets", "atencion", "raw_data.csv")
RUTA_MODELO = os.path.join("models",   "attention_model.joblib")
RUTA_JSON   = os.path.join("reports",  "metricas.json")
RUTA_CM     = os.path.join("reports",  "confusion_matrix.png")
RUTA_FI     = os.path.join("reports",  "feature_importance.png")
RUTA_TXT    = os.path.join("reports",  "reporte_clasificacion.txt")

os.makedirs("models",  exist_ok=True)
os.makedirs("reports", exist_ok=True)

FEATURES = ["ear", "pitch", "yaw", "ratio_h"]   # ratio_v descartado (inestable)
CLASES   = ["Enfocado", "Distraido", "Somnoliento"]
SEED     = 42

# ──────────────────────────────────────────────────────────────────────────────
# PALETA PARA GRÁFICAS (coherente con el dashboard)
# ──────────────────────────────────────────────────────────────────────────────
COLORES_CLASE = {
    "Enfocado":    "#10b981",   # verde esmeralda
    "Distraido":   "#ef4444",   # rojo
    "Somnoliento": "#f59e0b",   # naranja ámbar
}
BG_DARK   = "#0f172a"
BG_CARD   = "#1e293b"
INDIGO    = "#6366f1"
TEXT_GRAY = "#94a3b8"

# ──────────────────────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────────────────────
def separador(titulo="", ancho=62):
    if titulo:
        lado = (ancho - len(titulo) - 2) // 2
        print("-" * lado + f" {titulo} " + "-" * lado)
    else:
        print("-" * ancho)

def clip_iqr(df, columnas, factor=2.5):
    """Recorta outliers usando el método IQR sobre el conjunto completo."""
    df_out = df.copy()
    for col in columnas:
        q1, q3 = df_out[col].quantile(0.25), df_out[col].quantile(0.75)
        iqr = q3 - q1
        lo, hi = q1 - factor * iqr, q3 + factor * iqr
        df_out[col] = df_out[col].clip(lo, hi)
    return df_out

# ──────────────────────────────────────────────────────────────────────────────
# 1. CARGA Y LIMPIEZA
# ──────────────────────────────────────────────────────────────────────────────
print()
separador("EDU-INSIGHT · Entrenamiento del Clasificador")
print(f"  Fecha  : {datetime.datetime.now().strftime('%Y-%m-%d  %H:%M:%S')}")
print(f"  Dataset: {os.path.abspath(RUTA_CSV)}")
separador()

df = pd.read_csv(RUTA_CSV)
print(f"\n[1/6] CARGA DE DATOS")
print(f"  Filas totales cargadas : {len(df)}")
print(f"  Distribución original  :")
for c, n in df["label"].value_counts().items():
    print(f"    {c:<12}: {n:>4}  ({n/len(df)*100:.1f}%)")

# Filtrar solo las clases conocidas
df = df[df["label"].isin(CLASES)].copy()
df = df[FEATURES + ["label"]].dropna()

# Aplicar clipping IQR para limpiar outliers
antes = len(df)
df = clip_iqr(df, FEATURES, factor=2.5)
print(f"\n  Clipping IQR (factor=2.5) aplicado sobre: {FEATURES}")
print(f"  Filas disponibles para entrenamiento: {len(df)}")

# ──────────────────────────────────────────────────────────────────────────────
# 2. DIVISIÓN TRAIN / TEST  (70 / 30 estratificada)
# ──────────────────────────────────────────────────────────────────────────────
X = df[FEATURES].values
y = df["label"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, stratify=y, random_state=SEED
)

print(f"\n[2/6] DIVISIÓN TRAIN / TEST")
print(f"  Entrenamiento : {len(X_train)} muestras  (70%)")
print(f"  Test (hold-out): {len(X_test)} muestras  (30%)")

# ──────────────────────────────────────────────────────────────────────────────
# 3. BÚSQUEDA DE HIPERPARÁMETROS (GridSearchCV 5-fold estratificado)
# ──────────────────────────────────────────────────────────────────────────────
print(f"\n[3/6] OPTIMIZACIÓN DE HIPERPARÁMETROS (XGBoost + 5-Fold CV)")
print("  Esto puede tardar 30-60 segundos...")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

param_grid = {
    "n_estimators":    [100, 200, 300],
    "max_depth":       [3, 5, 7],
    "learning_rate":   [0.05, 0.1, 0.2],
    "subsample":       [0.8, 1.0],
    "colsample_bytree":[0.8, 1.0],
}

base_model = xgb.XGBClassifier(
    objective="multi:softprob",
    eval_metric="mlogloss",
    use_label_encoder=False,
    random_state=SEED,
    verbosity=0,
)

# Codificar etiquetas a enteros para XGBoost
le = LabelEncoder()
le.fit(CLASES)
y_train_enc = le.transform(y_train)
y_test_enc  = le.transform(y_test)

grid_search = GridSearchCV(
    base_model,
    param_grid,
    cv=cv,
    scoring="f1_macro",
    n_jobs=-1,
    verbose=0,
)
grid_search.fit(X_train, y_train_enc)

best_params = grid_search.best_params_
best_cv_f1  = grid_search.best_score_

print(f"  Mejor Macro F1 (CV)  : {best_cv_f1:.4f}")
print(f"  Mejores hiperparámetros:")
for k, v in best_params.items():
    print(f"    {k:<20}: {v}")

# ──────────────────────────────────────────────────────────────────────────────
# 4. ENTRENAMIENTO FINAL con todos los datos de train
# ──────────────────────────────────────────────────────────────────────────────
print(f"\n[4/6] ENTRENAMIENTO FINAL")
model = xgb.XGBClassifier(
    objective="multi:softprob",
    eval_metric="mlogloss",
    use_label_encoder=False,
    random_state=SEED,
    verbosity=0,
    **best_params,
)
model.fit(X_train, y_train_enc)
print("  Modelo entrenado correctamente.")

# ──────────────────────────────────────────────────────────────────────────────
# 5. EVALUACIÓN EN TEST SET (el 30% que nunca se usó)
# ──────────────────────────────────────────────────────────────────────────────
print(f"\n[5/6] EVALUACIÓN EN TEST SET (hold-out 30%)")

y_pred_enc  = model.predict(X_test)
y_pred      = le.inverse_transform(y_pred_enc)
y_pred_prob = model.predict_proba(X_test)

accuracy   = accuracy_score(y_test, y_pred)
macro_f1   = f1_score(y_test, y_pred, average="macro")
cm         = confusion_matrix(y_test, y_pred, labels=CLASES)
cr         = classification_report(y_test, y_pred, labels=CLASES,
                                   target_names=CLASES, output_dict=True)
cr_txt     = classification_report(y_test, y_pred, labels=CLASES,
                                   target_names=CLASES)

# Importancia de features
fi = dict(zip(FEATURES, model.feature_importances_))

separador()
print(f"  ACCURACY   : {accuracy:.4f}  ({accuracy*100:.2f}%)")
print(f"  MACRO F1   : {macro_f1:.4f}")
separador()
print(f"\n  REPORTE POR CLASE:\n")
print("  " + cr_txt.replace("\n", "\n  "))

separador()
print(f"  MATRIZ DE CONFUSIÓN:")
print(f"  {'':>14}", end="")
for c in CLASES: print(f"  {c[:8]:>10}", end="")
print()
for i, clase_real in enumerate(CLASES):
    print(f"  Real {clase_real[:8]:<9}", end="")
    for j in range(len(CLASES)):
        val = cm[i][j]
        mark = " OK" if i == j else "  "
        print(f"  {val:>10}{mark}" if i==j else f"  {val:>10}", end="")
    print()

separador()
print(f"\n  IMPORTANCIA DE FEATURES:")
for feat, imp in sorted(fi.items(), key=lambda x: -x[1]):
    barra = "#" * int(imp * 40)
    print(f"    {feat:<10}: {imp:.4f}  {barra}")

# ──────────────────────────────────────────────────────────────────────────────
# 6. GUARDAR ARTEFACTOS
# ──────────────────────────────────────────────────────────────────────────────
print(f"\n[6/6] GUARDANDO ARTEFACTOS")

# 6a. Modelo + metadatos
paquete_modelo = {
    "model":         model,
    "label_encoder": le,
    "features":      FEATURES,
    "clases":        CLASES,
    "best_params":   best_params,
    "accuracy":      accuracy,
    "macro_f1":      macro_f1,
    "fecha":         datetime.datetime.now().isoformat(timespec="seconds"),
}
joblib.dump(paquete_modelo, RUTA_MODELO)
print(f"  Modelo    → {RUTA_MODELO}")

# 6b. métricas.json (para el dashboard)
metricas = {
    "accuracy":            round(accuracy, 4),
    "macro_f1":            round(macro_f1, 4),
    "cv_macro_f1":         round(best_cv_f1, 4),
    "n_train":             int(len(X_train)),
    "n_test":              int(len(X_test)),
    "n_total":             int(len(X_train) + len(X_test)),
    "modelo":              "XGBoostClassifier",
    "fecha":               datetime.datetime.now().isoformat(timespec="seconds"),
    "features":            FEATURES,
    "clases":              CLASES,
    "best_params":         {k: str(v) for k, v in best_params.items()},
    "confusion_matrix":    cm.tolist(),
    "feature_importance":  {k: round(float(v), 4) for k, v in fi.items()},
    "por_clase":           {
        c: {
            "precision": round(cr[c]["precision"], 4),
            "recall":    round(cr[c]["recall"],    4),
            "f1":        round(cr[c]["f1-score"],  4),
            "support":   int(cr[c]["support"]),
        } for c in CLASES
    },
}
with open(RUTA_JSON, "w", encoding="utf-8") as f:
    json.dump(metricas, f, indent=2, ensure_ascii=False)
print(f"  Métricas  → {RUTA_JSON}")

# 6c. Reporte de texto
with open(RUTA_TXT, "w", encoding="utf-8") as f:
    f.write(f"EDU-INSIGHT — Reporte de Clasificación\n")
    f.write(f"Fecha: {metricas['fecha']}\n")
    f.write(f"Accuracy : {accuracy:.4f}\n")
    f.write(f"Macro F1 : {macro_f1:.4f}\n\n")
    f.write(cr_txt)
print(f"  Reporte   → {RUTA_TXT}")

# 6d. Imagen: Matriz de Confusión
fig, ax = plt.subplots(figsize=(7, 5.5))
fig.patch.set_facecolor(BG_DARK)
ax.set_facecolor(BG_CARD)

vmax = cm.max()
im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=vmax)

for i in range(len(CLASES)):
    for j in range(len(CLASES)):
        val = cm[i, j]
        color = "white" if val > vmax * 0.5 else TEXT_GRAY
        weight = "bold" if i == j else "normal"
        ax.text(j, i, str(val), ha="center", va="center",
                fontsize=16, color=color, fontweight=weight)

ax.set_xticks(range(len(CLASES)))
ax.set_yticks(range(len(CLASES)))
ax.set_xticklabels(CLASES, fontsize=11, color=TEXT_GRAY)
ax.set_yticklabels(CLASES, fontsize=11, color=TEXT_GRAY)
ax.set_xlabel("Predicción del modelo", fontsize=12, color=TEXT_GRAY, labelpad=10)
ax.set_ylabel("Etiqueta real", fontsize=12, color=TEXT_GRAY, labelpad=10)

ax.set_title(f"Matriz de Confusión — Accuracy: {accuracy*100:.1f}%  |  Macro F1: {macro_f1:.3f}",
             fontsize=12, color="white", pad=14, fontweight="bold")

for spine in ax.spines.values(): spine.set_edgecolor(TEXT_GRAY)
ax.tick_params(colors=TEXT_GRAY)

cbar = fig.colorbar(im, ax=ax)
cbar.ax.yaxis.set_tick_params(color=TEXT_GRAY)
plt.setp(cbar.ax.yaxis.get_ticklabels(), color=TEXT_GRAY)

plt.tight_layout()
plt.savefig(RUTA_CM, dpi=150, bbox_inches="tight", facecolor=BG_DARK)
plt.close()
print(f"  Conf.Mat. → {RUTA_CM}")

# 6e. Imagen: Feature Importance
fi_sorted = sorted(fi.items(), key=lambda x: x[1])
nombres   = [f[0] for f in fi_sorted]
valores   = [f[1] for f in fi_sorted]
bar_colors = [INDIGO] * len(nombres)

fig, ax = plt.subplots(figsize=(7, 3.5))
fig.patch.set_facecolor(BG_DARK)
ax.set_facecolor(BG_CARD)

bars = ax.barh(nombres, valores, color=bar_colors, height=0.55, zorder=3)
ax.set_xlim(0, max(valores) * 1.25)
ax.set_xlabel("Importancia relativa", fontsize=11, color=TEXT_GRAY)
ax.set_title("Importancia de Features (XGBoost)", fontsize=12,
             color="white", pad=12, fontweight="bold")

for bar, val in zip(bars, valores):
    ax.text(val + max(valores)*0.02, bar.get_y() + bar.get_height()/2,
            f"{val:.4f}", va="center", fontsize=10, color="white", fontweight="bold")

ax.tick_params(colors=TEXT_GRAY)
ax.set_yticklabels(nombres, fontsize=11, color=TEXT_GRAY)
for spine in ax.spines.values(): spine.set_edgecolor(TEXT_GRAY)
ax.xaxis.grid(True, color="#334155", zorder=0, linewidth=0.7)
ax.set_axisbelow(True)

plt.tight_layout()
plt.savefig(RUTA_FI, dpi=150, bbox_inches="tight", facecolor=BG_DARK)
plt.close()
print(f"  Feat.Imp. → {RUTA_FI}")

# ──────────────────────────────────────────────────────────────────────────────
# RESUMEN FINAL
# ──────────────────────────────────────────────────────────────────────────────
separador()
print(f"\n  RESUMEN EJECUTIVO PARA TU INGENIERO")
separador()
print(f"  Modelo      : XGBoost (n_est={best_params['n_estimators']}, "
      f"depth={best_params['max_depth']}, lr={best_params['learning_rate']})")
print(f"  Dataset     : {len(X_train)+len(X_test)} muestras  "
      f"(train={len(X_train)} / test={len(X_test)})")
print(f"  Features    : {', '.join(FEATURES)}")
print()
print(f"  ┌─────────────────────────────────────────┐")
print(f"  │  ACCURACY       :  {accuracy*100:>6.2f}%              │")
print(f"  │  MACRO F1-SCORE :   {macro_f1:.4f}              │")
print(f"  │  CV MACRO F1    :   {best_cv_f1:.4f}  (val. cruzada)│")
print(f"  └─────────────────────────────────────────┘")
print()
for c in CLASES:
    p = cr[c]["precision"]; r = cr[c]["recall"]; f = cr[c]["f1-score"]
    print(f"  {c:<12}  P={p:.3f}  R={r:.3f}  F1={f:.3f}")
print()
separador()
print()
print("  Siguiente paso: integrar el modelo en pruebas.py")
print(f"  (Ejecuta el servidor y verás la predicción ML en el dashboard)")
print()
