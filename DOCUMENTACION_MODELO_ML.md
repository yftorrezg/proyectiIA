# Documentación Técnica del Clasificador de Atención
## Proyecto EDU-INSIGHT — Módulo de Machine Learning
### `train_model.py` — Guía completa para defensa de presentación

---

## Índice

1. [Contexto del problema](#1-contexto-del-problema)
2. [¿Qué es XGBoost? — Teoría completa](#2-qué-es-xgboost--teoría-completa)
3. [Arquitectura del pipeline de entrenamiento](#3-arquitectura-del-pipeline-de-entrenamiento)
4. [Fase 1 — Carga y limpieza de datos](#4-fase-1--carga-y-limpieza-de-datos)
5. [Fase 2 — División Train / Test estratificada](#5-fase-2--división-train--test-estratificada)
6. [Fase 3 — Optimización de hiperparámetros (GridSearchCV)](#6-fase-3--optimización-de-hiperparámetros-gridsearchcv)
7. [Fase 4 — Entrenamiento final](#7-fase-4--entrenamiento-final)
8. [Fase 5 — Evaluación y métricas](#8-fase-5--evaluación-y-métricas)
9. [Fase 6 — Serialización de artefactos](#9-fase-6--serialización-de-artefactos)
10. [Resultados obtenidos](#10-resultados-obtenidos)
11. [Preguntas frecuentes del ingeniero — respuestas preparadas](#11-preguntas-frecuentes-del-ingeniero--respuestas-preparadas)

---

## 1. Contexto del problema

### ¿Qué problema resuelve este código?

El sistema EDU-INSIGHT necesita determinar, frame a frame (30 veces por segundo), si un alumno está en uno de tres estados de atención:

| Clase | Descripción |
|-------|-------------|
| **Enfocado** | Mira la pantalla, ojos abiertos, cabeza al frente |
| **Distraido** | Gira la cabeza o los ojos hacia otro lado |
| **Somnoliento** | Ojos semicerrados, parpadeo lento, cabeza caída |

**Antes de este módulo** la detección se hacía con reglas fijas escritas a mano:
```python
# Versión heurística (lógica fija, no aprende de datos)
if ear < 0.22 durante 20 frames:
    estado = "Somnoliento"
elif pitch < -30 or pitch > 35 or yaw > 35:
    estado = "Distraido"
else:
    estado = "Enfocado"
```

**El problema** de las reglas fijas es que los umbrales (0.22, -30, 35) son arbitrarios y no se adaptan a cada persona. El modelo de Machine Learning **aprende automáticamente** los límites óptimos a partir de 3,571 muestras reales.

### Features (variables de entrada)

El modelo recibe 4 valores numéricos por cada frame, todos calculados por MediaPipe:

| Feature | Qué mide | Rango típico |
|---------|----------|--------------|
| `ear` | Eye Aspect Ratio — qué tan abiertos están los ojos | 0.05 – 0.45 |
| `pitch` | Inclinación vertical de la cabeza (grados) | -80° a +35° |
| `yaw` | Rotación horizontal de la cabeza (grados) | -70° a +70° |
| `ratio_h` | Posición horizontal del iris (0=extremo izq, 1=extremo der) | 0.4 – 0.6 |

> **Nota técnica:** Se descartó `ratio_v` (posición vertical del iris) porque cuando los ojos están casi cerrados (clase Somnoliento), el denominador del cálculo se acerca a cero, generando valores extremos como −587 o +296. El 22.3% de las filas de Somnoliento tenían `ratio_v` corrupto. Las 4 features restantes son numéricamente estables.

---

## 2. ¿Qué es XGBoost? — Teoría completa

### 2.1 La idea central: Gradient Boosting

XGBoost (eXtreme Gradient Boosting) es un algoritmo de **aprendizaje supervisado basado en árboles de decisión** que construye su modelo de forma incremental, añadiendo un árbol a la vez donde cada árbol nuevo **corrige los errores del árbol anterior**.

La idea es análoga a estudiar para un examen:
1. Haces un primer intento → cometes errores
2. Revisas solo lo que fallaste → te concentras en los errores
3. Vuelves a intentarlo → mejoras
4. Repites el proceso N veces hasta que los errores son mínimos

### 2.2 Árbol de decisión: el componente base

Cada "árbol" en XGBoost es un árbol de decisión binario que hace preguntas del tipo:

```
¿EAR < 0.18?
    ├── SÍ → ¿pitch < -5?
    │           ├── SÍ → Somnoliento (confianza: 0.97)
    │           └── NO → Somnoliento (confianza: 0.94)
    └── NO → ¿yaw > 28?
                ├── SÍ → Distraido (confianza: 0.96)
                └── NO → ¿ratio_h < 0.42?
                              ├── SÍ → Distraido (confianza: 0.88)
                              └── NO → Enfocado (confianza: 0.99)
```

Cada nodo hace una pregunta binaria sobre una feature. Las hojas dan la clase predicha. La **profundidad** del árbol (`max_depth`) controla qué tan complejo puede ser cada árbol.

### 2.3 El proceso de Boosting paso a paso

```
Árbol 1:  Hace predicciones → calcula errores residuales
    ↓
Árbol 2:  Se entrena sobre los ERRORES del árbol 1 → calcula nuevos residuales
    ↓
Árbol 3:  Se entrena sobre los errores restantes
    ↓
    ...
    ↓
Árbol N:  Predicción final = suma ponderada de TODOS los árboles
```

La predicción final es:
```
Ŷ = árbol_1(X) * lr + árbol_2(X) * lr + ... + árbol_N(X) * lr
```

donde `lr` es el `learning_rate` (tasa de aprendizaje, en este caso 0.1). Un `learning_rate` bajo significa que cada árbol aporta poco — el modelo aprende despacio pero generaliza mejor.

### 2.4 La función objetivo de XGBoost

XGBoost minimiza matemáticamente:

```
Objetivo = Σ L(yᵢ, ŷᵢ) + Σ Ω(fₖ)
              ↑                ↑
         Error de predicción   Penalización por complejidad
         (log-loss multiclase) (regularización para evitar overfitting)
```

- **L(yᵢ, ŷᵢ)**: pérdida logarítmica multiclase (`multi:softprob`). Penaliza predicciones incorrectas o inciertas.
- **Ω(fₖ)**: regularización. Penaliza árboles muy complejos. Controlada por `reg_lambda` (L2) y `reg_alpha` (L1).

### 2.5 Por qué XGBoost para este problema

| Criterio | XGBoost | Red Neuronal | Regresión Logística |
|----------|---------|--------------|---------------------|
| Datos pequeños (~3500 muestras) | Excelente | Mediocre | Bueno |
| Features numéricas mixtas | Excelente | Requiere normalización | Requiere normalización |
| Sin escalado requerido | Sí | No | No |
| Interpretabilidad | Alta (importancia de features) | Baja (caja negra) | Media |
| Velocidad de predicción | Muy alta (microsegundos) | Media | Alta |
| Manejo de outliers | Bueno (árboles) | Malo | Malo |

**Conclusión:** Para 4 features numéricas, 3 clases y ~3500 muestras, XGBoost es la elección técnicamente más sólida.

### 2.6 `objective="multi:softprob"`

Para clasificación multiclase, XGBoost usa **softmax** que convierte los scores de cada árbol en probabilidades:

```
P(clase=k | X) = exp(score_k) / Σⱼ exp(score_j)
```

Esto significa que para cada frame, el modelo no solo dice "Enfocado" sino que devuelve:
```python
[P(Enfocado)=0.97, P(Distraido)=0.02, P(Somnoliento)=0.01]
```

La clase con mayor probabilidad gana. El valor máximo es la **confianza** que se muestra en el dashboard (en este caso 97%).

---

## 3. Arquitectura del pipeline de entrenamiento

```
raw_data.csv (3,571 filas)
       │
       ▼
┌─────────────────────────────┐
│  FASE 1: Limpieza de datos  │
│  - Filtrar clases válidas   │
│  - Eliminar NaN             │
│  - Clipping IQR (factor 2.5)│
│  - Descartar ratio_v        │
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│  FASE 2: División 70/30     │
│  - Estratificada por clase  │
│  - random_state=42 (reprod.)│
│  - X_train: 2,499 muestras  │
│  - X_test:  1,072 muestras  │
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│  FASE 3: GridSearchCV       │
│  - 5-Fold CV estratificado  │
│  - 108 combinaciones        │
│  - Optimiza: Macro F1       │
│  - Mejor CV F1: 0.9638      │
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│  FASE 4: Entrenamiento      │
│  - XGBoost con best_params  │
│  - Sobre X_train completo   │
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│  FASE 5: Evaluación         │
│  - Solo sobre X_test (30%)  │
│  - Accuracy, F1, CM, FI     │
│  - Datos nunca vistos       │
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│  FASE 6: Artefactos         │
│  - attention_model.joblib   │
│  - metricas.json            │
│  - confusion_matrix.png     │
│  - feature_importance.png   │
└─────────────────────────────┘
```

---

## 4. Fase 1 — Carga y limpieza de datos

### Código

```python
FEATURES = ["ear", "pitch", "yaw", "ratio_h"]   # ratio_v descartado (inestable)
CLASES   = ["Enfocado", "Distraido", "Somnoliento"]
SEED     = 42

df = pd.read_csv(RUTA_CSV)

# Filtrar solo las clases conocidas y eliminar valores vacíos
df = df[df["label"].isin(CLASES)].copy()
df = df[FEATURES + ["label"]].dropna()

# Aplicar clipping IQR para limpiar outliers
df = clip_iqr(df, FEATURES, factor=2.5)
```

### La función `clip_iqr` explicada

```python
def clip_iqr(df, columnas, factor=2.5):
    df_out = df.copy()
    for col in columnas:
        q1, q3 = df_out[col].quantile(0.25), df_out[col].quantile(0.75)
        iqr = q3 - q1
        lo, hi = q1 - factor * iqr, q3 + factor * iqr
        df_out[col] = df_out[col].clip(lo, hi)
    return df_out
```

**¿Qué hace el IQR (Rango Intercuartílico)?**

```
Datos ordenados:
[0.1, 0.2, 0.25, 0.28, 0.30, 0.32, 0.33, 0.35, 0.38, 1.51]
                  ↑ Q1=0.25         ↑ Q3=0.35

IQR = Q3 - Q1 = 0.35 - 0.25 = 0.10

Límite inferior = 0.25 - 2.5 * 0.10 = 0.00
Límite superior = 0.35 + 2.5 * 0.10 = 0.60

El valor 1.51 (EAR imposible) se recorta a 0.60
```

**¿Por qué no se eliminan las filas con outliers?** Porque perderíamos muestras de la clase Somnoliento que son valiosas. El clipping conserva la fila pero con un valor razonable en lugar del valor aberrante.

**¿Por qué factor 2.5 y no 1.5 (el estándar)?** El factor 1.5 es muy agresivo para datos biomecánicos donde la variabilidad natural es alta. Factor 2.5 elimina solo los outliers realmente extremos (como `ratio_v = -587`).

### ¿Por qué se descartó `ratio_v`?

Durante el análisis exploratorio del dataset se encontró:

| Feature | % filas corruptas Somnoliento |
|---------|-------------------------------|
| ear | 0% |
| pitch | 0.08% |
| yaw | 0% |
| ratio_h | 1.5% |
| **ratio_v** | **22.3%** — INUSABLE |

Cuando los ojos están casi cerrados (clase Somnoliento), la altura del ojo es mínima. El cálculo `ratio_v = (iris_y - top_y) / (bot_y - top_y)` tiene denominador `bot_y - top_y ≈ 0`, produciendo divisiones por cero que generan valores como −587 o +296. El EAR (que usa distancias euclidianas) no tiene este problema.

---

## 5. Fase 2 — División Train / Test estratificada

### Código

```python
X = df[FEATURES].values   # matriz 3571 x 4
y = df["label"].values    # vector 3571 etiquetas

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.30,    # 30% para test
    stratify=y,        # proporciones de clase iguales en train y test
    random_state=42    # reproducibilidad
)
# Resultado: X_train (2499 x 4), X_test (1072 x 4)
```

### ¿Por qué 70/30 y no 80/20?

Con 3,571 muestras totales, el 30% de test = 1,072 muestras. Eso da un margen estadístico sólido para cada clase (~350 muestras por clase en test). Si usáramos 80/20, solo tendríamos ~238 muestras por clase en test — menos representativo.

### ¿Qué significa `stratify=y`?

Sin estratificación, podría ocurrir por azar que el test set tenga 40% Somnoliento y 20% Distraido — no sería representativo del dataset real. Con `stratify=y`, se garantiza que tanto train como test tengan la misma distribución de clases que el dataset original:

```
Dataset total:   Enfocado 31.6%  |  Distraido 36.1%  |  Somnoliento 32.3%
Train (2499):    Enfocado 31.6%  |  Distraido 36.1%  |  Somnoliento 32.3%
Test  (1072):    Enfocado 31.6%  |  Distraido 36.1%  |  Somnoliento 32.3%
```

### ¿Por qué `random_state=42`?

Garantiza reproducibilidad. Si el ingeniero ejecuta el script mañana con los mismos datos, obtendrá exactamente la misma división y por lo tanto las mismas métricas. Hace el experimento **reproducible y verificable**.

### El principio fundamental: el test set NUNCA se toca

El 30% de test se guarda en una caja sellada y **solo se abre una vez**: en la evaluación final. Esto simula cómo funcionará el modelo con datos completamente nuevos del mundo real. Si se usara el test set para ajustar hiperparámetros, las métricas serían engañosamente optimistas (data leakage).

```
     Train (70%)                    Test (30%)
┌──────────────────────┐          ┌────────────┐
│ GridSearchCV 5-Fold  │          │            │
│ ┌───┬───┬───┬───┬───┐│          │  SELLADO   │
│ │ 1 │ 2 │ 3 │ 4 │ 5 ││  →→→→    │   hasta    │
│ └───┴───┴───┴───┴───┘│          │ evaluación │
│ Entrenamiento Final  │          │   final    │
└──────────────────────┘          └────────────┘
```

---

## 6. Fase 3 — Optimización de hiperparámetros (GridSearchCV)

### Código

```python
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

param_grid = {
    "n_estimators":     [100, 200, 300],    # número de árboles
    "max_depth":        [3, 5, 7],          # profundidad máxima de cada árbol
    "learning_rate":    [0.05, 0.1, 0.2],   # contribución de cada árbol
    "subsample":        [0.8, 1.0],         # fracción de datos para cada árbol
    "colsample_bytree": [0.8, 1.0],         # fracción de features por árbol
}

grid_search = GridSearchCV(
    base_model,
    param_grid,
    cv=cv,
    scoring="f1_macro",   # optimizar F1 promedio entre clases
    n_jobs=-1,            # usar todos los núcleos del CPU
)
grid_search.fit(X_train, y_train_enc)
```

### ¿Qué es GridSearchCV?

GridSearchCV prueba **todas las combinaciones posibles** de hiperparámetros. En este caso:

```
3 × 3 × 3 × 2 × 2 = 108 combinaciones
```

Para cada combinación, entrena y evalúa el modelo con **validación cruzada 5-fold**:

```
Iteración 1: Entrena en [2,3,4,5] → Evalúa en [1]
Iteración 2: Entrena en [1,3,4,5] → Evalúa en [2]
Iteración 3: Entrena en [1,2,4,5] → Evalúa en [3]
Iteración 4: Entrena en [1,2,3,5] → Evalúa en [4]
Iteración 5: Entrena en [1,2,3,4] → Evalúa en [5]

CV Score = promedio de los 5 resultados
```

**Total de modelos entrenados**: 108 combinaciones × 5 folds = **540 entrenamientos**, ejecutados en paralelo (`n_jobs=-1`).

### ¿Por qué validación cruzada estratificada?

`StratifiedKFold` garantiza que en cada fold haya proporciones iguales de las 3 clases. Esto es crítico para que el `f1_macro` calculado en cada fold sea representativo.

### ¿Por qué optimizar `f1_macro` y no `accuracy`?

El **accuracy** puede ser engañoso con clases desbalanceadas. Si el modelo siempre predijera "Distraido" (la clase más frecuente, 36%), tendría un accuracy del 36% sin aprender nada.

El **Macro F1** promedia el F1-score de cada clase dando el mismo peso a todas. Así, si el modelo falla en detectar "Somnoliento" (que podría ser la clase menos frecuente), el Macro F1 lo penaliza igual que si fallara en "Enfocado".

### Hiperparámetros — explicación detallada

| Hiperparámetro | Qué controla | Valor elegido | Por qué |
|----------------|--------------|---------------|---------|
| `n_estimators` | Número de árboles en el ensemble | 100 | Con datos bien separables, 100 árboles ya converge |
| `max_depth` | Profundidad máxima de cada árbol | 5 | Balancea complejidad y generalización; 3 es muy simple, 7 puede overfit |
| `learning_rate` | Cuánto aporta cada árbol nuevo | 0.1 | Estándar industrial; más bajo = más robusto pero más lento |
| `subsample` | Fracción de filas por árbol | 0.8 | Introduce aleatoriedad → previene overfitting (como bagging) |
| `colsample_bytree` | Fracción de features por árbol | 1.0 | Con solo 4 features, usar todas es óptimo |

### Resultado de la búsqueda

```
Mejor Macro F1 (CV) : 0.9638
Mejores parámetros  :
    n_estimators     = 100
    max_depth        = 5
    learning_rate    = 0.1
    subsample        = 0.8
    colsample_bytree = 1.0
```

---

## 7. Fase 4 — Entrenamiento final

### Código

```python
le = LabelEncoder()
le.fit(CLASES)
y_train_enc = le.transform(y_train)   # "Enfocado"→0, "Distraido"→1, "Somnoliento"→2

model = xgb.XGBClassifier(
    objective="multi:softprob",    # clasificación multiclase con probabilidades
    eval_metric="mlogloss",        # log-loss para evaluación interna
    use_label_encoder=False,       # no usar el encoder obsoleto de XGBoost
    random_state=SEED,
    verbosity=0,
    **best_params,                 # desempaqueta los mejores parámetros encontrados
)
model.fit(X_train, y_train_enc)
```

### ¿Por qué `LabelEncoder`?

XGBoost solo acepta etiquetas numéricas enteras, no strings. El `LabelEncoder` convierte:
```
"Enfocado"    → 0
"Distraido"   → 1
"Somnoliento" → 2
```

El encoder se guarda junto con el modelo para poder revertir la transformación en producción:
```python
modelo["label_encoder"].inverse_transform([0, 1, 2])
# → ["Enfocado", "Distraido", "Somnoliento"]
```

### ¿Por qué entrenar sobre todos los datos de train y no usar el mejor fold?

El modelo del GridSearchCV entrena en el 80% de cada fold para encontrar los mejores hiperparámetros. El entrenamiento final usa el **100% de X_train** (2,499 muestras), lo que da un modelo más robusto con más datos.

---

## 8. Fase 5 — Evaluación y métricas

### Código

```python
y_pred_enc  = model.predict(X_test)
y_pred      = le.inverse_transform(y_pred_enc)    # numérico → string
y_pred_prob = model.predict_proba(X_test)          # probabilidades por clase

accuracy = accuracy_score(y_test, y_pred)
macro_f1 = f1_score(y_test, y_pred, average="macro")
cm       = confusion_matrix(y_test, y_pred, labels=CLASES)
cr       = classification_report(y_test, y_pred, labels=CLASES, output_dict=True)

# Importancia de features (score de ganancia de información)
fi = dict(zip(FEATURES, model.feature_importances_))
```

### Métrica 1: Accuracy

```
Accuracy = Predicciones correctas / Total de predicciones
         = (336 + 375 + 340) / 1072
         = 1051 / 1072
         = 98.04%
```

**Interpretación:** De cada 100 frames analizados, el modelo clasifica correctamente el estado de atención 98 veces. Solo falla 2 veces.

### Métrica 2: Precision

```
Precision(clase C) = TP_C / (TP_C + FP_C)
```

Responde: **"De todas las veces que el modelo dijo C, ¿cuántas realmente eran C?"**

```
Precision(Enfocado)    = 336 / (336 + 6 + 0)  = 336/342 = 98.25%
Precision(Distraido)   = 375 / (1 + 375 + 7)  = 375/383 = 97.91%
Precision(Somnoliento) = 340 / (1 + 6 + 340)  = 340/347 = 97.98%
```

### Métrica 3: Recall (Sensibilidad)

```
Recall(clase C) = TP_C / (TP_C + FN_C)
```

Responde: **"De todos los casos reales de C, ¿cuántos detectó el modelo?"**

```
Recall(Enfocado)    = 336 / (336 + 1 + 1)  = 336/338 = 99.41%
Recall(Distraido)   = 375 / (6 + 375 + 6)  = 375/387 = 96.90%
Recall(Somnoliento) = 340 / (0 + 7 + 340)  = 340/347 = 97.98%
```

**Recall de Somnoliento = 97.98%** — Esto es crítico en el contexto educativo: el modelo detecta al alumno somnoliento casi siempre.

### Métrica 4: F1-Score

```
F1(clase C) = 2 × (Precision × Recall) / (Precision + Recall)
```

Es la **media armónica** de Precision y Recall. Penaliza si alguno de los dos es bajo.

```
F1(Enfocado)    = 2 × (0.9825 × 0.9941) / (0.9825 + 0.9941) = 98.82%
F1(Distraido)   = 2 × (0.9791 × 0.9690) / (0.9791 + 0.9690) = 97.40%
F1(Somnoliento) = 2 × (0.9798 × 0.9798) / (0.9798 + 0.9798) = 97.98%
```

### Métrica 5: Macro F1-Score

```
Macro F1 = (F1_Enfocado + F1_Distraido + F1_Somnoliento) / 3
         = (0.9882 + 0.9740 + 0.9798) / 3
         = 0.9807
```

Da el mismo peso a cada clase sin importar cuántas muestras tenga.

### La Matriz de Confusión — interpretación completa

```
                   PREDICCIÓN DEL MODELO
                 Enfocado  Distraido  Somnoliento
REAL Enfocado      336  ✓      1           1
REAL Distraido       6        375  ✓        6
REAL Somnoliento     0          7         340  ✓
```

| Celda | Valor | Significado |
|-------|-------|-------------|
| (Enfocado, Enfocado) | **336** | El modelo dijo "Enfocado" y era correcto |
| (Enfocado, Distraido) | 1 | El modelo se equivocó: era Enfocado pero dijo Distraido |
| (Enfocado, Somnoliento) | 1 | Era Enfocado pero dijo Somnoliento |
| (Distraido, Enfocado) | 6 | Era Distraido pero dijo Enfocado |
| (Distraido, Distraido) | **375** | Correcto |
| (Distraido, Somnoliento) | 6 | Era Distraido pero dijo Somnoliento |
| (Somnoliento, Enfocado) | 0 | Nunca confundió Somnoliento con Enfocado |
| (Somnoliento, Distraido) | 7 | Era Somnoliento pero dijo Distraido |
| (Somnoliento, Somnoliento) | **340** | Correcto |

**Análisis de errores:**
- Los errores más frecuentes son Distraido↔Somnoliento (6 + 7 = 13 casos). Esto tiene sentido: cuando el alumno está somnoliento puede inclinar la cabeza de lado, lo que MediaPipe interpreta como un yaw elevado (similar a Distraido).
- Nunca hubo confusión Somnoliento→Enfocado (0 casos). El modelo nunca catalogó como "atento" a un alumno somnoliento.

### Importancia de features

```
ear       : 64.09%  ████████████████████████████████████████
ratio_h   : 17.73%  ████████████
pitch     :  9.85%  ████████
yaw       :  8.32%  ██████
```

El **EAR domina con 64%**. Esto tiene sentido matemático: el EAR separa Somnoliento (media 0.093) de las otras dos clases (0.328 y 0.396) con una diferencia de casi 3 desviaciones estándar. Es el predictor más potente.

`ratio_h` con 17.73% captura cuando el iris se desvía horizontalmente (alumno mirando al costado). `pitch` y `yaw` discriminan entre Enfocado y Distraido cuando la cabeza se mueve.

**La importancia de features en XGBoost** se calcula como la **ganancia de información promedio** que aporta cada feature en todas las divisiones de todos los árboles donde esa feature fue usada.

---

## 9. Fase 6 — Serialización de artefactos

### El archivo `attention_model.joblib`

```python
paquete_modelo = {
    "model":         model,          # el objeto XGBClassifier entrenado
    "label_encoder": le,             # convierte int → nombre de clase
    "features":      FEATURES,       # ["ear","pitch","yaw","ratio_h"]
    "clases":        CLASES,         # ["Enfocado","Distraido","Somnoliento"]
    "best_params":   best_params,    # hiperparámetros óptimos encontrados
    "accuracy":      accuracy,       # 0.9804
    "macro_f1":      macro_f1,       # 0.9807
    "fecha":         "2026-04-15...", # cuándo se entrenó
}
joblib.dump(paquete_modelo, "models/attention_model.joblib")
```

`joblib` es más eficiente que `pickle` para objetos NumPy/scikit-learn. El archivo pesa ~513 KB y contiene todo lo necesario para hacer predicciones en producción sin necesidad de volver a entrenar.

### Cómo se usa en producción (`pruebas.py`)

```python
# Al iniciar el servidor
pkg = joblib.load("models/attention_model.joblib")

# En cada frame (30 veces por segundo)
features = np.array([[ear_promedio, pitch, yaw, ratio_h]])
pred_idx  = pkg["model"].predict(features)[0]
pred_prob = pkg["model"].predict_proba(features)[0]
pred_label = pkg["label_encoder"].inverse_transform([pred_idx])[0]
confidence = float(np.max(pred_prob))

# pred_label = "Enfocado" | "Distraido" | "Somnoliento"
# confidence = 0.0 – 1.0 (se muestra en el dashboard)
```

La predicción toma **menos de 1 milisegundo** por frame, mucho menos que los 33ms disponibles a 30fps.

---

## 10. Resultados obtenidos

### Resumen ejecutivo

| Métrica | Valor |
|---------|-------|
| **Accuracy** | **98.04%** |
| **Macro F1-Score** | **0.9807** |
| CV Macro F1 (validación cruzada) | 0.9638 |
| Muestras de entrenamiento | 2,499 |
| Muestras de test (nunca vistas) | 1,072 |
| Total del dataset | 3,571 |
| Modelo | XGBoost (100 árboles, profundidad 5) |
| Features utilizadas | 4 (ear, pitch, yaw, ratio_h) |

### Resultados por clase

| Clase | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| **Enfocado** | 98.25% | 99.41% | **98.82%** | 338 |
| **Distraido** | 97.91% | 96.90% | **97.40%** | 387 |
| **Somnoliento** | 97.98% | 97.98% | **97.98%** | 347 |

### Comparación: Heurístico vs ML

| Aspecto | Modo Heurístico | Modo XGBoost ML |
|---------|-----------------|-----------------|
| Método | Reglas fijas manuales | Aprendizaje desde datos |
| Umbrales | Arbitrarios | Optimizados matemáticamente |
| Adaptabilidad | No se adapta a la persona | Entrenado con tus datos reales |
| Confianza | No disponible | Sí, por predicción |
| Mantenimiento | Cambiar código | Reentrenar con más datos |
| Accuracy medible | No (no hay ground truth) | **98.04%** en test independiente |

---

## 11. Preguntas frecuentes del ingeniero — respuestas preparadas

**P: ¿Por qué no usaron una red neuronal?**
> Con 3,571 muestras y solo 4 features, las redes neuronales no tienen ventaja sobre XGBoost. Requieren más datos para generalizar bien y son más difíciles de interpretar. XGBoost con este tamaño de dataset típicamente supera a las redes neuronales en tabular data, como lo confirman benchmarks de Kaggle y la literatura (Chen & Guestrin, 2016).

**P: ¿No hay riesgo de overfitting con 100 árboles?**
> Se mitiga con tres mecanismos: (1) `subsample=0.8` usa solo el 80% de los datos en cada árbol, (2) `learning_rate=0.1` limita la contribución de cada árbol, y (3) la validación cruzada de 5 folds detectaría overfitting si el CV score fuera mucho menor que el test score. El CV F1 fue 0.9638 y el test F1 fue 0.9807 — diferencia de solo 0.017, indicando buen balance bias-varianza.

**P: ¿El modelo funciona con otras personas o solo contigo?**
> Actualmente está entrenado con datos de una sola persona. Para generalizar, habría que recolectar datos de múltiples personas. Sin embargo, para el propósito del sistema (monitoreo de un alumno específico) es un clasificador personalizado, lo cual es una fortaleza: se adapta a la geometría facial y hábitos de movimiento de esa persona en particular.

**P: ¿Cómo se garantiza que el 98% de accuracy sea real y no inflado?**
> El test set (30%, 1,072 muestras) se separó antes de cualquier entrenamiento y optimización. Nunca se usó para ajustar hiperparámetros ni para tomar decisiones de diseño. La diferencia entre el CV score (0.9638) y el test score (0.9807) es positiva, lo que indica que el modelo generaliza bien y no hubo data leakage.

**P: ¿Por qué se descartó `ratio_v`?**
> El análisis exploratorio mostró que el 22.3% de las muestras de la clase Somnoliento tenían valores de `ratio_v` fuera de rango (como −587 o +296). Esto ocurre porque cuando los ojos están casi cerrados, el denominador del cálculo del iris vertical tiende a cero, generando divisiones por cero. Incluir esta feature añadiría ruido sin información, reduciendo la calidad del modelo.

**P: ¿Qué pasa si el modelo tiene baja confianza?**
> El dashboard muestra el porcentaje de confianza en tiempo real. Si la confianza cae por debajo del 60% (indicador ámbar) o del 40% (indicador rojo), significa que los valores de las features están en una zona ambigua. En ese caso, el sistema puede configurarse para volver al modo heurístico como fallback, garantizando robustez.

**P: ¿Qué significan los 7 errores de Somnoliento→Distraido?**
> Son casos donde el alumno, al quedarse somnoliento, inclinó la cabeza hacia el costado. Esta postura hace que el `yaw` suba y el `pitch` se desvíe, características propias de la clase Distraido. Es el único caso de confusión lógicamente justificable. Para eliminarlo, habría que añadir la feature de `roll` (inclinación lateral) que MediaPipe también calcula.

---

*Documentación generada para el proyecto EDU-INSIGHT — Clasificador de Atención Personalizado*
*Modelo: XGBoost v3.2.0 | scikit-learn v1.7.2 | Dataset: 3,571 muestras propias*
