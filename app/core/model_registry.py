"""
ModelRegistry — Edu-Insight MLOps
Gestiona el catálogo de modelos disponibles, el modelo activo por categoría,
y el ciclo de vida (carga/descarga) con control de VRAM.
"""

import gc
import logging
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
#  ENUMS Y DATACLASSES
# ─────────────────────────────────────────────

class ModelCategory(str, Enum):
    ATENCION  = "atencion"
    AUDIO     = "audio"
    EMOCION   = "emocion"
    SEMANTICA = "semantica"


@dataclass
class ModelConfig:
    id: str
    name: str
    category: ModelCategory
    description: str
    vram_mb: int          # VRAM estimada en MB (0 = CPU-only)
    trainable: bool       # ¿Puede reentrenarse con dataset local?
    tags: List[str] = field(default_factory=list)
    # Info extra para el frontend (dataset requerido, tiempo estimado, etc.)
    metadata: Dict[str, Any] = field(default_factory=dict)


# ─────────────────────────────────────────────
#  CATÁLOGO COMPLETO DE MODELOS
# ─────────────────────────────────────────────

MODEL_CATALOG: Dict[ModelCategory, List[ModelConfig]] = {

    # ── ATENCIÓN ─────────────────────────────────────────────────────────────
    # Dataset: datasets/atencion/raw_data.csv  (3,571 muestras)
    # Features: EAR, pitch, yaw, ratio_h
    # Clases:   Enfocado | Distraído | Somnoliento
    # ─────────────────────────────────────────────────────────────────────────
    ModelCategory.ATENCION: [
        ModelConfig(
            id="xgboost",
            name="XGBoost",
            category=ModelCategory.ATENCION,
            description="Gradient Boosting sobre árboles de decisión. Rápido, interpretable y con alta precisión en datos tabulares. Modelo actual en producción.",
            vram_mb=0,
            trainable=True,
            tags=["cpu", "actual", "recomendado"],
            metadata={
                "hyperparams": {
                    "n_estimators":   {"default": 100, "min": 50,   "max": 500,  "step": 50,   "tip": "Más árboles = más precisión pero más lento. 100–200 es el punto óptimo."},
                    "max_depth":      {"default": 5,   "min": 3,    "max": 10,   "step": 1,    "tip": "Profundidad del árbol. >6 puede causar overfitting con este dataset."},
                    "learning_rate":  {"default": 0.1, "min": 0.01, "max": 0.3,  "step": 0.01, "tip": "Learning rate bajo (0.05) con más epochs da mejores resultados generalizables."},
                    "subsample":      {"default": 0.8, "min": 0.5,  "max": 1.0,  "step": 0.1,  "tip": "Fracción de muestras por árbol. 0.8 previene overfitting."},
                },
                "dataset": "datasets/atencion/raw_data.csv",
                "estimated_time_sec": 30,
            },
        ),
        ModelConfig(
            id="random_forest",
            name="Random Forest",
            category=ModelCategory.ATENCION,
            description="Ensemble de árboles de decisión independientes. Muy robusto al overfitting y fácil de interpretar.",
            vram_mb=0,
            trainable=True,
            tags=["cpu", "robusto"],
            metadata={
                "hyperparams": {
                    "n_estimators":  {"default": 100, "min": 50,  "max": 500, "step": 50,  "tip": "200 árboles dan buen balance. Más de 300 tiene rendimiento decreciente."},
                    "max_depth":     {"default": 10,  "min": 3,   "max": 20,  "step": 1,   "tip": "None permite crecer ilimitado. Para evitar overfitting usa 10–15."},
                    "min_samples_split": {"default": 2, "min": 2, "max": 20, "step": 1,   "tip": "Mínimo de muestras para dividir un nodo. Aumentar a 5 reduce overfitting."},
                    "max_features":  {"default": "sqrt", "options": ["sqrt", "log2", "none"], "tip": "'sqrt' es estándar para clasificación. 'log2' es más conservador."},
                },
                "dataset": "datasets/atencion/raw_data.csv",
                "estimated_time_sec": 20,
            },
        ),
        ModelConfig(
            id="svm",
            name="SVM (Kernel RBF)",
            category=ModelCategory.ATENCION,
            description="Support Vector Machine con kernel radial. Maximiza el margen de separación. Excelente con datasets pequeños y bien normalizados.",
            vram_mb=0,
            trainable=True,
            tags=["cpu", "alta-precision"],
            metadata={
                "hyperparams": {
                    "C":       {"default": 1.0,   "min": 0.01, "max": 100.0, "step": 0.1,  "tip": "Costo de margen. C alto = menos regularización. Prueba C=10 primero."},
                    "gamma":   {"default": "scale","options": ["scale", "auto"],             "tip": "'scale' usa 1/(n_features*X.var()). Generalmente mejor que 'auto'."},
                    "kernel":  {"default": "rbf",  "options": ["rbf", "linear", "poly"],    "tip": "'rbf' es el mejor para separación no lineal. 'linear' si los datos son simples."},
                },
                "dataset": "datasets/atencion/raw_data.csv",
                "estimated_time_sec": 45,
                "note": "Requiere normalización automática (StandardScaler aplicado internamente).",
            },
        ),
        ModelConfig(
            id="logistic_regression",
            name="Regresión Logística",
            category=ModelCategory.ATENCION,
            description="Modelo lineal probabilístico. Rápido, interpretable y útil como baseline. Funciona bien si las clases son linealmente separables.",
            vram_mb=0,
            trainable=True,
            tags=["cpu", "interpretable", "baseline"],
            metadata={
                "hyperparams": {
                    "C":          {"default": 1.0,  "min": 0.001, "max": 100.0, "step": 0.1,  "tip": "Inverso de la regularización. C pequeño = más regularización (útil para evitar overfitting)."},
                    "max_iter":   {"default": 1000, "min": 100,   "max": 5000,  "step": 100,  "tip": "Máximo de iteraciones para convergencia. Aumentar si aparece ConvergenceWarning."},
                    "solver":     {"default": "lbfgs", "options": ["lbfgs", "saga", "liblinear"], "tip": "'lbfgs' es eficiente para multiclase. 'saga' funciona bien con L1."},
                },
                "dataset": "datasets/atencion/raw_data.csv",
                "estimated_time_sec": 10,
            },
        ),
    ],

    # ── AUDIO ─────────────────────────────────────────────────────────────────
    # Faster-Whisper: intercambiable (no entrenable sin corpus de audio)
    # Tradeoff: velocidad (tiny) ↔ precisión (medium)
    # ─────────────────────────────────────────────────────────────────────────
    ModelCategory.AUDIO: [
        ModelConfig(
            id="whisper_tiny",
            name="Whisper Tiny",
            category=ModelCategory.AUDIO,
            description="39M parámetros. Latencia ~200ms. Ideal cuando la velocidad es crítica y el audio es claro.",
            vram_mb=400,
            trainable=False,
            tags=["gpu", "ultra-rapido", "bajo-vram"],
            metadata={"whisper_size": "tiny", "params_m": 39, "wer_es": "~15%"},
        ),
        ModelConfig(
            id="whisper_base",
            name="Whisper Base",
            category=ModelCategory.AUDIO,
            description="74M parámetros. Balance óptimo velocidad/precisión. Modelo actual en producción.",
            vram_mb=800,
            trainable=False,
            tags=["gpu", "balanceado", "actual"],
            metadata={"whisper_size": "base", "params_m": 74, "wer_es": "~10%"},
        ),
        ModelConfig(
            id="whisper_small",
            name="Whisper Small",
            category=ModelCategory.AUDIO,
            description="244M parámetros. Mejor reconocimiento en español con acentos regionales. +0.5s latencia.",
            vram_mb=1200,
            trainable=False,
            tags=["gpu", "preciso"],
            metadata={"whisper_size": "small", "params_m": 244, "wer_es": "~7%"},
        ),
        ModelConfig(
            id="whisper_medium",
            name="Whisper Medium",
            category=ModelCategory.AUDIO,
            description="769M parámetros. Máxima precisión disponible. Requiere ~1.5GB VRAM libre adicional.",
            vram_mb=1500,
            trainable=False,
            tags=["gpu", "alta-precision", "alto-vram"],
            metadata={"whisper_size": "medium", "params_m": 769, "wer_es": "~5%"},
        ),
    ],

    # ── EMOCIÓN ───────────────────────────────────────────────────────────────
    # Dataset FER-2013: 28,709 train + 7,178 test
    # Clases: angry, disgust, fear, happy, neutral, sad, surprise
    # VGG-Face: backend de DeepFace (no requiere reentrenamiento)
    # MobileNetV2 / ResNet18 / EfficientNet-B0: CNNs entrenables con FER-2013
    # ─────────────────────────────────────────────────────────────────────────
    ModelCategory.EMOCION: [
        ModelConfig(
            id="deepface_vggface",
            name="VGG-Face (DeepFace)",
            category=ModelCategory.EMOCION,
            description="CNN clásica pre-entrenada en VGGFace2. Alta precisión facial, modelo actual en producción. No requiere entrenamiento.",
            vram_mb=500,
            trainable=False,
            tags=["gpu", "actual", "deepface"],
            metadata={"backend": "VGG-Face", "input_size": 224},
        ),
        ModelConfig(
            id="mobilenet_v2_fer",
            name="MobileNetV2 (FER-2013)",
            category=ModelCategory.EMOCION,
            description="3.4M params. CNN liviana con depthwise separable convolutions. Mejor velocidad, menor VRAM. Entrenable con FER-2013.",
            vram_mb=300,
            trainable=True,
            tags=["gpu", "ligero", "entrenable"],
            metadata={
                "architecture": "mobilenet_v2",
                "input_size": 96,
                "dataset": "datasets/emotion",
                "classes": ["angry","disgust","fear","happy","neutral","sad","surprise"],
                "hyperparams": {
                    "epochs":         {"default": 15,  "min": 3,    "max": 50,   "step": 1,    "tip": "Con freeze=OFF necesitas 15-20 epochs para converger. Con freeze=ON 8-10 bastan (pero accuracy será menor)."},
                    "learning_rate":  {"default": 5e-5, "min": 1e-6, "max": 1e-3, "step": 1e-6, "tip": "CRÍTICO: con freeze=OFF usa 5e-5 o menos para no destruir los pesos ImageNet. Con freeze=ON puedes usar 1e-4."},
                    "batch_size":     {"default": 64,  "min": 16,   "max": 128,  "step": 16,   "tip": "64 aprovecha mejor la RTX 3060. Baja a 32 si hay error de memoria VRAM."},
                    "img_size":       {"default": 96,  "min": 64,   "max": 224,  "step": 32,   "tip": "96px da buen balance velocidad/precisión. 64px = más rápido pero pierde detalles faciales."},
                    "patience":       {"default": 5,   "min": 0,    "max": 10,   "step": 1,    "tip": "Early stopping: para si la val_acc no mejora en N epochs. 0 = desactivado. El mejor epoch siempre se guarda."},
                    "freeze_layers":  {"default": False, "tip": "RECOMENDADO: OFF. Con capas descongeladas el backbone se adapta a caras/emociones → +15-20% accuracy. Con ON solo el clasificador final aprende (accuracy ~30-40%)."},
                },
                "estimated_time_min": 8,
            },
        ),
        ModelConfig(
            id="resnet18_fer",
            name="ResNet-18 (FER-2013)",
            category=ModelCategory.EMOCION,
            description="11M params. Arquitectura residual. Entrenamiento estable, buena precisión. El estándar de facto para clasificación de imágenes.",
            vram_mb=600,
            trainable=True,
            tags=["gpu", "estable", "entrenable"],
            metadata={
                "architecture": "resnet18",
                "input_size": 96,
                "dataset": "datasets/emotion",
                "classes": ["angry","disgust","fear","happy","neutral","sad","surprise"],
                "hyperparams": {
                    "epochs":         {"default": 15,   "min": 5,    "max": 50,   "step": 1,    "tip": "15-20 epochs con freeze=OFF. ResNet converge más rápido que MobileNet gracias a las conexiones residuales."},
                    "learning_rate":  {"default": 5e-5,  "min": 1e-6, "max": 1e-3, "step": 1e-6, "tip": "CRÍTICO: usa 5e-5 con freeze=OFF. Solo layer4+fc descongeladas por defecto. Valores >1e-4 pueden deteriorar las capas residuales."},
                    "batch_size":     {"default": 64,   "min": 16,   "max": 128,  "step": 16,   "tip": "64 con AMP activo es seguro en RTX 3060. ResNet-18 usa menos VRAM que MobileNetV2 con el mismo batch."},
                    "img_size":       {"default": 96,  "min": 64,   "max": 224,  "step": 32,   "tip": "96px para FER-2013 da buen equilibrio. ResNet puede beneficiarse más de 112-128px que MobileNet."},
                    "patience":       {"default": 5,   "min": 0,    "max": 10,   "step": 1,    "tip": "Early stopping. El mejor epoch se guarda automáticamente aunque el entrenamiento continúe."},
                    "freeze_layers":  {"default": False, "tip": "RECOMENDADO: OFF. Con ResNet-18 solo layer4+fc se descongelan por default, dando fine-tuning parcial seguro con LR bajo."},
                },
                "estimated_time_min": 10,
            },
        ),
        ModelConfig(
            id="resnet50_fer",
            name="ResNet-50 (FER-2013)",
            category=ModelCategory.EMOCION,
            description="25M params. Arquitectura residual profunda. Mayor capacidad que ResNet-18, mejor para datasets con alta variabilidad intra-clase como FER-2013.",
            vram_mb=900,
            trainable=True,
            tags=["gpu", "profundo", "entrenable"],
            metadata={
                "architecture": "resnet50",
                "input_size": 112,
                "dataset": "datasets/emotion",
                "classes": ["angry","disgust","fear","happy","neutral","sad","surprise"],
                "hyperparams": {
                    "epochs":         {"default": 20,   "min": 5,    "max": 50,   "step": 1,    "tip": "ResNet-50 converge más lento que ResNet-18. Usa 20-25 epochs con freeze=OFF para resultados óptimos."},
                    "learning_rate":  {"default": 5e-5,  "min": 1e-6, "max": 1e-3, "step": 1e-6, "tip": "CRÍTICO: 5e-5 con freeze=OFF. ResNet-50 con LR alto destruye los pesos residuales ImageNet."},
                    "batch_size":     {"default": 32,   "min": 8,    "max": 64,   "step": 8,    "tip": "32 es seguro en RTX 3060 (25M params). Puedes subir a 64 con AMP activo."},
                    "img_size":       {"default": 112,  "min": 64,   "max": 224,  "step": 32,   "tip": "112px da buen balance. ResNet-50 se beneficia más de resoluciones altas que las versiones pequeñas."},
                    "patience":       {"default": 6,   "min": 0,    "max": 15,   "step": 1,    "tip": "Early stopping. Con 20 epochs usa patience=6 para detectar estancamiento sin cortar demasiado pronto."},
                    "freeze_layers":  {"default": False, "tip": "RECOMENDADO: OFF. ResNet-50 tiene más capas especializadas que adaptar a reconocimiento facial."},
                },
                "estimated_time_min": 15,
            },
        ),
        ModelConfig(
            id="efficientnet_b0_fer",
            name="EfficientNet-B0 (FER-2013)",
            category=ModelCategory.EMOCION,
            description="5.3M params. Mejor ratio precisión/parámetros de la familia EfficientNet. Escala eficientemente con más datos.",
            vram_mb=500,
            trainable=True,
            tags=["gpu", "sota", "entrenable"],
            metadata={
                "architecture": "efficientnet_b0",
                "input_size": 96,
                "dataset": "datasets/emotion",
                "classes": ["angry","disgust","fear","happy","neutral","sad","surprise"],
                "hyperparams": {
                    "epochs":         {"default": 15,   "min": 5,    "max": 50,   "step": 1,    "tip": "15 epochs con freeze=OFF. EfficientNet-B0 es el más preciso de los 3 pero requiere más epochs para estabilizarse."},
                    "learning_rate":  {"default": 5e-5,  "min": 1e-6, "max": 1e-3, "step": 1e-6, "tip": "CRÍTICO: 5e-5 con freeze=OFF. EfficientNet es sensible al LR. Valores >2e-4 causan divergencia."},
                    "batch_size":     {"default": 64,   "min": 16,   "max": 128,  "step": 16,   "tip": "64 con AMP. EfficientNet-B0 es el más eficiente en VRAM de los 3 modelos."},
                    "img_size":       {"default": 96,  "min": 64,   "max": 224,  "step": 32,   "tip": "EfficientNet-B0 fue diseñado para 224px. Con 96px es más rápido; con 128px da mejor balance precisión/velocidad."},
                    "patience":       {"default": 5,   "min": 0,    "max": 10,   "step": 1,    "tip": "Early stopping guarda siempre el mejor epoch. Útil si entrenas 20+ epochs para evitar overfitting."},
                    "freeze_layers":  {"default": False, "tip": "RECOMENDADO: OFF. EfficientNet con freeze=ON tiene accuracy de ~30-40%. Con freeze=OFF y LR bajo alcanza 55-65%."},
                },
                "estimated_time_min": 10,
            },
        ),
    ],

    # ── SEMÁNTICA ─────────────────────────────────────────────────────────────
    # Análisis de sentimiento: intercambiables (no entrenables sin corpus de texto)
    # ─────────────────────────────────────────────────────────────────────────
    ModelCategory.SEMANTICA: [
        ModelConfig(
            id="robertuito",
            name="RoBERTuito (pysentimiento)",
            category=ModelCategory.SEMANTICA,
            description="RoBERTa fine-tuneado en 60M tweets en español. Excelente en lenguaje coloquial y expresiones orales. Modelo actual.",
            vram_mb=700,
            trainable=False,
            tags=["gpu", "español", "actual", "coloquial"],
            metadata={"hf_model": "pysentimiento/robertuito-sentiment-analysis", "lang": "es"},
        ),
        ModelConfig(
            id="beto",
            name="BETO (BERT Español)",
            category=ModelCategory.SEMANTICA,
            description="BERT pre-entrenado en Wikipedia + Opus español. Robusto en texto formal, académico y técnico.",
            vram_mb=700,
            trainable=False,
            tags=["gpu", "español", "formal"],
            metadata={"hf_model": "dccuchile/bert-base-spanish-wwm-cased", "lang": "es"},
        ),
        ModelConfig(
            id="multilingual_bert",
            name="mBERT Multilingüe",
            category=ModelCategory.SEMANTICA,
            description="BERT entrenado en 104 idiomas. Útil si los estudiantes mezclan español e inglés (code-switching).",
            vram_mb=700,
            trainable=False,
            tags=["gpu", "multilingüe"],
            metadata={"hf_model": "nlptown/bert-base-multilingual-uncased-sentiment", "lang": "multilingual"},
        ),
        ModelConfig(
            id="xlm_roberta_sentiment",
            name="XLM-RoBERTa Sentiment",
            category=ModelCategory.SEMANTICA,
            description="XLM-RoBERTa fine-tuneado en Twitter multilingüe. Excelente en texto corto y expresiones informales.",
            vram_mb=800,
            trainable=False,
            tags=["gpu", "multilingüe", "twitter"],
            metadata={"hf_model": "cardiffnlp/twitter-xlm-roberta-base-sentiment", "lang": "multilingual"},
        ),
    ],
}


# ─────────────────────────────────────────────
#  MODEL REGISTRY — SINGLETON
# ─────────────────────────────────────────────

class ModelRegistry:
    """
    Singleton que gestiona:
    - El catálogo de modelos disponibles
    - El modelo activo por categoría
    - La carga/descarga con limpieza agresiva de VRAM
    """

    _instance: Optional["ModelRegistry"] = None

    def __new__(cls) -> "ModelRegistry":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return
        self._initialized = True

        # Config activa por categoría (por defecto: primer modelo del catálogo)
        self.active_configs: Dict[ModelCategory, ModelConfig] = {
            ModelCategory.ATENCION:  MODEL_CATALOG[ModelCategory.ATENCION][0],   # XGBoost
            ModelCategory.AUDIO:     MODEL_CATALOG[ModelCategory.AUDIO][1],      # Whisper base
            ModelCategory.EMOCION:   MODEL_CATALOG[ModelCategory.EMOCION][0],    # VGG-Face
            ModelCategory.SEMANTICA: MODEL_CATALOG[ModelCategory.SEMANTICA][0],  # RoBERTuito
        }

        # Instancias cargadas (pobladas por inference_engine.py)
        self.loaded_instances: Dict[ModelCategory, Any] = {}

        logger.info("ModelRegistry inicializado.")

    # ── VRAM ──────────────────────────────────────────────────────────────────

    def get_vram_info(self) -> Dict[str, int]:
        """Retorna VRAM libre, usada y total en MB."""
        try:
            import torch
            if not torch.cuda.is_available():
                return {"free_mb": 0, "used_mb": 0, "total_mb": 0}
            free, total = torch.cuda.mem_get_info()
            used = total - free
            return {
                "free_mb":  free  // (1024 * 1024),
                "used_mb":  used  // (1024 * 1024),
                "total_mb": total // (1024 * 1024),
            }
        except Exception:
            return {"free_mb": 0, "used_mb": 0, "total_mb": 0}

    def _release_vram(self) -> None:
        """Limpieza agresiva de VRAM: del + gc + empty_cache."""
        try:
            import torch
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        except Exception:
            pass

    # ── CARGA / DESCARGA ──────────────────────────────────────────────────────

    def register_instance(self, category: ModelCategory, instance: Any) -> None:
        """Registra una instancia cargada por inference_engine."""
        self.loaded_instances[category] = instance
        logger.info(f"[Registry] Instancia registrada: {category.value}")

    def unload_instance(self, category: ModelCategory) -> None:
        """Descarga la instancia activa y libera memoria."""
        if category in self.loaded_instances:
            instance = self.loaded_instances.pop(category)
            del instance
            self._release_vram()
            logger.info(f"[Registry] Instancia descargada: {category.value}")

    # ── ACTIVACIÓN DE MODELOS ─────────────────────────────────────────────────

    def set_active_model(
        self,
        category: ModelCategory,
        model_id: str,
        trained_model_path: Optional[str] = None,
    ) -> ModelConfig:
        """
        Cambia el modelo activo de una categoría.
        - Si el nuevo modelo usa GPU, verifica VRAM disponible.
        - Descarga la instancia anterior antes de cambiar.
        - trained_model_path: ruta a un .joblib/.pt entrenado (opcional).
        """
        # Buscar en catálogo
        catalog = MODEL_CATALOG.get(category, [])
        config = next((m for m in catalog if m.id == model_id), None)
        if config is None:
            raise ValueError(f"Modelo '{model_id}' no encontrado en categoría '{category.value}'.")

        # Verificar VRAM si el nuevo modelo usa GPU
        current = self.active_configs[category]
        if config.vram_mb > 0:
            vram = self.get_vram_info()
            # VRAM que se liberará al descargar el actual + la libre disponible
            effective_free = vram["free_mb"] + current.vram_mb
            if effective_free < config.vram_mb:
                raise RuntimeError(
                    f"VRAM insuficiente. Necesitas {config.vram_mb} MB, "
                    f"disponibles efectivos: {effective_free} MB. "
                    f"Considera descargar otros modelos primero."
                )

        # Descargar instancia actual
        self.unload_instance(category)

        # Actualizar config activa
        self.active_configs[category] = config
        if trained_model_path:
            config.metadata["trained_model_path"] = trained_model_path

        logger.info(f"[Registry] Modelo activo [{category.value}]: {model_id}")
        return config

    # ── CONSULTAS ─────────────────────────────────────────────────────────────

    def get_catalog(self) -> Dict[str, List[Dict]]:
        """Catálogo completo con flag is_active por modelo."""
        result = {}
        for cat, models in MODEL_CATALOG.items():
            active_id = self.active_configs[cat].id
            result[cat.value] = [
                {
                    "id":          m.id,
                    "name":        m.name,
                    "description": m.description,
                    "vram_mb":     m.vram_mb,
                    "trainable":   m.trainable,
                    "tags":        m.tags,
                    "metadata":    m.metadata,
                    "is_active":   m.id == active_id,
                    "is_loaded":   cat in self.loaded_instances,
                }
                for m in models
            ]
        return result

    def get_status(self) -> Dict:
        """Estado global del registry (VRAM + modelos activos)."""
        try:
            import torch
            gpu_available = torch.cuda.is_available()
            gpu_name = torch.cuda.get_device_name(0) if gpu_available else "N/A"
        except Exception:
            gpu_available = False
            gpu_name = "N/A"

        return {
            "gpu": {
                "available": gpu_available,
                "name": gpu_name,
                "vram": self.get_vram_info(),
            },
            "active_models": {
                cat.value: {
                    "id":        config.id,
                    "name":      config.name,
                    "vram_mb":   config.vram_mb,
                    "trainable": config.trainable,
                    "is_loaded": cat in self.loaded_instances,
                }
                for cat, config in self.active_configs.items()
            },
        }

    def get_hyperparams_guide(self, category: ModelCategory, model_id: str) -> Dict:
        """
        Devuelve los hiperparámetros ajustables y su guía de recomendaciones
        para el Training Lab del admin panel.
        """
        catalog = MODEL_CATALOG.get(category, [])
        config = next((m for m in catalog if m.id == model_id), None)
        if config is None:
            raise ValueError(f"Modelo '{model_id}' no encontrado.")
        if not config.trainable:
            raise ValueError(f"El modelo '{config.name}' no es entrenable.")

        hyperparams = config.metadata.get("hyperparams", {})
        return {
            "model_id":   config.id,
            "model_name": config.name,
            "category":   category.value,
            "dataset":    config.metadata.get("dataset", "N/A"),
            "estimated_time": config.metadata.get("estimated_time_sec") or config.metadata.get("estimated_time_min"),
            "time_unit":  "segundos" if "estimated_time_sec" in config.metadata else "minutos",
            "hyperparams": hyperparams,
            "note":       config.metadata.get("note", ""),
        }


# ── Instancia global del Registry ─────────────────────────────────────────────
registry = ModelRegistry()
