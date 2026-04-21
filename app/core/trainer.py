"""
trainer.py — Edu-Insight MLOps Fase 2
Pipeline de entrenamiento asíncrono con emisión de progreso por queue thread-safe.

Modelos soportados:
  Atención  : XGBoost | RandomForest | SVM | LogisticRegression (CPU, sklearn)
  Emoción   : MobileNetV2 | ResNet18 | EfficientNet-B0 (GPU, torchvision / FER-2013)

Flujo:
  POST /api/train/start
      → TrainingManager.start_training()
          → Thread(target=_run_training)
              → _emit() → progress_queue (thread-safe)
  WS /ws/training
      → polling progress_queue → send_json al cliente
"""

import gc
import json
import logging
import queue
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
#  RUTAS
# ─────────────────────────────────────────────
STORAGE_DIR      = Path("app/storage/trained_models")
METRICS_DIR      = Path("app/storage/metrics")
DATASET_ATENCION = Path("datasets/atencion/raw_data.csv")
DATASET_EMOCION  = Path("datasets/emotion")

STORAGE_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────
#  DATACLASS DEL JOB
# ─────────────────────────────────────────────

@dataclass
class TrainingJob:
    job_id: str
    category: str          # "atencion" | "emocion"
    model_id: str
    hyperparams: Dict[str, Any]
    status: str = "pending"    # pending | running | completed | failed
    progress: int = 0          # 0–100
    current_epoch: int = 0
    total_epochs: int = 0
    message: str = ""
    metrics: Optional[Dict]    = None
    model_path: Optional[str]  = None
    metrics_path: Optional[str] = None
    error: Optional[str]       = None
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    completed_at: Optional[str] = None


# ─────────────────────────────────────────────
#  TRAINING MANAGER — SINGLETON
# ─────────────────────────────────────────────

class TrainingManager:
    """
    Gestiona un único trabajo de entrenamiento a la vez.
    El entrenamiento corre en un Thread daemon separado.
    El progreso se comunica via queue.Queue (thread-safe).
    """

    _instance: Optional["TrainingManager"] = None

    def __new__(cls) -> "TrainingManager":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return
        self._initialized = True
        self.current_job: Optional[TrainingJob] = None
        self.progress_queue: queue.Queue = queue.Queue(maxsize=500)
        self.all_jobs: Dict[str, TrainingJob] = {}
        self._lock = threading.Lock()

    # ── Estado ────────────────────────────────────────────────────────────────

    @property
    def is_busy(self) -> bool:
        return self.current_job is not None and self.current_job.status == "running"

    # ── Inicio de entrenamiento ───────────────────────────────────────────────

    def start_training(
        self,
        category: str,
        model_id: str,
        hyperparams: Dict[str, Any],
    ) -> TrainingJob:
        if self.is_busy:
            raise RuntimeError(
                "Ya hay un entrenamiento en curso. "
                f"Espera a que termine el job '{self.current_job.job_id}'."
            )

        job = TrainingJob(
            job_id=str(uuid.uuid4())[:8],
            category=category,
            model_id=model_id,
            hyperparams=hyperparams,
        )

        with self._lock:
            self.current_job = job
            self.all_jobs[job.job_id] = job

        # Vaciar queue de jobs anteriores
        self._flush_queue()

        thread = threading.Thread(
            target=self._run_training,
            args=(job,),
            daemon=True,
            name=f"trainer-{job.job_id}",
        )
        thread.start()
        return job

    def _flush_queue(self) -> None:
        while not self.progress_queue.empty():
            try:
                self.progress_queue.get_nowait()
            except queue.Empty:
                break

    # ── Emisión de eventos ────────────────────────────────────────────────────

    def _emit(self, job: TrainingJob, event_type: str, **extra) -> None:
        event = {
            "type":          event_type,
            "job_id":        job.job_id,
            "progress":      job.progress,
            "current_epoch": job.current_epoch,
            "total_epochs":  job.total_epochs,
            "message":       job.message,
            "status":        job.status,
            **extra,
        }
        try:
            self.progress_queue.put_nowait(event)
        except queue.Full:
            # Si el cliente no está leyendo, descarta el evento más viejo
            try:
                self.progress_queue.get_nowait()
                self.progress_queue.put_nowait(event)
            except Exception:
                pass

    # ── Dispatcher ────────────────────────────────────────────────────────────

    def _run_training(self, job: TrainingJob) -> None:
        try:
            job.status  = "running"
            job.message = "Iniciando entrenamiento..."
            self._emit(job, "start")

            if job.category == "atencion":
                self._train_atencion(job)
            elif job.category == "emocion":
                self._train_emocion(job)
            else:
                raise ValueError(f"Categoría no entrenable: '{job.category}'")

            job.status       = "completed"
            job.completed_at = datetime.now().isoformat()
            self._emit(job, "complete", metrics=job.metrics, model_path=job.model_path)

        except Exception as exc:
            job.status  = "failed"
            job.error   = str(exc)
            job.message = f"Error: {exc}"
            logger.exception(f"[Trainer] Job {job.job_id} falló")
            self._emit(job, "error", error=str(exc))

        finally:
            with self._lock:
                self.current_job = None

    # ──────────────────────────────────────────────────────────────────────────
    #  ATENCION — Modelos sklearn (CPU)
    #  Dataset: datasets/atencion/raw_data.csv
    #  Features: ear, pitch, yaw, ratio_h
    #  Clases: Enfocado | Distraído | Somnoliento
    # ──────────────────────────────────────────────────────────────────────────

    def _train_atencion(self, job: TrainingJob) -> None:
        hp = job.hyperparams

        # ── Carga y preprocesamiento ──────────────────────────────────
        job.message  = "Cargando dataset de atención..."
        job.progress = 5
        self._emit(job, "progress")

        df = pd.read_csv(DATASET_ATENCION)
        feature_cols = ["ear", "pitch", "yaw", "ratio_h"]
        label_col    = "label"

        df = df.dropna(subset=feature_cols + [label_col])
        # Limpieza de outliers IQR (mismo criterio que train_model.py original)
        for col in feature_cols:
            q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
            iqr    = q3 - q1
            df     = df[(df[col] >= q1 - 2.5 * iqr) & (df[col] <= q3 + 2.5 * iqr)]

        X      = df[feature_cols].values
        y_raw  = df[label_col].values
        le     = LabelEncoder()
        y      = le.fit_transform(y_raw)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, stratify=y, random_state=42
        )

        job.message  = f"Dataset listo: {len(X_train)} train / {len(X_test)} test · clases: {le.classes_.tolist()}"
        job.progress = 15
        self._emit(job, "progress")

        # ── Normalización (SVM y LogReg) ──────────────────────────────
        needs_scaling = job.model_id in ("svm", "logistic_regression")
        scaler = StandardScaler()
        if needs_scaling:
            X_train = scaler.fit_transform(X_train)
            X_test  = scaler.transform(X_test)
            job.message  = "Datos normalizados con StandardScaler."
            job.progress = 20
            self._emit(job, "progress")

        # ── Construcción del modelo ───────────────────────────────────
        model = self._build_sklearn_model(job.model_id, hp)

        # ── Entrenamiento con progreso ────────────────────────────────
        if job.model_id == "xgboost":
            self._fit_xgboost(job, model, X_train, y_train, hp)
        elif job.model_id == "random_forest":
            self._fit_random_forest(job, model, X_train, y_train, hp)
        else:
            # SVM y LogReg: entrenamiento único, progreso por etapas
            job.total_epochs = 1
            job.current_epoch = 0
            job.progress = 30
            job.message  = f"Ajustando {job.model_id}..."
            self._emit(job, "progress")
            model.fit(X_train, y_train)
            job.current_epoch = 1
            job.progress = 75
            job.message  = "Modelo ajustado. Evaluando..."
            self._emit(job, "progress")

        # ── Evaluación ────────────────────────────────────────────────
        job.message  = "Calculando métricas..."
        job.progress = 85
        self._emit(job, "progress")

        y_pred      = model.predict(X_test)
        job.metrics = self._compute_metrics(y_test, y_pred, le.classes_.tolist())
        job.metrics["n_train"] = int(len(X_train))
        job.metrics["n_test"]  = int(len(X_test))

        # ── Guardado ──────────────────────────────────────────────────
        job.message  = "Guardando modelo y métricas..."
        job.progress = 92
        self._emit(job, "progress")

        ts             = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path     = STORAGE_DIR / f"{job.model_id}_{ts}.joblib"
        metrics_path   = METRICS_DIR  / f"{job.model_id}_{ts}.json"

        joblib.dump(
            {
                "model":         model,
                "scaler":        scaler if needs_scaling else None,
                "label_encoder": le,
                "features":      feature_cols,
                "hyperparams":   hp,
                "job_id":        job.job_id,
                "trained_at":    ts,
            },
            model_path,
        )

        job.metrics.update(
            {
                "job_id":      job.job_id,
                "model_id":    job.model_id,
                "model_name":  job.model_id,
                "category":    job.category,
                "hyperparams": hp,
                "trained_at":  ts,
                "model_path":  str(model_path),
            }
        )
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(job.metrics, f, ensure_ascii=False, indent=2)

        job.model_path   = str(model_path)
        job.metrics_path = str(metrics_path)
        job.progress     = 100
        job.message      = f"✅ Completado — Accuracy: {job.metrics['accuracy']:.1%} | F1: {job.metrics['f1_macro']:.1%}"
        self._emit(job, "progress")

    # ── Helpers sklearn ───────────────────────────────────────────────────────

    def _build_sklearn_model(self, model_id: str, hp: Dict) -> Any:
        if model_id == "xgboost":
            from xgboost import XGBClassifier
            return XGBClassifier(
                n_estimators  = int(hp.get("n_estimators", 100)),
                max_depth     = int(hp.get("max_depth", 5)),
                learning_rate = float(hp.get("learning_rate", 0.1)),
                subsample     = float(hp.get("subsample", 0.8)),
                eval_metric   = "mlogloss",
                verbosity     = 0,
                random_state  = 42,
            )
        if model_id == "random_forest":
            from sklearn.ensemble import RandomForestClassifier
            mf = hp.get("max_features", "sqrt")
            return RandomForestClassifier(
                n_estimators    = int(hp.get("n_estimators", 100)),
                max_depth       = int(hp.get("max_depth", 10)) or None,
                min_samples_split = int(hp.get("min_samples_split", 2)),
                max_features    = None if mf == "none" else mf,
                random_state    = 42,
                n_jobs          = -1,
            )
        if model_id == "svm":
            from sklearn.svm import SVC
            return SVC(
                C           = float(hp.get("C", 1.0)),
                kernel      = hp.get("kernel", "rbf"),
                gamma       = hp.get("gamma", "scale"),
                probability = True,
                random_state = 42,
            )
        if model_id == "logistic_regression":
            from sklearn.linear_model import LogisticRegression
            return LogisticRegression(
                C          = float(hp.get("C", 1.0)),
                max_iter   = int(hp.get("max_iter", 1000)),
                solver     = hp.get("solver", "lbfgs"),
                multi_class = "multinomial",
                random_state = 42,
                n_jobs     = -1,
            )
        raise ValueError(f"Modelo sklearn desconocido: {model_id}")

    def _fit_xgboost(self, job, model, X_train, y_train, hp) -> None:
        n_total = int(hp.get("n_estimators", 100))
        chunk   = max(5, n_total // 20)
        job.total_epochs = n_total

        trained = 0
        booster = None
        while trained < n_total:
            batch   = min(chunk, n_total - trained)
            trained += batch
            model.set_params(n_estimators=trained)
            model.fit(X_train, y_train, xgb_model=booster, verbose=False)
            booster = model.get_booster()

            job.current_epoch = trained
            job.progress      = 20 + int((trained / n_total) * 62)
            job.message       = f"XGBoost: árbol {trained}/{n_total}"
            self._emit(job, "progress")

    def _fit_random_forest(self, job, model, X_train, y_train, hp) -> None:
        n_total = int(hp.get("n_estimators", 100))
        chunk   = max(5, n_total // 20)
        job.total_epochs = n_total

        model.set_params(warm_start=True)
        trained = 0
        while trained < n_total:
            batch   = min(chunk, n_total - trained)
            trained += batch
            model.set_params(n_estimators=trained)
            model.fit(X_train, y_train)

            job.current_epoch = trained
            job.progress      = 20 + int((trained / n_total) * 62)
            job.message       = f"RandomForest: árbol {trained}/{n_total}"
            self._emit(job, "progress")

    # ──────────────────────────────────────────────────────────────────────────
    #  EMOCION — CNNs torchvision con transfer learning (GPU)
    #  Dataset: datasets/emotion/train/ y datasets/emotion/test/
    #  7 clases: angry, disgust, fear, happy, neutral, sad, surprise
    # ──────────────────────────────────────────────────────────────────────────

    def _train_emocion(self, job: TrainingJob) -> None:
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from torch.utils.data import DataLoader
        from torchvision import datasets, transforms

        hp          = job.hyperparams
        num_epochs  = int(hp.get("epochs", 10))
        lr          = float(hp.get("learning_rate", 1e-4))
        batch_size  = int(hp.get("batch_size", 64))   # 64 > 32: mejor utilización GPU
        freeze      = bool(hp.get("freeze_layers", True))
        img_size    = int(hp.get("img_size", 96))      # 96 en vez de 224: ~5× más rápido

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        job.total_epochs = num_epochs

        # cudnn benchmark: elige el algoritmo de convolución más rápido para el tamaño fijo
        if device.type == "cuda":
            torch.backends.cudnn.benchmark = True

        # ── Transforms ────────────────────────────────────────────────
        job.message  = "Preparando dataset FER-2013 (7 emociones)..."
        job.progress = 5
        self._emit(job, "progress")

        # FER-2013: imágenes 48×48 grayscale → RGB img_size×img_size
        # Grayscale primero (antes de Resize) para operar en 1 canal y evitar trabajo extra
        train_tf = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]),
        ])
        val_tf = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]),
        ])

        train_ds = datasets.ImageFolder(str(DATASET_EMOCION / "train"), transform=train_tf)
        val_ds   = datasets.ImageFolder(str(DATASET_EMOCION / "test"),  transform=val_tf)

        num_classes = len(train_ds.classes)
        classes     = train_ds.classes

        # ── WeightedRandomSampler: corrige desbalance de clases (disgust ~547 vs happy ~8989) ──
        # Cada imagen tiene probabilidad proporcional a 1/frecuencia_de_su_clase
        targets     = [s[1] for s in train_ds.samples]
        class_count = [targets.count(i) for i in range(num_classes)]
        weights     = [1.0 / class_count[t] for t in targets]
        sampler     = torch.utils.data.WeightedRandomSampler(
            weights, num_samples=len(weights), replacement=True
        )

        # Windows + Thread: num_workers>0 falla con spawn desde hilos no-principales.
        num_workers = 0

        train_loader = DataLoader(
            train_ds, batch_size=batch_size,
            sampler=sampler,               # reemplaza shuffle=True
            num_workers=num_workers,
            pin_memory=(device.type == "cuda"),
        )
        val_loader = DataLoader(
            val_ds, batch_size=batch_size * 2,
            shuffle=False, num_workers=num_workers,
            pin_memory=(device.type == "cuda"),
        )

        job.message  = (
            f"Dataset: {len(train_ds):,} train / {len(val_ds):,} val | "
            f"{num_classes} emociones | input {img_size}×{img_size}"
        )
        job.progress = 10
        self._emit(job, "progress")

        # ── Modelo ────────────────────────────────────────────────────
        model_nn = self._build_cnn(job.model_id, num_classes, freeze).to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, model_nn.parameters()),
            lr=lr,
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

        # Automatic Mixed Precision: usa FP16 en los Tensor Cores → 1.5-2× más rápido
        use_amp = device.type == "cuda"
        scaler  = torch.cuda.amp.GradScaler(enabled=use_amp)

        job.message  = (
            f"Modelo {job.model_id} en {device} | AMP={'ON' if use_amp else 'OFF'} | "
            f"batch={batch_size} | img={img_size}px | WRS=ON"
        )
        job.progress = 12
        self._emit(job, "progress")

        # ── Loop de entrenamiento ─────────────────────────────────────
        history: Dict[str, List] = {"train_loss": [], "val_loss": [], "val_acc": []}
        last_preds: List[int]  = []
        last_labels: List[int] = []

        # Early stopping + checkpoint del mejor epoch
        best_val_acc    = 0.0
        best_state_dict = None
        patience        = int(hp.get("patience", 5))
        epochs_no_improve = 0

        for epoch in range(1, num_epochs + 1):
            # — Train con AMP —
            model_nn.train()
            running_loss = 0.0
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad(set_to_none=True)   # más rápido que zero_grad()
                with torch.autocast(device_type=device.type, enabled=use_amp):
                    loss = criterion(model_nn(inputs), labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                running_loss += loss.item()

            scheduler.step()

            # — Validation —
            model_nn.eval()
            val_loss = 0.0
            correct  = 0
            total    = 0
            ep_preds: List[int]  = []
            ep_labels: List[int] = []

            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    with torch.autocast(device_type=device.type, enabled=use_amp):
                        outputs = model_nn(inputs)
                        val_loss += criterion(outputs, labels).item()
                    _, predicted = torch.max(outputs, 1)
                    total   += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    ep_preds.extend(predicted.cpu().numpy().tolist())
                    ep_labels.extend(labels.cpu().numpy().tolist())

            train_loss_avg = running_loss / len(train_loader)
            val_loss_avg   = val_loss   / len(val_loader)
            val_acc        = correct / total

            history["train_loss"].append(round(train_loss_avg, 4))
            history["val_loss"].append(round(val_loss_avg, 4))
            history["val_acc"].append(round(val_acc, 4))
            last_preds  = ep_preds
            last_labels = ep_labels

            # Checkpoint: guardar el mejor estado en memoria
            if val_acc > best_val_acc:
                best_val_acc    = val_acc
                best_state_dict = {k: v.cpu().clone() for k, v in model_nn.state_dict().items()}
                epochs_no_improve = 0
                best_marker = " ★ mejor"
            else:
                epochs_no_improve += 1
                best_marker = f" (sin mejora {epochs_no_improve}/{patience})"

            job.current_epoch = epoch
            job.progress      = 12 + int((epoch / num_epochs) * 75)
            job.message       = (
                f"Epoch {epoch}/{num_epochs} — "
                f"loss: {train_loss_avg:.4f} — "
                f"val_acc: {val_acc:.2%}{best_marker}"
            )
            self._emit(
                job, "progress",
                train_loss=round(train_loss_avg, 4),
                val_loss=round(val_loss_avg, 4),
                val_acc=round(val_acc, 4),
            )

            # Early stopping
            if patience > 0 and epochs_no_improve >= patience:
                job.message = f"Early stopping en epoch {epoch}: sin mejora en {patience} epochs consecutivos."
                self._emit(job, "progress")
                break

        # ── Restaurar el mejor estado antes de guardar ────────────────
        if best_state_dict is not None:
            model_nn.load_state_dict({k: v.to(device) for k, v in best_state_dict.items()})

        # ── Métricas finales ──────────────────────────────────────────
        job.message  = f"Calculando métricas finales (mejor val_acc: {best_val_acc:.2%})..."
        job.progress = 90
        self._emit(job, "progress")

        metrics          = self._compute_metrics(np.array(last_labels), np.array(last_preds), classes)
        metrics["history"] = history
        metrics["n_train"] = len(train_ds)
        metrics["n_test"]  = len(val_ds)
        job.metrics      = metrics

        # ── Guardado ──────────────────────────────────────────────────
        job.message  = "Guardando modelo (.pt) y métricas (.json)..."
        job.progress = 95
        self._emit(job, "progress")

        ts           = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path   = STORAGE_DIR / f"{job.model_id}_{ts}.pt"
        metrics_path = METRICS_DIR  / f"{job.model_id}_{ts}.json"

        import torch
        torch.save(
            {
                "model_state_dict": model_nn.state_dict(),
                "model_id":         job.model_id,
                "num_classes":      num_classes,
                "classes":          classes,
                "hyperparams":      hp,
                "job_id":           job.job_id,
                "trained_at":       ts,
            },
            model_path,
        )

        metrics.update(
            {
                "job_id":      job.job_id,
                "model_id":    job.model_id,
                "model_name":  job.model_id,
                "category":    job.category,
                "hyperparams": hp,
                "trained_at":  ts,
                "model_path":  str(model_path),
            }
        )
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)

        job.model_path   = str(model_path)
        job.metrics_path = str(metrics_path)
        job.progress     = 100
        job.message      = (
            f"✅ Completado — Accuracy: {metrics['accuracy']:.1%} | F1: {metrics['f1_macro']:.1%}"
        )
        self._emit(job, "progress")

        # Liberar VRAM inmediatamente
        del model_nn
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ── Helper: construcción de CNNs ──────────────────────────────────────────

    def _build_cnn(self, model_id: str, num_classes: int, freeze: bool):
        import torch.nn as nn
        from torchvision import models

        if model_id == "mobilenet_v2_fer":
            m = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
            if freeze:
                for p in m.features.parameters():
                    p.requires_grad = False
            m.classifier[1] = nn.Linear(m.last_channel, num_classes)
            return m

        if model_id == "resnet18_fer":
            m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            if freeze:
                for name, p in m.named_parameters():
                    if "layer4" not in name and "fc" not in name:
                        p.requires_grad = False
            m.fc = nn.Linear(m.fc.in_features, num_classes)
            return m

        if model_id == "resnet50_fer":
            m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
            if freeze:
                for name, p in m.named_parameters():
                    if "layer4" not in name and "fc" not in name:
                        p.requires_grad = False
            m.fc = nn.Linear(m.fc.in_features, num_classes)
            return m

        if model_id == "efficientnet_b0_fer":
            m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
            if freeze:
                for p in m.features.parameters():
                    p.requires_grad = False
            in_feat = m.classifier[1].in_features
            m.classifier[1] = nn.Linear(in_feat, num_classes)
            return m

        if model_id == "efficientnet_b3_fer":
            m = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.IMAGENET1K_V1)
            if freeze:
                for p in m.features.parameters():
                    p.requires_grad = False
            in_feat = m.classifier[1].in_features
            m.classifier[1] = nn.Linear(in_feat, num_classes)
            return m

        raise ValueError(f"CNN desconocida: {model_id}")

    # ── Helper: métricas científicas ──────────────────────────────────────────

    def _compute_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        class_names: List[str],
    ) -> Dict:
        """
        Calcula: Accuracy, F1-macro, Recall-macro, Precision-macro,
        Matriz de Confusión, y VP/VN/FP/FN por clase.
        """
        acc      = accuracy_score(y_true, y_pred)
        f1       = f1_score(y_true, y_pred, average="macro",  zero_division=0)
        recall   = recall_score(y_true, y_pred, average="macro",  zero_division=0)
        prec     = precision_score(y_true, y_pred, average="macro", zero_division=0)
        cm       = confusion_matrix(y_true, y_pred)

        f1_pc   = f1_score(y_true, y_pred, average=None, zero_division=0)
        rec_pc  = recall_score(y_true, y_pred, average=None, zero_division=0)
        prec_pc = precision_score(y_true, y_pred, average=None, zero_division=0)

        # VP / VN / FP / FN por clase (one-vs-rest)
        per_class: Dict[str, Dict] = {}
        for i, cls in enumerate(class_names):
            tp = int(cm[i, i])
            fp = int(cm[:, i].sum() - tp)
            fn = int(cm[i, :].sum() - tp)
            tn = int(cm.sum() - tp - fp - fn)
            per_class[cls] = {
                "VP": tp,
                "VN": tn,
                "FP": fp,
                "FN": fn,
                "precision": round(float(prec_pc[i]), 4),
                "recall":    round(float(rec_pc[i]),  4),
                "f1":        round(float(f1_pc[i]),   4),
            }

        return {
            "accuracy":         round(float(acc),    4),
            "f1_macro":         round(float(f1),     4),
            "recall":           round(float(recall), 4),
            "precision":        round(float(prec),   4),
            "confusion_matrix": cm.tolist(),
            "class_names":      list(class_names),
            "per_class":        per_class,
        }


# ── Instancia global ───────────────────────────────────────────────────────────
training_manager = TrainingManager()
