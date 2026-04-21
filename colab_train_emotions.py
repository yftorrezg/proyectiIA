"""
colab_train_emotions.py
=======================
Script para entrenar el modelo de emociones en Google Colab con GPU.
Compatible directo con inference_engine.py del proyecto (guarda en formato .pt PyTorch).

INSTRUCCIONES:
1. Sube este archivo a Google Colab (o copia su contenido)
2. Activa GPU: Entorno → Cambiar tipo de entorno de ejecución → GPU (T4 o A100)
3. Sube el dataset a Google Drive con esta estructura:
       MyDrive/IA3Drive/emotion/train/angry/...
       MyDrive/IA3Drive/emotion/test/angry/...
4. Ejecuta celda por celda
5. Al finalizar, descarga el .pt y ponlo en:
       app/storage/trained_models/
6. Actívalo desde el Admin Lab → Historial

COMPATIBILIDAD: el checkpoint .pt generado es idéntico al formato
que produce trainer.py → load_emocion_cnn() lo carga sin modificaciones.
"""

# ─────────────────────────────────────────────────────────────
# CELDA 1: Instalar dependencias (ejecutar solo una vez)
# ─────────────────────────────────────────────────────────────
# !pip install -q torchvision scikit-learn

# ─────────────────────────────────────────────────────────────
# CELDA 2: Montar Drive
# ─────────────────────────────────────────────────────────────
# from google.colab import drive
# drive.mount('/content/drive')

# ─────────────────────────────────────────────────────────────
# CELDA 3: Configuración — AJUSTA AQUÍ
# ─────────────────────────────────────────────────────────────
import os, json, copy
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms, models
from sklearn.metrics import (
    accuracy_score, f1_score, recall_score,
    precision_score, confusion_matrix,
)

# ── PARÁMETROS (ajusta según tu sesión de Colab) ──────────────
DATASET_PATH = "/content/drive/MyDrive/IA3Drive/emotion"   # ← cambia si es necesario
MODEL_ID     = "mobilenet_v2_fer"   # opciones: mobilenet_v2_fer | resnet18_fer | efficientnet_b0_fer
NUM_EPOCHS   = 25
LR_HEAD      = 1e-3     # fase 1: solo clasificador
LR_FULL      = 5e-5     # fase 2: fine-tuning completo
BATCH_SIZE   = 64
IMG_SIZE     = 112      # En Colab con A100 puedes usar 112 o 160; con T4 usa 96
PATIENCE     = 6        # early stopping
SAVE_DIR     = "/content/drive/MyDrive/IA3Drive/modelos"   # donde guardar el .pt
# ──────────────────────────────────────────────────────────────

os.makedirs(SAVE_DIR, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if device.type == "cuda":
    torch.backends.cudnn.benchmark = True
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# ─────────────────────────────────────────────────────────────
# CELDA 4: Dataset y DataLoaders
# ─────────────────────────────────────────────────────────────
train_tf = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.1),
    transforms.RandomErasing(p=0.2, scale=(0.02, 0.1)),   # ocluye zonas pequeñas
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
val_tf = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

train_ds = datasets.ImageFolder(f"{DATASET_PATH}/train", transform=train_tf)
val_ds   = datasets.ImageFolder(f"{DATASET_PATH}/test",  transform=val_tf)

NUM_CLASSES = len(train_ds.classes)
CLASSES     = train_ds.classes
print(f"\nClases ({NUM_CLASSES}): {CLASSES}")
print(f"Train: {len(train_ds):,} | Val: {len(val_ds):,}")

# Distribución de clases
targets     = [s[1] for s in train_ds.samples]
class_count = [targets.count(i) for i in range(NUM_CLASSES)]
print("\nDistribución:")
for i, (cls, cnt) in enumerate(zip(CLASSES, class_count)):
    print(f"  {cls:12s}: {cnt:5d} ({cnt/len(train_ds)*100:.1f}%)")

# WeightedRandomSampler — corrige desbalance (disgust muy poco frecuente)
weights = [1.0 / class_count[t] for t in targets]
sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

# En Colab num_workers=2 funciona correctamente (no hay restricción de spawn)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler,
                          num_workers=2, pin_memory=True, persistent_workers=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE * 2, shuffle=False,
                          num_workers=2, pin_memory=True, persistent_workers=True)

# ─────────────────────────────────────────────────────────────
# CELDA 5: Modelo
# ─────────────────────────────────────────────────────────────
def build_model(model_id: str, num_classes: int) -> nn.Module:
    if model_id == "mobilenet_v2_fer":
        m = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        m.classifier[1] = nn.Linear(m.last_channel, num_classes)
        return m
    if model_id == "resnet18_fer":
        m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        return m
    if model_id == "efficientnet_b0_fer":
        m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
        return m
    raise ValueError(f"Modelo desconocido: {model_id}")

model = build_model(MODEL_ID, NUM_CLASSES).to(device)
use_amp = device.type == "cuda"
scaler  = torch.cuda.amp.GradScaler(enabled=use_amp)

# ─────────────────────────────────────────────────────────────
# CELDA 6: Función de entrenamiento de una época
# ─────────────────────────────────────────────────────────────
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)   # suaviza labels ruidosas de FER-2013

def run_epoch(loader, training: bool, optimizer=None):
    model.train() if training else model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for inputs, labels in loader:
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                logits = model(inputs)
                loss   = criterion(logits, labels)
            if training:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            total_loss += loss.item()
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total   += labels.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return total_loss / len(loader), correct / total, all_preds, all_labels

# ─────────────────────────────────────────────────────────────
# CELDA 7: Fase 1 — entrenar solo el clasificador (backbone congelado)
# ─────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("FASE 1: Entrenando solo el clasificador (backbone congelado)")
print("="*60)

# Congelar backbone
for name, p in model.named_parameters():
    if "classifier" not in name and "fc" not in name:
        p.requires_grad = False

opt1 = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR_HEAD)

best_acc      = 0.0
best_weights  = None
PHASE1_EPOCHS = min(5, NUM_EPOCHS // 4)

for epoch in range(1, PHASE1_EPOCHS + 1):
    tr_loss, tr_acc, _, _         = run_epoch(train_loader, True, opt1)
    val_loss, val_acc, vp, vl     = run_epoch(val_loader, False)
    marker = " ★" if val_acc > best_acc else ""
    if val_acc > best_acc:
        best_acc     = val_acc
        best_weights = copy.deepcopy(model.state_dict())
    print(f"[F1] Epoch {epoch}/{PHASE1_EPOCHS} | loss {tr_loss:.4f} | val_acc {val_acc:.2%}{marker}")

# ─────────────────────────────────────────────────────────────
# CELDA 8: Fase 2 — fine-tuning completo (todas las capas)
# ─────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("FASE 2: Fine-tuning completo (LR muy bajo)")
print("="*60)

# Descongelar todo
for p in model.parameters():
    p.requires_grad = True

opt2      = optim.Adam(model.parameters(), lr=LR_FULL)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    opt2, mode="max", factor=0.5, patience=3, verbose=True
)

epochs_no_improve = 0
PHASE2_EPOCHS = NUM_EPOCHS - PHASE1_EPOCHS

for epoch in range(1, PHASE2_EPOCHS + 1):
    tr_loss, tr_acc, _, _         = run_epoch(train_loader, True, opt2)
    val_loss, val_acc, vp, vl     = run_epoch(val_loader, False)
    scheduler.step(val_acc)

    marker = ""
    if val_acc > best_acc:
        best_acc     = val_acc
        best_weights = copy.deepcopy(model.state_dict())
        epochs_no_improve = 0
        marker = " ★ mejor"
    else:
        epochs_no_improve += 1
        marker = f" (sin mejora {epochs_no_improve}/{PATIENCE})"

    print(f"[F2] Epoch {epoch}/{PHASE2_EPOCHS} | loss {tr_loss:.4f} | val_acc {val_acc:.2%}{marker}")

    if epochs_no_improve >= PATIENCE:
        print(f"Early stopping en epoch {epoch}")
        break

# ─────────────────────────────────────────────────────────────
# CELDA 9: Métricas finales con el mejor modelo
# ─────────────────────────────────────────────────────────────
print("\n" + "="*60)
print(f"Mejor val_acc: {best_acc:.2%} — Cargando mejores pesos...")
print("="*60)

model.load_state_dict(best_weights)
_, _, y_pred, y_true = run_epoch(val_loader, False)

acc  = accuracy_score(y_true, y_pred)
f1   = f1_score(y_true, y_pred, average="macro", zero_division=0)
rec  = recall_score(y_true, y_pred, average="macro", zero_division=0)
prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
cm   = confusion_matrix(y_true, y_pred)

print(f"\nAccuracy : {acc:.4f} ({acc:.2%})")
print(f"F1 macro : {f1:.4f}")
print(f"Recall   : {rec:.4f}")
print(f"Precision: {prec:.4f}")

print("\nMatriz de Confusión:")
print("         " + "  ".join(f"{c[:5]:>5}" for c in CLASSES))
for i, row in enumerate(cm):
    print(f"{CLASSES[i]:8s} " + "  ".join(f"{v:5d}" for v in row))

# Por clase
print("\nPor clase:")
f1_pc   = f1_score(y_true, y_pred, average=None, zero_division=0)
rec_pc  = recall_score(y_true, y_pred, average=None, zero_division=0)
prec_pc = precision_score(y_true, y_pred, average=None, zero_division=0)
for i, cls in enumerate(CLASSES):
    tp = int(cm[i, i])
    fn = int(cm[i, :].sum() - tp)
    print(f"  {cls:12s} | P:{prec_pc[i]:.2%} R:{rec_pc[i]:.2%} F1:{f1_pc[i]:.2%} | VP:{tp} FN:{fn}")

# ─────────────────────────────────────────────────────────────
# CELDA 10: Guardar en formato compatible con inference_engine.py
# ─────────────────────────────────────────────────────────────
ts          = datetime.now().strftime("%Y%m%d_%H%M%S")
model_name  = f"{MODEL_ID}_{ts}_colab"
model_path  = f"{SAVE_DIR}/{model_name}.pt"
metrics_path = f"{SAVE_DIR}/{model_name}.json"

# Checkpoint idéntico al que genera trainer.py → compatible con load_emocion_cnn()
torch.save(
    {
        "model_state_dict": model.state_dict(),
        "model_id":         MODEL_ID,
        "num_classes":      NUM_CLASSES,
        "classes":          list(CLASSES),
        "hyperparams": {
            "img_size":       IMG_SIZE,
            "epochs":         NUM_EPOCHS,
            "learning_rate":  LR_FULL,
            "batch_size":     BATCH_SIZE,
            "freeze_layers":  False,
        },
        "trained_at": ts,
        "job_id":     f"colab_{ts}",
    },
    model_path,
)

# Métricas en el mismo formato que trainer.py → aparece en Admin Lab Historial
per_class = {}
for i, cls in enumerate(CLASSES):
    tp = int(cm[i, i])
    fp = int(cm[:, i].sum() - tp)
    fn = int(cm[i, :].sum() - tp)
    tn = int(cm.sum() - tp - fp - fn)
    per_class[cls] = {
        "VP": tp, "VN": tn, "FP": fp, "FN": fn,
        "precision": round(float(prec_pc[i]), 4),
        "recall":    round(float(rec_pc[i]),  4),
        "f1":        round(float(f1_pc[i]),   4),
    }

metrics = {
    "accuracy":         round(float(acc),  4),
    "f1_macro":         round(float(f1),   4),
    "recall":           round(float(rec),  4),
    "precision":        round(float(prec), 4),
    "confusion_matrix": cm.tolist(),
    "class_names":      list(CLASSES),
    "per_class":        per_class,
    "n_train":          len(train_ds),
    "n_test":           len(val_ds),
    "model_id":         MODEL_ID,
    "model_name":       MODEL_ID,
    "category":         "emocion",
    "trained_at":       ts,
    "model_path":       model_path,
    "hyperparams": {
        "img_size": IMG_SIZE, "epochs": NUM_EPOCHS,
        "batch_size": BATCH_SIZE, "freeze_layers": False,
    },
}

with open(metrics_path, "w", encoding="utf-8") as f:
    json.dump(metrics, f, ensure_ascii=False, indent=2)

print(f"\n✅ Modelo guardado : {model_path}")
print(f"✅ Métricas guardadas: {metrics_path}")
print(f"\nAccuracy final: {acc:.2%} | F1 macro: {f1:.2%}")
print("\n" + "="*60)
print("PRÓXIMOS PASOS:")
print("1. Descarga el .pt y el .json de Google Drive")
print("2. Copia ambos a:  app/storage/trained_models/  (el .pt)")
print("                   app/storage/metrics/         (el .json)")
print("3. Abre Admin Lab → Historial → Activar")
print("="*60)
