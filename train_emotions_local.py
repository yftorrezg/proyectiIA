"""
train_emotions_local.py  — Edu-Insight (versión avanzada)
==========================================================
Técnicas aplicadas para maximizar accuracy en FER-2013:
  1. Modelo más grande: resnet50_fer (25M params) o efficientnet_b3_fer
  2. CutMix augmentation  → +2-3% accuracy
  3. AdamW + weight decay → mejor regularización
  4. Warmup + CosineAnnealing → schedule de LR preciso
  5. Focal Loss           → penaliza ejemplos difíciles
  6. img_size=112         → más detalle facial que 96px
  7. WeightedRandomSampler → corrige desbalance disgust vs happy

Ejecutar (servidor APAGADO para tener GPU libre):
    cd C:/Users/Yafer Torrez/Desktop/proyectos/ProyectoIA
    .venv/Scripts/activate
    python train_emotions_local.py

Al terminar, el .pt y .json aparecen automáticamente en Admin Lab → Historial.
"""

import os, sys, json, copy, time, math
from datetime import datetime
from pathlib import Path

# ──────────────────────────────────────────────────────────────────────────────
#  CONFIGURACIÓN
# ──────────────────────────────────────────────────────────────────────────────
ROOT         = Path(__file__).parent
DATASET_PATH = ROOT / "datasets" / "emotion"

# ── Modelo ── (elige uno)
# "mobilenet_v2_fer"   →  3.4M params  — rápido, ~65%
# "resnet18_fer"       →   11M params  — equilibrado, ~66%
# "resnet50_fer"       →   25M params  — RECOMENDADO, ~68-70%
# "efficientnet_b0_fer"→  5.3M params  — eficiente
# "efficientnet_b3_fer"→   12M params  — RECOMENDADO, ~68-71%
MODEL_ID     = "resnet50_fer"

NUM_EPOCHS   = 30
LR_HEAD      = 1e-3    # fase 1: solo clasificador
LR_FULL      = 3e-5    # fase 2: fine-tuning completo (bajo para no romper pesos ImageNet)
WEIGHT_DECAY = 1e-4    # AdamW regularization
BATCH_SIZE   = 32      # resnet50 usa más VRAM; baja a 16 si hay OOM
IMG_SIZE     = 112     # más detalle facial que 96px
PATIENCE     = 7       # early stopping
CUTMIX_PROB  = 0.5     # probabilidad de aplicar CutMix por batch
WARMUP_EPOCHS = 3      # epochs de warmup lineal antes de cosine decay
PHASE1_EPOCHS = 5      # epochs con backbone congelado (solo head)
# ──────────────────────────────────────────────────────────────────────────────

MODELS_DIR  = ROOT / "app" / "storage" / "trained_models"
METRICS_DIR = ROOT / "app" / "storage" / "metrics"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)


# ── Focal Loss ────────────────────────────────────────────────────────────────
import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """
    Focal Loss: reduce el peso de ejemplos fáciles y fuerza al modelo a
    aprender de los difíciles (clases minoritarias como disgust).
    gamma=2 es el valor estándar del paper original.
    """
    def __init__(self, gamma: float = 2.0, label_smoothing: float = 0.1):
        super().__init__()
        self.gamma           = gamma
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Cross-entropy con label smoothing
        ce = F.cross_entropy(logits, targets, reduction="none",
                             label_smoothing=self.label_smoothing)
        pt = torch.exp(-ce)                         # probabilidad de la clase correcta
        focal = (1 - pt) ** self.gamma * ce         # ejemplos difíciles (pt bajo) reciben mayor peso
        return focal.mean()


# ── CutMix ────────────────────────────────────────────────────────────────────
def cutmix_batch(images: torch.Tensor, labels: torch.Tensor):
    """
    CutMix: recorta un parche de una imagen y lo pega en otra.
    Las etiquetas se mezclan proporcionalmente al área del parche.
    Mejora ~2-3% en FER-2013 según literatura.
    """
    B, C, H, W = images.shape
    lam = torch.distributions.Beta(1.0, 1.0).sample().item()

    # Tamaño del parche
    cut_ratio = math.sqrt(1.0 - lam)
    cut_h = int(H * cut_ratio)
    cut_w = int(W * cut_ratio)

    # Centro aleatorio
    cx = torch.randint(W, (1,)).item()
    cy = torch.randint(H, (1,)).item()
    x1 = max(cx - cut_w // 2, 0)
    y1 = max(cy - cut_h // 2, 0)
    x2 = min(cx + cut_w // 2, W)
    y2 = min(cy + cut_h // 2, H)

    # Proporción real de área mezclada
    lam_real = 1.0 - (x2 - x1) * (y2 - y1) / (H * W)

    # Índices mezclados (permutación aleatoria del batch)
    idx = torch.randperm(B, device=images.device)
    mixed = images.clone()
    mixed[:, :, y1:y2, x1:x2] = images[idx, :, y1:y2, x1:x2]

    return mixed, labels, labels[idx], lam_real


def main():
    import numpy as np
    import torch.optim as optim
    from torch.utils.data import DataLoader, WeightedRandomSampler
    from torchvision import datasets, transforms, models
    from sklearn.metrics import (
        accuracy_score, f1_score, recall_score,
        precision_score, confusion_matrix,
    )

    # ── Dispositivo ───────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 62)
    print(f"  train_emotions_local.py — MODO AVANZADO")
    print(f"  Modelo    : {MODEL_ID}")
    print(f"  Device    : {device}")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        props = torch.cuda.get_device_properties(0)
        print(f"  GPU       : {props.name}")
        print(f"  VRAM      : {props.total_memory/1e9:.1f} GB")
    print(f"  img_size  : {IMG_SIZE}px  |  batch={BATCH_SIZE}  |  epochs={NUM_EPOCHS}")
    print(f"  CutMix    : p={CUTMIX_PROB}  |  FocalLoss  |  AdamW  |  Cosine LR")
    print("=" * 62)

    train_dir = DATASET_PATH / "train"
    test_dir  = DATASET_PATH / "test"
    if not train_dir.exists():
        print(f"[ERROR] No se encontró: {train_dir}")
        sys.exit(1)

    # ── Transforms ────────────────────────────────────────────────────────────
    # Augmentation agresivo en train; mínimo en val
    train_tf = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((IMG_SIZE + 16, IMG_SIZE + 16)),   # resize más grande y luego crop
        transforms.RandomCrop(IMG_SIZE),                      # crop aleatorio
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2),
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.3, scale=(0.02, 0.15), ratio=(0.3, 3.3)),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    val_tf = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    train_ds = datasets.ImageFolder(str(train_dir), transform=train_tf)
    val_ds   = datasets.ImageFolder(str(test_dir),  transform=val_tf)

    NUM_CLASSES = len(train_ds.classes)
    CLASSES     = train_ds.classes

    print(f"\nClases ({NUM_CLASSES}): {list(CLASSES)}")
    print(f"Train: {len(train_ds):,}  |  Val: {len(val_ds):,}\n")

    targets     = [s[1] for s in train_ds.samples]
    class_count = [targets.count(i) for i in range(NUM_CLASSES)]
    for cls, cnt in zip(CLASSES, class_count):
        bar = "█" * int(cnt / max(class_count) * 28)
        print(f"  {cls:12s} {cnt:5d}  {bar}")

    # WeightedRandomSampler
    weights = [1.0 / class_count[t] for t in targets]
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, sampler=sampler,
        num_workers=4, pin_memory=True, persistent_workers=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE * 2, shuffle=False,
        num_workers=4, pin_memory=True, persistent_workers=True,
    )

    # ── Modelo ────────────────────────────────────────────────────────────────
    def build_model(model_id):
        if model_id == "mobilenet_v2_fer":
            m = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
            m.classifier[1] = nn.Linear(m.last_channel, NUM_CLASSES)
            return m
        if model_id == "resnet18_fer":
            m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            m.fc = nn.Linear(m.fc.in_features, NUM_CLASSES)
            return m
        if model_id == "resnet50_fer":
            m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
            m.fc = nn.Linear(m.fc.in_features, NUM_CLASSES)
            return m
        if model_id == "efficientnet_b0_fer":
            m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
            m.classifier[1] = nn.Linear(m.classifier[1].in_features, NUM_CLASSES)
            return m
        if model_id == "efficientnet_b3_fer":
            m = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.IMAGENET1K_V1)
            m.classifier[1] = nn.Linear(m.classifier[1].in_features, NUM_CLASSES)
            return m
        raise ValueError(f"Modelo desconocido: {model_id}")

    model     = build_model(MODEL_ID).to(device)
    criterion = FocalLoss(gamma=2.0, label_smoothing=0.1)
    use_amp   = device.type == "cuda"
    scaler    = torch.amp.GradScaler("cuda", enabled=use_amp)

    # ── Epoch function ────────────────────────────────────────────────────────
    def run_epoch(loader, training: bool, optimizer=None, use_cutmix: bool = False, sched=None):
        model.train() if training else model.eval()
        total_loss, correct, total = 0.0, 0, 0
        all_preds, all_labels = [], []
        t0 = time.time()

        ctx = torch.enable_grad() if training else torch.no_grad()
        with ctx:
            for images, labels in loader:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                if training and use_cutmix and torch.rand(1).item() < CUTMIX_PROB:
                    images, labels_a, labels_b, lam = cutmix_batch(images, labels)
                    if optimizer:
                        optimizer.zero_grad(set_to_none=True)
                    with torch.autocast(device_type=device.type, enabled=use_amp):
                        logits = model(images)
                        loss   = lam * criterion(logits, labels_a) + (1 - lam) * criterion(logits, labels_b)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    if sched is not None:
                        sched.step()
                    preds = logits.argmax(dim=1)
                    # Para accuracy reportada, usar labels_a (original)
                    correct += (preds == labels_a).sum().item()
                else:
                    if training and optimizer:
                        optimizer.zero_grad(set_to_none=True)
                    with torch.autocast(device_type=device.type, enabled=use_amp):
                        logits = model(images)
                        loss   = criterion(logits, labels)
                    if training:
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                        if sched is not None:
                            sched.step()
                    preds = logits.argmax(dim=1)
                    correct += (preds == labels).sum().item()

                total_loss += loss.item()
                total      += labels.size(0)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        return total_loss / len(loader), correct / total, all_preds, all_labels, time.time() - t0

    # ── FASE 1: Solo clasificador ─────────────────────────────────────────────
    print("\n" + "=" * 62)
    print(f"  FASE 1 — Head only ({PHASE1_EPOCHS} epochs, backbone congelado)")
    print("=" * 62)

    # Congelar backbone, descongelar solo la cabeza
    for name, p in model.named_parameters():
        is_head = any(k in name for k in ["classifier", "fc", "head"])
        p.requires_grad = is_head

    opt1 = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR_HEAD, weight_decay=WEIGHT_DECAY,
    )

    best_acc     = 0.0
    best_weights = None

    for epoch in range(1, PHASE1_EPOCHS + 1):
        tr_loss, tr_acc, _, _, t_tr = run_epoch(train_loader, True, opt1, use_cutmix=False)
        _,       val_acc, _,  _, t_v = run_epoch(val_loader, False)
        marker = " ★" if val_acc > best_acc else ""
        if val_acc > best_acc:
            best_acc     = val_acc
            best_weights = copy.deepcopy(model.state_dict())
        print(f"  [F1] {epoch:2d}/{PHASE1_EPOCHS} | loss {tr_loss:.4f} | "
              f"val_acc {val_acc:.2%}{marker} ({t_tr+t_v:.0f}s)")

    # ── FASE 2: Fine-tuning completo con CutMix ───────────────────────────────
    print("\n" + "=" * 62)
    print(f"  FASE 2 — Fine-tuning completo + CutMix ({NUM_EPOCHS - PHASE1_EPOCHS} epochs)")
    print("=" * 62)

    for p in model.parameters():
        p.requires_grad = True

    opt2 = optim.AdamW(model.parameters(), lr=LR_FULL, weight_decay=WEIGHT_DECAY)

    # OneCycleLR: warmup lineal + cosine decay en un solo scheduler por batch
    PHASE2_EPOCHS = NUM_EPOCHS - PHASE1_EPOCHS
    scheduler = optim.lr_scheduler.OneCycleLR(
        opt2,
        max_lr=LR_FULL,
        steps_per_epoch=len(train_loader),
        epochs=PHASE2_EPOCHS,
        pct_start=WARMUP_EPOCHS / PHASE2_EPOCHS,   # fracción dedicada al warmup
        anneal_strategy="cos",
        div_factor=10,          # LR inicial = max_lr / 10
        final_div_factor=1000,  # LR final   = max_lr / 1000
    )

    epochs_no_improve = 0
    prev_lr           = LR_FULL

    for epoch in range(1, PHASE2_EPOCHS + 1):
        tr_loss, tr_acc, _, _, t_tr   = run_epoch(train_loader, True, opt2, use_cutmix=True, sched=scheduler)
        _,       val_acc, vp, vl, t_v = run_epoch(val_loader, False)

        cur_lr = scheduler.get_last_lr()[0]

        if val_acc > best_acc:
            best_acc     = val_acc
            best_weights = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
            marker = " ★ mejor"
        else:
            epochs_no_improve += 1
            marker = f" (sin mejora {epochs_no_improve}/{PATIENCE})"

        print(f"  [F2] {epoch:2d}/{PHASE2_EPOCHS} | loss {tr_loss:.4f} | "
              f"val_acc {val_acc:.2%}{marker} | lr {cur_lr:.2e} ({t_tr+t_v:.0f}s)")

        if epochs_no_improve >= PATIENCE:
            print(f"\n  Early stopping. Mejor val_acc: {best_acc:.2%}")
            break

    # ── Métricas finales ──────────────────────────────────────────────────────
    print("\n" + "=" * 62)
    print(f"  Cargando mejor modelo (val_acc {best_acc:.2%})...")
    print("=" * 62)

    model.load_state_dict(best_weights)
    _, _, y_pred, y_true, _ = run_epoch(val_loader, False)

    acc  = accuracy_score(y_true, y_pred)
    f1   = f1_score(y_true, y_pred, average="macro", zero_division=0)
    rec  = recall_score(y_true, y_pred, average="macro", zero_division=0)
    prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    cm   = confusion_matrix(y_true, y_pred)
    f1_pc   = f1_score(y_true, y_pred, average=None, zero_division=0)
    rec_pc  = recall_score(y_true, y_pred, average=None, zero_division=0)
    prec_pc = precision_score(y_true, y_pred, average=None, zero_division=0)

    print(f"\n  Accuracy : {acc:.4f}  ({acc:.2%})")
    print(f"  F1 macro : {f1:.4f}")
    print(f"  Recall   : {rec:.4f}")
    print(f"  Precision: {prec:.4f}")

    print("\n  Matriz de Confusión (filas=real, cols=predicho):")
    print("             " + "  ".join(f"{c[:5]:>6}" for c in CLASSES))
    for i, row in enumerate(cm):
        print(f"  {CLASSES[i]:10s} " + "  ".join(f"{v:6d}" for v in row))

    print("\n  Por clase:")
    for i, cls in enumerate(CLASSES):
        tp = int(cm[i, i])
        fn = int(cm[i, :].sum() - tp)
        fp = int(cm[:, i].sum() - tp)
        print(f"    {cls:12s}  P:{prec_pc[i]:.2%}  R:{rec_pc[i]:.2%}  "
              f"F1:{f1_pc[i]:.2%}  VP:{tp}  FP:{fp}  FN:{fn}")

    # ── Guardar ───────────────────────────────────────────────────────────────
    ts           = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem         = f"{MODEL_ID}_{ts}_local"
    model_path   = MODELS_DIR / f"{stem}.pt"
    metrics_path = METRICS_DIR / f"{stem}.json"

    hp_saved = {
        "img_size": IMG_SIZE, "epochs": NUM_EPOCHS, "batch_size": BATCH_SIZE,
        "learning_rate": LR_FULL, "freeze_layers": False,
        "cutmix": CUTMIX_PROB, "focal_loss": True, "weight_decay": WEIGHT_DECAY,
    }

    torch.save({
        "model_state_dict": model.state_dict(),
        "model_id":         MODEL_ID,
        "num_classes":      NUM_CLASSES,
        "classes":          list(CLASSES),
        "hyperparams":      hp_saved,
        "trained_at":       ts,
        "job_id":           f"local_{ts}",
    }, model_path)

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

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump({
            "accuracy": round(float(acc), 4), "f1_macro": round(float(f1), 4),
            "recall":   round(float(rec), 4), "precision": round(float(prec), 4),
            "confusion_matrix": cm.tolist(), "class_names": list(CLASSES),
            "per_class": per_class, "n_train": len(train_ds), "n_test": len(val_ds),
            "model_id": MODEL_ID, "model_name": MODEL_ID, "category": "emocion",
            "trained_at": ts, "model_path": str(model_path), "hyperparams": hp_saved,
        }, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 62)
    print(f"  Modelo   → {model_path.name}")
    print(f"  Métricas → {metrics_path.name}")
    print(f"\n  Accuracy: {acc:.2%}  |  F1 macro: {f1:.2%}")
    print("\n  PRÓXIMO PASO: arranca el servidor y ve a Admin Lab → Historial → Activar")
    print("=" * 62)


if __name__ == "__main__":
    main()
