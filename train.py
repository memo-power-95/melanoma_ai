"""
train.py — Clasificador de melanoma (7 clases) con EfficientNet-B3

Clases:
    0  Extensión Superficial
    1  Lentiginoso Acral
    2  Lentigo Maligno
    3  Nodular
    4  Mucosas      (Otros/Mucosas/)
    5  Oculares     (Otros/Oculares/)
    6  No melanoma

Requisitos:
    pip install torch torchvision pillow scikit-learn tqdm

Uso:
    python train.py
    python train.py --epochs 50 --batch-size 32 --model efficientnet_b0
    python train.py --resume checkpoints/best_f1.pt
    python train.py --dataset dataset_aumentado --no-aug-samples
"""

import os
import sys
import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import models, transforms
from PIL import Image, UnidentifiedImageError
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm
import numpy as np

# ══════════════════════════════════════════════════════════════════════════════
# Configuración
# ══════════════════════════════════════════════════════════════════════════════

ROOT          = Path(__file__).parent
DATASET_DIR   = ROOT / "dataset_aumentado"
CHECKPOINT_DIR = ROOT / "checkpoints"

# Mapa: nombre de clase → ruta relativa dentro de dataset_aumentado/
CLASS_FOLDER_MAP = {
    "Extensión Superficial": "Extensión Superficial",
    "Lentiginoso Acral":     "Lentiginoso Acral",
    "Lentigo Maligno":       "Lentigo Maligno",
    "Nodular":               "Nodular",
    "Mucosas":               os.path.join("Otros", "Mucosas"),
    "Oculares":              os.path.join("Otros", "Oculares"),
    "No melanoma":           "No melanoma",
}

CLASSES     = list(CLASS_FOLDER_MAP.keys())
NUM_CLASSES = len(CLASSES)
IMG_SIZE    = 224          # EfficientNet-B3 acepta 300, B0 224; usamos 224 universal
VALID_EXTS  = {'.jpg', '.jpeg', '.png'}


# ══════════════════════════════════════════════════════════════════════════════
# Dataset
# ══════════════════════════════════════════════════════════════════════════════

class MelanomaDataset(Dataset):
    """
    Carga imágenes desde las 7 carpetas de dataset_aumentado/.
    Admite un subconjunto de índices (para split train/val/test).
    exclude_aug: si True excluye archivos con prefijo aug_ (solo imágenes reales).
    """

    def __init__(self, dataset_dir: Path, indices, transform=None,
                 exclude_aug: bool = False):
        self.transform = transform
        self.samples: list[tuple[Path, int]] = []

        all_samples = _collect_samples(dataset_dir, exclude_aug)
        self.samples = [all_samples[i] for i in indices]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            img = Image.open(path).convert("RGB")
        except (UnidentifiedImageError, OSError):
            # Imagen corrupta: devolver tensor negro
            img = Image.new("RGB", (IMG_SIZE, IMG_SIZE), 0)
        if self.transform:
            img = self.transform(img)
        return img, label


def _collect_samples(dataset_dir: Path,
                     exclude_aug: bool = False) -> list[tuple[Path, int]]:
    """Recopila (ruta, clase_idx) para todas las imágenes del dataset."""
    samples = []
    for cls_idx, cls_name in enumerate(CLASSES):
        folder = dataset_dir / CLASS_FOLDER_MAP[cls_name]
        if not folder.exists():
            print(f"  [AVISO] Carpeta no encontrada: {folder}")
            continue
        for f in sorted(folder.iterdir()):
            if f.suffix.lower() not in VALID_EXTS:
                continue
            if exclude_aug and f.name.startswith("aug_"):
                continue
            samples.append((f, cls_idx))
    return samples


# ══════════════════════════════════════════════════════════════════════════════
# Transforms
# ══════════════════════════════════════════════════════════════════════════════

# Normalización ImageNet (pesos preentrenados)
_MEAN = [0.485, 0.456, 0.406]
_STD  = [0.229, 0.224, 0.225]

TRAIN_TRANSFORMS = transforms.Compose([
    transforms.Resize((IMG_SIZE + 32, IMG_SIZE + 32)),
    transforms.RandomCrop(IMG_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.2, contrast=0.2,
                           saturation=0.2, hue=0.05),
    transforms.ToTensor(),
    transforms.Normalize(_MEAN, _STD),
])

VAL_TRANSFORMS = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(_MEAN, _STD),
])


# ══════════════════════════════════════════════════════════════════════════════
# Modelo
# ══════════════════════════════════════════════════════════════════════════════

AVAILABLE_MODELS = {
    "efficientnet_b0": models.efficientnet_b0,
    "efficientnet_b3": models.efficientnet_b3,
    "efficientnet_b4": models.efficientnet_b4,
    "resnet50":        models.resnet50,
    "resnet101":       models.resnet101,
}


def build_model(name: str, num_classes: int, freeze_backbone: bool = True):
    """
    Carga modelo preentrenado y reemplaza la cabeza de clasificación.
    Si freeze_backbone=True se congelan todos los pesos excepto la cabeza.
    """
    if name not in AVAILABLE_MODELS:
        raise ValueError(f"Modelo desconocido: {name}. "
                         f"Opciones: {list(AVAILABLE_MODELS)}")

    weights_map = {
        "efficientnet_b0": models.EfficientNet_B0_Weights.IMAGENET1K_V1,
        "efficientnet_b3": models.EfficientNet_B3_Weights.IMAGENET1K_V1,
        "efficientnet_b4": models.EfficientNet_B4_Weights.IMAGENET1K_V1,
        "resnet50":        models.ResNet50_Weights.IMAGENET1K_V2,
        "resnet101":       models.ResNet101_Weights.IMAGENET1K_V2,
    }

    model = AVAILABLE_MODELS[name](weights=weights_map[name])

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    # Reemplazar cabeza según arquitectura
    if name.startswith("efficientnet"):
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.4, inplace=True),
            nn.Linear(in_features, num_classes),
        )
    elif name.startswith("resnet"):
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(in_features, num_classes),
        )

    return model


def unfreeze_backbone(model):
    """Descongela todos los parámetros para fine-tuning completo."""
    for param in model.parameters():
        param.requires_grad = True


# ══════════════════════════════════════════════════════════════════════════════
# Utilidades de entrenamiento
# ══════════════════════════════════════════════════════════════════════════════

def compute_class_weights(dataset_dir: Path,
                          exclude_aug: bool = False) -> torch.Tensor:
    """Pesos inversamente proporcionales a la frecuencia de cada clase."""
    samples = _collect_samples(dataset_dir, exclude_aug)
    counts  = torch.zeros(NUM_CLASSES)
    for _, label in samples:
        counts[label] += 1
    weights = 1.0 / (counts + 1e-6)
    weights = weights / weights.sum() * NUM_CLASSES   # normalizar
    return weights


def make_sampler(labels: list[int]) -> WeightedRandomSampler:
    """WeightedRandomSampler para balancear mini-batches en entrenamiento."""
    counts = np.bincount(labels, minlength=NUM_CLASSES)
    weight_per_class = 1.0 / (counts + 1e-6)
    sample_weights   = [weight_per_class[l] for l in labels]
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = self.avg = self.sum = self.count = 0.0

    def update(self, val, n=1):
        self.val    = val
        self.sum   += val * n
        self.count += n
        self.avg    = self.sum / self.count


def accuracy(outputs, targets):
    preds = outputs.argmax(dim=1)
    return (preds == targets).float().mean().item()


def save_checkpoint(state: dict, path: Path):
    CHECKPOINT_DIR.mkdir(exist_ok=True)
    torch.save(state, path)


# ══════════════════════════════════════════════════════════════════════════════
# Bucle de entrenamiento
# ══════════════════════════════════════════════════════════════════════════════

def train_one_epoch(model, loader, criterion, optimizer, scaler, device):
    model.train()
    loss_m = AverageMeter()
    acc_m  = AverageMeter()

    pbar = tqdm(loader, desc="  Train", leave=False, unit="batch")
    for imgs, labels in pbar:
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()
        with torch.autocast(device_type=device.type,
                            enabled=(device.type == "cuda")):
            out  = model(imgs)
            loss = criterion(out, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        loss_m.update(loss.item(), imgs.size(0))
        acc_m.update(accuracy(out.detach(), labels), imgs.size(0))
        pbar.set_postfix(loss=f"{loss_m.avg:.4f}", acc=f"{acc_m.avg:.4f}")

    return loss_m.avg, acc_m.avg


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    loss_m = AverageMeter()
    acc_m  = AverageMeter()
    all_preds, all_labels = [], []

    for imgs, labels in tqdm(loader, desc="  Val  ", leave=False, unit="batch"):
        imgs, labels = imgs.to(device), labels.to(device)
        with torch.autocast(device_type=device.type,
                            enabled=(device.type == "cuda")):
            out  = model(imgs)
            loss = criterion(out, labels)

        loss_m.update(loss.item(), imgs.size(0))
        acc_m.update(accuracy(out, labels), imgs.size(0))
        all_preds.extend(out.argmax(dim=1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    return loss_m.avg, acc_m.avg, all_preds, all_labels


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="Entrena clasificador de melanoma con EfficientNet."
    )
    p.add_argument("--dataset",     default=str(DATASET_DIR),
                   help="Ruta al dataset_aumentado/")
    p.add_argument("--model",       default="efficientnet_b3",
                   choices=list(AVAILABLE_MODELS),
                   help="Arquitectura base (default: efficientnet_b3)")
    p.add_argument("--epochs",      type=int, default=40,
                   help="Épocas totales (default: 40)")
    p.add_argument("--batch-size",  type=int, default=32,
                   help="Tamaño de batch (default: 32)")
    p.add_argument("--lr",          type=float, default=1e-3,
                   help="Learning rate cabeza (default: 1e-3)")
    p.add_argument("--lr-finetune", type=float, default=1e-4,
                   help="LR al descongelar backbone (default: 1e-4)")
    p.add_argument("--unfreeze-epoch", type=int, default=10,
                   help="Época en que se descongela el backbone (default: 10)")
    p.add_argument("--val-split",   type=float, default=0.15,
                   help="Fracción de validación (default: 0.15)")
    p.add_argument("--test-split",  type=float, default=0.15,
                   help="Fracción de test (default: 0.15)")
    p.add_argument("--workers",     type=int, default=2,
                   help="DataLoader workers (default: 2)")
    p.add_argument("--seed",        type=int, default=42)
    p.add_argument("--resume",      default=None,
                   help="Checkpoint .pt para continuar entrenamiento")
    p.add_argument("--no-aug-samples", action="store_true",
                   help="Excluir imágenes aug_* del split (solo reales)")
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    dataset_dir = Path(args.dataset)
    if not dataset_dir.exists():
        print(f"[ERROR] Dataset no encontrado: {dataset_dir}")
        sys.exit(1)

    print("=" * 66)
    print("  ENTRENAMIENTO — Clasificador Melanoma (7 clases)")
    print("=" * 66)
    print(f"  Modelo     : {args.model}")
    print(f"  Épocas     : {args.epochs}")
    print(f"  Batch size : {args.batch_size}")
    print(f"  Dispositivo: {device}")
    print(f"  Dataset    : {dataset_dir}")

    # ── Split estratificado ────────────────────────────────────────────────
    all_samples = _collect_samples(dataset_dir,
                                   exclude_aug=args.no_aug_samples)
    all_labels  = [s[1] for s in all_samples]
    indices     = list(range(len(all_samples)))

    idx_train_val, idx_test = train_test_split(
        indices, test_size=args.test_split,
        stratify=all_labels, random_state=args.seed
    )
    labels_train_val = [all_labels[i] for i in idx_train_val]
    val_rel = args.val_split / (1.0 - args.test_split)
    idx_train, idx_val = train_test_split(
        idx_train_val, test_size=val_rel,
        stratify=labels_train_val, random_state=args.seed
    )

    train_labels = [all_labels[i] for i in idx_train]

    print(f"\n  Split  →  train: {len(idx_train)} | "
          f"val: {len(idx_val)} | test: {len(idx_test)}")
    print(f"\n  {'Clase':<26} {'Train':>7} {'Val':>7} {'Test':>7}")
    print(f"  {'-'*49}")
    for ci, cn in enumerate(CLASSES):
        tr = sum(1 for i in idx_train if all_labels[i] == ci)
        va = sum(1 for i in idx_val   if all_labels[i] == ci)
        te = sum(1 for i in idx_test  if all_labels[i] == ci)
        print(f"  {cn:<26} {tr:>7} {va:>7} {te:>7}")

    # ── Datasets y DataLoaders ─────────────────────────────────────────────
    train_ds = MelanomaDataset(dataset_dir, idx_train, TRAIN_TRANSFORMS,
                               args.no_aug_samples)
    val_ds   = MelanomaDataset(dataset_dir, idx_val,   VAL_TRANSFORMS,
                               args.no_aug_samples)
    test_ds  = MelanomaDataset(dataset_dir, idx_test,  VAL_TRANSFORMS,
                               args.no_aug_samples)

    sampler = make_sampler(train_labels)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              sampler=sampler, num_workers=args.workers,
                              pin_memory=(device.type == "cuda"))
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size,
                              shuffle=False, num_workers=args.workers,
                              pin_memory=(device.type == "cuda"))
    test_loader  = DataLoader(test_ds, batch_size=args.batch_size,
                              shuffle=False, num_workers=args.workers,
                              pin_memory=(device.type == "cuda"))

    # ── Modelo ────────────────────────────────────────────────────────────
    model = build_model(args.model, NUM_CLASSES, freeze_backbone=True)
    model = model.to(device)

    # ── Pérdida con pesos de clase ─────────────────────────────────────────
    class_weights = compute_class_weights(dataset_dir,
                                          args.no_aug_samples).to(device)
    criterion     = nn.CrossEntropyLoss(weight=class_weights)

    # ── Optimizador (solo cabeza inicialmente) ─────────────────────────────
    head_params = [p for p in model.parameters() if p.requires_grad]
    optimizer   = optim.AdamW(head_params, lr=args.lr, weight_decay=1e-4)
    scheduler   = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )
    scaler = torch.GradScaler(enabled=(device.type == "cuda"))

    # ── Reanudar checkpoint ────────────────────────────────────────────────
    start_epoch = 1
    best_val_f1 = 0.0

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_val_f1 = ckpt.get("best_val_f1", 0.0)
        print(f"\n  Checkpoint cargado: {args.resume} "
              f"(época {start_epoch - 1}, best_f1={best_val_f1:.4f})")

    # ── Entrenamiento ──────────────────────────────────────────────────────
    print(f"\n{'─'*66}")
    backbone_unfrozen = (args.resume is not None and
                         start_epoch > args.unfreeze_epoch)

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()

        # Descongelar backbone en la época indicada
        if epoch == args.unfreeze_epoch and not backbone_unfrozen:
            print(f"\n  [Época {epoch}] Descongelando backbone — "
                  f"lr={args.lr_finetune}")
            unfreeze_backbone(model)
            optimizer = optim.AdamW(model.parameters(),
                                    lr=args.lr_finetune, weight_decay=1e-4)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=args.epochs - epoch + 1,
                eta_min=1e-6,
            )
            backbone_unfrozen = True

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device
        )
        val_loss, val_acc, val_preds, val_labels = evaluate(
            model, val_loader, criterion, device
        )
        scheduler.step()

        # Macro-F1 sobre validación
        from sklearn.metrics import f1_score
        val_f1 = f1_score(val_labels, val_preds,
                          average="macro", zero_division=0)

        elapsed = time.time() - t0
        lr_now  = optimizer.param_groups[0]["lr"]

        print(f"  Época {epoch:>3}/{args.epochs}  "
              f"train_loss={train_loss:.4f}  train_acc={train_acc:.4f}  "
              f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}  "
              f"val_f1={val_f1:.4f}  lr={lr_now:.2e}  "
              f"({elapsed:.0f}s)")

        # Guardar mejor modelo (por macro-F1 en validación)
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            save_checkpoint(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "best_val_f1": best_val_f1,
                    "classes": CLASSES,
                    "model_name": args.model,
                },
                CHECKPOINT_DIR / "best_f1.pt",
            )
            print(f"    ✓ Nuevo mejor modelo guardado "
                  f"(val_f1={best_val_f1:.4f})")

        # Checkpoint periódico cada 10 épocas
        if epoch % 10 == 0:
            save_checkpoint(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "best_val_f1": best_val_f1,
                    "classes": CLASSES,
                    "model_name": args.model,
                },
                CHECKPOINT_DIR / f"epoch_{epoch:03d}.pt",
            )

    # ── Evaluación final sobre test ────────────────────────────────────────
    print(f"\n{'='*66}")
    print("  EVALUACIÓN FINAL — conjunto de test")
    print(f"{'='*66}")

    best_ckpt = CHECKPOINT_DIR / "best_f1.pt"
    if best_ckpt.exists():
        ckpt = torch.load(best_ckpt, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model"])
        print(f"  Modelo cargado: {best_ckpt} "
              f"(época {ckpt['epoch']}, val_f1={ckpt['best_val_f1']:.4f})")

    _, test_acc, test_preds, test_labels_list = evaluate(
        model, test_loader, criterion, device
    )

    print(f"\n  Accuracy en test: {test_acc:.4f}\n")
    print(classification_report(
        test_labels_list, test_preds,
        target_names=CLASSES, zero_division=0
    ))

    cm = confusion_matrix(test_labels_list, test_preds)
    print("  Matriz de confusión:")
    header = "  " + "".join(f"{c[:6]:>8}" for c in CLASSES)
    print(header)
    for i, row in enumerate(cm):
        print(f"  {CLASSES[i][:6]:<8}" +
              "".join(f"{v:>8}" for v in row))

    print(f"\n  Checkpoints guardados en: {CHECKPOINT_DIR}/")
    print(f"  Mejor val_f1 global: {best_val_f1:.4f}")
    print(f"{'='*66}\n")


if __name__ == "__main__":
    main()
