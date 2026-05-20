"""
train_balanceado.py — Clasificador de melanoma (7 clases) con EfficientNet-B3
Variante que usa dataset_balanceado/ en lugar de dataset_aumentado/.

La diferencia clave es el split por grupo:
  - dataset_aumentado usa el prefijo aug_ en el nombre de archivo.
  - dataset_balanceado usa manifest.csv con source_sha256 como clave de grupo,
    ya que todos los archivos tienen el mismo esquema de nombrado ISIC_SSM_XXXX.

El modelo producido (checkpoints/best_f1.pt) es idéntico en estructura y
formato al generado por train.py, por lo que export_model.py funciona igual.

Uso:
    python train_balanceado.py
    python train_balanceado.py --epochs 60 --batch-size 32
    python train_balanceado.py --resume checkpoints/best_f1.pt
    python train_balanceado.py --no-aug-samples   # solo imágenes originales
"""

import csv
import os
import sys
import argparse
import time
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import models, transforms
from PIL import Image, UnidentifiedImageError
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from tqdm import tqdm
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ══════════════════════════════════════════════════════════════════════════════
# Configuración
# ══════════════════════════════════════════════════════════════════════════════

ROOT           = Path(__file__).parent
DATASET_DIR    = ROOT / "dataset_balanceado"
CHECKPOINT_DIR = ROOT / "checkpoints"

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
IMG_SIZE    = 224          # 224 px — coincide con la inferencia en Android (ImageClassifier.kt)
VALID_EXTS  = {'.jpg', '.jpeg', '.png'}


# ══════════════════════════════════════════════════════════════════════════════
# Manifest — índice source_sha256 para evitar data leakage
# ══════════════════════════════════════════════════════════════════════════════

def _load_manifest(dataset_dir: Path) -> tuple[dict[str, str], set[str]]:
    """
    Lee manifest.csv y devuelve:
      - sha_map   : {output_name -> source_sha256}
      - aug_names : conjunto de output_name que son augmentaciones (no originales)
    """
    manifest_path = dataset_dir / "manifest.csv"
    sha_map: dict[str, str] = {}
    aug_names: set[str] = set()

    if not manifest_path.exists():
        return sha_map, aug_names

    with open(manifest_path, newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            name = row["output_name"]
            sha_map[name]  = row["source_sha256"]
            if row["augmentation"] != "original":
                aug_names.add(name)

    return sha_map, aug_names


# ══════════════════════════════════════════════════════════════════════════════
# Dataset
# ══════════════════════════════════════════════════════════════════════════════

def _collect_samples(dataset_dir: Path,
                     aug_names: set[str],
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
            if exclude_aug and f.name in aug_names:
                continue
            samples.append((f, cls_idx))
    return samples


class MelanomaDataset(Dataset):
    def __init__(self, dataset_dir: Path, indices: list[int],
                 transform=None, aug_names: set[str] = None,
                 exclude_aug: bool = False):
        self.transform = transform
        all_samples = _collect_samples(dataset_dir, aug_names or set(),
                                       exclude_aug)
        self.samples: list[tuple[Path, int]] = [all_samples[i] for i in indices]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            img = Image.open(path).convert("RGB")
        except (UnidentifiedImageError, OSError):
            img = Image.new("RGB", (IMG_SIZE, IMG_SIZE), 0)
        if self.transform:
            img = self.transform(img)
        return img, label


# ══════════════════════════════════════════════════════════════════════════════
# Transforms
# ══════════════════════════════════════════════════════════════════════════════

_MEAN = [0.485, 0.456, 0.406]
_STD  = [0.229, 0.224, 0.225]

TRAIN_TRANSFORMS = transforms.Compose([
    transforms.Resize((IMG_SIZE + 40, IMG_SIZE + 40)),
    transforms.RandomCrop(IMG_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(30),
    transforms.ColorJitter(brightness=0.3, contrast=0.3,
                           saturation=0.3, hue=0.08),
    transforms.RandomGrayscale(p=0.05),
    transforms.ToTensor(),
    transforms.Normalize(_MEAN, _STD),
    transforms.RandomErasing(p=0.25, scale=(0.02, 0.15)),
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
    weights_map = {
        "efficientnet_b0": models.EfficientNet_B0_Weights.IMAGENET1K_V1,
        "efficientnet_b3": models.EfficientNet_B3_Weights.IMAGENET1K_V1,
        "efficientnet_b4": models.EfficientNet_B4_Weights.IMAGENET1K_V1,
        "resnet50":        models.ResNet50_Weights.IMAGENET1K_V2,
        "resnet101":       models.ResNet101_Weights.IMAGENET1K_V2,
    }
    if name not in AVAILABLE_MODELS:
        raise ValueError(f"Modelo desconocido: {name}. "
                         f"Opciones: {list(AVAILABLE_MODELS)}")

    model = AVAILABLE_MODELS[name](weights=weights_map[name])

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

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
    for param in model.parameters():
        param.requires_grad = True


# ══════════════════════════════════════════════════════════════════════════════
# Utilidades de entrenamiento
# ══════════════════════════════════════════════════════════════════════════════

def compute_class_weights(dataset_dir: Path, aug_names: set[str],
                          exclude_aug: bool = False) -> torch.Tensor:
    samples = _collect_samples(dataset_dir, aug_names, exclude_aug)
    counts  = torch.zeros(NUM_CLASSES)
    for _, label in samples:
        counts[label] += 1
    weights = 1.0 / (counts + 1e-6)
    weights = weights / weights.sum() * NUM_CLASSES
    return weights


def make_sampler(labels: list[int]) -> WeightedRandomSampler:
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
# Gráficas
# ══════════════════════════════════════════════════════════════════════════════

PLOTS_DIR = ROOT / "plots"


def plot_training(history: dict, unfreeze_epoch: int | None = None):
    PLOTS_DIR.mkdir(exist_ok=True)
    epochs = list(range(1, len(history["train_loss"]) + 1))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("Curvas de entrenamiento — Clasificador Melanoma (balanceado)",
                 fontsize=13, fontweight="bold")

    specs = [
        ("train_loss", "val_loss",  "Loss",        "Loss"),
        ("train_acc",  "val_acc",   "Accuracy",    "Accuracy"),
        (None,         "val_f1",    "Val macro-F1","F1"),
    ]

    for ax, (tr_key, va_key, title, ylabel) in zip(axes, specs):
        if tr_key:
            ax.plot(epochs, history[tr_key], label="Train",
                    linewidth=1.8, color="steelblue")
        ax.plot(epochs, history[va_key], label="Val",
                linewidth=1.8, color="tomato")

        if unfreeze_epoch and unfreeze_epoch <= len(epochs):
            ax.axvline(x=unfreeze_epoch, color="gray", linestyle="--",
                       linewidth=1.2, label=f"Unfreeze (é{unfreeze_epoch})")

        ax.set_title(title)
        ax.set_xlabel("Época")
        ax.set_ylabel(ylabel)
        ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out = PLOTS_DIR / "training_curves_balanceado.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Gráfica guardada: {out}")


def plot_confusion_matrix(cm: np.ndarray, class_names: list[str]):
    PLOTS_DIR.mkdir(exist_ok=True)
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-9)

    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues",
                   vmin=0, vmax=1)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=35, ha="right", fontsize=9)
    ax.set_yticklabels(class_names, fontsize=9)
    ax.set_xlabel("Predicción", fontsize=10)
    ax.set_ylabel("Real", fontsize=10)
    ax.set_title("Matriz de confusión — dataset balanceado (normalizada por filas)",
                 fontsize=11, fontweight="bold")

    thresh = 0.5
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            val   = cm_norm[i, j]
            raw   = cm[i, j]
            color = "white" if val > thresh else "black"
            ax.text(j, i, f"{val:.0%}\n({raw})",
                    ha="center", va="center",
                    fontsize=7.5, color=color)

    fig.tight_layout()
    out = PLOTS_DIR / "confusion_matrix_balanceado.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Gráfica guardada: {out}")


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

        optimizer.zero_grad(set_to_none=True)
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
        description="Entrena clasificador de melanoma con dataset_balanceado/."
    )
    p.add_argument("--dataset",     default=str(DATASET_DIR),
                   help="Ruta al dataset_balanceado/")
    p.add_argument("--model",       default="efficientnet_b3",
                   choices=list(AVAILABLE_MODELS),
                   help="Arquitectura base (default: efficientnet_b3)")
    p.add_argument("--epochs",      type=int, default=60,
                   help="Épocas totales (default: 60)")
    p.add_argument("--batch-size",  type=int, default=32,
                   help="Tamaño de batch (default: 32)")
    p.add_argument("--lr",          type=float, default=1e-3,
                   help="Learning rate cabeza (default: 1e-3)")
    p.add_argument("--lr-finetune", type=float, default=5e-5,
                   help="LR al descongelar backbone (default: 5e-5)")
    p.add_argument("--unfreeze-epoch", type=int, default=10,
                   help="Época en que se descongela el backbone (default: 10)")
    p.add_argument("--early-stop",   type=int, default=15,
                   help="Patience de early stopping (0 = desactivado, default: 15)")
    p.add_argument("--no-melanoma-weight", type=float, default=2.0,
                   help="Multiplicador del peso de clase 'No melanoma' en la loss "
                        "(default: 2.0 — el doble que el calculado automáticamente)")
    p.add_argument("--val-split",   type=float, default=0.15,
                   help="Fracción de validación (default: 0.15)")
    p.add_argument("--test-split",  type=float, default=0.10,
                   help="Fracción de test (default: 0.15)")
    p.add_argument("--workers",     type=int, default=2,
                   help="DataLoader workers (default: 2)")
    p.add_argument("--seed",        type=int, default=42)
    p.add_argument("--resume",      default=None,
                   help="Checkpoint .pt para continuar entrenamiento")
    p.add_argument("--no-aug-samples", action="store_true",
                   help="Excluir imágenes augmentadas del split (solo originales)")
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
    print("  ENTRENAMIENTO — Clasificador Melanoma (dataset balanceado)")
    print("=" * 66)
    print(f"  Modelo     : {args.model}")
    print(f"  Épocas     : {args.epochs}")
    print(f"  Batch size : {args.batch_size}")
    print(f"  Dispositivo: {device}")
    print(f"  Dataset    : {dataset_dir}")

    # ── Cargar manifest para identificar augmentaciones ────────────────────
    sha_map, aug_names = _load_manifest(dataset_dir)
    if not sha_map:
        print("  [AVISO] manifest.csv no encontrado; "
              "el split no evitará data leakage.")

    # ── Recopilar todas las muestras ───────────────────────────────────────
    all_samples = _collect_samples(dataset_dir, aug_names,
                                   exclude_aug=args.no_aug_samples)
    all_labels  = [s[1] for s in all_samples]

    # ── Split por grupo (source_sha256) para evitar data leakage ──────────
    #
    # Todas las augmentaciones de una misma imagen original comparten el
    # mismo source_sha256.  Agrupamos los índices por (sha256, clase) y
    # hacemos el split a nivel de grupo, garantizando que original + todas
    # sus augmentaciones caigan siempre en la misma partición.
    groups: dict[tuple[str, int], list[int]] = defaultdict(list)
    for i, (path, label) in enumerate(all_samples):
        # Si no hay manifest, usamos el stem del archivo como clave
        group_key = sha_map.get(path.name, path.stem)
        groups[(group_key, label)].append(i)

    group_keys   = list(groups.keys())
    group_labels = [k[1] for k in group_keys]

    gk_train_val, gk_test = train_test_split(
        group_keys, test_size=args.test_split,
        stratify=group_labels, random_state=args.seed
    )
    gl_train_val = [k[1] for k in gk_train_val]
    val_rel      = args.val_split / (1.0 - args.test_split)
    gk_train, gk_val = train_test_split(
        gk_train_val, test_size=val_rel,
        stratify=gl_train_val, random_state=args.seed
    )

    idx_train = [i for gk in gk_train for i in groups[gk]]
    idx_val   = [i for gk in gk_val   for i in groups[gk]]
    idx_test  = [i for gk in gk_test  for i in groups[gk]]

    train_labels = [all_labels[i] for i in idx_train]

    print(f"\n  Split -> train: {len(idx_train)} | "
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
                               aug_names, args.no_aug_samples)
    val_ds   = MelanomaDataset(dataset_dir, idx_val,   VAL_TRANSFORMS,
                               aug_names, args.no_aug_samples)
    test_ds  = MelanomaDataset(dataset_dir, idx_test,  VAL_TRANSFORMS,
                               aug_names, args.no_aug_samples)

    sampler = make_sampler(train_labels)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              sampler=sampler, num_workers=args.workers,
                              pin_memory=(device.type == "cuda"),
                              persistent_workers=(args.workers > 0))
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size,
                              shuffle=False, num_workers=args.workers,
                              pin_memory=(device.type == "cuda"),
                              persistent_workers=(args.workers > 0))
    test_loader  = DataLoader(test_ds, batch_size=args.batch_size,
                              shuffle=False, num_workers=args.workers,
                              pin_memory=(device.type == "cuda"),
                              persistent_workers=(args.workers > 0))

    # ── Modelo ────────────────────────────────────────────────────────────
    model = build_model(args.model, NUM_CLASSES, freeze_backbone=True)
    model = model.to(device)

    # ── Pérdida con pesos de clase + label smoothing ───────────────────────
    class_weights = compute_class_weights(dataset_dir, aug_names,
                                          args.no_aug_samples).to(device)
    class_weights[CLASSES.index("No melanoma")] *= args.no_melanoma_weight
    criterion     = nn.CrossEntropyLoss(weight=class_weights,
                                        label_smoothing=0.1)

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

    # ── Historial de métricas ──────────────────────────────────────────────
    history = {"train_loss": [], "train_acc": [],
               "val_loss":   [], "val_acc":   [], "val_f1": []}

    # ── Early stopping ─────────────────────────────────────────────────────
    patience_counter = 0

    print(f"\n{'─'*66}")
    backbone_unfrozen = (args.resume is not None and
                         start_epoch > args.unfreeze_epoch)

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()

        if epoch == args.unfreeze_epoch and not backbone_unfrozen:
            print(f"\n  [Época {epoch}] Descongelando backbone — "
                  f"backbone lr={args.lr_finetune}, "
                  f"head lr={args.lr_finetune * 5:.1e}")
            unfreeze_backbone(model)
            # Differential LR: backbone con lr bajo, cabeza con lr más alto
            if args.model.startswith("efficientnet"):
                head_module = model.classifier
            else:
                head_module = model.fc
            head_ids = {id(p) for p in head_module.parameters()}
            backbone_params = [p for p in model.parameters()
                               if id(p) not in head_ids]
            head_params_ft  = [p for p in head_module.parameters()]
            optimizer = optim.AdamW([
                {"params": backbone_params, "lr": args.lr_finetune},
                {"params": head_params_ft,  "lr": args.lr_finetune * 5},
            ], weight_decay=1e-4)
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

        val_f1  = f1_score(val_labels, val_preds,
                           average="macro", zero_division=0)
        elapsed = time.time() - t0
        lr_now  = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_f1"].append(val_f1)

        print(f"  Época {epoch:>3}/{args.epochs}  "
              f"train_loss={train_loss:.4f}  train_acc={train_acc:.4f}  "
              f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}  "
              f"val_f1={val_f1:.4f}  lr={lr_now:.2e}  "
              f"({elapsed:.0f}s)")

        # Guardar gráfica cada 5 épocas (evita I/O en cada época)
        if epoch % 5 == 0 or epoch == args.epochs:
            plot_training(history, unfreeze_epoch=args.unfreeze_epoch)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
            save_checkpoint(
                {
                    "epoch":       epoch,
                    "model":       model.state_dict(),
                    "optimizer":   optimizer.state_dict(),
                    "scheduler":   scheduler.state_dict(),
                    "best_val_f1": best_val_f1,
                    "classes":     CLASSES,
                    "model_name":  args.model,
                },
                CHECKPOINT_DIR / "best_f1.pt",
            )
            print(f"    ✓ Nuevo mejor modelo guardado "
                  f"(val_f1={best_val_f1:.4f})")
        else:
            patience_counter += 1

        if epoch % 10 == 0:
            save_checkpoint(
                {
                    "epoch":       epoch,
                    "model":       model.state_dict(),
                    "optimizer":   optimizer.state_dict(),
                    "scheduler":   scheduler.state_dict(),
                    "best_val_f1": best_val_f1,
                    "classes":     CLASSES,
                    "model_name":  args.model,
                },
                CHECKPOINT_DIR / f"epoch_{epoch:03d}.pt",
            )

        # Early stopping
        if args.early_stop > 0 and patience_counter >= args.early_stop:
            print(f"\n  [Early stopping] Sin mejora en {patience_counter} épocas. "
                  f"Deteniendo en época {epoch}.")
            plot_training(history, unfreeze_epoch=args.unfreeze_epoch)
            break

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

    plot_confusion_matrix(cm, CLASSES)

    print(f"\n  Checkpoints guardados en : {CHECKPOINT_DIR}/")
    print(f"  Gráficas guardadas en    : {PLOTS_DIR}/")
    print(f"  Mejor val_f1 global      : {best_val_f1:.4f}")
    print(f"{'='*66}\n")


if __name__ == "__main__":
    main()
