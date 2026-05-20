"""
export_model.py — Exporta el checkpoint best_f1.pt a TorchScript Lite (.ptl)
para su uso con PyTorch Mobile en Android.

Uso:
    python export_model.py
    python export_model.py --model efficientnet_b3 --checkpoint checkpoints/best_f1.pt

Salida:
    checkpoints/melanoma_model.ptl   (listo para copiar a Android assets/)
"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models
from torch.utils.mobile_optimizer import optimize_for_mobile

# ── Clases (mismo orden que en train.py) ──────────────────────────────────────
CLASSES = [
    "Extensión Superficial",
    "Lentiginoso Acral",
    "Lentigo Maligno",
    "Nodular",
    "Mucosas",
    "Oculares",
    "No melanoma",
]
NUM_CLASSES = len(CLASSES)

AVAILABLE_MODELS = {
    "efficientnet_b0": (models.efficientnet_b0, models.EfficientNet_B0_Weights.IMAGENET1K_V1),
    "efficientnet_b3": (models.efficientnet_b3, models.EfficientNet_B3_Weights.IMAGENET1K_V1),
    "efficientnet_b4": (models.efficientnet_b4, models.EfficientNet_B4_Weights.IMAGENET1K_V1),
    "resnet50":        (models.resnet50,         models.ResNet50_Weights.IMAGENET1K_V2),
    "resnet101":       (models.resnet101,        models.ResNet101_Weights.IMAGENET1K_V2),
}


def build_model(name: str) -> nn.Module:
    fn, weights = AVAILABLE_MODELS[name]
    model = fn(weights=weights)

    if name.startswith("efficientnet"):
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.4, inplace=True),
            nn.Linear(in_features, NUM_CLASSES),
        )
    elif name.startswith("resnet"):
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(in_features, NUM_CLASSES),
        )
    return model


def load_checkpoint(model: nn.Module, checkpoint_path: Path, device: torch.device):
    state = torch.load(checkpoint_path, map_location=device)
    # Detectar automáticamente la clave del state_dict
    if isinstance(state, dict):
        for key in ("model", "model_state_dict", "state_dict"):
            if key in state:
                model.load_state_dict(state[key])
                epoch   = state.get("epoch", "?")
                best_f1 = state.get("best_val_f1", state.get("best_f1", "?"))
                print(f"  Checkpoint: época {epoch}, best_F1={best_f1}")
                return model
        # Si no tiene ninguna clave conocida, asumir que ES el state_dict directamente
        model.load_state_dict(state)
    else:
        model.load_state_dict(state)
    return model


def export(model_name: str, checkpoint_path: Path, output_path: Path):
    device = torch.device("cpu")   # Mobile siempre en CPU

    print(f"[1/4] Construyendo modelo: {model_name}")
    model = build_model(model_name)

    print(f"[2/4] Cargando checkpoint: {checkpoint_path}")
    model = load_checkpoint(model, checkpoint_path, device)
    model.eval()
    model.to(device)

    print("[3/4] Generando TorchScript con torch.jit.trace ...")
    example_input = torch.rand(1, 3, 224, 224)
    with torch.no_grad():
        traced = torch.jit.trace(model, example_input)

    print("[4/4] Optimizando para móvil y guardando .ptl ...")
    optimized = optimize_for_mobile(traced)
    optimized._save_for_lite_interpreter(str(output_path))

    print(f"\n✓ Modelo exportado a: {output_path}")
    size_mb = output_path.stat().st_size / 1_048_576
    print(f"  Tamaño: {size_mb:.1f} MB")
    print("\n  Copia este archivo a:  melanoma_android/app/src/main/assets/melanoma_model.ptl")


# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Exporta modelo a TorchScript Lite para Android")
    parser.add_argument("--model",      default="efficientnet_b3",
                        choices=list(AVAILABLE_MODELS), help="Arquitectura del modelo")
    parser.add_argument("--checkpoint", default="checkpoints/best_f1.pt",
                        help="Ruta al checkpoint .pt")
    parser.add_argument("--output",     default="checkpoints/melanoma_model.ptl",
                        help="Ruta de salida del archivo .ptl")
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint)
    output     = Path(args.output)

    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint no encontrado: {checkpoint}")

    output.parent.mkdir(parents=True, exist_ok=True)
    export(args.model, checkpoint, output)


if __name__ == "__main__":
    main()
