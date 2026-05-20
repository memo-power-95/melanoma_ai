"""
clean_dataset.py — Limpieza del dataset_aumentado

Operaciones:
  1. Detectar (y opcionalmente eliminar) imágenes corruptas que PIL no puede abrir.
  2. Detectar near-duplicates mediante hashing perceptual (imagehash):
       - Mismo clase, distancia == 0  →  elimina el duplicado (prefiere conservar
         el original sobre aug_; si son iguales conserva el de nombre más corto).
       - Clases distintas, distancia <= threshold  →  reporta en
         cross_class_duplicates.csv para revisión manual (NO elimina solo).
  3. Guardar resumen en clean_report.json.

Requisitos:
    pip install imagehash Pillow

Uso:
    python clean_dataset.py                        # dry-run (solo informa)
    python clean_dataset.py --apply                # aplica los cambios
    python clean_dataset.py --apply --threshold 6  # umbral hash más estricto
    python clean_dataset.py --dataset otra/ruta
"""

import os
import sys
import csv
import json
import argparse
from pathlib import Path
from collections import defaultdict

from PIL import Image, UnidentifiedImageError

try:
    import imagehash
    HAS_IMAGEHASH = True
except ImportError:
    HAS_IMAGEHASH = False

# ──────────────────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).parent
DATASET_DIR = ROOT / "dataset_aumentado"
VALID_EXTS  = {".jpg", ".jpeg", ".png"}

CLASS_FOLDER_MAP = {
    "Extensión Superficial": "Extensión Superficial",
    "Lentiginoso Acral":     "Lentiginoso Acral",
    "Lentigo Maligno":       "Lentigo Maligno",
    "Nodular":               "Nodular",
    "Mucosas":               os.path.join("Otros", "Mucosas"),
    "Oculares":              os.path.join("Otros", "Oculares"),
    "No melanoma":           "No melanoma",
}


# ──────────────────────────────────────────────────────────────────────────────
# Paso 1 — imágenes corruptas
# ──────────────────────────────────────────────────────────────────────────────

def scan_corrupt(dataset_dir: Path) -> list[Path]:
    """Retorna lista de rutas que PIL no puede abrir."""
    corrupt = []
    for cls_name, rel in CLASS_FOLDER_MAP.items():
        folder = dataset_dir / rel
        if not folder.exists():
            continue
        for f in sorted(folder.iterdir()):
            if f.suffix.lower() not in VALID_EXTS:
                continue
            try:
                with Image.open(f) as img:
                    img.verify()          # verifica integridad sin cargar pixeles
            except Exception:
                corrupt.append(f)
    return corrupt


# ──────────────────────────────────────────────────────────────────────────────
# Paso 2 — near-duplicates con pHash
# ──────────────────────────────────────────────────────────────────────────────

def _phash(path: Path):
    try:
        with Image.open(path).convert("RGB") as img:
            return imagehash.phash(img)
    except Exception:
        return None


def _keep_candidate(a: Path, b: Path) -> Path:
    """
    Entre dos duplicados decide cuál conservar:
    - Prefiere el que NO comienza con 'aug_'
    - En empate prefiere el de nombre más corto (más limpio/original)
    """
    a_aug = a.name.startswith("aug_")
    b_aug = b.name.startswith("aug_")
    if a_aug and not b_aug:
        return b
    if b_aug and not a_aug:
        return a
    return a if len(a.name) <= len(b.name) else b


def scan_duplicates(
    dataset_dir: Path,
    threshold: int = 8,
    skip_paths: set[Path] | None = None,
) -> tuple[list[Path], list[dict]]:
    """
    Calcula pHash para todas las imágenes válidas y detecta:
      - within_deletes : duplicados exactos (distancia=0) dentro de la misma clase.
                         Solo uno del par se elimina.
      - cross_report   : pares con distancia <= threshold en clases distintas
                         (para revisión manual).

    Retorna (within_deletes, cross_report).
    """
    if not HAS_IMAGEHASH:
        print("[AVISO] imagehash no instalado — omitiendo detección de duplicados.")
        print("        Instala con: pip install imagehash")
        return [], []

    skip = skip_paths or set()

    print("  Calculando hashes perceptuales...")
    # hash_table: hash_str -> list of (path, class_name)
    hash_table: dict[str, list[tuple[Path, str]]] = defaultdict(list)
    total = 0
    for cls_name, rel in CLASS_FOLDER_MAP.items():
        folder = dataset_dir / rel
        if not folder.exists():
            continue
        files = [f for f in sorted(folder.iterdir())
                 if f.suffix.lower() in VALID_EXTS and f not in skip]
        for f in files:
            h = _phash(f)
            if h is not None:
                hash_table[str(h)].append((f, cls_name))
                total += 1

    print(f"  Hashes calculados: {total} imágenes")

    within_deletes: list[Path] = []
    cross_report:   list[dict] = []

    # Para near-duplicates entre hashes distintos usamos fuerza bruta
    # solo sobre hashes únicos (mucho más rápido que comparar todas las imágenes)
    all_hashes_info: list[tuple[str, Path, str]] = []  # (hash_str, path, cls)
    for h_str, entries in hash_table.items():
        # Duplicados exactos (mismo hash)
        if len(entries) > 1:
            same_cls: dict[str, list[Path]] = defaultdict(list)
            diff_cls: list[tuple[Path, str]] = entries
            for path, cls in entries:
                same_cls[cls].append(path)

            # Within class: eliminar todos excepto el candidato a conservar
            for cls, paths in same_cls.items():
                if len(paths) > 1:
                    best = paths[0]
                    for p in paths[1:]:
                        best = _keep_candidate(best, p)
                    for p in paths:
                        if p != best:
                            within_deletes.append(p)

            # Cross-class: siempre reportar (distancia 0 = mismo pixel)
            classes_present = {cls for _, cls in diff_cls}
            if len(classes_present) > 1:
                paths_by_cls = defaultdict(list)
                for path, cls in diff_cls:
                    paths_by_cls[cls].append(str(path))
                cross_report.append({
                    "hash": h_str,
                    "distance": 0,
                    "classes": dict(paths_by_cls),
                })

        all_hashes_info.append((h_str, entries[0][0], entries[0][1]))

    # Near-duplicates (distancia 1..threshold) entre hashes distintos
    # Solo comparamos hashes únicos para eficiencia O(H²) donde H = hashes únicos
    unique_hashes = [(imagehash.hex_to_hash(h), info)
                     for h, *info in [
                         (h_str, path, cls)
                         for h_str, entries in hash_table.items()
                         for path, cls in [entries[0]]  # un representante por hash
                     ]]

    compared = 0
    for i in range(len(unique_hashes)):
        for j in range(i + 1, len(unique_hashes)):
            h_i, (path_i, cls_i) = unique_hashes[i]
            h_j, (path_j, cls_j) = unique_hashes[j]
            dist = h_i - h_j
            if 0 < dist <= threshold and cls_i != cls_j:
                cross_report.append({
                    "hash_a": str(h_i),
                    "hash_b": str(h_j),
                    "distance": dist,
                    "classes": {
                        cls_i: [str(path_i)],
                        cls_j: [str(path_j)],
                    },
                })
            compared += 1

    print(f"  Pares comparados (near-dup cross-class): {compared:,}")
    return within_deletes, cross_report


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Limpieza del dataset_aumentado: corruptas + duplicados."
    )
    p.add_argument("--dataset",   default=str(DATASET_DIR),
                   help="Ruta al dataset (default: dataset_aumentado/)")
    p.add_argument("--apply",     action="store_true",
                   help="Aplicar cambios (eliminar archivos). Sin este flag "
                        "es dry-run.")
    p.add_argument("--threshold", type=int, default=8,
                   help="Umbral de distancia pHash para cross-class near-dup "
                        "(default: 8). 0 = solo exactos.")
    p.add_argument("--no-hash",   action="store_true",
                   help="Omitir detección de duplicados (solo corruptas).")
    return p.parse_args()


def main():
    args        = parse_args()
    dataset_dir = Path(args.dataset)

    if not dataset_dir.exists():
        print(f"[ERROR] Dataset no encontrado: {dataset_dir}")
        sys.exit(1)

    mode = "APLICANDO CAMBIOS" if args.apply else "DRY-RUN (sin cambios reales)"
    print("=" * 66)
    print(f"  LIMPIEZA DATASET — {mode}")
    print("=" * 66)
    print(f"  Dataset   : {dataset_dir}")
    print(f"  Threshold : {args.threshold}")

    # ── Paso 1: imágenes corruptas ────────────────────────────────────────
    print("\n[1/2] Escaneando imágenes corruptas...")
    corrupt = scan_corrupt(dataset_dir)
    print(f"  Corruptas encontradas: {len(corrupt)}")
    for p in corrupt:
        print(f"    - {p.relative_to(dataset_dir)}")

    if args.apply and corrupt:
        for p in corrupt:
            p.unlink()
        print(f"  ✓ {len(corrupt)} imágenes corruptas eliminadas.")

    # ── Paso 2: near-duplicates ───────────────────────────────────────────
    within_deletes: list[Path] = []
    cross_report:   list[dict] = []

    if not args.no_hash:
        print("\n[2/2] Escaneando near-duplicates (pHash)...")
        within_deletes, cross_report = scan_duplicates(
            dataset_dir,
            threshold=args.threshold,
            skip_paths=set(corrupt),
        )
        print(f"  Duplicados exactos dentro de clase (a eliminar): "
              f"{len(within_deletes)}")
        for p in within_deletes:
            print(f"    - {p.relative_to(dataset_dir)}")

        print(f"  Near-duplicates cross-clase (reportados): "
              f"{len(cross_report)}")

        if args.apply and within_deletes:
            for p in within_deletes:
                p.unlink()
            print(f"  ✓ {len(within_deletes)} duplicados dentro de clase eliminados.")

        # Guardar reporte cross-class
        report_path = ROOT / "cross_class_duplicates.csv"
        if cross_report:
            with open(report_path, "w", newline="", encoding="utf-8") as fh:
                writer = csv.writer(fh)
                writer.writerow(["distance", "classes_and_paths"])
                for entry in cross_report:
                    writer.writerow([
                        entry["distance"],
                        json.dumps(entry["classes"], ensure_ascii=False),
                    ])
            print(f"\n  Reporte cross-class guardado en: {report_path.name}")
            print("  Revisa manualmente estos pares — podrían ser etiquetas erróneas.")
    else:
        print("\n[2/2] Detección de duplicados omitida (--no-hash).")

    # ── Resumen ───────────────────────────────────────────────────────────
    total_removed = (len(corrupt) + len(within_deletes)) if args.apply else 0
    report = {
        "mode": "apply" if args.apply else "dry-run",
        "dataset": str(dataset_dir),
        "corrupt_found": len(corrupt),
        "corrupt_removed": len(corrupt) if args.apply else 0,
        "within_class_duplicates_found": len(within_deletes),
        "within_class_duplicates_removed": len(within_deletes) if args.apply else 0,
        "cross_class_near_duplicates_reported": len(cross_report),
        "total_removed": total_removed,
    }

    report_path = ROOT / "clean_report.json"
    with open(report_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)

    print(f"\n{'='*66}")
    print(f"  Corruptas     : {len(corrupt)}")
    print(f"  Dup. exactos  : {len(within_deletes)}")
    print(f"  Cross-clase   : {len(cross_report)}")
    print(f"  Eliminadas    : {total_removed}"
          + (" (dry-run)" if not args.apply else ""))
    print(f"  Reporte       : clean_report.json")
    print(f"{'='*66}")

    if not args.apply and (corrupt or within_deletes):
        print("\n  Ejecuta con --apply para aplicar los cambios.")


if __name__ == "__main__":
    main()
