"""
Construye dataset_aumentado (~3 500–5 000 imágenes) desde tres fuentes:

  [0] BASE       dataset_balanceado/ — las 1 200 imágenes ya existentes.
                 Se copian tal cual con prefijo  base_  para trazabilidad.

  [1] HAM10000   Kaggle: kmader/skin-lesion-analysis-toward-melanoma-detection
                 ~10 000 imágenes dermoscópicas; incluye 7 diagnósticos.

  [2] DermNet NZ Kaggle: shubhamgoel27/dermnet
                 Solo se usan categorías NO-melanoma → clase "No melanoma".
                 (La carpeta de melanoma se omite intencionalmente para evitar
                 mezclar nevus/melanoma sin etiquetado fino.)

  [3] ISIC API   https://api.isic-archive.com/api/v2/images/
                 Complemento con clasificación fina de subtipos de melanoma
                 (igual que download_isic_completo.py).

Requisitos:
    pip install requests kaggle Pillow

    Credenciales Kaggle (necesarias para fuentes 1 y 2):
      1. Ve a  https://www.kaggle.com/settings → Account → Create New API Token
      2. Guarda el archivo kaggle.json en:
           Windows : C:\\Users\\<tu_usuario>\\.kaggle\\kaggle.json
           Linux   : ~/.kaggle/kaggle.json
      3. Permisos  : chmod 600 ~/.kaggle/kaggle.json  (solo Linux/Mac)

Uso:
    python download_dataset_aumentado.py
    python download_dataset_aumentado.py --skip-ham
    python download_dataset_aumentado.py --skip-dermnet
    python download_dataset_aumentado.py --skip-isic
    python download_dataset_aumentado.py --no-cleanup   # conserva _temp_aumentado/

Estructura de salida:
    dataset_aumentado/
      Extensión Superficial/   ← base_*.jpg  |  ham_*.jpg  |  isic_*.jpg
      Lentiginoso Acral/       ← base_*.jpg  |  ham_*.jpg  |  isic_*.jpg
      Lentigo Maligno/         ← base_*.jpg  |  isic_*.jpg
      Nodular/                 ← base_*.jpg  |  isic_*.jpg
      Otros/
        Mucosas/               ← base_*.jpg  |  ham_*.jpg  |  isic_*.jpg
        Oculares/              ← base_*.jpg  |  isic_*.jpg
      No melanoma/             ← base_*.jpg  |  ham_*.jpg  |  dermnet_*.jpg  |  isic_*.jpg

Nomenclatura de archivos:
    base_{nombre_original}.jpg → origen dataset_balanceado (base)
    ham_{image_id}.jpg         → origen HAM10000
    dermnet_{cat}_{file}.jpg   → origen DermNet NZ
    isic_{isic_id}.jpg         → origen ISIC Archive API
"""

import os
import sys
import csv
import time
import shutil
import random
import argparse
import requests

BASE_DIR         = os.path.dirname(os.path.abspath(__file__))   # dataset_aumentado/
OUTPUT_DIR       = BASE_DIR                                       # las imágenes van aquí
TEMP_DIR         = os.path.join(BASE_DIR, "_temp")
BASE_DATASET_DIR = os.path.join(os.path.dirname(BASE_DIR), "dataset_balanceado")

# Mapeo de carpeta relativa en dataset_balanceado → nombre de clase
BASE_FOLDER_MAP = {
    "Extensión Superficial":      "Extensión Superficial",
    "Lentiginoso Acral":          "Lentiginoso Acral",
    "Lentigo Maligno":            "Lentigo Maligno",
    "Nodular":                    "Nodular",
    os.path.join("Otros", "Mucosas"):   "Mucosas",
    os.path.join("Otros", "Oculares"): "Oculares",
    "No melanoma":                "No melanoma",
}

# ─── Metas por clase ──────────────────────────────────────────────────────────
# Objetivo total ≈ 4 700 imágenes (rango 3 500–5 000)
TARGETS = {
    "Extensión Superficial": 850,   # HAM10000 mel (no-acral) + ISIC superficial
    "Lentiginoso Acral":     280,   # HAM10000 mel/acral + ISIC acral
    "Lentigo Maligno":       280,   # ISIC (raro en HAM10000)
    "Nodular":               390,   # ISIC
    "Mucosas":               150,   # ISIC + HAM10000 mel/genital
    "Oculares":              150,   # ISIC
    "No melanoma":          2600,   # HAM10000 benignos + DermNet + ISIC benign
}
# TOTAL = 4 700

# ─── Mapeo de carpetas de salida ──────────────────────────────────────────────
CLASS_PATHS = {
    "Extensión Superficial": "Extensión Superficial",
    "Lentiginoso Acral":     "Lentiginoso Acral",
    "Lentigo Maligno":       "Lentigo Maligno",
    "Nodular":               "Nodular",
    "Mucosas":               os.path.join("Otros", "Mucosas"),
    "Oculares":              os.path.join("Otros", "Oculares"),
    "No melanoma":           "No melanoma",
}

# ─── HAM10000: códigos de diagnóstico ─────────────────────────────────────────
HAM_BENIGN      = {"nv", "bcc", "akiec", "bkl", "df", "vasc"}
HAM_ACRAL_LOCS  = {"acral", "foot", "hand"}
HAM_MUCOSAL_LOCS = {"genital", "oral"}

# ─── DermNet NZ: categorías benignas (excluye carpeta de melanoma) ─────────────
DERMNET_BENIGN_LOWER = {
    "acne and rosacea photos",
    "actinic keratosis basal cell carcinoma and other malignant lesions",
    "atopic dermatitis photos",
    "bullous disease photos",
    "cellulitis impetigo and other bacterial infections",
    "eczema photos",
    "exanthems and drug eruptions",
    "hair loss photos alopecia and other hair diseases",
    "herpes hpv and other stds photos",
    "light diseases and disorders of pigmentation",
    "lupus and other connective tissue diseases",
    "nail fungus and other nail disease",
    "poison ivy photos and other contact dermatitis",
    "psoriasis pictures lichen planus and related diseases",
    "scabies lyme disease and other infestations and bites",
    "seborrheic keratoses and other benign tumors",
    "systemic disease",
    "tinea ringworm candidiasis and other fungal infections",
    "urticaria hives",
    "vascular tumors",
    "vasculitis photos",
    "warts molluscum and other viral infections",
}

# ─── ISIC API ─────────────────────────────────────────────────────────────────
ISIC_API_URL  = "https://api.isic-archive.com/api/v2/images/"

MUCOSAL_SITES = {"oral/genital", "oral", "genital", "vulvar", "mucosal",
                 "anorectal", "sinonasal", "esophagus"}
OCULAR_SITES  = {"eye", "conjunctiva", "eyelid", "orbital", "ocular",
                 "choroid", "uvea", "retina"}
MUCOSAL_DIAG  = {"mucosal", "oral", "vulvar", "anorectal", "sinonasal"}
OCULAR_DIAG   = {"uveal", "conjunctival", "ocular", "choroidal", "ophthalmic",
                 "intraocular", "eye"}


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def dest_dir(cls_name: str) -> str:
    return os.path.join(OUTPUT_DIR, CLASS_PATHS[cls_name])


def count_images(folder: str) -> int:
    valid = ('.jpg', '.jpeg', '.png')
    if not os.path.exists(folder):
        return 0
    return sum(1 for f in os.listdir(folder) if f.lower().endswith(valid))


def remaining(downloaded: dict, needed: dict) -> int:
    return sum(max(0, needed[c] - downloaded[c]) for c in needed)


def _contains_any(text: str, keywords: set) -> bool:
    t = text.lower()
    return any(k in t for k in keywords)


def copy_or_convert(src: str, dst: str) -> bool:
    """
    Copia imagen al destino.  Si el origen no es JPEG la convierte usando
    Pillow para garantizar que el archivo .jpg sea un JPEG válido.
    Retorna True si tuvo éxito.
    """
    try:
        if os.path.getsize(src) < 1_000:
            return False
        ext = os.path.splitext(src)[1].lower()
        if ext in ('.jpg', '.jpeg'):
            shutil.copy2(src, dst)
        else:
            from PIL import Image
            with Image.open(src) as img:
                img.convert('RGB').save(dst, 'JPEG', quality=95)
        return True
    except Exception:
        return False


def _kaggle_download(dataset_slug: str, dest_path: str) -> bool:
    """
    Descarga y descomprime un dataset de Kaggle en dest_path.
    Retorna True si fue exitoso.
    """
    import subprocess
    try:
        result = subprocess.run(
            ["kaggle", "datasets", "download",
             "-d", dataset_slug,
             "-p", dest_path, "--unzip"],
            capture_output=True, text=True, timeout=3600,
        )
        if result.returncode != 0:
            print(f"  [ERROR Kaggle] {result.stderr[:400]}")
            return False
        return True
    except FileNotFoundError:
        print("  [ERROR] kaggle CLI no encontrado. Instala con:  pip install kaggle")
        return False
    except subprocess.TimeoutExpired:
        print("  [TIMEOUT] La descarga superó 60 min. Inténtalo manualmente.")
        return False


# ══════════════════════════════════════════════════════════════════════════════
# Clasificadores
# ══════════════════════════════════════════════════════════════════════════════

def classify_ham(dx: str, localization: str):
    """
    Clasifica una entrada de HAM10000 según dx y localización anatómica.
    Retorna nombre de clase o None.
    """
    dx  = dx.strip().lower()
    loc = localization.strip().lower()

    if dx == "mel":
        if loc in HAM_ACRAL_LOCS:
            return "Lentiginoso Acral"
        if loc in HAM_MUCOSAL_LOCS:
            return "Mucosas"
        return "Extensión Superficial"

    if dx in HAM_BENIGN:
        return "No melanoma"

    return None


def classify_isic(item: dict):
    """
    Clasifica una entrada de la ISIC API v2.
    Retorna nombre de clase o None.
    (Lógica idéntica a download_isic_completo.py)
    """
    meta     = item.get("metadata", {})
    clinical = meta.get("clinical", {})

    diag1 = (clinical.get("diagnosis_1") or "").strip()
    diag3 = (clinical.get("diagnosis_3") or "").strip().lower()
    diag4 = (clinical.get("diagnosis_4") or "").strip().lower()

    site_gen  = (clinical.get("anatom_site_general") or "").strip().lower()
    site1     = (clinical.get("anatom_site_1") or "").strip().lower()
    site2     = (clinical.get("anatom_site_2") or "").strip().lower()
    all_sites = f"{site_gen} {site1} {site2}"

    if diag1 == "Malignant":
        if _contains_any(all_sites, OCULAR_SITES) or \
                _contains_any(diag3 + " " + diag4, OCULAR_DIAG):
            return "Oculares"
        if _contains_any(all_sites, MUCOSAL_SITES) or \
                _contains_any(diag3 + " " + diag4, MUCOSAL_DIAG):
            return "Mucosas"
        if "nodular" in diag4:
            return "Nodular"
        if "acral" in diag4:
            return "Lentiginoso Acral"
        if "lentigo" in diag4 or "lentigo maligna" in diag3:
            return "Lentigo Maligno"
        if "superficial" in diag4 or "superficial spreading" in diag3:
            return "Extensión Superficial"
        if "melanoma" in diag3:
            return "Extensión Superficial"
        return None

    if diag1 == "Benign":
        return "No melanoma"

    return None


# ══════════════════════════════════════════════════════════════════════════════
# FUENTE 0 — dataset_balanceado (base existente)
# ══════════════════════════════════════════════════════════════════════════════

def run_base_copy(downloaded: dict, needed: dict) -> dict:
    """
    Copia las imágenes de dataset_balanceado/ a dataset_aumentado/
    añadiendo el prefijo  base_  al nombre del archivo para identificar
    su origen.  Evita duplicar archivos que ya existan en el destino.
    """
    print("\n" + "─" * 62)
    print("  FUENTE 0 — dataset_balanceado (1 200 imágenes base)")
    print("─" * 62)

    if not os.path.isdir(BASE_DATASET_DIR):
        print(f"  [AVISO] No se encontró {BASE_DATASET_DIR}. Saltando.")
        return downloaded

    valid = ('.jpg', '.jpeg', '.png')
    copied  = 0
    skipped = 0

    for rel_folder, cls in BASE_FOLDER_MAP.items():
        src_folder = os.path.join(BASE_DATASET_DIR, rel_folder)
        if not os.path.isdir(src_folder):
            continue

        for fname in sorted(os.listdir(src_folder)):
            if not fname.lower().endswith(valid):
                continue

            src = os.path.join(src_folder, fname)
            stem = os.path.splitext(fname)[0]
            new_fname = f"base_{stem}.jpg"
            dst = os.path.join(dest_dir(cls), new_fname)

            if os.path.exists(dst):
                downloaded[cls] += 1
                skipped += 1
                continue

            if copy_or_convert(src, dst):
                downloaded[cls] += 1
                copied += 1

    print(f"  Copiadas  : {copied}")
    print(f"  Ya existían: {skipped}")
    for cls in TARGETS:
        print(f"    {cls:<26} → {downloaded[cls]} copiadas")
    return downloaded


# ══════════════════════════════════════════════════════════════════════════════
# FUENTE 1 — HAM10000
# ══════════════════════════════════════════════════════════════════════════════

def run_ham10000(downloaded: dict, needed: dict) -> dict:
    """
    Descarga HAM10000 vía Kaggle y copia las imágenes clasificadas al dataset.

    Si la descarga falla, el script imprime instrucciones para descarga manual:
      1. Descargar desde https://www.kaggle.com/datasets/kmader/skin-lesion-analysis-toward-melanoma-detection
      2. Extraer en  _temp_aumentado/ham10000/
      3. Relanzar el script con  --skip-ham  si aún falla
    """
    print("\n" + "─" * 62)
    print("  FUENTE 1 — HAM10000")
    print("─" * 62)

    ham_dir = os.path.join(TEMP_DIR, "ham10000")

    # ── Descarga si no existe ya ────────────────────────────────────────────
    has_csv = any(
        f.lower().endswith('.csv') and 'metadata' in f.lower()
        for _, _, files in os.walk(ham_dir) for f in files
    ) if os.path.exists(ham_dir) else False

    if not has_csv:
        print("  Descargando HAM10000 desde Kaggle (~3 GB)...")
        print("  Dataset: kmader/skin-lesion-analysis-toward-melanoma-detection")
        os.makedirs(ham_dir, exist_ok=True)
        if not _kaggle_download(
            "kmader/skin-lesion-analysis-toward-melanoma-detection", ham_dir
        ):
            print("\n  ── Descarga manual ──────────────────────────────────────")
            print("  1. Descarga desde:")
            print("     https://www.kaggle.com/datasets/kmader/"
                  "skin-lesion-analysis-toward-melanoma-detection")
            print("  2. Extrae el contenido en:")
            print(f"     {ham_dir}")
            print("  3. Relanza el script (detectará los archivos automáticamente).")
            print("  ─" * 31)
            return downloaded
    else:
        print(f"  Usando datos existentes en: {ham_dir}")

    # ── Localizar CSV de metadatos ──────────────────────────────────────────
    metadata_path = None
    for root, _, files in os.walk(ham_dir):
        for f in files:
            if f.lower().endswith('.csv') and 'metadata' in f.lower():
                metadata_path = os.path.join(root, f)
                break
        if metadata_path:
            break

    if not metadata_path:
        print("  [ERROR] No se encontró HAM10000_metadata.csv")
        return downloaded

    # ── Indexar todas las imágenes del directorio ───────────────────────────
    print("  Indexando imágenes HAM10000...")
    img_lookup: dict[str, str] = {}
    for root, _, files in os.walk(ham_dir):
        for f in files:
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                stem = os.path.splitext(f)[0]
                img_lookup[stem] = os.path.join(root, f)

    print(f"  {len(img_lookup)} imágenes encontradas")

    # ── Procesar metadatos y copiar ─────────────────────────────────────────
    copied  = 0
    skipped = 0

    with open(metadata_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if remaining(downloaded, needed) == 0:
                break

            image_id = row.get('image_id', '').strip()
            dx       = row.get('dx',           '').strip()
            loc      = row.get('localization',  '').strip()

            cls = classify_ham(dx, loc)
            if cls is None:
                skipped += 1
                continue

            if downloaded[cls] >= needed[cls]:
                continue

            src = img_lookup.get(image_id)
            if not src:
                continue

            filename = f"ham_{image_id}.jpg"
            dst      = os.path.join(dest_dir(cls), filename)

            if os.path.exists(dst):
                downloaded[cls] += 1
                continue

            if copy_or_convert(src, dst):
                downloaded[cls] += 1
                copied += 1
                print(
                    f"  [HAM/{cls:<22}] {image_id}"
                    f"  ({downloaded[cls]}/{needed[cls]})"
                    f"  restante total: {remaining(downloaded, needed)}"
                )

    print(f"\n  HAM10000 completado: {copied} imágenes copiadas"
          f" | {skipped} filas sin clase")
    return downloaded


# ══════════════════════════════════════════════════════════════════════════════
# FUENTE 2 — DermNet NZ
# ══════════════════════════════════════════════════════════════════════════════

def run_dermnet(downloaded: dict, needed: dict) -> dict:
    """
    Descarga DermNet NZ vía Kaggle y usa solo las categorías benignas para
    enriquecer la clase "No melanoma".

    La carpeta "Melanoma Skin Cancer Nevi and Moles" se omite intencionalmente
    porque mezcla nevus (benigno) con melanoma (maligno) sin etiquetado fino,
    lo que podría contaminar ambas clases.

    Si la descarga falla:
      1. Descarga desde https://www.kaggle.com/datasets/shubhamgoel27/dermnet
      2. Extrae en  _temp_aumentado/dermnet/
    """
    print("\n" + "─" * 62)
    print("  FUENTE 2 — DermNet NZ (solo No melanoma)")
    print("─" * 62)

    if downloaded["No melanoma"] >= needed["No melanoma"]:
        print("  Clase 'No melanoma' ya alcanzó el objetivo. Saltando DermNet.")
        return downloaded

    dermnet_dir = os.path.join(TEMP_DIR, "dermnet")
    train_dir   = os.path.join(dermnet_dir, "train")

    # ── Descarga si no existe ya ────────────────────────────────────────────
    if not os.path.isdir(train_dir):
        print("  Descargando DermNet desde Kaggle (~1.7 GB)...")
        print("  Dataset: shubhamgoel27/dermnet")
        os.makedirs(dermnet_dir, exist_ok=True)
        if not _kaggle_download("shubhamgoel27/dermnet", dermnet_dir):
            print("\n  ── Descarga manual ──────────────────────────────────────")
            print("  1. Descarga desde:")
            print("     https://www.kaggle.com/datasets/shubhamgoel27/dermnet")
            print("  2. Extrae en:")
            print(f"     {dermnet_dir}")
            print("  ─" * 31)
            return downloaded
    else:
        print(f"  Usando datos existentes en: {dermnet_dir}")

    # ── Recorrer splits train/ y test/ ─────────────────────────────────────
    copied   = 0
    skipped  = 0

    for split in ("train", "test"):
        split_path = os.path.join(dermnet_dir, split)
        if not os.path.isdir(split_path):
            continue

        for category in sorted(os.listdir(split_path)):
            if downloaded["No melanoma"] >= needed["No melanoma"]:
                break

            cat_path = os.path.join(split_path, category)
            if not os.path.isdir(cat_path):
                continue

            # Verificar si la categoría es benigna/no-melanoma
            cat_lower = category.strip().lower()
            if cat_lower not in DERMNET_BENIGN_LOWER:
                # Búsqueda parcial tolerante
                matched = any(bc in cat_lower or cat_lower in bc
                              for bc in DERMNET_BENIGN_LOWER)
                if not matched:
                    skipped += 1
                    continue

            # Slug corto para el nombre del archivo (máx 25 chars)
            cat_slug = (category.strip()
                        .replace(' ', '_')
                        .replace('/', '-')
                        .replace('\\', '-'))[:25]

            for fname in sorted(os.listdir(cat_path)):
                if downloaded["No melanoma"] >= needed["No melanoma"]:
                    break

                ext = os.path.splitext(fname)[1].lower()
                if ext not in ('.jpg', '.jpeg', '.png'):
                    continue

                src = os.path.join(cat_path, fname)

                # Nombre destino: dermnet_{split[0]}_{cat_slug}_{fname}.jpg
                safe_fname = os.path.splitext(fname)[0].replace(' ', '_')
                new_fname  = f"dermnet_{split[0]}_{cat_slug}_{safe_fname}.jpg"
                dst        = os.path.join(dest_dir("No melanoma"), new_fname)

                if os.path.exists(dst):
                    downloaded["No melanoma"] += 1
                    continue

                if copy_or_convert(src, dst):
                    downloaded["No melanoma"] += 1
                    copied += 1
                    if copied % 200 == 0 or copied <= 5:
                        print(
                            f"  [DERMNET/No melanoma] {new_fname}"
                            f"  ({downloaded['No melanoma']}/{needed['No melanoma']})"
                        )

    print(f"\n  DermNet completado: {copied} imágenes copiadas"
          f" | {skipped} categorías omitidas (melanoma/desconocida)")
    return downloaded


# ══════════════════════════════════════════════════════════════════════════════
# FUENTE 3 — ISIC Archive API
# ══════════════════════════════════════════════════════════════════════════════

def run_isic(downloaded: dict, needed: dict) -> dict:
    """
    Descarga imágenes desde la ISIC Archive API v2 con clasificación fina
    de subtipos de melanoma.  Complementa especialmente Lentigo Maligno,
    Nodular y Oculares, que HAM10000 y DermNet no cubren bien.

    Evita descargar imágenes que ya existan (prefijo isic_) en cualquier
    clase del dataset_aumentado.
    """
    print("\n" + "─" * 62)
    print("  FUENTE 3 — ISIC Archive API")
    print("─" * 62)

    if remaining(downloaded, needed) == 0:
        print("  Todas las clases alcanzaron su objetivo. Saltando ISIC.")
        return downloaded

    # Construir set de IDs isic ya presentes (evitar duplicados)
    existing_isic: set[str] = set()
    for cls in TARGETS:
        d = dest_dir(cls)
        if os.path.exists(d):
            for f in os.listdir(d):
                if f.startswith("isic_") and f.lower().endswith('.jpg'):
                    existing_isic.add(os.path.splitext(f)[0])  # "isic_ISIC_xxxxx"

    url       = ISIC_API_URL
    params    = {"limit": 100}
    pages     = 0
    max_pages = 3_000
    errors    = 0
    copied    = 0

    while url and pages < max_pages:
        if remaining(downloaded, needed) == 0:
            break

        try:
            resp = requests.get(url, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException as e:
            print(f"  [ERROR API] {e} — reintentando en 10 s...")
            time.sleep(10)
            pages += 1
            continue
        except ValueError:
            print("  [ERROR] Respuesta no JSON. Saltando página.")
            pages += 1
            continue

        results = data.get("results", [])
        if not results:
            break

        for item in results:
            if remaining(downloaded, needed) == 0:
                break

            cls = classify_isic(item)
            if cls is None:
                continue

            if downloaded[cls] >= needed[cls]:
                continue

            isic_id = item.get("isic_id", "")
            if not isic_id:
                continue

            file_key = f"isic_{isic_id}"
            if file_key in existing_isic:
                continue

            filename = f"{file_key}.jpg"
            filepath = os.path.join(dest_dir(cls), filename)

            if os.path.exists(filepath):
                downloaded[cls] += 1
                existing_isic.add(file_key)
                continue

            img_url = (item.get("files", {})
                       .get("full", {}).get("url", ""))
            if not img_url:
                img_url = (item.get("files", {})
                           .get("thumbnail_256", {}).get("url", ""))
            if not img_url:
                continue

            try:
                r = requests.get(img_url, timeout=30)
                r.raise_for_status()
                if len(r.content) < 1_000:
                    continue
                with open(filepath, "wb") as f:
                    f.write(r.content)
                downloaded[cls] += 1
                existing_isic.add(file_key)
                copied += 1
                kb  = os.path.getsize(filepath) // 1024
                rem = remaining(downloaded, needed)
                print(
                    f"  [ISIC/{cls:<22}] {isic_id}  {kb:>4} KB"
                    f"  ({downloaded[cls]}/{needed[cls]})"
                    f"  restante: {rem}"
                )
                time.sleep(0.15)   # throttle suave
            except requests.RequestException:
                errors += 1

        url    = data.get("next")
        params = {}
        pages += 1

    print(f"\n  ISIC completado: {copied} imágenes descargadas"
          f" | {errors} errores | {pages} páginas revisadas")
    return downloaded


# ══════════════════════════════════════════════════════════════════════════════
# FUENTE 4 — Aumentación sintética con Pillow
# ══════════════════════════════════════════════════════════════════════════════

def _augment_image(img, rng: random.Random):
    """
    Aplica una combinación aleatoria de transformaciones a una imagen PIL.
    """
    from PIL import ImageEnhance

    # Rotación (múltiplos de 90° para no perder región útil)
    angle = rng.choice([0, 90, 180, 270])
    if angle:
        img = img.rotate(angle, expand=True)

    # Flip horizontal
    if rng.random() > 0.5:
        from PIL import Image as _Img
        img = img.transpose(_Img.FLIP_LEFT_RIGHT)

    # Flip vertical
    if rng.random() > 0.5:
        from PIL import Image as _Img
        img = img.transpose(_Img.FLIP_TOP_BOTTOM)

    # Brillo
    img = ImageEnhance.Brightness(img).enhance(rng.uniform(0.75, 1.25))

    # Contraste
    img = ImageEnhance.Contrast(img).enhance(rng.uniform(0.80, 1.20))

    # Saturación
    img = ImageEnhance.Color(img).enhance(rng.uniform(0.80, 1.20))

    return img


def run_augmentation(downloaded: dict, needed: dict) -> dict:
    """
    Fuente 4 — Aumentación sintética.
    Para cada clase que siga por debajo del objetivo tras todas las descargas,
    genera variantes de las imágenes existentes (rotación, flip, brillo,
    contraste, saturación) hasta alcanzar la meta.  Los archivos generados
    reciben el prefijo  aug_  para distinguirlos de imágenes reales.
    """
    print("\n" + "─" * 62)
    print("  FUENTE 4 — Aumentación sintética (Pillow)")
    print("─" * 62)

    try:
        from PIL import Image
    except ImportError:
        print("  [ERROR] Pillow no instalado.  pip install Pillow")
        return downloaded

    total_generated = 0

    for cls in TARGETS:
        # needed[cls] = cuántas faltan según conteo real en disco
        falta = needed[cls]
        if falta <= 0:
            continue

        folder = dest_dir(cls)
        # Solo imágenes reales (no aug) como fuente de variantes
        sources = sorted(
            f for f in os.listdir(folder)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
            and not f.startswith('aug_')
        )

        if not sources:
            print(f"  [{cls}] Sin imágenes base. Saltando.")
            continue

        target_total = TARGETS[cls]
        print(f"  [{cls:<26}] generando {falta} variantes "
              f"(base: {len(sources)} imágenes)...")

        generated = 0
        idx       = 0

        while generated < falta:
            src_name = sources[idx % len(sources)]
            src_path = os.path.join(folder, src_name)

            # Semilla determinista: cambia con idx para cada variante
            rng      = random.Random(idx * 7919 + hash(src_name) % 99991)
            stem     = os.path.splitext(src_name)[0][:30]
            aug_name = f"aug_{idx:05d}_{stem}.jpg"
            dst_path = os.path.join(folder, aug_name)

            if not os.path.exists(dst_path):
                try:
                    with Image.open(src_path) as img:
                        aug = _augment_image(img.convert('RGB'), rng)
                        aug.save(dst_path, 'JPEG', quality=92)
                    total_generated += 1
                except Exception:
                    idx += 1
                    continue

            generated += 1
            idx       += 1

            if generated % 100 == 0 or generated == falta:
                real_now = count_images(folder)
                print(f"    {generated}/{falta}  ({real_now}/{target_total})")

        print(f"  [{cls:<26}] +{generated} imágenes sintéticas")

    print(f"\n  Aumentación completada: {total_generated} imágenes generadas")
    return downloaded


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Construye dataset_aumentado desde HAM10000, DermNet e ISIC."
    )
    parser.add_argument("--skip-base",    action="store_true",
                        help="Omitir copia de dataset_balanceado")
    parser.add_argument("--skip-ham",     action="store_true",
                        help="Omitir fuente HAM10000 (Kaggle)")
    parser.add_argument("--skip-dermnet", action="store_true",
                        help="Omitir fuente DermNet NZ (Kaggle)")
    parser.add_argument("--skip-isic",    action="store_true",
                        help="Omitir fuente ISIC Archive API")
    parser.add_argument("--no-cleanup",   action="store_true",
                        help="Conservar carpeta _temp_aumentado/ tras finalizar")
    parser.add_argument("--skip-augment", action="store_true",
                        help="Omitir aumentación sintética (Pillow)")
    args = parser.parse_args()

    # ── Crear estructura de carpetas de salida ─────────────────────────────
    for cls in TARGETS:
        os.makedirs(dest_dir(cls), exist_ok=True)
    os.makedirs(TEMP_DIR, exist_ok=True)

    # ── Estado inicial ─────────────────────────────────────────────────────
    needed = {
        cls: max(0, TARGETS[cls] - count_images(dest_dir(cls)))
        for cls in TARGETS
    }
    downloaded = {cls: TARGETS[cls] - needed[cls] for cls in TARGETS}

    print("=" * 66)
    print("  DATASET AUMENTADO — CONSTRUCCIÓN MULTI-FUENTE")
    print("=" * 66)
    print(f"\n  {'Clase':<26} {'Tienes':>8} {'Objetivo':>9} {'Faltan':>8}")
    print(f"  {'-'*57}")
    for cls in TARGETS:
        have = count_images(dest_dir(cls))
        print(f"  {cls:<26} {have:>8} {TARGETS[cls]:>9} {needed[cls]:>8}")

    total_target  = sum(TARGETS.values())
    total_missing = sum(needed.values())
    print(f"\n  Total objetivo   : {total_target}")
    print(f"  Total por añadir : {total_missing}")

    if total_missing == 0:
        print("\n  ¡Dataset ya completo! Nada que descargar.")
        return

    print(f"\n  Carpeta de salida : {OUTPUT_DIR}")
    print(f"  Carpeta temporal  : {TEMP_DIR}")
    print()

    # ── Ejecutar fuentes en orden ──────────────────────────────────────────
    if not args.skip_base:
        downloaded = run_base_copy(downloaded, needed)
        # Recalcular needed tras copia base
        needed = {
            cls: max(0, TARGETS[cls] - count_images(dest_dir(cls)))
            for cls in TARGETS
        }

    if not args.skip_ham:
        downloaded = run_ham10000(downloaded, needed)

    if not args.skip_dermnet:
        downloaded = run_dermnet(downloaded, needed)

    if not args.skip_isic:
        downloaded = run_isic(downloaded, needed)

    if not args.skip_augment:
        # Recalcular needed con conteo real antes de aumentar
        needed = {
            cls: max(0, TARGETS[cls] - count_images(dest_dir(cls)))
            for cls in TARGETS
        }
        downloaded = run_augmentation(downloaded, needed)

    # ── Limpiar temporales ─────────────────────────────────────────────────
    if not args.no_cleanup and os.path.exists(TEMP_DIR):
        print(f"\n  Limpiando archivos temporales ({TEMP_DIR})...")
        shutil.rmtree(TEMP_DIR, ignore_errors=True)
        print("  Listo.")

    # ── Resumen final ──────────────────────────────────────────────────────
    print(f"\n{'='*66}")
    print(f"  RESUMEN FINAL — dataset_aumentado/")
    print(f"{'='*66}")
    print(f"\n  {'Clase':<26} {'Total':>8} {'Objetivo':>9}  {'Estado':>11}")
    print(f"  {'-'*60}")

    grand_total = 0
    for cls in TARGETS:
        total_now = count_images(dest_dir(cls))
        grand_total += total_now
        status = "✓ OK" if total_now >= TARGETS[cls] else "INCOMPLETO"
        print(f"  {cls:<26} {total_now:>8} {TARGETS[cls]:>9}  {status:>11}")

    print(f"\n  TOTAL: {grand_total} imágenes en dataset_aumentado/")
    if grand_total < 3500:
        print(f"  AVISO: No se alcanzó el mínimo de 3 500. Revisa los errores.")
    elif grand_total >= 5000:
        print(f"  NOTA : Se superó el máximo estimado de 5 000.")
    else:
        print(f"  Rango objetivo [3 500–5 000] alcanzado.")

    print(f"\n  Fuentes de imágenes:")
    print(f"    base_*    → dataset_balanceado (origen base)")
    print(f"    ham_*     → HAM10000 (Kaggle)")
    print(f"    dermnet_* → DermNet NZ (Kaggle)")
    print(f"    isic_*    → ISIC Archive API")
    print(f"    aug_*     → Aumentación sintética (Pillow)")
    print(f"{'='*66}\n")


if __name__ == "__main__":
    main()
