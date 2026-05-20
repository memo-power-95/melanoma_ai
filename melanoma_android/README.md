# Instrucciones para usar el proyecto Android

## Paso 1 — Exportar el modelo como TorchScript Lite

Desde la raíz del repositorio melanoma_ai:

```bash
pip install torch torchvision
python export_model.py
```

Esto genera: `checkpoints/melanoma_model.ptl`

## Paso 2 — Copiar el modelo a los assets de Android

```bash
copy checkpoints\melanoma_model.ptl melanoma_android\app\src\main\assets\melanoma_model.ptl
```

(Linux/macOS: `cp checkpoints/melanoma_model.ptl melanoma_android/app/src/main/assets/`)

## Paso 3 — Abrir en Android Studio

1. Abre Android Studio → **Open** → selecciona la carpeta `melanoma_android/`
2. Espera a que Gradle sincronice
3. Conecta un dispositivo Android (API 26+) o usa el emulador
4. Pulsa **Run ▶**

## Arquitectura del modelo

| Parámetro      | Valor                          |
|---------------|-------------------------------|
| Arquitectura  | EfficientNet-B3               |
| Clases        | 7 tipos de melanoma           |
| Entrada       | 224 × 224 RGB                 |
| Normalización | ImageNet (mean/std)           |
| Checkpoint    | `checkpoints/best_f1.pt`      |
| Exportado     | `checkpoints/melanoma_model.ptl` |

## Clases

| Índice | Nombre                 | Riesgo      |
|--------|------------------------|-------------|
| 0      | Extensión Superficial  | Alto        |
| 1      | Lentiginoso Acral      | Alto        |
| 2      | Lentigo Maligno        | Alto        |
| 3      | Nodular                | Alto        |
| 4      | Mucosas                | Alto        |
| 5      | Oculares               | Alto        |
| 6      | No melanoma            | Bajo        |

## Estructura del proyecto Android

```
melanoma_android/
├── app/
│   ├── build.gradle
│   ├── proguard-rules.pro
│   └── src/main/
│       ├── AndroidManifest.xml
│       ├── assets/
│       │   └── melanoma_model.ptl      ← copiar aquí
│       ├── java/com/melanoma/ai/
│       │   ├── MainActivity.kt         ← pantalla principal (galería + cámara)
│       │   ├── CameraActivity.kt       ← visor en tiempo real
│       │   └── ImageClassifier.kt      ← inferencia con PyTorch Mobile
│       └── res/
│           ├── layout/
│           │   ├── activity_main.xml
│           │   └── activity_camera.xml
│           └── values/
│               ├── strings.xml
│               ├── colors.xml
│               └── themes.xml
├── build.gradle
├── settings.gradle
├── gradle.properties
└── gradle/
    ├── libs.versions.toml
    └── wrapper/gradle-wrapper.properties
```

## Dependencias principales

- `org.pytorch:pytorch_android_lite:2.1.0` — inferencia móvil
- `org.pytorch:pytorch_android_torchvision:2.1.0` — utils de imagen
- `androidx.camera:camera-*:1.3.3` — CameraX
- `kotlinx-coroutines-android:1.8.1` — ejecución asíncrona
