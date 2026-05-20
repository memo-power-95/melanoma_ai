# Contexto completo del proyecto — MelanomaAI Android

## 1. Descripción del proyecto

Aplicación Android que clasifica imágenes de lesiones cutáneas usando un modelo de deep learning entrenado localmente. El usuario toma una foto con la cámara o selecciona una de su galería, y la app devuelve el diagnóstico con nivel de riesgo y porcentaje de confianza.

---

## 2. Modelo de IA

| Parámetro       | Valor                                        |
|-----------------|----------------------------------------------|
| Arquitectura    | EfficientNet-B3 (torchvision)                |
| Framework       | PyTorch 2.x                                  |
| Clases          | 7                                            |
| Tamaño entrada  | 224 × 224 × 3 (RGB)                          |
| Normalización   | ImageNet: mean=[0.485,0.456,0.406] std=[0.229,0.224,0.225] |
| Checkpoint      | `checkpoints/best_f1.pt`                     |
| Formato mobile  | TorchScript Lite `.ptl` (PyTorch Mobile 2.1) |
| Archivo mobile  | `checkpoints/melanoma_model.ptl` (40.6 MB)   |

### Clases (índice → nombre → riesgo)

| Índice | Clase                  | Riesgo     |
|--------|------------------------|------------|
| 0      | Extensión Superficial  | Alto       |
| 1      | Lentiginoso Acral      | Alto       |
| 2      | Lentigo Maligno        | Alto       |
| 3      | Nodular                | Alto       |
| 4      | Mucosas                | Alto       |
| 5      | Oculares               | Alto       |
| 6      | No melanoma            | Bajo       |

Clases 0–5 = melanoma maligno (alto riesgo). Clase 6 = benigno.

### Cabeza del modelo (cómo se reemplazó en train.py)
```python
model.classifier = nn.Sequential(
    nn.Dropout(p=0.4, inplace=True),
    nn.Linear(in_features, 7),   # in_features = 1536 para B3
)
```

### Checkpoint — estructura del dict guardado
```python
{
    "epoch": int,
    "model": OrderedDict,          # ← state_dict está en esta clave
    "optimizer": ...,
    "scheduler": ...,
    "best_val_f1": float,
    "classes": list[str],
    "model_name": str
}
```

---

## 3. Estructura de archivos del workspace

```
melanoma_ai/
├── train.py                        # Script de entrenamiento
├── export_model.py                 # Exporta best_f1.pt → melanoma_model.ptl
├── checkpoints/
│   ├── best_f1.pt                  # Checkpoint entrenado (43 MB)
│   └── melanoma_model.ptl          # Modelo exportado para Android (40.6 MB) ✓ LISTO
└── melanoma_android/               # Proyecto Android Studio
    ├── settings.gradle
    ├── build.gradle
    ├── gradle.properties
    ├── gradle/
    │   ├── libs.versions.toml
    │   └── wrapper/gradle-wrapper.properties
    └── app/
        ├── build.gradle
        ├── proguard-rules.pro
        └── src/main/
            ├── AndroidManifest.xml
            ├── assets/
            │   └── melanoma_model.ptl   # ✓ Ya copiado aquí
            ├── java/com/melanoma/ai/
            │   ├── ImageClassifier.kt   # Carga modelo, ejecuta inferencia
            │   └── MainActivity.kt      # Pantalla principal
            └── res/
                ├── layout/
                │   └── activity_main.xml
                └── values/
                    ├── strings.xml
                    ├── colors.xml
                    └── themes.xml
```

---

## 4. Dependencias (gradle/libs.versions.toml)

```toml
[versions]
agp             = "8.4.0"
kotlin          = "1.9.24"
coreKtx         = "1.13.1"
appcompat       = "1.7.0"
material        = "1.12.0"
constraintlayout = "2.1.4"
activity        = "1.9.0"
pytorchAndroid  = "2.1.0"
cameraX         = "1.3.3"
coroutines      = "1.8.1"

[libraries]
androidx-core-ktx          = { group = "androidx.core",              name = "core-ktx",              version.ref = "coreKtx" }
androidx-appcompat         = { group = "androidx.appcompat",         name = "appcompat",             version.ref = "appcompat" }
material                   = { group = "com.google.android.material", name = "material",              version.ref = "material" }
androidx-constraintlayout  = { group = "androidx.constraintlayout",  name = "constraintlayout",      version.ref = "constraintlayout" }
androidx-activity          = { group = "androidx.activity",          name = "activity-ktx",          version.ref = "activity" }
pytorch-android-lite       = { group = "org.pytorch",                name = "pytorch_android_lite",  version.ref = "pytorchAndroid" }
pytorch-torchvision-lite   = { group = "org.pytorch",                name = "pytorch_android_torchvision", version.ref = "pytorchAndroid" }
camerax-core               = { group = "androidx.camera",            name = "camera-core",           version.ref = "cameraX" }
camerax-camera2            = { group = "androidx.camera",            name = "camera-camera2",        version.ref = "cameraX" }
camerax-lifecycle          = { group = "androidx.camera",            name = "camera-lifecycle",      version.ref = "cameraX" }
camerax-view               = { group = "androidx.camera",            name = "camera-view",           version.ref = "cameraX" }
coroutines-android         = { group = "org.jetbrains.kotlinx",      name = "kotlinx-coroutines-android", version.ref = "coroutines" }

[plugins]
android-application = { id = "com.android.application", version.ref = "agp" }
kotlin-android      = { id = "org.jetbrains.kotlin.android", version.ref = "kotlin" }
```

---

## 5. app/build.gradle

```kotlin
plugins {
    alias(libs.plugins.android.application)
    alias(libs.plugins.kotlin.android)
}

android {
    namespace  = "com.melanoma.ai"
    compileSdk = 34

    defaultConfig {
        applicationId   = "com.melanoma.ai"
        minSdk          = 26        // Android 8.0 mínimo
        targetSdk       = 34
        versionCode     = 1
        versionName     = "1.0"
    }

    buildFeatures { viewBinding = true }

    sourceSets {
        getByName("main") { assets.srcDirs("src/main/assets") }
    }

    packaging {
        resources {
            excludes += setOf(
                "META-INF/INDEX.LIST",
                "META-INF/io.netty.versions.properties"
            )
        }
    }
}

dependencies {
    implementation(libs.androidx.core.ktx)
    implementation(libs.androidx.appcompat)
    implementation(libs.material)
    implementation(libs.androidx.constraintlayout)
    implementation(libs.androidx.activity)
    implementation(libs.pytorch.android.lite)
    implementation(libs.pytorch.torchvision.lite)
    implementation(libs.camerax.core)
    implementation(libs.camerax.camera2)
    implementation(libs.camerax.lifecycle)
    implementation(libs.camerax.view)
    implementation(libs.coroutines.android)
}
```

---

## 6. AndroidManifest.xml

```xml
<manifest xmlns:android="http://schemas.android.com/apk/res/android">
    <uses-permission android:name="android.permission.CAMERA" />
    <uses-permission android:name="android.permission.READ_EXTERNAL_STORAGE" android:maxSdkVersion="32" />
    <uses-permission android:name="android.permission.READ_MEDIA_IMAGES" />
    <uses-permission android:name="android.permission.WRITE_EXTERNAL_STORAGE" android:maxSdkVersion="28" />
    <uses-feature android:name="android.hardware.camera" android:required="false" />

    <application
        android:theme="@style/Theme.MelanomaAI"
        android:largeHeap="true">

        <activity android:name=".MainActivity" android:exported="true" android:screenOrientation="portrait">
            <intent-filter>
                <action android:name="android.intent.action.MAIN" />
                <category android:name="android.intent.category.LAUNCHER" />
            </intent-filter>
        </activity>
    </application>
</manifest>
```

---

## 7. ImageClassifier.kt (inferencia completa)

```kotlin
package com.melanoma.ai

import android.content.Context
import android.graphics.Bitmap
import org.pytorch.IValue
import org.pytorch.LiteModuleLoader
import org.pytorch.Module
import org.pytorch.Tensor
import org.pytorch.torchvision.TensorImageUtils
import java.io.File
import java.io.FileOutputStream

class ImageClassifier(context: Context) {

    val classes = listOf(
        "Extensión Superficial", "Lentiginoso Acral", "Lentigo Maligno",
        "Nodular", "Mucosas", "Oculares", "No melanoma"
    )

    val highRiskClasses = setOf(
        "Extensión Superficial", "Lentiginoso Acral", "Lentigo Maligno",
        "Nodular", "Mucosas", "Oculares"
    )

    private val MEAN = floatArrayOf(0.485f, 0.456f, 0.406f)
    private val STD  = floatArrayOf(0.229f, 0.224f, 0.225f)
    private val IMG_SIZE = 224
    private val module: Module

    init {
        module = LiteModuleLoader.load(assetFilePath(context, "melanoma_model.ptl"))
    }

    data class Prediction(val className: String, val confidence: Float, val isHighRisk: Boolean)
    data class ClassificationResult(val top1: Prediction, val top3: List<Prediction>)

    fun classify(bitmap: Bitmap): ClassificationResult {
        val resized = Bitmap.createScaledBitmap(bitmap, IMG_SIZE, IMG_SIZE, true)
        val inputTensor: Tensor = TensorImageUtils.bitmapToFloat32Tensor(resized, MEAN, STD)
        val output = module.forward(IValue.from(inputTensor)).toTensor()
        val softmax = softmax(output.dataAsFloatArray)
        val predictions = softmax.mapIndexed { idx, score ->
            Prediction(classes[idx], score, classes[idx] in highRiskClasses)
        }.sortedByDescending { it.confidence }
        return ClassificationResult(top1 = predictions[0], top3 = predictions.take(3))
    }

    private fun softmax(logits: FloatArray): FloatArray {
        val max = logits.max()
        val exp = logits.map { Math.exp((it - max).toDouble()).toFloat() }
        val sum = exp.sum()
        return exp.map { it / sum }.toFloatArray()
    }

    private fun assetFilePath(context: Context, assetName: String): String {
        val file = File(context.filesDir, assetName)
        if (file.exists() && file.length() > 0) return file.absolutePath
        context.assets.open(assetName).use { input ->
            FileOutputStream(file).use { output ->
                val buffer = ByteArray(4 * 1024)
                var read: Int
                while (input.read(buffer).also { read = it } != -1) output.write(buffer, 0, read)
                output.flush()
            }
        }
        return file.absolutePath
    }

    fun close() { module.destroy() }
}
```

---

## 8. MainActivity.kt (pantalla única)

```kotlin
package com.melanoma.ai

import android.Manifest
import android.content.ContentValues
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.provider.MediaStore
import android.view.View
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import androidx.lifecycle.lifecycleScope
import com.melanoma.ai.databinding.ActivityMainBinding
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.IOException

class MainActivity : AppCompatActivity() {

    private lateinit var binding: ActivityMainBinding
    private var classifier: ImageClassifier? = null
    private var selectedBitmap: Bitmap? = null
    private var cameraImageUri: Uri? = null

    private val galleryLauncher = registerForActivityResult(ActivityResultContracts.GetContent()) { uri ->
        uri?.let { loadBitmapFromUri(it) }
    }

    private val cameraLauncher = registerForActivityResult(ActivityResultContracts.TakePicture()) { success ->
        if (success) cameraImageUri?.let { loadBitmapFromUri(it) }
    }

    private val permissionLauncher = registerForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
        if (granted) launchCamera()
        else Toast.makeText(this, getString(R.string.permission_camera_rationale), Toast.LENGTH_LONG).show()
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)
        initClassifier()
        setupListeners()
    }

    private fun initClassifier() {
        lifecycleScope.launch(Dispatchers.IO) {
            try {
                val c = ImageClassifier(applicationContext)
                withContext(Dispatchers.Main) {
                    classifier = c
                    binding.btnAnalyze.isEnabled = selectedBitmap != null
                }
            } catch (e: Exception) {
                withContext(Dispatchers.Main) { showError(getString(R.string.error_model)) }
            }
        }
    }

    private fun setupListeners() {
        binding.btnGallery.setOnClickListener { galleryLauncher.launch("image/*") }
        binding.btnCamera.setOnClickListener {
            if (hasCameraPermission()) launchCamera()
            else permissionLauncher.launch(Manifest.permission.CAMERA)
        }
        binding.btnAnalyze.setOnClickListener {
            selectedBitmap?.let { runClassification(it) } ?: showError("Primero selecciona una imagen")
        }
    }

    private fun launchCamera() {
        val values = ContentValues().apply {
            put(MediaStore.Images.Media.DISPLAY_NAME, "melanoma_${System.currentTimeMillis()}.jpg")
            put(MediaStore.Images.Media.MIME_TYPE, "image/jpeg")
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q)
                put(MediaStore.Images.Media.RELATIVE_PATH, "Pictures/MelanomaAI")
        }
        cameraImageUri = contentResolver.insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, values)
        cameraImageUri?.let { cameraLauncher.launch(it) }
    }

    private fun loadBitmapFromUri(uri: Uri) {
        lifecycleScope.launch(Dispatchers.IO) {
            try {
                val bmp = contentResolver.openInputStream(uri)?.use { BitmapFactory.decodeStream(it) }
                withContext(Dispatchers.Main) {
                    if (bmp != null) setImage(bmp) else showError("No se pudo cargar la imagen")
                }
            } catch (e: IOException) {
                withContext(Dispatchers.Main) { showError("No se pudo cargar la imagen") }
            }
        }
    }

    private fun setImage(bitmap: Bitmap) {
        selectedBitmap = bitmap
        binding.imagePreview.setImageBitmap(bitmap)
        binding.imagePreview.visibility = View.VISIBLE
        binding.tvHint.visibility = View.GONE
        binding.cardResult.visibility = View.GONE
        binding.btnAnalyze.isEnabled = classifier != null
    }

    private fun runClassification(bitmap: Bitmap) {
        val clf = classifier ?: run { showError(getString(R.string.error_model)); return }
        binding.progressBar.visibility = View.VISIBLE
        binding.cardResult.visibility  = View.GONE
        binding.btnAnalyze.isEnabled   = false
        binding.btnCamera.isEnabled    = false
        binding.btnGallery.isEnabled   = false

        lifecycleScope.launch(Dispatchers.Default) {
            try {
                val result = clf.classify(bitmap)
                withContext(Dispatchers.Main) { showResult(result) }
            } catch (e: Exception) {
                withContext(Dispatchers.Main) { showError(getString(R.string.error_inference)) }
            } finally {
                withContext(Dispatchers.Main) {
                    binding.progressBar.visibility = View.GONE
                    binding.btnAnalyze.isEnabled   = true
                    binding.btnCamera.isEnabled    = true
                    binding.btnGallery.isEnabled   = true
                }
            }
        }
    }

    private fun showResult(result: ImageClassifier.ClassificationResult) {
        val top1 = result.top1
        val riskColor = if (top1.isHighRisk)
            ContextCompat.getColor(this, R.color.colorHighRisk)
        else
            ContextCompat.getColor(this, R.color.colorLowRisk)

        binding.tvResultClass.text = top1.className
        binding.tvResultClass.setTextColor(riskColor)
        binding.tvConfidence.text  = "%.1f%%".format(top1.confidence * 100)
        binding.tvRiskBadge.text   = if (top1.isHighRisk) "⚠ ALTO RIESGO" else "✓ No melanoma"
        binding.tvRiskBadge.setTextColor(riskColor)
        binding.tvTop3.text = result.top3.mapIndexed { i, p ->
            "${i + 1}. ${p.className}  ${"%.1f%%".format(p.confidence * 100)}"
        }.joinToString("\n")
        binding.cardResult.visibility = View.VISIBLE
    }

    private fun hasCameraPermission() =
        ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED

    private fun showError(message: String) = Toast.makeText(this, message, Toast.LENGTH_LONG).show()

    override fun onDestroy() { super.onDestroy(); classifier?.close() }
}
```

---

## 9. activity_main.xml (layout completo)

```xml
<?xml version="1.0" encoding="utf-8"?>
<LinearLayout xmlns:android="http://schemas.android.com/apk/res/android"
    xmlns:app="http://schemas.android.com/apk/res-auto"
    xmlns:tools="http://schemas.android.com/tools"
    android:layout_width="match_parent"
    android:layout_height="match_parent"
    android:orientation="vertical"
    android:background="@color/colorBackground"
    tools:context=".MainActivity">

    <!-- Toolbar azul con spinner de carga -->
    <LinearLayout android:layout_width="match_parent" android:layout_height="56dp"
        android:orientation="horizontal" android:gravity="center_vertical"
        android:paddingStart="16dp" android:paddingEnd="16dp"
        android:background="@color/colorPrimary">
        <TextView android:layout_width="0dp" android:layout_height="wrap_content"
            android:layout_weight="1" android:text="@string/app_name"
            android:textColor="#FFFFFF" android:textSize="20sp" android:textStyle="bold" />
        <ProgressBar android:id="@+id/progressBar" android:layout_width="28dp"
            android:layout_height="28dp" android:indeterminateTint="#FFFFFF" android:visibility="gone" />
    </LinearLayout>

    <ScrollView android:layout_width="match_parent" android:layout_height="0dp"
        android:layout_weight="1" android:fillViewport="true">
        <LinearLayout android:layout_width="match_parent" android:layout_height="wrap_content"
            android:orientation="vertical" android:padding="16dp">

            <!-- Preview de imagen (300dp, fondo azul claro) -->
            <androidx.cardview.widget.CardView android:layout_width="match_parent"
                android:layout_height="300dp" app:cardCornerRadius="12dp"
                app:cardElevation="3dp" android:layout_marginBottom="16dp">
                <FrameLayout android:layout_width="match_parent" android:layout_height="match_parent"
                    android:background="#E8EAF6">
                    <ImageView android:id="@+id/imagePreview" android:layout_width="match_parent"
                        android:layout_height="match_parent" android:scaleType="centerCrop"
                        android:visibility="gone" />
                    <!-- id tvHint es un LinearLayout (no TextView) para compatibilidad con View binding -->
                    <LinearLayout android:id="@+id/tvHint" android:layout_width="wrap_content"
                        android:layout_height="wrap_content" android:layout_gravity="center"
                        android:orientation="vertical" android:gravity="center">
                        <TextView android:layout_width="wrap_content" android:layout_height="wrap_content"
                            android:text="🖼️" android:textSize="48sp" android:gravity="center"
                            android:layout_marginBottom="8dp" />
                        <TextView android:layout_width="wrap_content" android:layout_height="wrap_content"
                            android:text="Toma una foto o elige\nuna imagen de tu galería"
                            android:textSize="14sp" android:textColor="#888888" android:gravity="center" />
                    </LinearLayout>
                </FrameLayout>
            </androidx.cardview.widget.CardView>

            <!-- Botones: Cámara (relleno azul) | Galería (outline) -->
            <LinearLayout android:layout_width="match_parent" android:layout_height="wrap_content"
                android:orientation="horizontal" android:layout_marginBottom="12dp">
                <com.google.android.material.button.MaterialButton android:id="@+id/btnCamera"
                    android:layout_width="0dp" android:layout_height="52dp" android:layout_weight="1"
                    android:layout_marginEnd="6dp" android:text="📷  Cámara" android:textSize="15sp"
                    app:backgroundTint="@color/colorPrimary" />
                <com.google.android.material.button.MaterialButton android:id="@+id/btnGallery"
                    android:layout_width="0dp" android:layout_height="52dp" android:layout_weight="1"
                    android:layout_marginStart="6dp" android:text="🖼  Galería" android:textSize="15sp"
                    style="@style/Widget.MaterialComponents.Button.OutlinedButton" />
            </LinearLayout>

            <!-- Botón analizar (naranja, deshabilitado al inicio) -->
            <com.google.android.material.button.MaterialButton android:id="@+id/btnAnalyze"
                android:layout_width="match_parent" android:layout_height="56dp"
                android:text="Analizar imagen" android:textSize="16sp" android:textStyle="bold"
                android:enabled="false" android:layout_marginBottom="16dp"
                app:backgroundTint="@color/colorAccent" />

            <!-- Tarjeta de resultados (oculta al inicio) -->
            <androidx.cardview.widget.CardView android:id="@+id/cardResult"
                android:layout_width="match_parent" android:layout_height="wrap_content"
                app:cardCornerRadius="12dp" app:cardElevation="4dp"
                android:layout_marginBottom="12dp" android:visibility="gone">
                <LinearLayout android:layout_width="match_parent" android:layout_height="wrap_content"
                    android:orientation="vertical" android:padding="16dp">

                    <!-- Badge ⚠ ALTO RIESGO o ✓ No melanoma -->
                    <TextView android:id="@+id/tvRiskBadge" android:layout_width="wrap_content"
                        android:layout_height="wrap_content" android:textSize="12sp"
                        android:textStyle="bold" android:layout_marginBottom="12dp" />

                    <TextView android:layout_width="wrap_content" android:layout_height="wrap_content"
                        android:text="DIAGNÓSTICO" android:textSize="11sp" android:textColor="#888888"
                        android:letterSpacing="0.1" />
                    <TextView android:id="@+id/tvResultClass" android:layout_width="wrap_content"
                        android:layout_height="wrap_content" android:textSize="24sp"
                        android:textStyle="bold" android:layout_marginBottom="8dp" />

                    <TextView android:layout_width="wrap_content" android:layout_height="wrap_content"
                        android:text="CONFIANZA" android:textSize="11sp" android:textColor="#888888"
                        android:letterSpacing="0.1" />
                    <TextView android:id="@+id/tvConfidence" android:layout_width="wrap_content"
                        android:layout_height="wrap_content" android:textSize="20sp"
                        android:textStyle="bold" android:textColor="@color/colorPrimary"
                        android:layout_marginBottom="12dp" />

                    <View android:layout_width="match_parent" android:layout_height="1dp"
                        android:background="#E0E0E0" android:layout_marginBottom="12dp" />

                    <TextView android:layout_width="wrap_content" android:layout_height="wrap_content"
                        android:text="TOP 3 RESULTADOS" android:textSize="11sp"
                        android:textColor="#888888" android:letterSpacing="0.1"
                        android:layout_marginBottom="4dp" />
                    <TextView android:id="@+id/tvTop3" android:layout_width="match_parent"
                        android:layout_height="wrap_content" android:textSize="13sp"
                        android:lineSpacingExtra="6dp" android:fontFamily="monospace" />
                </LinearLayout>
            </androidx.cardview.widget.CardView>

            <!-- Disclaimer médico -->
            <TextView android:layout_width="match_parent" android:layout_height="wrap_content"
                android:text="⚠️ Herramienta de apoyo. No reemplaza el diagnóstico médico profesional."
                android:textSize="11sp" android:textColor="@color/colorWarning"
                android:gravity="center" android:padding="8dp" />
        </LinearLayout>
    </ScrollView>
</LinearLayout>
```

---

## 10. res/values/

### strings.xml
```xml
<resources>
    <string name="app_name">Melanoma AI</string>
    <string name="permission_camera_rationale">La cámara es necesaria para capturar imágenes de lesiones cutáneas.</string>
    <string name="error_model">Error al cargar el modelo. Verifique el archivo melanoma_model.ptl en assets/.</string>
    <string name="error_inference">Error durante el análisis. Intente con otra imagen.</string>
</resources>
```

### colors.xml
```xml
<resources>
    <color name="colorPrimary">#1565C0</color>
    <color name="colorPrimaryDark">#0D47A1</color>
    <color name="colorAccent">#FF6F00</color>
    <color name="colorBackground">#F5F5F5</color>
    <color name="colorWarning">#F57F17</color>
    <color name="colorHighRisk">#C62828</color>
    <color name="colorLowRisk">#2E7D32</color>
    <color name="colorCardBorder">#E0E0E0</color>
</resources>
```

### themes.xml
```xml
<resources>
    <style name="Theme.MelanomaAI" parent="Theme.MaterialComponents.DayNight.DarkActionBar">
        <item name="colorPrimary">@color/colorPrimary</item>
        <item name="colorPrimaryDark">@color/colorPrimaryDark</item>
        <item name="colorAccent">@color/colorAccent</item>
        <item name="android:windowBackground">@color/colorBackground</item>
    </style>
</resources>
```

---

## 11. Estado actual — qué está hecho y qué falta

### ✅ Completado
- Modelo EfficientNet-B3 exportado a `melanoma_model.ptl` (40.6 MB)
- Archivo `.ptl` copiado a `melanoma_android/app/src/main/assets/`
- Todos los archivos del proyecto Android creados y funcionales
- Lógica de inferencia con PyTorch Mobile Lite
- UI de pantalla única (cámara + galería + resultados)

### ⚠️ Pendiente / posibles mejoras
1. **Icono de la app** — Falta crear `res/mipmap-*/ic_launcher.png` (actualmente usará el placeholder de Android Studio al generar el proyecto desde IDE)
2. **`@drawable/badge_background`** — El XML del badge en `tvRiskBadge` referencia este drawable que no existe; se puede eliminar la línea `android:background="@drawable/badge_background"` del tvRiskBadge o crear el archivo
3. **Prueba en dispositivo físico** — El modelo es pesado (40 MB), puede tardar ~2-3 seg en cargar en el primer arranque
4. **Versión mínima SDK 26** — Requiere Android 8.0+

### Nota sobre el DeprecationWarning del export
`Lite Interpreter is deprecated` — solo es un warning. `pytorch_android_lite:2.1.0` sigue soportando `.ptl`. Si se quiere migrar, usar ExecuTorch, pero requiere cambios mayores.

---

## 12. Cómo abrir y ejecutar en Android Studio

1. **File → Open** → seleccionar carpeta `melanoma_android/`
2. Esperar sync de Gradle (~2-3 min primera vez)
3. Conectar dispositivo Android (API 26+) con depuración USB activada
4. **Run ▶** (Shift+F10)
