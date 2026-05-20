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

/**
 * ImageClassifier — carga el modelo TorchScript Lite y ejecuta inferencia.
 *
 * Preprocesado idéntico a VAL_TRANSFORMS de train.py:
 *   Resize(224,224) → ToTensor → Normalize(ImageNet mean/std)
 */
class ImageClassifier(context: Context) {

    // ── Clases (mismo orden que CLASSES en train.py) ──────────────────────────
    val classes = listOf(
        "Extensión Superficial",
        "Lentiginoso Acral",
        "Lentigo Maligno",
        "Nodular",
        "Mucosas",
        "Oculares",
        "No melanoma"
    )

    // Clases consideradas de alto riesgo (melanoma maligno)
    val highRiskClasses = setOf(
        "Extensión Superficial",
        "Lentiginoso Acral",
        "Lentigo Maligno",
        "Nodular",
        "Mucosas",
        "Oculares"
    )

    // ImageNet normalization — idéntica a train.py
    private val MEAN = floatArrayOf(0.485f, 0.456f, 0.406f)
    private val STD  = floatArrayOf(0.229f, 0.224f, 0.225f)

    private val IMG_SIZE = 224

    /**
     * Umbral de confianza para reportar melanoma.
     * Si la probabilidad COMBINADA de todas las clases malignas (0–5) es
     * inferior a este valor, el resultado se fuerza a "No melanoma".
     *
     * Valores orientativos:
     *   0.50 → sin sesgo (argmax puro)
     *   0.60 → sesgo moderado hacia "No melanoma"  ← recomendado
     *   0.70 → sesgo fuerte  (menos alarmas, más riesgo de falsos negativos)
     *
     * ⚠️ ADVERTENCIA MÉDICA: bajar este umbral reduce falsas alarmas pero
     * puede hacer que melanomas reales no sean detectados. Ajustar con
     * cautela y siempre bajo supervisión clínica.
     */
    private val MELANOMA_CONFIDENCE_THRESHOLD = 0.60f

    private val module: Module

    init {
        // PyTorch Mobile requiere el modelo en almacenamiento interno, no en assets
        val modelPath = assetFilePath(context, "melanoma_model.ptl")
        module = LiteModuleLoader.load(modelPath)
    }

    data class Prediction(
        val className: String,
        val confidence: Float,
        val isHighRisk: Boolean
    )

    data class ClassificationResult(
        val top1: Prediction,
        val top3: List<Prediction>
    )

    /**
     * Clasifica un [Bitmap]. Se ejecuta en el hilo que lo llame —
     * usar desde una coroutine con Dispatchers.Default.
     */
    fun classify(bitmap: Bitmap): ClassificationResult {
        // Redimensionar a 224×224
        val resized = Bitmap.createScaledBitmap(bitmap, IMG_SIZE, IMG_SIZE, true)

        // Convertir a tensor normalizado [1, 3, 224, 224]
        val inputTensor: Tensor = TensorImageUtils.bitmapToFloat32Tensor(resized, MEAN, STD)

        // Inferencia
        val output = module.forward(IValue.from(inputTensor)).toTensor()
        val scores = output.dataAsFloatArray

        // Softmax
        val softmax = softmax(scores)

        // Probabilidad combinada de todas las clases malignas (índices 0–5)
        val melanomaTotalProb = (0..5).sumOf { softmax[it].toDouble() }.toFloat()

        // Ordenar por probabilidad descendente
        val indexed = softmax.mapIndexed { idx, score -> idx to score }
            .sortedByDescending { it.second }

        // Aplicar umbral: si la confianza melanoma total < threshold → "No melanoma"
        val finalIndexed = if (melanomaTotalProb < MELANOMA_CONFIDENCE_THRESHOLD) {
            val noMelIdx = 6
            listOf(noMelIdx to (1f - melanomaTotalProb))
                .plus(indexed.filter { it.first != noMelIdx })
        } else {
            indexed
        }

        val predictions = finalIndexed.map { (idx, score) ->
            Prediction(
                className   = classes[idx],
                confidence  = score,
                isHighRisk  = classes[idx] in highRiskClasses
            )
        }

        return ClassificationResult(
            top1 = predictions[0],
            top3 = predictions.take(3)
        )
    }

    private fun softmax(logits: FloatArray): FloatArray {
        val max = logits.max()
        val exp = logits.map { Math.exp((it - max).toDouble()).toFloat() }
        val sum = exp.sum()
        return exp.map { it / sum }.toFloatArray()
    }

    /** Copia un asset al almacenamiento interno y devuelve su ruta absoluta. */
    private fun assetFilePath(context: Context, assetName: String): String {
        val file = File(context.filesDir, assetName)
        if (file.exists() && file.length() > 0) return file.absolutePath

        context.assets.open(assetName).use { input ->
            FileOutputStream(file).use { output ->
                val buffer = ByteArray(4 * 1024)
                var read: Int
                while (input.read(buffer).also { read = it } != -1) {
                    output.write(buffer, 0, read)
                }
                output.flush()
            }
        }
        return file.absolutePath
    }

    fun close() {
        module.destroy()
    }
}
