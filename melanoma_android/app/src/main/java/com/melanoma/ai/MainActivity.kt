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

    // URI temporal para la foto tomada con la cámara
    private var cameraImageUri: Uri? = null

    // ── Launchers ──────────────────────────────────────────────────────────────

    // Galería — devuelve un URI
    private val galleryLauncher = registerForActivityResult(
        ActivityResultContracts.GetContent()
    ) { uri: Uri? ->
        uri?.let { loadBitmapFromUri(it) }
    }

    // Cámara — captura a URI (imagen de tamaño completo, no thumbnail)
    private val cameraLauncher = registerForActivityResult(
        ActivityResultContracts.TakePicture()
    ) { success ->
        if (success) {
            cameraImageUri?.let { loadBitmapFromUri(it) }
        }
    }

    // Permisos de cámara
    private val permissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { granted ->
        if (granted) launchCamera()
        else Toast.makeText(this, getString(R.string.permission_camera_rationale), Toast.LENGTH_LONG).show()
    }

    // ── Lifecycle ──────────────────────────────────────────────────────────────

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
                withContext(Dispatchers.Main) {
                    showError(getString(R.string.error_model))
                }
            }
        }
    }

    private fun setupListeners() {
        binding.btnGallery.setOnClickListener {
            galleryLauncher.launch("image/*")
        }

        binding.btnCamera.setOnClickListener {
            if (hasCameraPermission()) launchCamera()
            else permissionLauncher.launch(Manifest.permission.CAMERA)
        }

        binding.btnAnalyze.setOnClickListener {
            selectedBitmap?.let { bmp -> runClassification(bmp) }
                ?: showError("Primero selecciona una imagen")
        }
    }

    // ── Image loading ──────────────────────────────────────────────────────────

    private fun launchCamera() {
        // Crea un URI en MediaStore para guardar la foto de tamaño completo
        val values = ContentValues().apply {
            put(MediaStore.Images.Media.DISPLAY_NAME, "melanoma_${System.currentTimeMillis()}.jpg")
            put(MediaStore.Images.Media.MIME_TYPE, "image/jpeg")
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                put(MediaStore.Images.Media.RELATIVE_PATH, "Pictures/MelanomaAI")
            }
        }
        cameraImageUri = contentResolver.insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, values)
        cameraImageUri?.let { cameraLauncher.launch(it) }
    }

    private fun loadBitmapFromUri(uri: Uri) {
        lifecycleScope.launch(Dispatchers.IO) {
            try {
                val bmp = contentResolver.openInputStream(uri)?.use { BitmapFactory.decodeStream(it) }
                withContext(Dispatchers.Main) {
                    if (bmp != null) setImage(bmp)
                    else showError("No se pudo cargar la imagen")
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
        // Habilitar analizar solo si el clasificador ya cargó
        binding.btnAnalyze.isEnabled = classifier != null
    }

    // ── Classification ─────────────────────────────────────────────────────────

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
        val pct  = "%.1f%%".format(top1.confidence * 100)

        val riskColor = if (top1.isHighRisk)
            ContextCompat.getColor(this, R.color.colorHighRisk)
        else
            ContextCompat.getColor(this, R.color.colorLowRisk)

        binding.tvResultClass.text = top1.className
        binding.tvResultClass.setTextColor(riskColor)
        binding.tvConfidence.text  = pct
        binding.tvRiskBadge.text   = if (top1.isHighRisk) "⚠ ALTO RIESGO" else "✓ No melanoma"
        binding.tvRiskBadge.setTextColor(riskColor)

        val top3Text = result.top3.mapIndexed { i, p ->
            "${i + 1}. ${p.className}  ${"%.1f%%".format(p.confidence * 100)}"
        }.joinToString("\n")
        binding.tvTop3.text = top3Text

        binding.cardResult.visibility = View.VISIBLE
    }

    // ── Permissions ────────────────────────────────────────────────────────────

    private fun hasCameraPermission() =
        ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) ==
                PackageManager.PERMISSION_GRANTED

    // ── Helpers ────────────────────────────────────────────────────────────────

    private fun showError(message: String) {
        Toast.makeText(this, message, Toast.LENGTH_LONG).show()
    }

    override fun onDestroy() {
        super.onDestroy()
        classifier?.close()
    }
}
