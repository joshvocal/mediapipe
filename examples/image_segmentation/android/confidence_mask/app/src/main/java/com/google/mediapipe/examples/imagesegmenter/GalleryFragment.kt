/*
 * Copyright 2023 The TensorFlow Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *             http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package com.google.mediapipe.examples.imagesegmenter

import android.content.res.Configuration
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.ImageDecoder
import android.graphics.Paint
import android.graphics.PorterDuff
import android.graphics.PorterDuffXfermode
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.provider.MediaStore
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.AdapterView
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.core.view.drawToBitmap
import androidx.fragment.app.Fragment
import androidx.fragment.app.activityViewModels
import com.google.mediapipe.examples.imagesegmenter.databinding.FragmentGalleryBinding
import com.google.mediapipe.examples.imagesegmenter.mediapipe.ImageSegmenterHelper
import com.google.mediapipe.examples.imagesegmenter.tensorflow.InterpreterImageSegmenterHelper
import com.google.mediapipe.examples.imagesegmenter.tensorflow.TensorImageSegmenterHelper
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.framework.image.ByteBufferExtractor
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.imagesegmenter.ImageSegmenterResult
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.cancel
import kotlinx.coroutines.launch
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.withContext
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.task.vision.segmenter.Segmentation
import java.nio.ByteBuffer

class GalleryFragment : Fragment(), ImageSegmenterHelper.SegmenterListener,
    TensorImageSegmenterHelper.TensorSegmentationListener {
    enum class MediaType {
        IMAGE, VIDEO, UNKNOWN
    }

    private val maskPaint: Paint = Paint(Paint.ANTI_ALIAS_FLAG)

    init {
        maskPaint.isAntiAlias = true
        maskPaint.style = Paint.Style.FILL
        maskPaint.xfermode = PorterDuffXfermode(PorterDuff.Mode.DST_IN)
    }

    private var tensorImageSegmenterHelper: TensorImageSegmenterHelper? = null
    private var interpreterImageSegmenterHelper: InterpreterImageSegmenterHelper? = null

    private val viewModel: MainViewModel by activityViewModels()

    private var _fragmentGalleryBinding: FragmentGalleryBinding? = null
    private val fragmentGalleryBinding
        get() = _fragmentGalleryBinding!!
    private var imageSegmenterHelper: ImageSegmenterHelper? = null
    private var backgroundScope: CoroutineScope? = null

    private val getContent =
        registerForActivityResult(ActivityResultContracts.OpenDocument()) { uri: Uri? ->
            // Handle the returned Uri
            uri?.let { mediaUri ->
                when (val mediaType = loadMediaType(mediaUri)) {
//                    MediaType.IMAGE -> runSegmentationOnImage(mediaUri)
//                    MediaType.IMAGE -> runTensorSegmentationOnImage(mediaUri)
                    MediaType.IMAGE -> runInterpreterSegmentationOnImage(mediaUri)
                    MediaType.UNKNOWN -> {
                        updateDisplayView(mediaType)
                        Toast.makeText(
                            requireContext(),
                            "Unsupported data type.",
                            Toast.LENGTH_SHORT
                        ).show()
                    }
                }
            }
        }

    override fun onCreateView(
        inflater: LayoutInflater,
        container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View {
        _fragmentGalleryBinding = FragmentGalleryBinding.inflate(inflater, container, false)

        return fragmentGalleryBinding.root
    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)

        fragmentGalleryBinding.fabGetContent.setOnClickListener {
            stopAllTasks()
            getContent.launch(arrayOf("image/*", "video/*"))
            updateDisplayView(MediaType.UNKNOWN)
        }
        initBottomSheetControls()
    }

    override fun onPause() {
        stopAllTasks()
        super.onPause()
    }

    override fun onConfigurationChanged(newConfig: Configuration) {
        super.onConfigurationChanged(newConfig)
        stopAllTasks()
        setUiEnabled(true)
    }

    private fun initBottomSheetControls() {

        // When clicked, change the underlying hardware used for inference. Current options are CPU
        // GPU, and NNAPI
        fragmentGalleryBinding.bottomSheetLayout.spinnerDelegate.setSelection(
            viewModel.currentDelegate, false
        )
        fragmentGalleryBinding.bottomSheetLayout.spinnerDelegate.onItemSelectedListener =
            object : AdapterView.OnItemSelectedListener {
                override fun onItemSelected(
                    p0: AdapterView<*>?, p1: View?, p2: Int, p3: Long
                ) {

                    viewModel.setDelegate(p2)
                    stopAllTasks()
                }

                override fun onNothingSelected(p0: AdapterView<*>?) {/* no op */
                }
            }

        fragmentGalleryBinding.bottomSheetLayout.spinnerModel.setSelection(
            viewModel.currentModel, false
        )

        fragmentGalleryBinding.bottomSheetLayout.spinnerModel.onItemSelectedListener =
            object : AdapterView.OnItemSelectedListener {
                override fun onItemSelected(
                    parent: AdapterView<*>?,
                    view: View?,
                    position: Int,
                    id: Long
                ) {
                    viewModel.setModel(position)
                    stopAllTasks()
                }

                override fun onNothingSelected(parent: AdapterView<*>?) {
                    /* no op */
                }
            }
    }

    private fun stopAllTasks() {
        // cancel all jobs
        backgroundScope?.cancel()
        backgroundScope = null

        // clear Image Segmenter
        imageSegmenterHelper?.clearListener()
        imageSegmenterHelper?.clearImageSegmenter()
        imageSegmenterHelper = null

        with(fragmentGalleryBinding) {
            // clear overlay view
            overlayView.clear()
            progress.visibility = View.GONE
        }
        updateDisplayView(MediaType.UNKNOWN)
    }

    private fun runInterpreterSegmentationOnImage(uri: Uri) {
        fragmentGalleryBinding.overlayView.setRunningMode(RunningMode.IMAGE)
        setUiEnabled(false)
        updateDisplayView(MediaType.IMAGE)

        var inputImage = uri.toBitmap()
        val outputWidth = inputImage.width
        val outputHeight = inputImage.height
        inputImage = inputImage.scaleDown2(INPUT_IMAGE_MAX_WIDTH)
        val inputWidth = inputImage.width
        val inputHeight = inputImage.height

        fragmentGalleryBinding.imageResult.setImageBitmap(inputImage)

        backgroundScope = CoroutineScope(Dispatchers.IO)

        interpreterImageSegmenterHelper =
            InterpreterImageSegmenterHelper(context = requireContext())

        backgroundScope?.launch {
            val result = interpreterImageSegmenterHelper?.segmentImageWithInterpreter(
                bitmap = inputImage,
            )

            result?.let {
                updateTensorOverlay(
                    results = result,
                    width = inputWidth,
                    height = inputHeight,
                )
            }
        }
    }

    private fun runTensorSegmentationOnImage(uri: Uri) {
        fragmentGalleryBinding.overlayView.setRunningMode(RunningMode.IMAGE)
        setUiEnabled(false)
        updateDisplayView(MediaType.IMAGE)

        var inputImage = uri.toBitmap()
        inputImage = inputImage.scaleDown2(INPUT_IMAGE_MAX_WIDTH)

        fragmentGalleryBinding.imageResult.setImageBitmap(inputImage)

        backgroundScope = CoroutineScope(Dispatchers.IO)

        tensorImageSegmenterHelper = TensorImageSegmenterHelper(
            context = requireContext()
        )

        backgroundScope?.launch {
            // Create preprocessor for the image.
            // See https://www.tensorflow.org/lite/inference_with_metadata/
            //            lite_support#imageprocessor_architecture
            val imageProcessor = ImageProcessor.Builder()
                .build()

            // Preprocess the image and convert it into a TensorImage for segmentation.
            val tensorImage = imageProcessor.process(TensorImage.fromBitmap(inputImage))

            val result = tensorImageSegmenterHelper?.segmentTensorImage(
                tensorImage = tensorImage,
            )

            result?.let {
                updateTensorOverlay(
                    result = it,
                    imageWidth = tensorImage.width,
                    imageHeight = tensorImage.height,
                )
            }
        }
    }

    private fun updateTensorOverlay(result: List<Segmentation>, imageWidth: Int, imageHeight: Int) {
        if (_fragmentGalleryBinding != null) {
            runBlocking {
                withContext(Dispatchers.Main) {
                    setUiEnabled(true)

                    val maskTensor: TensorImage = result[0].masks[0]
                    val maskArray: ByteBuffer = maskTensor.buffer

                    fragmentGalleryBinding.overlayView.setTensorResults(
                        byteBuffer = maskArray,
                        maskTensor = maskTensor,
                        outputWidth = imageWidth,
                        outputHeight = imageHeight,
                    )

                    fragmentGalleryBinding.overlayView.visibility = View.GONE

                    fragmentGalleryBinding.imageResult.setImageBitmap(
                        getMaskedImage(
                            input = fragmentGalleryBinding.imageResult.drawToBitmap(),
                            mask = fragmentGalleryBinding.overlayView.drawToBitmap(),
                        )
                    )
                }
            }
        }
    }

    // Load and display the image.
    private fun runSegmentationOnImage(uri: Uri) {
        fragmentGalleryBinding.overlayView.setRunningMode(RunningMode.IMAGE)
        setUiEnabled(false)
        updateDisplayView(MediaType.IMAGE)

        var inputImage = uri.toBitmap()
        inputImage = inputImage.scaleDown2(INPUT_IMAGE_MAX_WIDTH)

        // display image on UI
        fragmentGalleryBinding.imageResult.setImageBitmap(inputImage)

        backgroundScope = CoroutineScope(Dispatchers.IO)

        imageSegmenterHelper = ImageSegmenterHelper(
            context = requireContext(),
            runningMode = RunningMode.IMAGE,
            currentDelegate = viewModel.currentDelegate,
            imageSegmenterListener = this
        )

        // Run image segmentation on the input image
        backgroundScope?.launch {
            val mpImage = BitmapImageBuilder(inputImage).build()
            val result = imageSegmenterHelper?.segmentImageFile(mpImage)
            updateOverlay(result!!)
        }
    }

    private fun updateDisplayView(mediaType: MediaType) {
        fragmentGalleryBinding.imageResult.visibility =
            if (mediaType == MediaType.IMAGE) View.VISIBLE else View.GONE
    }

    // Check the type of media that user selected.
    private fun loadMediaType(uri: Uri): MediaType {
        val mimeType = context?.contentResolver?.getType(uri)

        mimeType?.let {
            if (mimeType.startsWith("image")) {
                return MediaType.IMAGE
            }

            if (mimeType.startsWith("video")) {
                return MediaType.VIDEO
            }
        }

        return MediaType.UNKNOWN
    }

    private fun setUiEnabled(enabled: Boolean) {
        fragmentGalleryBinding.fabGetContent.isEnabled = enabled
    }

    private fun updateOverlay(result: ImageSegmenterResult) {
        val newImage = result.categoryMask().get()
        val results = ByteBufferExtractor.extract(newImage)

        updateOverlay(
            results = results,
            width = newImage.width,
            height = newImage.height,
            inferenceTime = result.timestampMs(),
        )
    }

    private fun updateTensorOverlay(
        results: ByteBuffer,
        width: Int,
        height: Int,
    ) {
        if (_fragmentGalleryBinding != null) {

            runBlocking {
                withContext(Dispatchers.Main) {
                    setUiEnabled(true)

                    fragmentGalleryBinding.overlayView.setResults(
                        byteBuffer = results,
                        outputWidth = width,
                        outputHeight = height,
                    )

//                    fragmentGalleryBinding.overlayView.visibility = View.GONE
//
//                    fragmentGalleryBinding.imageResult.setImageBitmap(
//                        getMaskedImage(
//                            input = fragmentGalleryBinding.imageResult.drawToBitmap(),
//                            mask = fragmentGalleryBinding.overlayView.drawToBitmap(),
//                        )
//                    )
                }
            }
        }
    }

    private fun updateOverlay(
        results: ByteBuffer,
        width: Int,
        height: Int,
        inferenceTime: Long,
    ) {
        if (_fragmentGalleryBinding != null) {

            runBlocking {
                withContext(Dispatchers.Main) {
                    setUiEnabled(true)

                    String.format("%d ms", inferenceTime)

                    fragmentGalleryBinding.overlayView.setResults(
                        byteBuffer = results,
                        outputWidth = width,
                        outputHeight = height,
                    )

                    fragmentGalleryBinding.overlayView.visibility = View.GONE

                    fragmentGalleryBinding.imageResult.setImageBitmap(
                        getMaskedImage(
                            input = fragmentGalleryBinding.imageResult.drawToBitmap(),
                            mask = fragmentGalleryBinding.overlayView.drawToBitmap(),
                        )
                    )
                }
            }
        }
    }

    private fun getMaskedImage(input: Bitmap, mask: Bitmap): Bitmap {
        val result = Bitmap.createBitmap(mask.width, mask.height, Bitmap.Config.ARGB_8888)
        val mCanvas = Canvas(result)

        mCanvas.drawBitmap(input, 0f, 0f, null)
        mCanvas.drawBitmap(mask, 0f, 0f, maskPaint)
        return result
    }

    private fun segmentationError() {
        stopAllTasks()
        setUiEnabled(true)
    }

    // convert Uri to bitmap image.
    private fun Uri.toBitmap(): Bitmap {
        return if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.P) {
            val source = ImageDecoder.createSource(
                requireActivity().contentResolver, this
            )
            ImageDecoder.decodeBitmap(source)
        } else {
            MediaStore.Images.Media.getBitmap(
                requireActivity().contentResolver, this
            )
        }.copy(Bitmap.Config.ARGB_8888, true)
    }

    /**
     * Scales down the given bitmap to the specified target width while maintaining aspect ratio.
     * If the original image is already smaller than the target width, the original image is returned.
     */
    private fun Bitmap.scaleDown(targetWidth: Float): Bitmap {
        // if this image smaller than widthSize, return original image
        if (targetWidth >= width) return this
        val scaleFactor = targetWidth / width
        return Bitmap.createScaledBitmap(
            this,
            (width * scaleFactor).toInt(),
            (height * scaleFactor).toInt(),
            false
        )
    }

    private fun Bitmap.scaleDown2(targetWidth: Float): Bitmap {
        if (targetWidth >= width) {
            return this
        }

        val aspectRatio: Float = width.toFloat() / height.toFloat()

        // Calculate the target height based on the target width and aspect ratio
        val targetHeight: Int = (targetWidth / aspectRatio).toInt()

        // Create a scaled bitmap
        return Bitmap.createScaledBitmap(this, targetWidth.toInt(), targetHeight, true)

    }

    override fun onError(error: String, errorCode: Int) {
        backgroundScope?.launch {
            withContext(Dispatchers.Main) {
                segmentationError()
                Toast.makeText(requireContext(), error, Toast.LENGTH_SHORT)
                    .show()
                if (errorCode == ImageSegmenterHelper.GPU_ERROR) {
                    fragmentGalleryBinding.bottomSheetLayout.spinnerDelegate.setSelection(
                        ImageSegmenterHelper.DELEGATE_CPU, false
                    )
                }
            }
        }
    }

    override fun onResults(
        results: ByteBuffer,
        width: Int,
        height: Int,
        inferenceTime: Long,
    ) {
        updateOverlay(
            results,
            width,
            height,
            inferenceTime,
        )
    }

    companion object {
        private const val INPUT_IMAGE_MAX_WIDTH = 512F
    }

    override fun onError(error: String) {
        TODO("Not yet implemented")
    }

    override fun onResults(
        results: List<Segmentation>?,
        inferenceTime: Long,
        imageHeight: Int,
        imageWidth: Int
    ) {
        TODO("Not yet implemented")
    }
}
