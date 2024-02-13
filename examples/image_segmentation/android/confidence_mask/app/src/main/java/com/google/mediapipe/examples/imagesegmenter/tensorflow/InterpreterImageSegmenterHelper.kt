package com.google.mediapipe.examples.imagesegmenter.tensorflow

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.CompatibilityList
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.task.core.BaseOptions
import org.tensorflow.lite.task.vision.segmenter.ImageSegmenter
import org.tensorflow.lite.task.vision.segmenter.OutputType
import org.tensorflow.lite.task.vision.segmenter.Segmentation
import java.io.FileInputStream
import java.io.IOException
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

class InterpreterImageSegmenterHelper(
    var numThreads: Int = 2,
    var currentDelegate: Int = 0,
    val context: Context,
) {

    private var interpreter: Interpreter?

    init {
        interpreter = Interpreter(loadModelFile(context, MODEL_DEEPLABV3))
        interpreter?.allocateTensors()
    }

    @Throws(IOException::class)
    private fun loadModelFile(context: Context, modelPath: String): MappedByteBuffer {
        val fileDescriptor = context.assets.openFd(modelPath)
        val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        val startOffset = fileDescriptor.startOffset
        val declaredLength = fileDescriptor.declaredLength
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    }

    fun clearImageSegmenter() {
        interpreter = null
    }

    fun segmentImageWithInterpreter(
        bitmap: Bitmap,
    ): ByteBuffer {
        val scaledBitmap = Bitmap.createScaledBitmap(bitmap, 257, 257, true)

        val input = ByteBuffer.allocateDirect(257*257*3*4).order(ByteOrder.nativeOrder())
//        for (y in 0 until 257) {
//            for (x in 0 until 257) {
//                val px = scaledBitmap.getPixel(x, y)
//
//                // Get channel values from the pixel value.
//                val r = Color.red(px)
//                val g = Color.green(px)
//                val b = Color.blue(px)
//
//                // Normalize channel values to [-1.0, 1.0]. This requirement depends on the model.
//                // For example, some models might require values to be normalized to the range
//                // [0.0, 1.0] instead.
//                val rf = (r - 127) / 255f
//                val gf = (g - 127) / 255f
//                val bf = (b - 127) / 255f
//
//                input.putFloat(rf)
//                input.putFloat(gf)
//                input.putFloat(bf)
//            }
//        }

        val bufferSize = 5548116 * java.lang.Float.SIZE / java.lang.Byte.SIZE
        val modelOutput = ByteBuffer.allocateDirect(bufferSize).order(ByteOrder.nativeOrder())

        interpreter?.run(input, modelOutput)

        return modelOutput
    }

    data class SegmentationResult(
        val result: List<Segmentation>?,
        val tensorImage: TensorImage,
    )

    interface TensorSegmentationListener {
        fun onError(error: String)
        fun onResults(
            results: List<Segmentation>?,
            inferenceTime: Long,
            imageHeight: Int,
            imageWidth: Int
        )
    }

    companion object {
        const val DELEGATE_CPU = 0
        const val DELEGATE_GPU = 1
        const val DELEGATE_NNAPI = 2
        const val MODEL_DEEPLABV3 = "deeplabv3.tflite"

        private const val TAG = "Image Segmentation Helper"
    }

}