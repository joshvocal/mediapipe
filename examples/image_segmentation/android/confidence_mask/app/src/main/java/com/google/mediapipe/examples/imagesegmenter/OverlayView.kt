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

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.util.AttributeSet
import android.view.View
import com.google.mediapipe.tasks.vision.core.RunningMode
import org.tensorflow.lite.support.image.TensorImage
import java.nio.ByteBuffer
import kotlin.math.min

class OverlayView(context: Context?, attrs: AttributeSet?) : View(context, attrs) {
    private var scaleBitmap: Bitmap? = null
    private var runningMode: RunningMode = RunningMode.IMAGE

    fun clear() {
        scaleBitmap = null
        invalidate()
    }

    override fun draw(canvas: Canvas) {
        super.draw(canvas)
        scaleBitmap?.let {
            canvas.drawBitmap(it, 0f, 0f, null)
        }
    }

    fun setRunningMode(runningMode: RunningMode) {
        this.runningMode = runningMode
    }

    fun setTensorResults(
        byteBuffer: ByteBuffer,
        maskTensor: TensorImage,
        outputWidth: Int,
        outputHeight: Int
    ) {
        // Create the mask bitmap with colors and the set of detected labels.
        val pixels = IntArray(byteBuffer.capacity())

        for (i in pixels.indices) {
            val index = byteBuffer.get(i).toUInt() % 20U

            val color = if (index == 0U) Color.TRANSPARENT else Color.BLUE

            pixels[i] = color
        }
        val image = Bitmap.createBitmap(
            pixels,
            maskTensor.width,
            maskTensor.height,
            Bitmap.Config.ARGB_8888
        )

        // RunningMode.IMAGE
        val scaleFactor = min(width * 1f / outputWidth, height * 1f / outputHeight)

        val scaleWidth = (outputWidth * scaleFactor).toInt()
        val scaleHeight = (outputHeight * scaleFactor).toInt()

        scaleBitmap = Bitmap.createScaledBitmap(
            image, scaleWidth, scaleHeight, false
        )

        invalidate()
    }

    fun setResults(
        byteBuffer: ByteBuffer,
        outputWidth: Int,
        outputHeight: Int
    ) {
        // Create the mask bitmap with colors and the set of detected labels.
        val pixels = IntArray(byteBuffer.capacity())

        for (i in pixels.indices) {
            val index = byteBuffer.get(i).toUInt() % 20U

            val color = if (index == 0U) Color.TRANSPARENT else Color.BLUE

            pixels[i] = color
        }
        val image = Bitmap.createBitmap(
            pixels,
            outputWidth,
            outputHeight,
            Bitmap.Config.ARGB_8888
        )

        // RunningMode.IMAGE
        val scaleFactor = min(width * 1f / outputWidth, height * 1f / outputHeight)

        val scaleWidth = (outputWidth * scaleFactor).toInt()
        val scaleHeight = (outputHeight * scaleFactor).toInt()

        scaleBitmap = Bitmap.createScaledBitmap(
            image, scaleWidth, scaleHeight, false
        )

        invalidate()
    }

    companion object {
        const val ALPHA_COLOR = 128
    }
}

fun Int.toAlphaColor(): Int {
    return Color.argb(
        OverlayView.ALPHA_COLOR,
        Color.red(this),
        Color.green(this),
        Color.blue(this)
    )
}