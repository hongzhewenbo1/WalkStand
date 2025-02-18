package com.example.walkstand

import android.content.Context
import android.graphics.Color
import android.os.Handler
import android.os.HandlerThread
import org.tensorflow.lite.Interpreter
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel

class InferenceEngine(private val context: Context, modelFileName: String) {

    private val interpreter: Interpreter
    private val inputBuffer: ByteBuffer
    private val outputBuffer: ByteBuffer
    private val inferenceThread: HandlerThread
    private val inferenceHandler: Handler

    init {
        interpreter = Interpreter(loadModelFile(modelFileName))
        // 模型输入为 100 个采样点 * 2 个特征，每个 float 4 字节，共 800 字节
        inputBuffer = ByteBuffer.allocateDirect(800).order(ByteOrder.nativeOrder())
        // 模型输出为 2 个 float（预测 standing/walking），共 8 字节
        outputBuffer = ByteBuffer.allocateDirect(8).order(ByteOrder.nativeOrder())
        inferenceThread = HandlerThread("InferenceThread").apply { start() }
        inferenceHandler = Handler(inferenceThread.looper)
    }

    /**
     * 异步运行推理，传入数据数组，并在推理完成后通过 onResult 返回预测结果
     */
    fun runInference(inputData: FloatArray, onResult: (String, Int) -> Unit) {
        inferenceHandler.post {
            inputBuffer.rewind()
            // 将输入数据写入 ByteBuffer
            inputData.forEach { inputBuffer.putFloat(it) }
            outputBuffer.rewind()
            // 运行模型推理
            interpreter.run(inputBuffer, outputBuffer)
            outputBuffer.rewind()
            val outputArray = FloatArray(2)
            outputBuffer.asFloatBuffer().get(outputArray)
            // 根据输出确定预测类别
            val maxIndex = if (outputArray[0] > outputArray[1]) 0 else 1
            val predictionText = if (maxIndex == 1) "Walking" else "Standing"
            val predictionColor = if (maxIndex == 1) Color.GREEN else Color.RED
            onResult(predictionText, predictionColor)
        }
    }

    fun destroy() {
        interpreter.close()
        inferenceThread.quitSafely()
    }

    /**
     * 从 assets 加载 TFLite 模型文件
     */
    private fun loadModelFile(fileName: String): ByteBuffer {
        val assetFileDescriptor = context.assets.openFd(fileName)
        val fileInputStream = assetFileDescriptor.createInputStream()
        val fileChannel = fileInputStream.channel
        return fileChannel.map(
            FileChannel.MapMode.READ_ONLY,
            assetFileDescriptor.startOffset,
            assetFileDescriptor.declaredLength
        )
    }
}
