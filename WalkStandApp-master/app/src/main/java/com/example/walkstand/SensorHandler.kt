package com.example.walkstand

import android.content.Context
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import kotlin.math.sqrt

class SensorHandler(
    private val context: Context,
    private val inferenceEngine: InferenceEngine,
    // 当获得预测结果后调用
    private val onPrediction: (String, Int) -> Unit
) : SensorEventListener {

    private val sensorManager =
        context.getSystemService(Context.SENSOR_SERVICE) as SensorManager

    // 获取需要的传感器
    private val magnetometer = sensorManager.getDefaultSensor(Sensor.TYPE_MAGNETIC_FIELD)
    private val accelerometer = sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER)
    private val gyroscope = sensorManager.getDefaultSensor(Sensor.TYPE_GYROSCOPE)
    private val stepCounter = sensorManager.getDefaultSensor(Sensor.TYPE_STEP_COUNTER)

    companion object {
        // 高频传感器采样周期
        private const val SAMPLING_PERIOD_US = 10000
        // 缓冲区大小
        private const val BUFFER_SIZE = 100
    }

    // 标识是否正在采集数据
    private var running = false

    // 数据缓冲区：每个采样点存储
    private val combinedBuffer = FloatArray(BUFFER_SIZE * 2)
    private var bufferIndex = 0

    fun start() {
        running = true
        // 注册所有需要的传感器
        magnetometer?.let { sensorManager.registerListener(this, it, SAMPLING_PERIOD_US) }
        accelerometer?.let { sensorManager.registerListener(this, it, SAMPLING_PERIOD_US) }
        gyroscope?.let { sensorManager.registerListener(this, it, SAMPLING_PERIOD_US) }
        stepCounter?.let { sensorManager.registerListener(this, it, SensorManager.SENSOR_DELAY_NORMAL) }
    }

    fun stop() {
        running = false
        sensorManager.unregisterListener(this)
        // 重置采样索引
        bufferIndex = 0
    }

    fun isRunning() = running

    override fun onSensorChanged(event: SensorEvent) {
        if (!running) return

        when (event.sensor.type) {
            Sensor.TYPE_GYROSCOPE -> {
                // 计算陀螺仪数据的幅值
                val gyro = event.values
                val gyroMagnitude = sqrt(
                    gyro[0] * gyro[0] + gyro[1] * gyro[1] + gyro[2] * gyro[2]
                )
                // 将陀螺仪幅值存入缓冲区对应位置
                if (bufferIndex < BUFFER_SIZE) {
                    combinedBuffer[bufferIndex * 2 + 1] = gyroMagnitude
                }
            }
            Sensor.TYPE_MAGNETIC_FIELD -> {
                // 计算磁力计数据的幅值
                val mag = event.values
                val magAbs = sqrt(
                    mag[0] * mag[0] + mag[1] * mag[1] + mag[2] * mag[2]
                )
                if (bufferIndex < BUFFER_SIZE) {
                    combinedBuffer[bufferIndex * 2] = magAbs
                    bufferIndex++
                }
                // 当采集到足够数据后，调用推理引擎
                if (bufferIndex >= BUFFER_SIZE) {

                    val inputData = combinedBuffer.copyOf()
                    inferenceEngine.runInference(inputData) { predictionText, predictionColor ->
                        onPrediction(predictionText, predictionColor)
                    }

                    bufferIndex = 0
                }
            }

        }
    }

    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) {
        // 不处理精度变化
    }

    fun destroy() {
        stop()

    }
}
