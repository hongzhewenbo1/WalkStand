package com.example.walkstand
import kotlin.math.absoluteValue

/**
 * SensorFilterManager handles filtering for various sensor data.
 * It supports 1D Kalman filtering (and optionally multi-dimensional filtering)
 * with a simple example of dynamic parameter adjustment.
 */
class SensorFilterManager {

    // 1D Kalman filters for each axis (accelerometer, gyroscope, magnetometer)
    private val accFilter = Array(3) { KalmanFilter(0.0005, 0.05, 0.0, 1.0) }
    private val gyroFilter = Array(3) { KalmanFilter(0.001, 0.1, 0.0, 1.0) }
    private val magFilter = Array(3) { KalmanFilter(0.0007, 0.07, 0.0, 1.0) }

    // Optionally, you can add a multi-dimensional filter (e.g., for accelerometer fusion)
    // private val multiDimAccFilter = MultiDimKalmanFilter(3)

    /**
     * Update accelerometer data.
     * Dynamically adjusts filtering parameters based on raw values.
     * @param raw Raw accelerometer data (3-axis).
     * @return Filtered 3-axis accelerometer data.
     */
    fun updateAccelerometer(raw: FloatArray): FloatArray {
        val filtered = FloatArray(3)
        for (i in 0 until 3) {
            // Example: adjust process noise based on raw value magnitude
            if (raw[i].absoluteValue > 15) {
                accFilter[i].q = 0.001
            } else {
                accFilter[i].q = 0.0005
            }
            filtered[i] = accFilter[i].update(raw[i].toDouble()).toFloat()
        }
        return filtered
    }

    /**
     * Update gyroscope data.
     * Adjusts filtering parameters based on raw values.
     * @param raw Raw gyroscope data (3-axis).
     * @return Filtered 3-axis gyroscope data.
     */
    fun updateGyroscope(raw: FloatArray): FloatArray {
        val filtered = FloatArray(3)
        for (i in 0 until 3) {
            // Example: adjust measurement noise based on raw value magnitude
            gyroFilter[i].r = if (raw[i].absoluteValue > 1) 0.15 else 0.1
            filtered[i] = gyroFilter[i].update(raw[i].toDouble()).toFloat()
        }
        return filtered
    }

    /**
     * Update magnetometer data.
     * @param raw Raw magnetometer data (3-axis).
     * @return Filtered 3-axis magnetometer data.
     */
    fun updateMagnetometer(raw: FloatArray): FloatArray {
        val filtered = FloatArray(3)
        for (i in 0 until 3) {
            filtered[i] = magFilter[i].update(raw[i].toDouble()).toFloat()
        }
        return filtered
    }

    /**
     * Simple 1D Kalman filter implementation.
     *
     * @param q Process noise
     * @param r Measurement noise
     * @param x Initial state estimate
     * @param p Initial error covariance
     */
    class KalmanFilter(
        var q: Double, // Process noise
        var r: Double, // Measurement noise
        var x: Double, // State estimate
        var p: Double  // Error covariance
    ) {
        private var k: Double = 0.0 // Kalman gain

        /**
         * Update the filter with a new measurement.
         * @param measurement The new sensor measurement.
         * @return Updated state estimate.
         */
        fun update(measurement: Double): Double {
            p += q
            k = p / (p + r)
            x += k * (measurement - x)
            p *= (1 - k)
            return x
        }
    }
}
