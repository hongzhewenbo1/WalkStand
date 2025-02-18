package com.example.walkstand

/**
 * Multi-dimensional filter manager for accelerometer, gyroscope, and magnetometer data.
 * Uses a MultiDimKalmanFilter to filter 3D data and capture inter-axis correlations.
 */
class SensorFilterManagerMultiDim {

    // Create a 3D Kalman filter for each sensor type.
    private val accFilter = MultiDimKalmanFilter(3).apply {
        // Adjust Q and R based on data characteristics.
        Q = Array(3) { i -> DoubleArray(3) { if (i == it) 0.0005 else 0.0 } }
        R = Array(3) { i -> DoubleArray(3) { if (i == it) 0.05 else 0.0 } }
    }
    private val gyroFilter = MultiDimKalmanFilter(3).apply {
        Q = Array(3) { i -> DoubleArray(3) { if (i == it) 0.001 else 0.0 } }
        R = Array(3) { i -> DoubleArray(3) { if (i == it) 0.1 else 0.0 } }
    }
    private val magFilter = MultiDimKalmanFilter(3).apply {
        Q = Array(3) { i -> DoubleArray(3) { if (i == it) 0.0007 else 0.0 } }
        R = Array(3) { i -> DoubleArray(3) { if (i == it) 0.07 else 0.0 } }
    }

    /**
     * Update accelerometer data.
     * @param raw Raw 3D accelerometer data.
     * @return Filtered 3D accelerometer data.
     */
    fun updateAccelerometer(raw: FloatArray): FloatArray {
        val rawDouble = raw.map { it.toDouble() }.toDoubleArray()
        val filtered = accFilter.update(rawDouble)
        return filtered.map { it.toFloat() }.toFloatArray()
    }

    /**
     * Update gyroscope data.
     * @param raw Raw 3D gyroscope data.
     * @return Filtered 3D gyroscope data.
     */
    fun updateGyroscope(raw: FloatArray): FloatArray {
        val rawDouble = raw.map { it.toDouble() }.toDoubleArray()
        val filtered = gyroFilter.update(rawDouble)
        return filtered.map { it.toFloat() }.toFloatArray()
    }

    /**
     * Update magnetometer data.
     * @param raw Raw 3D magnetometer data.
     * @return Filtered 3D magnetometer data.
     */
    fun updateMagnetometer(raw: FloatArray): FloatArray {
        val rawDouble = raw.map { it.toDouble() }.toDoubleArray()
        val filtered = magFilter.update(rawDouble)
        return filtered.map { it.toFloat() }.toFloatArray()
    }
}
