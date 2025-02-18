package com.example.walkstand

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Color
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.location.Location
import android.os.Bundle
import android.os.Handler
import android.os.HandlerThread
import android.os.Looper
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import com.example.walkstand.databinding.ActivityMainBinding
import com.google.android.gms.location.*
import org.tensorflow.lite.Interpreter
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.abs
import kotlin.math.sqrt
import java.nio.channels.FileChannel

class MainActivity : AppCompatActivity(), SensorEventListener {

    // View binding
    private lateinit var binding: ActivityMainBinding

    // Sensor manager and sensors
    private lateinit var sensorManager: SensorManager
    private lateinit var magnetometer: Sensor
    private var accelerometer: Sensor? = null
    private var stepCounter: Sensor? = null
    private lateinit var gyroscope: Sensor

    // TFLite model interpreter and buffers
    private lateinit var interpreter: Interpreter
    private lateinit var outputBuffer: ByteBuffer
    private lateinit var inputBuffer: ByteBuffer

    // Inference thread and handler
    private lateinit var inferenceThread: HandlerThread
    private lateinit var inferenceHandler: Handler
    private val mainHandler = Handler(Looper.getMainLooper())

    // Data buffers for model inference
    private val bufferSize = 100
    private var bufferIndex = 0
    private val combinedBuffer = FloatArray(bufferSize * 2)

    // Current prediction color and timer for red color
    private var currentPredictionColor: Int = Color.WHITE
    private var lastRedTime: Long = 0

    // Latest accelerometer and magnetometer data (for orientation)
    private var accelValues: FloatArray? = null
    private var magValues: FloatArray? = null

    // Kalman filter for smoothing azimuth (1D)
    private val kalmanAzimuth = SensorFilterManager.KalmanFilter(0.0005, 0.05, 0.0, 1.0)

    // Latest gyroscope magnitude
    private var latestGyroMagnitude: Float = 0.0f

    // Pedometer data
    private var initialStepCount: Float = -1f
    private var lastStepCount: Float = -1f
    private var stepCountAtInference: Float = -1f

    // Location variables
    private lateinit var fusedLocationClient: FusedLocationProviderClient
    private lateinit var locationCallback: LocationCallback
    private var lastLocation: Location? = null
    private val locationMovementThreshold = 3.0f

    // Orientation variables
    private var lastOrientationDeg: Double? = null
    private var stableOrientationStartTime: Long = 0
    private var isOrientationStable: Boolean = false
    private var standingOrientation: Double? = null
    private var currentAzimuthGlobal: Double = 0.0

    companion object {
        private const val REQUEST_LOCATION_PERMISSION = 1
        private const val ORIENTATION_STABLE_DURATION = 1000L
        private const val ORIENTATION_STABLE_THRESHOLD = 20.0
        // Sensor sampling period: 10ms (10000 microseconds)
        private const val SAMPLING_PERIOD_US = 10000
    }

    // App running state
    private var isRunning: Boolean = false

    // Multi-dimensional sensor filter
    private val sensorFilterManagerMultiDim = SensorFilterManagerMultiDim()

    // Calibration variables
    private var isCalibrating = false
    private var isCalibrated = false
    private val accelCalibrationData = mutableListOf<FloatArray>()
    private val gyroCalibrationData = mutableListOf<FloatArray>()
    private val magCalibrationData = mutableListOf<FloatArray>()
    private var accelBias = FloatArray(3) { 0f }
    private var gyroBias = FloatArray(3) { 0f }
    private var magBias = FloatArray(3) { 0f }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        // Set UI text sizes
        binding.predictionLabelTextView.textSize = 24f
        binding.predictionDataTextView.textSize = 24f
        binding.startStopButton.textSize = 24f
        binding.calibrateButton.textSize = 24f

        // Initialize sensor manager and sensors
        sensorManager = getSystemService(SENSOR_SERVICE) as SensorManager
        magnetometer = sensorManager.getDefaultSensor(Sensor.TYPE_MAGNETIC_FIELD)!!
        accelerometer = sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER)
        stepCounter = sensorManager.getDefaultSensor(Sensor.TYPE_STEP_COUNTER)
        gyroscope = sensorManager.getDefaultSensor(Sensor.TYPE_GYROSCOPE)!!

        if (accelerometer == null) {
            binding.accelerometerDataTextView.text = "Accelerometer: Not available"
        }
        if (stepCounter == null) {
            binding.pedometerDataTextView.text = "Pedometer: Not available"
        }

        // Initialize TFLite interpreter (ensure cnn tflite  in assets)
        interpreter = Interpreter(loadModelFile("cnn_2.tflite"))
        inputBuffer = ByteBuffer.allocateDirect(800)
        inputBuffer.order(ByteOrder.nativeOrder())
        outputBuffer = ByteBuffer.allocateDirect(8)
        outputBuffer.order(ByteOrder.nativeOrder())

        // Initialize location client and callback
        fusedLocationClient = LocationServices.getFusedLocationProviderClient(this)
        setupLocationCallback()

        inferenceThread = HandlerThread("InferenceThread")
        inferenceThread.start()
        inferenceHandler = Handler(inferenceThread.looper)

        binding.calibrateButton.setOnClickListener {
            startCalibration()
        }

        // Start/Stop button click listener
        binding.startStopButton.setOnClickListener {
            if (!isRunning) {
                // Ensure calibration is done before starting
                if (!isCalibrated) {
                    binding.predictionDataTextView.text = "Please calibrate your sensors first."
                    return@setOnClickListener
                }
                isRunning = true
                binding.startStopButton.text = "Stop"
                sensorManager.registerListener(this, magnetometer, SAMPLING_PERIOD_US)
                accelerometer?.let { sensorManager.registerListener(this, it, SAMPLING_PERIOD_US) }
                stepCounter?.let { sensorManager.registerListener(this, it, SensorManager.SENSOR_DELAY_NORMAL) }
                sensorManager.registerListener(this, gyroscope, SAMPLING_PERIOD_US)
                startLocationUpdates()
            } else {
                isRunning = false
                binding.startStopButton.text = "Start"
                sensorManager.unregisterListener(this)
                stopLocationUpdates()
                mainHandler.removeCallbacksAndMessages(null)
                binding.colorFrame.setBackgroundColor(Color.WHITE)
                binding.predictionDataTextView.text = ""
                currentPredictionColor = Color.WHITE
            }
        }
    }

    // Setup location callback
    private fun setupLocationCallback() {
        locationCallback = object : LocationCallback() {
            override fun onLocationResult(locationResult: LocationResult) {
                for (location in locationResult.locations) {
                    lastLocation?.let { lastLoc ->
                        if (location.distanceTo(lastLoc) >= locationMovementThreshold) {
                            mainHandler.post {
                                binding.colorFrame.setBackgroundColor(Color.GREEN)
                                binding.predictionDataTextView.text = "Walking (via GPS)"
                                binding.predictionDataTextView.setTextColor(Color.GREEN)
                            }
                        }
                    }
                    lastLocation = location
                }
            }
        }
    }

    // Start location updates
    private fun startLocationUpdates() {
        if (ActivityCompat.checkSelfPermission(this, Manifest.permission.ACCESS_FINE_LOCATION)
            != PackageManager.PERMISSION_GRANTED &&
            ActivityCompat.checkSelfPermission(this, Manifest.permission.ACCESS_COARSE_LOCATION)
            != PackageManager.PERMISSION_GRANTED
        ) {
            ActivityCompat.requestPermissions(
                this,
                arrayOf(Manifest.permission.ACCESS_FINE_LOCATION, Manifest.permission.ACCESS_COARSE_LOCATION),
                REQUEST_LOCATION_PERMISSION
            )
            return
        }
        val locationRequest = LocationRequest.create().apply {
            interval = 2000
            fastestInterval = 1000
            priority = LocationRequest.PRIORITY_HIGH_ACCURACY
        }
        fusedLocationClient.requestLocationUpdates(locationRequest, locationCallback, Looper.getMainLooper())
    }

    // Stop location updates
    private fun stopLocationUpdates() {
        fusedLocationClient.removeLocationUpdates(locationCallback)
    }

    override fun onSensorChanged(event: SensorEvent) {
        // During calibration, record raw data and return
        if (isCalibrating) {
            when (event.sensor.type) {
                Sensor.TYPE_ACCELEROMETER -> accelCalibrationData.add(event.values.clone())
                Sensor.TYPE_GYROSCOPE -> gyroCalibrationData.add(event.values.clone())
                Sensor.TYPE_MAGNETIC_FIELD -> magCalibrationData.add(event.values.clone())
            }
            return
        }

        // Prompt calibration if not done
        if (!isCalibrated) {
            binding.predictionDataTextView.text = "Please calibrate your sensors"
            return
        }

        when (event.sensor.type) {
            Sensor.TYPE_ACCELEROMETER -> {
                val rawAcc = event.values
                // Apply calibration bias
                val calibratedAcc = floatArrayOf(
                    rawAcc[0] - accelBias[0],
                    rawAcc[1] - accelBias[1],
                    rawAcc[2] - accelBias[2]
                )
                // Update accelerometer data using multi-dim filter
                val filteredAcc = sensorFilterManagerMultiDim.updateAccelerometer(calibratedAcc)
                binding.accelerometerDataTextView.text =
                    "Accelerometer Filtered:\nx=%.2f, y=%.2f, z=%.2f"
                        .format(filteredAcc[0], filteredAcc[1], filteredAcc[2])
                // Save for orientation calculation
                accelValues = calibratedAcc.clone()
            }
            Sensor.TYPE_GYROSCOPE -> {
                val rawGyro = event.values
                val calibratedGyro = floatArrayOf(
                    rawGyro[0] - gyroBias[0],
                    rawGyro[1] - gyroBias[1],
                    rawGyro[2] - gyroBias[2]
                )
                val filteredGyro = sensorFilterManagerMultiDim.updateGyroscope(calibratedGyro)
                latestGyroMagnitude = sqrt(
                    filteredGyro[0] * filteredGyro[0] +
                            filteredGyro[1] * filteredGyro[1] +
                            filteredGyro[2] * filteredGyro[2]
                )
                binding.gyroscopeDataTextView?.text =
                    "Gyroscope Filtered:\nx=%.2f, y=%.2f, z=%.2f"
                        .format(filteredGyro[0], filteredGyro[1], filteredGyro[2])
            }
            Sensor.TYPE_MAGNETIC_FIELD -> {
                val rawMag = event.values
                val calibratedMag = floatArrayOf(
                    rawMag[0] - magBias[0],
                    rawMag[1] - magBias[1],
                    rawMag[2] - magBias[2]
                )
                val filteredMag = sensorFilterManagerMultiDim.updateMagnetometer(calibratedMag)
                binding.magnetometerDataTextView.text =
                    "Magnetometer Filtered:\nx=%.2f, y=%.2f, z=%.2f"
                        .format(filteredMag[0], filteredMag[1], filteredMag[2])
                // Save for orientation calculation
                magValues = filteredMag

                // Calculate magnetometer magnitude and store with latest gyroscope magnitude
                val magAbs = sqrt(
                    filteredMag[0] * filteredMag[0] +
                            filteredMag[1] * filteredMag[1] +
                            filteredMag[2] * filteredMag[2]
                )
                combinedBuffer[bufferIndex * 2] = magAbs
                combinedBuffer[bufferIndex * 2 + 1] = latestGyroMagnitude
                bufferIndex++

                // Run inference every 100 samples
                if (bufferIndex >= bufferSize) {
                    bufferIndex = 0
                    inputBuffer.rewind()
                    for (value in combinedBuffer) {
                        inputBuffer.putFloat(value)
                    }
                    inferenceHandler.post {
                        outputBuffer.rewind()
                        interpreter.run(inputBuffer, outputBuffer)
                        outputBuffer.rewind()
                        val outputArray = FloatArray(2)
                        outputBuffer.asFloatBuffer().get(outputArray)
                        val maxIndex = if (outputArray[0] > outputArray[1]) 0 else 1

                        var newColor = when (maxIndex) {
                            0 -> Color.RED    // Standing
                            1 -> Color.GREEN  // Walking
                            else -> Color.WHITE
                        }
                        var newPrediction = if (maxIndex == 1) "Walking" else "Standing"

                        // Supplemental check: step counter
                        if (lastStepCount - stepCountAtInference > 1) {
                            newPrediction = "Walking"
                            newColor = Color.GREEN
                        }
                        stepCountAtInference = lastStepCount

                        // Supplemental check: orientation stability
                        if (newPrediction == "Walking" && getStandingOrientation() != null) {
                            val orientationDiff = abs(currentAzimuthGlobal - standingOrientation!!)
                            if (orientationDiff < ORIENTATION_STABLE_THRESHOLD) {
                                newPrediction = "Standing"
                                newColor = Color.RED
                            }
                        }
                        currentPredictionColor = newColor
                        if (newColor == Color.RED) {
                            lastRedTime = System.currentTimeMillis()
                        }
                        mainHandler.post {
                            binding.colorFrame.setBackgroundColor(newColor)
                            binding.predictionDataTextView.text = newPrediction
                            binding.predictionDataTextView.setTextColor(newColor)
                        }
                    }
                }
            }
            Sensor.TYPE_STEP_COUNTER -> {
                val steps = event.values[0]
                if (initialStepCount < 0) {
                    initialStepCount = steps
                    lastStepCount = steps
                    stepCountAtInference = steps
                }
                lastStepCount = steps
                binding.pedometerDataTextView.text =
                    "Pedometer: %.0f steps".format(steps - initialStepCount)
            }
        }

        // Orientation calculation using accelerometer and magnetometer data
        if (accelValues != null && magValues != null) {
            val rMatrix = FloatArray(9)
            val success = SensorManager.getRotationMatrix(rMatrix, null, accelValues, magValues)
            if (success) {
                val orientation = FloatArray(3)
                SensorManager.getOrientation(rMatrix, orientation)
                var currentAzimuth = Math.toDegrees(orientation[0].toDouble())
                currentAzimuth = kalmanAzimuth.update(currentAzimuth)
                currentAzimuthGlobal = currentAzimuth
                binding.orientationDataTextView.text = "Orientation (Azimuth): %.2f°".format(currentAzimuth)

                val currentTime = System.currentTimeMillis()
                if (lastOrientationDeg == null) {
                    lastOrientationDeg = currentAzimuth
                    stableOrientationStartTime = currentTime
                    isOrientationStable = false
                } else {
                    val diff = abs(currentAzimuth - lastOrientationDeg!!)
                    if (diff < ORIENTATION_STABLE_THRESHOLD) {
                        if (currentTime - stableOrientationStartTime > ORIENTATION_STABLE_DURATION) {
                            isOrientationStable = true
                            standingOrientation = lastOrientationDeg
                        }
                    } else {
                        lastOrientationDeg = currentAzimuth
                        stableOrientationStartTime = currentTime
                        isOrientationStable = false
                        standingOrientation = null
                    }
                }
            }
        }
    }

    private fun getStandingOrientation() = standingOrientation

    override fun onAccuracyChanged(sensor: Sensor, accuracy: Int) {
        // No action on accuracy changes
    }

    // Load TFLite model file from assets
    private fun loadModelFile(filename: String): ByteBuffer {
        val assetFileDescriptor = assets.openFd(filename)
        val fileInputStream = assetFileDescriptor.createInputStream()
        val fileChannel = fileInputStream.channel
        val startOffset = assetFileDescriptor.startOffset
        val declaredLength = assetFileDescriptor.declaredLength
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    }

    override fun onDestroy() {
        super.onDestroy()
        interpreter.close()
        sensorManager.unregisterListener(this)
        stopLocationUpdates()
        inferenceThread.quitSafely()
    }

    // ------------------------------------------- Calibration Process ------------------------------------------------------

    // Start sensor calibration
    private fun startCalibration() {
        if (isCalibrating) return
        isCalibrating = true
        isCalibrated = false

        // Clear previous calibration data
        accelCalibrationData.clear()
        gyroCalibrationData.clear()
        magCalibrationData.clear()

        binding.predictionDataTextView.text = "Calibration in progress..."

        // Register sensors for calibration sampling
        accelerometer?.let { sensorManager.registerListener(this, it, SAMPLING_PERIOD_US) }
        sensorManager.registerListener(this, gyroscope, SAMPLING_PERIOD_US)
        sensorManager.registerListener(this, magnetometer, SAMPLING_PERIOD_US)

        // End calibration after 5 seconds and compute biases
        mainHandler.postDelayed({
            computeCalibrationBiases()
            isCalibrating = false
            isCalibrated = true
            binding.predictionDataTextView.text = "Calibration done!"

            // Unregister calibration listeners
            accelerometer?.let { sensorManager.unregisterListener(this, it) }
            sensorManager.unregisterListener(this, gyroscope)
            sensorManager.unregisterListener(this, magnetometer)
        }, 5000)
    }

    // Compute sensor biases from calibration data
    private fun computeCalibrationBiases() {
        // Accelerometer: average the values and subtract gravity on Z-axis
        if (accelCalibrationData.isNotEmpty()) {
            val sum = FloatArray(3) { 0f }
            accelCalibrationData.forEach { values ->
                sum[0] += values[0]
                sum[1] += values[1]
                sum[2] += values[2]
            }
            accelBias[0] = sum[0] / accelCalibrationData.size
            accelBias[1] = sum[1] / accelCalibrationData.size
            accelBias[2] = (sum[2] / accelCalibrationData.size) - SensorManager.GRAVITY_EARTH
        }

        // Gyroscope: average values (should be 0 at rest)
        if (gyroCalibrationData.isNotEmpty()) {
            val sum = FloatArray(3) { 0f }
            gyroCalibrationData.forEach { values ->
                sum[0] += values[0]
                sum[1] += values[1]
                sum[2] += values[2]
            }
            gyroBias[0] = sum[0] / gyroCalibrationData.size
            gyroBias[1] = sum[1] / gyroCalibrationData.size
            gyroBias[2] = sum[2] / gyroCalibrationData.size
        }

        // Magnetometer: average each axis as bias
        if (magCalibrationData.isNotEmpty()) {
            val sum = FloatArray(3) { 0f }
            magCalibrationData.forEach { values ->
                sum[0] += values[0]
                sum[1] += values[1]
                sum[2] += values[2]
            }
            magBias[0] = sum[0] / magCalibrationData.size
            magBias[1] = sum[1] / magCalibrationData.size
            magBias[2] = sum[2] / magCalibrationData.size
        }
    }
}
