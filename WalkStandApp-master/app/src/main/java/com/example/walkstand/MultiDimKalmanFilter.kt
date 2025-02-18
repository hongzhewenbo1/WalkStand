package com.example.walkstand

/**
 * Multi-dimensional Kalman Filter for an n-dimensional state vector.
 * Assumes both the state transition (F) and observation (H) matrices are identity matrices.
 */
class MultiDimKalmanFilter(val n: Int) {
    var x: DoubleArray = DoubleArray(n) { 0.0 }              // State vector
    var P: Array<DoubleArray> = identityMatrix(n)            // Covariance matrix (initialized as identity)
    val F: Array<DoubleArray> = identityMatrix(n)            // State transition matrix (identity)
    val H: Array<DoubleArray> = identityMatrix(n)            // Observation matrix (identity)
    var Q: Array<DoubleArray> = Array(n) { i ->             // Process noise (default diagonal 0.0003)
        DoubleArray(n) { if (i == it) 0.0003 else 0.0 }
    }
    var R: Array<DoubleArray> = Array(n) { i ->             // Measurement noise (default diagonal 0.03)
        DoubleArray(n) { if (i == it) 0.03 else 0.0 }
    }

    /**
     * Update the filter with an observation vector z.
     * Returns the updated state vector.
     */
    fun update(z: DoubleArray): DoubleArray {
        // Prediction
        val xPred = x.copyOf()
        val PPred = matAdd(P, Q)
        // Innovation
        val y = subtractVectors(z, xPred)
        val S = matAdd(PPred, R)
        val SInv = matInverse3x3(S)  // Only works for 3x3 matrices (n must be 3)
        val K = matMul(PPred, SInv)
        // Update state and covariance
        val Ky = matMulVec(K, y)
        x = addVectors(xPred, Ky)
        val I = identityMatrix(n)
        val I_minus_K = subtractMatrices(I, K)
        P = matMul(I_minus_K, PPred)
        return x
    }
}

/* --------- Helper Matrix Functions --------- */

// Create an n x n identity matrix
fun identityMatrix(n: Int): Array<DoubleArray> =
    Array(n) { i -> DoubleArray(n) { if (i == it) 1.0 else 0.0 } }

// Matrix addition: A + B
fun matAdd(A: Array<DoubleArray>, B: Array<DoubleArray>): Array<DoubleArray> {
    val n = A.size
    val m = A[0].size
    return Array(n) { i ->
        DoubleArray(m) { j ->
            A[i][j] + B[i][j]
        }
    }
}

// Vector addition: a + b
fun addVectors(a: DoubleArray, b: DoubleArray): DoubleArray {
    return DoubleArray(a.size) { i -> a[i] + b[i] }
}

// Vector subtraction: a - b
fun subtractVectors(a: DoubleArray, b: DoubleArray): DoubleArray {
    return DoubleArray(a.size) { i -> a[i] - b[i] }
}

// Matrix multiplication: A * B
fun matMul(A: Array<DoubleArray>, B: Array<DoubleArray>): Array<DoubleArray> {
    val n = A.size
    val m = B[0].size
    val p = A[0].size
    return Array(n) { i ->
        DoubleArray(m) { j ->
            var sum = 0.0
            for (k in 0 until p) {
                sum += A[i][k] * B[k][j]
            }
            sum
        }
    }
}

// Multiply matrix A with vector v: A * v
fun matMulVec(A: Array<DoubleArray>, v: DoubleArray): DoubleArray {
    val n = A.size
    val m = A[0].size
    return DoubleArray(n) { i ->
        var sum = 0.0
        for (j in 0 until m) {
            sum += A[i][j] * v[j]
        }
        sum
    }
}

// Matrix subtraction: A - B
fun subtractMatrices(A: Array<DoubleArray>, B: Array<DoubleArray>): Array<DoubleArray> {
    val n = A.size
    val m = A[0].size
    return Array(n) { i ->
        DoubleArray(m) { j ->
            A[i][j] - B[i][j]
        }
    }
}

// Inverse of a 3x3 matrix
fun matInverse3x3(A: Array<DoubleArray>): Array<DoubleArray> {
    val a = A[0][0]; val b = A[0][1]; val c = A[0][2]
    val d = A[1][0]; val e = A[1][1]; val f = A[1][2]
    val g = A[2][0]; val h = A[2][1]; val i = A[2][2]
    val det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)
    if (det == 0.0) throw Exception("Matrix is singular")
    val invDet = 1.0 / det
    return arrayOf(
        doubleArrayOf((e * i - f * h) * invDet, (c * h - b * i) * invDet, (b * f - c * e) * invDet),
        doubleArrayOf((f * g - d * i) * invDet, (a * i - c * g) * invDet, (c * d - a * f) * invDet),
        doubleArrayOf((d * h - e * g) * invDet, (b * g - a * h) * invDet, (a * e - b * d) * invDet)
    )
}
