# Walking-Standing Detection App

## Project Overview
This project is an **Android-based** **walking-standing detection app** that uses **TensorFlow Lite** to classify user movements as **standing or walking**.

- **Supports Android devices**
-  **Uses sensor data** (Magnetometer, Accelerometer, Gyroscope)
-  **Powered by CNN Model** (TensorFlow Lite version)
-  **Real-time movement detection**
-  **Color-coded UI feedback**
-  **Displays real-time sensor data**

---

##  Features
- ** Sensor Support**
  - **Accelerometer**
  - **Gyroscope**
  - **Magnetometer**
(In additonal to support detection: GPS,Pedometer, Geometry.)
- ** Machine Learning**
  - Uses **CNN model** (`cnn.tflite or cnn_2.tflite`) for inference.
  - **Input shape `[1, 100, 2]`** (100 time steps, 2 features)
  - **Runs on TensorFlow Lite**

- ** UI Feedback**
  - **🔴Red** = Standing
  - **🟢Green** = Walking
  - **⚪️White** = Unknown state

---

##  Installation & Running the App
#### **📦 Method : Using Android Studio**
####To run the app, you need to enable “android developer” mode
```sh
# Clone the repository
git clone https://github.com/your-repo/walking-standing-app.git
cd walking-standing-app

# Open the project in Android Studio
# Run the app
2️⃣ Dependencies
Android Studio
TensorFlow Lite
Android device (API 21+)

##  Project Structure

app/
├── manifests/                   # AndroidManifest.xml
├── kotlin+java/
│   ├── com.example.walkstand/
│   │   ├── MainActivity.kt       # Core logic: Sensor handling & model inference
│   │   ├── ui.theme/
├── assets/
│   ├── cnn.tflite                # TensorFlow Lite machine learning model
├── res/
│   ├── layout/                   # UI layout files
├── build.gradle.kts              # Gradle configuration
