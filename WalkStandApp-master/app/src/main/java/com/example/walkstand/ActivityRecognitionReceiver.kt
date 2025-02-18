package com.example.walkstand

import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.util.Log
import com.google.android.gms.location.ActivityRecognitionResult
import com.google.android.gms.location.DetectedActivity

class ActivityRecognitionReceiver : BroadcastReceiver() {
    override fun onReceive(context: Context, intent: Intent) {
        if (ActivityRecognitionResult.hasResult(intent)) {
            val result = ActivityRecognitionResult.extractResult(intent) ?: return
            val detectedActivities = result.probableActivities
            for (activity in detectedActivities) {
                when (activity.type) {
                    DetectedActivity.WALKING -> {
                        Log.d("ActivityRecognition", "Walking detected with confidence ${activity.confidence}")
                    }
                    DetectedActivity.RUNNING -> {
                        Log.d("ActivityRecognition", "Running detected with confidence ${activity.confidence}")
                    }
                    DetectedActivity.STILL -> {
                        Log.d("ActivityRecognition", "Still detected with confidence ${activity.confidence}")
                    }
                    else -> {
                        Log.d("ActivityRecognition", "Other activity: ${activity.type} with confidence ${activity.confidence}")
                    }
                }
            }
        }
    }
}
