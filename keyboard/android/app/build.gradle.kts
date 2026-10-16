plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

android {
    namespace = "mn.translit.keyboard"
    compileSdk = 34

    defaultConfig {
        applicationId = "mn.translit.keyboard"
        minSdk = 26
        targetSdk = 34
        versionCode = 1
        versionName = "0.1"
    }
    buildTypes {
        release {
            isMinifyEnabled = false
        }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlinOptions {
        jvmTarget = "17"
    }
    // the ONNX graphs and the dictionary are copied here by tools/pack_keyboard_assets.py (not in git)
    androidResources {
        noCompress += listOf("onnx", "json")
    }
}

dependencies {
    implementation("com.microsoft.onnxruntime:onnxruntime-android:1.19.2")
    implementation("androidx.core:core-ktx:1.13.1")
}
