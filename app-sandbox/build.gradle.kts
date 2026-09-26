plugins {
    alias(libs.plugins.android.application)
    // AGP 9+ 內建 Kotlin 支援，不再套 kotlin.android
}

// LLM key **不再**於 build 時注入 APK（2026-09-26 拔掉：APK 外流＝key 外流）。sandbox 改在 app 內輸入、存 prefs；
// repo 根的 api-keys.properties 只剩桌面 parity 腳本（translate_parity.py）在讀。
android {
    namespace = "li.joye.yakuyomi.sandbox"
    compileSdk = 37

    defaultConfig {
        applicationId = "li.joye.yakuyomi.sandbox"
        minSdk = 26
        targetSdk = 36
        versionCode = 2
        versionName = "0.3-ncnn"
        // 只打 arm64-v8a（實機）：NCNN 原生層由 :engine 提供（libyakuyomi_ncnn，ncnn 預編庫本來就只出 arm64），
        // 其他 ABI 沒有可用的推論後端，打進去只是白佔空間；sandbox 不再自帶 native build。
        ndk { abiFilters += "arm64-v8a" }
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro",
            )
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    buildFeatures {
        viewBinding = true
        buildConfig = true
    }

    // 模型分離（BYOM）：模型檔（.ncnn.param/.bin）不放 assets、由 app 從使用者選的資料夾複製到 filesDir 載入，
    // APK 只帶測試圖與 :engine 的字典/字型。★ ignoreAssetsPattern 仍要留著：engine/src/main/assets/models/ 是 gitignore 的
    // 本機夾，開發機上常殘留舊 .onnx／實驗用 .param/.bin（曾因拿掉這條 APK 從 78MB 變 415MB）。
    androidResources {
        ignoreAssetsPattern = "*.onnx:*.param:*.bin"
    }
    // （舊的 ORT QNN jniLibs 瘦身規則隨 ORT 一起拔掉：引擎已無 onnxruntime 相依，沒有 libQnn* 可排除。）
}

// AGP 9 內建 Kotlin：jvmTarget 改在 kotlin{} 設
kotlin {
    compilerOptions {
        jvmTarget.set(org.jetbrains.kotlin.gradle.dsl.JvmTarget.JVM_17)
    }
}

dependencies {
    implementation(project(":engine"))
    // 夜讀：上機驗證用。只接 :nightread 核心（純 Kotlin 管線，includeBuild 靠 group:name 替換）；人物遮罩推論走 :engine 的
    // NCNN 分割器（CharSegmenter：YoloSegSegmenter／CsegSegmenter），:nightread-ort（ORT 版 CharMaskOrt）已退役不再依賴。
    implementation("li.joye.yakuyomi:nightread:0.1.0")
    implementation(libs.androidx.core.ktx)
    implementation(libs.androidx.appcompat)
    implementation(libs.androidx.lifecycle.runtime.ktx)
    implementation(libs.kotlinx.coroutines.android)
    implementation("androidx.documentfile:documentfile:1.0.1") // SAF 資料夾讀檔
    implementation("androidx.activity:activity-ktx:1.8.2")      // registerForActivityResult / OpenDocumentTree
    // NCNN 推論全在 :engine 內（libyakuyomi_ncnn）；sandbox 只透過引擎介面（Detector/Ocr/Inpainter/Yakuyomi/CharSegmenter），不直接依賴 runtime。
}
