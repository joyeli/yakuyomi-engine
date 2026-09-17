import java.util.Properties

plugins {
    alias(libs.plugins.android.application)
    // AGP 9+ 內建 Kotlin 支援，不再套 kotlin.android
}

// BYOK：build 時從 repo 根 api-keys.properties（gitignored）讀 key 注入 debug BuildConfig。
// 缺檔則為空字串，App 端會略過翻譯。正式版走 Android Keystore + 設定頁，不走這裡。
val apiKeys = Properties().apply {
    val f = rootProject.file("api-keys.properties")
    if (f.exists()) f.inputStream().use { load(it) }
}

android {
    namespace = "li.joye.yakuyomi.sandbox"
    compileSdk = 37

    defaultConfig {
        applicationId = "li.joye.yakuyomi.sandbox"
        minSdk = 26
        targetSdk = 36
        versionCode = 2
        versionName = "0.2-qnn"
        buildConfigField(
            "String",
            "DEEPSEEK_API_KEY",
            "\"${apiKeys.getProperty("DEEPSEEK_API_KEY", "")}\"",
        )
        // 只打 arm64-v8a（實機）：砍掉 armeabi-v7a/x86/x86_64 的 ORT native libs ≈ 省 55MB，雲端手動安裝快很多
        // NCNN 原生層由 :engine 提供（libyakuyomi_ncnn）；sandbox 不再自帶 native build。
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

    // 模型分離（BYOM）：*.onnx 不打進 APK，改由 app 從使用者選的資料夾載入 → APK 變小
    androidResources {
        ignoreAssetsPattern = "*.onnx"
    }

    // QNN 庫瘦身：qnn-runtime 預設打進所有 Hexagon 版本(V68/69/73/75)+DSP+GPU。
    // 此測試機＝SD 8 Gen 3＝Hexagon V75，只需 V75 skel/stub + 共用的 libQnnHtp/Prepare/System。
    // 砍掉其餘 ≈ 省 ~50MB，OneDrive 手動安裝快很多。★換非 8Gen3 的機測試時要把對應 Vxx 加回來。
    packaging {
        jniLibs {
            excludes += listOf(
                "**/libQnnHtpV68Skel.so", "**/libQnnHtpV68Stub.so",
                "**/libQnnHtpV69Skel.so", "**/libQnnHtpV69Stub.so",
                "**/libQnnHtpV73Skel.so", "**/libQnnHtpV73Stub.so",
                "**/libQnnDsp.so", "**/libQnnDspV66Skel.so", "**/libQnnDspV66Stub.so",
                "**/libQnnGpu.so",
            )
        }
    }
}

// AGP 9 內建 Kotlin：jvmTarget 改在 kotlin{} 設
kotlin {
    compilerOptions {
        jvmTarget.set(org.jetbrains.kotlin.gradle.dsl.JvmTarget.JVM_17)
    }
}

dependencies {
    implementation(project(":engine"))
    // 夜讀：上機驗證用。含 :nightread 核心（純 Kotlin）與 :nightread-ort（人物遮罩推論）。
    implementation("li.joye.yakuyomi:nightread-ort:0.1.0")
    implementation(libs.androidx.core.ktx)
    implementation(libs.androidx.appcompat)
    implementation(libs.androidx.lifecycle.runtime.ktx)
    implementation(libs.kotlinx.coroutines.android)
    implementation("androidx.documentfile:documentfile:1.0.1") // SAF 資料夾讀檔
    implementation("androidx.activity:activity-ktx:1.8.2")      // registerForActivityResult / OpenDocumentTree
    // ORT/NCNN 推論全在 :engine 內；sandbox 只透過引擎介面（Detector/Ocr/Inpainter/Yakuyomi），不直接依賴 runtime。
}
