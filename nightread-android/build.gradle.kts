plugins {
    alias(libs.plugins.android.library)
    // AGP 9+ 內建 Kotlin 支援，不再套 kotlin.android
}

// 夜讀的 Android 膠水：NightReadRenderer（Bitmap ↔ nightread 輸入、分群、縮圖、計時）＋人物分割（YOLO11-seg、
// CartoonSegmentation，NCNN）。偵測與 NCNN 原生層在 :inference-core；演算法在 nightread 函式庫（根 settings 的
// includeBuild("yakuyomi-nightread") 以 group:name 替換成 submodule 原始碼）。不依賴翻譯引擎 :engine。
// 模組不叫 :nightread：座標 li.joye.yakuyomi:nightread 已是函式庫的，同名會讓 composite build 的替換撞名。
group = "li.joye.yakuyomi"
version = "0.5.0"

android {
    namespace = "li.joye.yakuyomi.nightread.android"
    compileSdk = 37

    defaultConfig {
        minSdk = 26
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
}

// AGP 9 內建 Kotlin：jvmTarget 改在 kotlin{} 設。人物分割用到核心的 NcnnBackend（@InternalEngineApi），整模組 opt-in。
kotlin {
    compilerOptions {
        jvmTarget.set(org.jetbrains.kotlin.gradle.dsl.JvmTarget.JVM_17)
        optIn.add("li.joye.yakuyomi.engine.InternalEngineApi")
    }
}

dependencies {
    // Detector、Detection、TextLine、NcnnFlavor 出現在公開 API
    api(project(":inference-core"))
    // NightReadParams、NightTier 等出現在公開 API（fork 要拿到 nightread 型別）
    api("li.joye.yakuyomi:nightread:0.1.0")

    testImplementation("junit:junit:4.13.2")
}
