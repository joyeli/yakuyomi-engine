plugins {
    alias(libs.plugins.android.library)
    // AGP 9+ 內建 Kotlin 支援，不再套 kotlin.android（見 kotl.in/gradle/agp-built-in-kotlin）
}

// Yakuyomi 翻譯引擎（OCR／去字／排版／LLM／Pipeline）。Yakuyomi fork 以 Gradle composite build（includeBuild）接此引擎，
// 靠 group:name 替換依賴。原生層（NCNN JNI、libyakuyomi_ncnn.so）與 DBNet 偵測、分群、模型下載在 :inference-core——用 api：
// EngineConfig.detector、PageAnalysis.regions、OcrAbResult.quads 都露出核心型別。不依賴夜讀（:nightread-android）。
group = "li.joye.yakuyomi"
version = "0.5.0"

android {
    namespace = "li.joye.yakuyomi.engine"
    compileSdk = 37

    defaultConfig {
        minSdk = 26
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
}

// AGP 9 內建 Kotlin：jvmTarget 改在 kotlin{} 設（取代已移除的 android.kotlinOptions）。
// Ocr／Inpainter／Yakuyomi 用到核心的 NcnnBackend、Geometry（@InternalEngineApi），整模組 opt-in。
kotlin {
    compilerOptions {
        jvmTarget.set(org.jetbrains.kotlin.gradle.dsl.JvmTarget.JVM_17)
        optIn.add("li.joye.yakuyomi.engine.InternalEngineApi")
    }
}

dependencies {
    api(project(":inference-core"))
    implementation(libs.kotlinx.coroutines.android)
    implementation(libs.okhttp)

    testImplementation("junit:junit:4.13.2")
}
