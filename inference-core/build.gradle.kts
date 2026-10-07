plugins {
    alias(libs.plugins.android.library)
    // AGP 9+ 內建 Kotlin 支援，不再套 kotlin.android（見 kotl.in/gradle/agp-built-in-kotlin）
}

// Yakuyomi 推論核心：NCNN JNI（libyakuyomi_ncnn.so 唯一的出處）＋ DBNet 偵測＋分群＋模型下載＋推論 trace。
// 翻譯（:engine）與夜讀（:nightread-android）都以 api 依賴這裡、兩者互不依賴；fork 經 includeBuild 以 group:name 替換。
// 原生庫只能在這裡編一份：ncnn 是靜態庫，兩個 .so 各帶一份就有兩個 SimpleOMP 常駐池，NcnnBackend 的建池規則與全域鎖只管得到一個。
group = "li.joye.yakuyomi"
version = "0.5.0"

android {
    namespace = "li.joye.yakuyomi.inference"
    compileSdk = 37
    ndkVersion = "28.2.13676358" // NCNN 原生層；釘住版本讓 CI/fork submodule 建置一致（fork app 的 ndkVersion 跟這裡同版，strip 原生庫要用）

    defaultConfig {
        minSdk = 26
        consumerProguardFiles("consumer-rules.pro")

        // NCNN 原生後端（偵測／OCR／去字／人物分割，引擎唯一的推論 runtime；ORT 2026-09-26 拔除）：只出 arm64（ncnn 預編庫＝arm64-v8a、SimpleOMP、不含 Vulkan）
        ndk {
            abiFilters += "arm64-v8a"
        }
        externalNativeBuild {
            cmake {
                cppFlags += "-std=c++17"
                arguments += listOf("-DANDROID_STL=c++_static")
            }
        }
    }

    externalNativeBuild {
        cmake {
            path = file("src/main/cpp/CMakeLists.txt")
            version = "3.22.1"
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
}

// AGP 9 內建 Kotlin：jvmTarget 改在 kotlin{} 設（取代已移除的 android.kotlinOptions）。
// NcnnBackend／Geometry 標了 @InternalEngineApi（拆模組前是 internal），本模組與兩個兄弟模組整模組 opt-in。
kotlin {
    compilerOptions {
        jvmTarget.set(org.jetbrains.kotlin.gradle.dsl.JvmTarget.JVM_17)
        optIn.add("li.joye.yakuyomi.engine.InternalEngineApi")
    }
}

dependencies {
    implementation(libs.kotlinx.coroutines.android) // ModelDownloader
    implementation(libs.okhttp)                     // ModelDownloader

    testImplementation("junit:junit:4.13.2")
}
