plugins {
    alias(libs.plugins.android.library)
    // AGP 9+ 內建 Kotlin 支援，不再套 kotlin.android（見 kotl.in/gradle/agp-built-in-kotlin）
}

// Yakuyomi fork 以 Gradle composite build（includeBuild）接此引擎，靠 group:name 替換依賴
group = "li.joye.yakuyomi"
version = "0.3.0"

android {
    namespace = "li.joye.yakuyomi.engine"
    compileSdk = 37
    ndkVersion = "28.2.13676358" // NCNN 原生層（Detector/Ocr/Inpainter）；釘住版本讓 CI/fork submodule 建置一致

    defaultConfig {
        minSdk = 26
        consumerProguardFiles("consumer-rules.pro")

        // NCNN 原生後端（去字／偵測／OCR，引擎唯一的推論 runtime；ORT 2026-09-26 拔除）：只出 arm64（ncnn 預編庫即 arm64-v8a Vulkan 版）
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

// AGP 9 內建 Kotlin：jvmTarget 改在 kotlin{} 設（取代已移除的 android.kotlinOptions）
kotlin {
    compilerOptions {
        jvmTarget.set(org.jetbrains.kotlin.gradle.dsl.JvmTarget.JVM_17)
    }
}

dependencies {
    implementation(libs.kotlinx.coroutines.android)
    implementation(libs.okhttp)

    testImplementation("junit:junit:4.13.2")
}
