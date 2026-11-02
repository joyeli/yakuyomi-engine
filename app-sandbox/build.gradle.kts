plugins {
    alias(libs.plugins.android.application)
    // AGP 9+ 內建 Kotlin 支援，不再套 kotlin.android
}

// 夜讀 C++ 探針的守門頁：照 nightread 的 fixture 打包進 assets/nightprobe/page，不在本 repo 留第二份。
// 清單跟探針的 GuardDigest.fixtureFiles() 相同（GuardDigestJvmTest 檢查檔案都在）。
abstract class NightProbeFixtures : DefaultTask() {
    @get:InputFiles
    @get:PathSensitive(PathSensitivity.NAME_ONLY)
    abstract val pages: ConfigurableFileCollection

    @get:OutputDirectory
    abstract val outDir: DirectoryProperty

    @TaskAction
    fun copy() {
        val dir = outDir.get().asFile.resolve("nightprobe/page")
        dir.deleteRecursively()
        dir.mkdirs()
        pages.files.forEach { it.copyTo(dir.resolve(it.name)) }
    }
}

val nightProbeFixtures = tasks.register<NightProbeFixtures>("nightProbeFixtures") {
    val src = rootProject.file("yakuyomi-nightread/nightread/src/test/resources/page")
    val guardPages = listOf("ch34_011", "demo01", "demo02", "demo05", "demo06", "ch34_015", "syn_more", "syn_v4")
    pages.from(
        guardPages.flatMap { p -> listOf("gray.png", "seg.png", "char.png", "chroma.png", "regions.txt").map { src.resolve("${p}_$it") } } +
            src.resolve("syn_v4_inpaint.png"),
    )
    outDir.set(layout.buildDirectory.dir("generated/nightprobe"))
}

androidComponents {
    onVariants { variant ->
        variant.sources.assets?.addGeneratedSourceDirectory(nightProbeFixtures, NightProbeFixtures::outDir)
    }
}

// LLM key **不再**於 build 時注入 APK（2026-09-26 拔掉：APK 外流＝key 外流）。sandbox 改在 app 內輸入、存 prefs；
// repo 根的 api-keys.properties 只剩桌面 parity 腳本（translate_parity.py）在讀。
android {
    namespace = "li.joye.yakuyomi.sandbox"
    compileSdk = 37
    ndkVersion = "28.2.13676358" // 跟 :inference-core 同版（打包時 strip .so 要用；探針要記真正的 .so 大小）

    defaultConfig {
        applicationId = "li.joye.yakuyomi.sandbox"
        minSdk = 26
        targetSdk = 36
        versionCode = 3
        versionName = "0.4-nightprobe"
        // 只打 arm64-v8a（實機）：NCNN 原生層由 :inference-core 提供（libyakuyomi_ncnn，ncnn 預編庫本來就只出 arm64），
        // 其他 ABI 沒有可用的推論後端，打進去只是白佔空間；sandbox 不再自帶 native build。
        ndk { abiFilters += "arm64-v8a" }
    }

    buildTypes {
        // 夜讀 C++ 探針用這個：不可除錯（可除錯的 ART 慢很多，C++ 的倍數會被高估）、不混淆、用 debug key 簽（裝得上、不必另備金鑰）。
        // 建置：./gradlew :app-sandbox:assembleRelease -Pyakuyomi.nightProbe=true（數值自測橋才進 .so）。
        release {
            isMinifyEnabled = false
            signingConfig = signingConfigs.getByName("debug")
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
    testOptions {
        unitTests.all { it.maxHeapSize = "2g" } // 守門摘要跑整頁重繪
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
        optIn.add("li.joye.yakuyomi.engine.InternalEngineApi") // 探針直接呼叫 NcnnBackend 的 probe*
    }
}

dependencies {
    implementation(project(":engine"))
    // 夜讀上機驗證：:nightread-android（NightReadRenderer＋NCNN 人物分割 CharSegmenter：YoloSegSegmenter／CsegSegmenter；
    // api 帶 nightread 函式庫與 :inference-core）。:nightread-ort（ORT 版 CharMaskOrt）已退役不再依賴。
    implementation(project(":nightread-android"))
    implementation(libs.androidx.core.ktx)
    implementation(libs.androidx.appcompat)
    implementation(libs.androidx.lifecycle.runtime.ktx)
    implementation(libs.kotlinx.coroutines.android)
    implementation("androidx.documentfile:documentfile:1.0.1") // SAF 資料夾讀檔
    implementation("androidx.activity:activity-ktx:1.8.2")      // registerForActivityResult / OpenDocumentTree
    testImplementation("junit:junit:4.13.2")
    // NCNN 推論全在 :inference-core 內（libyakuyomi_ncnn）；sandbox 只透過引擎介面（Detector/Ocr/Inpainter/Yakuyomi/CharSegmenter），不直接依賴 runtime。
}
