pluginManagement {
    repositories {
        google()
        mavenCentral()
        gradlePluginPortal()
    }
}

dependencyResolutionManagement {
    repositoriesMode.set(RepositoriesMode.FAIL_ON_PROJECT_REPOS)
    repositories {
        google()
        mavenCentral()
    }
}

rootProject.name = "yakuyomi-engine"

// 共用推論核心：NCNN JNI（libyakuyomi_ncnn.so 唯一的出處）＋ DBNet 偵測＋分群＋模型下載
include(":inference-core")
// 翻譯引擎：api 依賴 :inference-core；不依賴夜讀
include(":engine")
// 夜讀 Android 膠水（NightReadRenderer＋人物分割）：api 依賴 :inference-core 與 nightread 函式庫；不依賴 :engine
include(":nightread-android")
include(":app-sandbox")

// 夜讀函式庫（submodule）：:nightread-android 以 api 依賴 li.joye.yakuyomi:nightread，靠這行 includeBuild 以 group:name
// 替換成 submodule 原始碼（fork 經 includeBuild 本 repo 一併拿到）。只要翻譯的人也需要這個目錄（設定階段就要）。
// 開發流程仍是「桌面 research（nightread repo）→ sandbox 真機驗 → 才進產品」。
includeBuild("yakuyomi-nightread")
