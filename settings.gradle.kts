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

include(":engine")
include(":app-sandbox")

// 夜讀函式庫（submodule）：夜讀膠水（NightReadRenderer）與人物分割 NCNN 都在 :engine，所以 :engine 以 api 依賴它
// （li.joye.yakuyomi:nightread 靠這行 includeBuild 以 group:name 替換成 submodule 原始碼；fork 經 includeBuild 本 repo 一併拿到）。
// :app-sandbox 也直接依賴它做真機 A/B。開發流程仍是「桌面 research（nightread repo）→ sandbox 真機驗 → 才進產品」。
includeBuild("yakuyomi-nightread")
