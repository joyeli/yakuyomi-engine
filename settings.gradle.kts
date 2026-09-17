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

// 夜讀模組（submodule）：只有 :app-sandbox 用它跑上機驗證。
// :engine 不依賴夜讀，發佈的函式庫也不帶——開發流程是「桌面 research → sandbox 真機驗 → 才進產品」。
includeBuild("yakuyomi-nightread")
