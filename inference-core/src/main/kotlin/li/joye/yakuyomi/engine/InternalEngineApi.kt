package li.joye.yakuyomi.engine

/**
 * 推論核心給 :engine（翻譯）與 :nightread-android（夜讀）用的內部 API。
 *
 * 拆模組前這些宣告是 `internal`；Kotlin 的 internal 跨不了 Gradle 模組，兄弟模組又要用，只好公開，
 * 改用這個 opt-in 標記擋住外部呼叫端（三個模組在各自的 build.gradle.kts 整模組 opt-in）。不屬公開 API，
 * 隨時可能改，外部程式不要 opt-in 來用。
 */
@RequiresOptIn(
    message = "Yakuyomi 推論核心的內部 API，只給 :engine 與 :nightread-android 用",
    level = RequiresOptIn.Level.ERROR,
)
@Retention(AnnotationRetention.BINARY)
@Target(AnnotationTarget.CLASS)
annotation class InternalEngineApi
