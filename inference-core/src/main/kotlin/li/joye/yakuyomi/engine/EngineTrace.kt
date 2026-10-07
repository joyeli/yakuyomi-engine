package li.joye.yakuyomi.engine

/**
 * 引擎診斷 trace hook（app 端的診斷紀錄開關控制；預設關＝零開銷）。
 *
 * 抓的是 logcat 與內建 crash log 都抓不到的那類 crash：原生 SIGSEGV/abort、OOM 被 lowmemorykiller
 * SIGKILL——都秒殺行程、不彈崩潰畫面。引擎不知道 log 檔在哪、也不碰 Android 儲存/權限——只把「階段訊息」
 * 丟給 [sink]（由 app 端接去落盤的 trace 檔，每行 flush、native crash 也留得住）。
 * **[sink]=null（預設）＝零開銷、完全不記**——app 端的診斷開關關閉時就不設 sink，等於這整套不存在。
 *
 * 慣例：一段原生呼叫記 `xxx.enter`（進函式）→ `xxx.call`（緊接原生呼叫前）→ `xxx.exit`（回來後）。
 * 若行程死在原生內，最後一行會是 `xxx.call` 而**沒有** `xxx.exit` → 精準定位死在哪個原生呼叫。
 * 死在鎖上（deadlock/久候）則停在 `xxx.enter`（連 `.call` 都沒到）。
 */
object EngineTrace {
    @Volatile
    @JvmStatic
    var sink: ((String) -> Unit)? = null

    fun log(msg: String) {
        sink?.invoke(msg)
    }
}
