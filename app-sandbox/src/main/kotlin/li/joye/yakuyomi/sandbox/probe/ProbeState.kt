package li.joye.yakuyomi.sandbox.probe

import android.content.Context

/**
 * 探針的跨行程狀態（SharedPreferences，一律 commit）：這次的紀錄檔、正在跑哪一步、做完哪些步、哪些步閃退過。
 * 原生閃退會帶走整個行程：下次開 app 靠這裡知道是跑到哪一步死的，也才能跳過那一步繼續跑（一個組合閃退，其他資料還拿得到）。
 */
class ProbeState(ctx: Context) {
    private val p = ctx.getSharedPreferences("nightprobe", Context.MODE_PRIVATE)

    var fileName: String?
        get() = p.getString("file", null)
        set(v) { p.edit().putString("file", v).commit() }
    var startedAt: Long
        get() = p.getLong("started", 0L)
        set(v) { p.edit().putLong("started", v).commit() }
    var finished: Boolean
        get() = p.getBoolean("finished", false)
        set(v) { p.edit().putBoolean("finished", v).commit() }
    var inProgress: String?
        get() = p.getString("step", null)
        set(v) { p.edit().putString("step", v).commit() }

    /** 上次中斷已經寫進紀錄了（同一次中斷不重複寫）。 */
    var recovered: Boolean
        get() = p.getBoolean("recovered", false)
        set(v) { p.edit().putBoolean("recovered", v).commit() }

    val done: Set<String> get() = p.getStringSet("done", emptySet()) ?: emptySet()
    val crashed: Set<String> get() = p.getStringSet("crashed", emptySet()) ?: emptySet()

    fun addDone(id: String) { p.edit().putStringSet("done", done + id).commit() }
    fun addCrashed(id: String) { p.edit().putStringSet("crashed", crashed + id).commit() }

    fun reset(file: String, now: Long) {
        p.edit().clear().putString("file", file).putLong("started", now).commit()
    }

    /** 有沒有沒跑完、可以接著跑的一次。 */
    val resumable: Boolean get() = fileName != null && !finished
}
