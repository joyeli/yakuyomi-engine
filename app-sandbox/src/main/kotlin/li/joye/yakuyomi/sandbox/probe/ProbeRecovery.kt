package li.joye.yakuyomi.sandbox.probe

import android.app.ActivityManager
import android.app.ApplicationExitInfo
import android.content.Context
import android.net.Uri
import android.os.Build
import androidx.documentfile.provider.DocumentFile
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

/**
 * 探針跑到一半行程死掉（原生閃退、被系統殺、OOM）：下次開 app 把 ApplicationExitInfo 寫進同一份紀錄，tombstone 原始位元組存成同名 `_crash.pb`
 *（桌面用 NDK llvm-symbolizer 配未 strip 的 .so 解；要原始位元組，當文字讀會把 varint 弄壞），那一步記成閃退、之後接著跑時跳過。
 */
object ProbeRecovery {
    /** 回傳給畫面看的一句話；沒事回 null。開 app 時呼叫，同一次中斷只處理一次。 */
    fun check(ctx: Context, treeUri: String?): String? {
        if (ProbeSession.running) return null
        val st = ProbeState(ctx)
        val file = st.fileName ?: return null
        val step = st.inProgress
        if (st.finished || step == null || st.recovered) return null
        st.recovered = true
        val tree = treeUri?.let { runCatching { DocumentFile.fromTreeUri(ctx, Uri.parse(it)) }.getOrNull() }
        val report = ProbeReport(ctx, tree, file)
        var why = "未知（系統沒有留下離開原因）"
        var nativeCrash = false
        if (Build.VERSION.SDK_INT >= 30) {
            runCatching {
                val am = ctx.getSystemService(ActivityManager::class.java)
                val exit = am.getHistoricalProcessExitReasons(ctx.packageName, 0, 8).firstOrNull { it.timestamp >= st.startedAt }
                if (exit != null) {
                    nativeCrash = exit.reason == ApplicationExitInfo.REASON_CRASH_NATIVE || exit.reason == ApplicationExitInfo.REASON_SIGNALED ||
                        exit.reason == ApplicationExitInfo.REASON_CRASH || exit.reason == ApplicationExitInfo.REASON_LOW_MEMORY ||
                        exit.reason == ApplicationExitInfo.REASON_EXCESSIVE_RESOURCE_USAGE || exit.reason == ApplicationExitInfo.REASON_ANR
                    why = "reason=${exit.reason} status=${exit.status} desc=${exit.description}"
                    val raw = runCatching { exit.traceInputStream?.use { it.readBytes() } }.getOrNull()
                    if (raw != null && raw.isNotEmpty()) {
                        val pb = file.removeSuffix(".txt") + "_crash.pb"
                        report.publishBytes(pb, "application/octet-stream", raw)
                        why += " tombstone=$pb(${raw.size} bytes)"
                    }
                    report.line(
                        "EXIT", "reason" to exit.reason, "status" to exit.status, "desc" to exit.description, "pssMB" to exit.pss / 1024, "rssMB" to exit.rss / 1024,
                        "time" to SimpleDateFormat("MMdd_HHmmss", Locale.US).format(Date(exit.timestamp)),
                    )
                }
            }
        }
        report.line("CRASH", "step" to step, "native" to nativeCrash, "why" to why)
        if (nativeCrash) st.addCrashed(step)
        st.inProgress = null
        report.publish()
        report.close()
        return "上次探針跑到「$step」時中斷：$why\n已寫進 $file。按「接著跑」會跳過這一步。"
    }
}
