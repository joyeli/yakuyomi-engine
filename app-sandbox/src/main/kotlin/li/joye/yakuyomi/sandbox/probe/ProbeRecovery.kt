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
 * 探針跑到一半行程死掉（原生閃退、被系統殺、OOM）：下次開 app 把 ApplicationExitInfo 寫進同一份紀錄，tombstone 原始位元組另存一個 `.pb`
 *（桌面用 NDK llvm-symbolizer 配未 strip 的 .so 解；要原始位元組，當文字讀會把 varint 弄壞），那一步記成閃退、之後接著跑時跳過。
 * 步驟之外的閃退（理論上不該有，原生呼叫都包在步驟裡）也照樣寫 EXIT 與 tombstone，不會無聲無息。
 */
object ProbeRecovery {
    private fun crashLike(reason: Int) =
        reason == ApplicationExitInfo.REASON_CRASH_NATIVE || reason == ApplicationExitInfo.REASON_SIGNALED ||
            reason == ApplicationExitInfo.REASON_CRASH || reason == ApplicationExitInfo.REASON_LOW_MEMORY ||
            reason == ApplicationExitInfo.REASON_EXCESSIVE_RESOURCE_USAGE || reason == ApplicationExitInfo.REASON_ANR

    /** 回傳給畫面看的一句話；沒事回 null。開 app 時呼叫，同一次中斷只處理一次。 */
    fun check(ctx: Context, treeUri: String?): String? {
        if (ProbeSession.running) return null
        val st = ProbeState(ctx)
        val file = st.fileName ?: return null
        if (st.finished) return null
        val exit: ApplicationExitInfo? = if (Build.VERSION.SDK_INT >= 30) {
            runCatching {
                ctx.getSystemService(ActivityManager::class.java).getHistoricalProcessExitReasons(ctx.packageName, 0, 8)
                    .firstOrNull { it.timestamp >= st.startedAt && it.timestamp > st.lastExitTs }
            }.getOrNull()
        } else {
            null
        }
        if (exit != null) st.lastExitTs = exit.timestamp
        // 看門狗自己結束行程：HANG 與閃退記號都已經寫了
        st.selfKill?.let { hung ->
            st.selfKill = null
            return "上次探針卡在「$hung」，看門狗已把堆疊寫進 $file。按「接著跑」會跳過這一步。"
        }
        val step = st.inProgress
        val crash = exit != null && crashLike(exit.reason)
        // 步驟之間的正常離開（使用者關掉 app）不用記；步驟之間的閃退要記
        if (step == null && !crash) return null
        val tree = treeUri?.let { runCatching { DocumentFile.fromTreeUri(ctx, Uri.parse(it)) }.getOrNull() }
        val report = ProbeReport(ctx, tree, file)
        var why = "未知（系統沒有留下離開原因）"
        if (exit != null) {
            why = "reason=${exit.reason} status=${exit.status} desc=${exit.description}"
            val time = SimpleDateFormat("MMdd_HHmmss", Locale.US).format(Date(exit.timestamp))
            val raw = runCatching { exit.traceInputStream?.use { it.readBytes() } }.getOrNull()
            if (raw != null && raw.isNotEmpty()) {
                val pb = ProbeRecords.tombstoneName(file, step, time)
                report.publishBytes(pb, "application/octet-stream", raw)
                why += " tombstone=$pb(${raw.size} bytes)"
            }
            report.line(
                "EXIT", "reason" to exit.reason, "status" to exit.status, "desc" to exit.description, "pssMB" to exit.pss / 1024, "rssMB" to exit.rss / 1024,
                "time" to time,
            )
        }
        report.line("CRASH", "step" to (step ?: "步驟之外"), "native" to crash, "why" to why)
        if (step != null) {
            if (crash || step in st.interrupted) {
                st.addCrashed(step) // 第二次死在同一步、看不出原因也當閃退：不然接著跑永遠過不去
            } else {
                st.addInterrupted(step)
            }
            st.inProgress = null
        }
        report.publish()
        report.close()
        val skipped = step != null && step in st.crashed
        return "上次探針跑到「${step ?: "步驟之外"}」時中斷：$why\n已寫進 $file。按「接著跑」${if (skipped) "會跳過這一步" else "會接著做"}。"
    }
}
