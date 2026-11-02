package li.joye.yakuyomi.sandbox.probe

import android.content.Context
import android.net.Uri
import androidx.documentfile.provider.DocumentFile
import java.io.File
import java.io.FileOutputStream

/**
 * 探針紀錄。每一行寫進 app 私有目錄就 fsync（閃退也留得下），分步時把整份複製到使用者選的資料夾（[publish]；SAF 不保證能附加，一律整份覆寫）。
 *
 * 格式：UTF-8 文字，一行一筆，欄位以 Tab 分開：第一欄是種類，其餘是 `鍵=值`。人讀的摘要在最後的 `SUMMARY` 段。
 */
class ProbeReport(private val ctx: Context, private val tree: DocumentFile?, val fileName: String) {
    val local: File = File(dirOf(ctx), fileName).also { it.parentFile?.mkdirs() }
    private val out = FileOutputStream(local, true)
    private var treeUri: Uri? = null
    private var publishFailed = false
    private var lastPublish = 0L

    /** 每行都會通知（畫面顯示進度）；在呼叫 [line] 的執行緒上。 */
    @Volatile
    var onLine: ((String) -> Unit)? = null

    @Synchronized
    fun line(kind: String, vararg kv: Pair<String, Any?>) {
        val sb = StringBuilder(kind)
        for ((k, v) in kv) sb.append('\t').append(k).append('=').append(v.toString().replace('\t', ' ').replace('\n', ' '))
        raw(sb.toString())
    }

    @Synchronized
    fun raw(s: String) {
        // 收尾之後才到的一行（切走的警告）寫不進去就算了，不能讓它變成未捕捉的例外
        runCatching {
            out.write((s + "\n").toByteArray(Charsets.UTF_8))
            out.flush()
            out.fd.sync()
        }
        onLine?.invoke(s)
    }

    /** 複製一份到使用者選的資料夾。[force]＝false 時最多 15 秒做一次（資料夾可能是雲端）。失敗只記一次，不中斷探針。 */
    @Synchronized
    fun publish(force: Boolean = true) {
        val now = System.currentTimeMillis()
        if (!force && now - lastPublish < 15_000) return
        lastPublish = now
        val t = tree ?: return
        try {
            val uri = treeUri ?: (t.findFile(fileName) ?: t.createFile("text/plain", fileName))?.uri?.also { treeUri = it }
                ?: error("建不了 $fileName")
            ctx.contentResolver.openOutputStream(uri, "wt")!!.use { o -> local.inputStream().use { it.copyTo(o) } }
        } catch (e: Throwable) {
            if (!publishFailed) {
                publishFailed = true
                raw("WARN\twhat=寫不進所選資料夾\terror=${e.javaClass.simpleName}: ${e.message}")
            }
        }
    }

    /** 把一個檔案（閃退的 tombstone）放進所選資料夾。 */
    fun publishBytes(name: String, mime: String, bytes: ByteArray) {
        File(dirOf(ctx), name).writeBytes(bytes)
        val t = tree ?: return
        runCatching {
            val uri = (t.findFile(name) ?: t.createFile(mime, name))?.uri ?: return
            ctx.contentResolver.openOutputStream(uri, "wt")!!.use { it.write(bytes) }
        }
    }

    @Synchronized
    fun close() {
        runCatching { out.close() }
    }

    companion object {
        fun dirOf(ctx: Context) = File(ctx.filesDir, "nightprobe")
    }
}
