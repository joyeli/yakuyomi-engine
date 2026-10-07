package li.joye.yakuyomi.engine

import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import okhttp3.OkHttpClient
import okhttp3.Request
import org.json.JSONObject
import java.io.File
import java.security.MessageDigest
import java.util.concurrent.TimeUnit

/** manifest（models.json）裡的一顆遠端模型。 */
data class RemoteModel(
    val role: String,   // manifest 的 role 字串（現有 detector／ocr／inpainter／charseg；引擎不寫死清單）
    val name: String,   // 落地檔名
    val url: String,
    val size: Long,
    val sha256: String,
)

/** 下載/驗證進度回報（reader 做 UI 用）。 */
sealed interface ModelProgress {
    val role: String
    val name: String
    data class Downloading(override val role: String, override val name: String, val bytes: Long, val total: Long) : ModelProgress
    data class Verifying(override val role: String, override val name: String) : ModelProgress
    data class Done(override val role: String, override val name: String) : ModelProgress
    data class Failed(override val role: String, override val name: String, val error: String) : ModelProgress
}

/**
 * 模型 hosted 下載 + sha256 驗證——BYOM 的「自動版」，與手動放檔**並存**（下載進同一個 models 資料夾，
 * 下游 :engine 的 `ModelSet` 解析不變）。引擎只管「抓 + 驗 + 落檔」；觸發時機、進度 UI、目標資料夾由 reader 決定
 * （對照 :engine 的 `LlmModels` 的引擎/fork 分法）。
 *
 * manifest（models.json）＝單一真理來源：列每顆模型的 url / size / sha256 / role，**版本化** ⇒ 雜湊永遠對得上
 * （解掉「權重更新 → checksum 誤判」的舊顧慮）。下載後逐顆 sha256 驗證＝確定抓到的跟發行版是同一份。
 *
 * 記憶體：下載與算雜湊都走 64KB 串流，**不把整顆模型讀進 JVM heap**（避開 512MB heap OOM，與模型載入同原則）。
 */
object ModelDownloader {

    /** 預設 manifest 位置（引擎 repo 公開後生效；reader 可覆寫成 release 資產或自架）。 */
    const val DEFAULT_MANIFEST_URL =
        "https://raw.githubusercontent.com/joyeli/yakuyomi-engine/main/models.json"

    private val client = OkHttpClient.Builder()
        .connectTimeout(30, TimeUnit.SECONDS)
        .readTimeout(120, TimeUnit.SECONDS)
        .build()

    /** 撈 manifest → 模型清單。失敗拋例外（reader 接住顯示）。 */
    suspend fun fetchManifest(url: String = DEFAULT_MANIFEST_URL): List<RemoteModel> =
        withContext(Dispatchers.IO) {
            val req = Request.Builder().url(url).get().build()
            client.newCall(req).execute().use { resp ->
                val text = resp.body.string() // okhttp5：body 非空
                if (!resp.isSuccessful) throw RuntimeException("HTTP ${resp.code}")
                parseManifest(text)
            }
        }

    /** 解析 models.json（`{ version, models: [{role,name,url,size,sha256,...}] }`）。只給 [fetchManifest] 用。 */
    private fun parseManifest(json: String): List<RemoteModel> {
        val arr = JSONObject(json).getJSONArray("models")
        return (0 until arr.length()).map {
            val m = arr.getJSONObject(it)
            RemoteModel(
                role = m.getString("role"),
                name = m.getString("name"),
                url = m.getString("url"),
                size = m.getLong("size"),
                sha256 = m.getString("sha256").lowercase(),
            )
        }
    }

    /**
     * 確保 [models] 都在 [destDir]：已存在且 size+sha256 相符的跳過，其餘下載後驗證。
     * 回傳 role→本機檔。任一顆失敗即拋例外（已下好的保留、壞檔刪除）。
     */
    suspend fun ensure(
        models: List<RemoteModel>,
        destDir: File,
        onProgress: (ModelProgress) -> Unit = {},
    ): Map<String, File> = withContext(Dispatchers.IO) {
        if (!destDir.exists()) destDir.mkdirs()
        val out = LinkedHashMap<String, File>()
        for (m in models) {
            val file = File(destDir, m.name)
            if (file.exists() && file.length() == m.size && sha256(file) == m.sha256) {
                onProgress(ModelProgress.Done(m.role, m.name)) // 已是正確版本
                out[m.role] = file
                continue
            }
            try {
                downloadOne(m, file, onProgress)
                onProgress(ModelProgress.Verifying(m.role, m.name))
                val got = sha256(file)
                if (got != m.sha256) {
                    file.delete()
                    throw RuntimeException("sha256 不符（期望 ${m.sha256.take(12)}…，實得 ${got.take(12)}…）")
                }
                onProgress(ModelProgress.Done(m.role, m.name))
                out[m.role] = file
            } catch (t: Throwable) {
                onProgress(ModelProgress.Failed(m.role, m.name, t.message ?: t.javaClass.simpleName))
                throw t
            }
        }
        out
    }

    /** 串流下載到 `<name>.part` 再 rename（避免半成品被當完整檔）。 */
    private fun downloadOne(m: RemoteModel, dest: File, onProgress: (ModelProgress) -> Unit) {
        val req = Request.Builder().url(m.url).get().build()
        client.newCall(req).execute().use { resp ->
            if (!resp.isSuccessful) throw RuntimeException("HTTP ${resp.code}")
            val body = resp.body
            val total = if (m.size > 0) m.size else body.contentLength()
            val tmp = File(dest.parentFile, "${dest.name}.part")
            body.byteStream().use { input ->
                tmp.outputStream().use { output ->
                    val buf = ByteArray(1 shl 16)
                    var acc = 0L
                    var lastReport = 0L
                    while (true) {
                        val read = input.read(buf)
                        if (read < 0) break
                        output.write(buf, 0, read)
                        acc += read
                        if (acc - lastReport >= (1 shl 20)) { // 每 ~1MB 回報
                            onProgress(ModelProgress.Downloading(m.role, m.name, acc, total))
                            lastReport = acc
                        }
                    }
                }
            }
            if (!tmp.renameTo(dest)) {
                tmp.copyTo(dest, overwrite = true)
                tmp.delete()
            }
        }
    }

    /** 串流算 sha256（64KB buffer，不把整顆模型讀進 heap）。 */
    fun sha256(file: File): String {
        val md = MessageDigest.getInstance("SHA-256")
        file.inputStream().use { input ->
            val buf = ByteArray(1 shl 16)
            while (true) {
                val read = input.read(buf)
                if (read < 0) break
                md.update(buf, 0, read)
            }
        }
        return md.digest().joinToString("") { "%02x".format(it.toInt() and 0xff) }
    }
}
