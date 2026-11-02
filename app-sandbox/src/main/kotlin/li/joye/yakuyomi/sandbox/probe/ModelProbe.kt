package li.joye.yakuyomi.sandbox.probe

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.content.Context
import android.os.Process
import androidx.documentfile.provider.DocumentFile
import li.joye.yakuyomi.engine.CsegPost
import li.joye.yakuyomi.engine.Detector
import li.joye.yakuyomi.engine.NcnnBackend
import li.joye.yakuyomi.engine.YoloSegPost
import java.io.File
import java.nio.ByteBuffer
import java.security.MessageDigest
import java.util.concurrent.CyclicBarrier
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicInteger
import java.util.concurrent.atomic.AtomicReference

/**
 * 模型運算的同時跑實驗：DBNet、yolo、cseg、AOT（768）用不同緒數跑，輸出比 sha256，並量吞吐。
 *
 * 全部直接呼叫 [NcnnBackend] 的 `probe*`（不進全域鎖），前處理用產品的公開函式；只有 AOT 的輸入是這裡照 Inpainter 的寫法做
 *（那幾個函式是 private），遮罩用 DBNet 的筆畫遮罩，前向的計算量跟遮罩內容無關。
 */
internal class ModelProbe(private val ctx: Context, private val tree: DocumentFile, private val report: ProbeReport, private val env: ProbeEnv) {

    enum class M(val key: String) { DBNET("dbnet"), YOLO("manga_seg"), CSEG("cartoonseg"), AOT("aot") }

    class Page(
        val name: String,
        val db: FloatArray, val dbW: Int, val dbH: Int,
        val yolo: FloatArray,
        val cseg: FloatArray,
        val aotImg: FloatArray, val aotMask: FloatArray,
    )

    private val paths = HashMap<M, Pair<String, String>>()

    private fun candidates(): List<DocumentFile> {
        val out = arrayListOf(tree)
        tree.findFile("models")?.takeIf { it.isDirectory }?.let { out += it }
        return out
    }

    private fun copyLocal(doc: DocumentFile): File {
        val out = File(ctx.filesDir, doc.name ?: error("模型沒有檔名"))
        if (out.exists() && out.length() == doc.length()) return out
        val tmp = File(out.path + ".tmp")
        ctx.contentResolver.openInputStream(doc.uri)!!.use { i -> tmp.outputStream().use { o -> i.copyTo(o, 1 shl 16) } }
        check(tmp.renameTo(out)) { "改名失敗：$tmp" }
        return out
    }

    private fun sha16(f: File): String {
        val md = MessageDigest.getInstance("SHA-256")
        f.inputStream().use { i ->
            val b = ByteArray(1 shl 20)
            while (true) {
                val n = i.read(b)
                if (n <= 0) break
                md.update(b, 0, n)
            }
        }
        return md.digest().take(8).joinToString("") { "%02x".format(it) }
    }

    /** 從所選資料夾（或它底下的 models）把 4 顆模型放進 filesDir。缺模型丟例外。 */
    fun prepareModels() {
        val docs = candidates().flatMap { it.listFiles().toList() }
        for (m in M.values()) {
            fun find(suffix: String) = docs.firstOrNull { d ->
                val n = d.name?.lowercase() ?: return@firstOrNull false
                n.contains(m.key) && n.endsWith(suffix)
            } ?: error("所選資料夾（或 models 子資料夾）缺 ${m.key} 的 $suffix")
            val t0 = System.nanoTime()
            val p = copyLocal(find(".param"))
            val b = copyLocal(find(".bin"))
            paths[m] = p.path to b.path
            report.line(
                "MODEL", "model" to m.name, "param" to p.name, "bin" to b.name, "binBytes" to b.length(), "binSha16" to sha16(b),
                "copyMs" to (System.nanoTime() - t0) / 1_000_000,
            )
        }
    }

    private fun assetBitmap(path: String): Bitmap = ctx.assets.open(path).use { BitmapFactory.decodeStream(it) }

    /** 一頁的所有輸入（只算一次，之後每次前向都用同一份）。 */
    fun preparePage(asset: String): Page {
        val bmp = assetBitmap(asset)
        val w = bmp.width
        val h = bmp.height
        val px = IntArray(w * h)
        bmp.getPixels(px, 0, w, 0, 0, w, h)
        val db = Detector.preprocess(bmp)
        val yolo = YoloSegPost.preprocess(px, w, h)
        val cseg = CsegPost.preprocess(px, w, h)
        val textMask = Detector(paths.getValue(M.DBNET).first).use { it.detect(bmp).textMask }
        val n = AOT_SIZE
        val imgScaled = Bitmap.createScaledBitmap(bmp, n, n, true)
        val maskScaled = Bitmap.createScaledBitmap(textMask, n, n, false)
        val area = n * n
        val ip = IntArray(area).also { imgScaled.getPixels(it, 0, n, 0, 0, n, n) }
        val mp = IntArray(area).also { maskScaled.getPixels(it, 0, n, 0, 0, n, n) }
        val img = FloatArray(3 * area)
        val mask = FloatArray(area)
        for (i in 0 until area) {
            if ((mp[i] and 0xFF) > 127) { // 洞＝0（Inpainter.aotImageChw）
                mask[i] = 1f
                continue
            }
            val p = ip[i]
            img[i] = ((p shr 16) and 0xFF) / 127.5f - 1f
            img[area + i] = ((p shr 8) and 0xFF) / 127.5f - 1f
            img[2 * area + i] = (p and 0xFF) / 127.5f - 1f
        }
        if (imgScaled !== bmp) imgScaled.recycle()
        if (maskScaled !== textMask) maskScaled.recycle()
        textMask.recycle()
        bmp.recycle()
        val page = Page(asset.substringAfterLast('/'), db.chw, db.w, db.h, yolo.chw, cseg.chw, img, mask)
        report.line(
            "PAGE", "page" to page.name, "w" to w, "h" to h, "dbInput" to "${db.w}x${db.h}",
            "inputSha16" to shaOf(arrayOf(page.db, page.yolo, page.cseg, page.aotImg, page.aotMask)).take(16),
        )
        return page
    }

    fun alloc(m: M, pg: Page): Array<FloatArray> = when (m) {
        M.DBNET -> arrayOf(FloatArray(2 * pg.dbW * pg.dbH), FloatArray(pg.dbW * pg.dbH))
        M.YOLO -> YoloSegPost.allocOutputs()
        M.CSEG -> CsegPost.allocOutputs()
        M.AOT -> arrayOf(FloatArray(3 * AOT_SIZE * AOT_SIZE))
    }

    /** 一次前向；回奈秒。失敗丟例外（rc 記在訊息）。 */
    fun forward(m: M, handle: Long, pg: Page, outs: Array<FloatArray>): Long {
        val t0 = System.nanoTime()
        val rc = when (m) {
            M.DBNET -> NcnnBackend.probeDetectDbnet(handle, pg.db, pg.dbW, pg.dbH, outs[0], outs[1])
            M.YOLO -> NcnnBackend.probeExtract(handle, pg.yolo, YoloSegPost.SIZE, YoloSegPost.SIZE, 3, YoloSegPost.OUT_NAMES, outs)
            M.CSEG -> NcnnBackend.probeExtract(handle, pg.cseg, CsegPost.SIZE, CsegPost.SIZE, 3, CsegPost.OUT_NAMES, outs)
            M.AOT -> NcnnBackend.probeInpaintAot(handle, pg.aotImg, pg.aotMask, AOT_SIZE, outs[0])
        }
        val dt = System.nanoTime() - t0
        val ok = if (m == M.DBNET) rc > 0 else rc == 0
        check(ok) { "${m.name} 前向失敗 rc=$rc" }
        return dt
    }

    /** 一組 Net（同一個緒數）。 */
    inner class NetSet(val threads: Int) : AutoCloseable {
        val handles = HashMap<M, Long>()

        init {
            for (m in M.values()) {
                val (p, b) = paths.getValue(m)
                val t0 = System.nanoTime()
                val h = NcnnBackend.createNetEx(p, b, threads, fp16Storage = true, fp16Arith = true)
                check(h != 0L) { "${m.name} 載入失敗（$threads 緒）" }
                handles[m] = h
                report.line("LOAD", "model" to m.name, "threads" to threads, "ms" to (System.nanoTime() - t0) / 1_000_000)
            }
        }

        operator fun get(m: M): Long = handles.getValue(m)

        override fun close() {
            handles.values.forEach { NcnnBackend.releaseNet(it) }
            handles.clear()
        }
    }

    /** 單緒順序跑一次每顆模型（Net 第一次前向會做 lazy 初始化，要在同時跑之前做完；多頁同時打進未初始化的 Net 曾經原生閃退）。 */
    fun warm(set: NetSet, pg: Page) {
        for (m in M.values()) forward(m, set[m], pg, alloc(m, pg))
    }

    /**
     * 兩條執行緒同時跑 [a]、[b]。短的那邊在對方還沒跑完前會一直重複（最多 [MAX_REPEAT] 次），每次都比 sha。
     * [expect]：模型 → 這頁的預期 sha。回傳不同的次數。
     */
    fun pair(a: M, b: M, set: NetSet, pg: Page, expect: (M) -> String?, sink: (Array<Pair<String, Any?>>) -> Unit): Int {
        val barrier = CyclicBarrier(2)
        val mismatches = AtomicInteger(0)
        val firstError = AtomicReference<Throwable?>(null)
        val doneA = AtomicBoolean(false)
        val doneB = AtomicBoolean(false)
        val start = LongArray(2)
        val end = LongArray(2)
        val runs = IntArray(2)
        val sumNs = LongArray(2)
        fun side(i: Int, m: M, me: AtomicBoolean, other: AtomicBoolean) = Thread {
            try {
                Process.setThreadPriority(Process.THREAD_PRIORITY_DEFAULT)
                val outs = alloc(m, pg)
                barrier.await()
                start[i] = System.nanoTime()
                do {
                    sumNs[i] += forward(m, set[m], pg, outs)
                    end[i] = System.nanoTime()
                    runs[i]++
                    val sha = shaOf(outs)
                    val want = expect(m)
                    if (want != null && want != sha) mismatches.incrementAndGet()
                    me.set(true)
                } while (!other.get() && runs[i] < MAX_REPEAT)
            } catch (t: Throwable) {
                firstError.compareAndSet(null, t)
                barrier.reset()
                me.set(true)
            }
        }.also { it.name = "probe-pair-$i"; it.start() }
        val t0 = System.nanoTime()
        val ta = side(0, a, doneA, doneB)
        val tb = side(1, b, doneB, doneA)
        ta.join()
        tb.join()
        val wallMs = (System.nanoTime() - t0) / 1_000_000
        firstError.get()?.let { throw it }
        val overlap = (minOf(end[0], end[1]) - maxOf(start[0], start[1])) / 1_000_000
        sink(
            arrayOf(
                "page" to pg.name, "a" to a.name, "b" to b.name, "threads" to set.threads,
                "runsA" to runs[0], "runsB" to runs[1], "msA" to sumNs[0] / runs[0] / 1_000_000, "msB" to sumNs[1] / runs[1] / 1_000_000,
                "wallMs" to wallMs, "overlapMs" to overlap, "mismatches" to mismatches.get(),
            ),
        )
        return mismatches.get()
    }

    /** 吞吐實驗的結果。 */
    class Throughput(val wallNs: Long, val cpuMs: Long, val tasks: Int, val medianMs: Map<M, Long>, val mismatches: Int)

    /**
     * [workers] 條執行緒從同一份工作清單拿工作（[ROUND] × [rounds]：跟合併模式一頁實際用到的模型一樣），第 i 條用 [sets] 的第 i 組。
     * 只量前向與 JNI 複製；結尾才比 sha（每條執行緒每顆模型最後一次的輸出）。
     */
    fun throughput(rounds: Int, workers: Int, sets: List<NetSet>, pg: Page, expect: (M) -> String?): Throughput {
        val tasks = ArrayList<M>()
        repeat(rounds) { tasks += ROUND }
        val next = AtomicInteger(0)
        val barrier = CyclicBarrier(workers)
        val ms = LongArray(tasks.size)
        val firstError = AtomicReference<Throwable?>(null)
        val perWorker = List(workers) { M.values().associateWith { m -> alloc(m, pg) } }
        val threads = List(workers) { w ->
            Thread {
                try {
                    Process.setThreadPriority(Process.THREAD_PRIORITY_DEFAULT)
                    val set = sets[w]
                    val outs = perWorker[w]
                    barrier.await()
                    while (true) {
                        val i = next.getAndIncrement()
                        if (i >= tasks.size) break
                        ms[i] = forward(tasks[i], set[tasks[i]], pg, outs.getValue(tasks[i]))
                    }
                } catch (t: Throwable) {
                    firstError.compareAndSet(null, t)
                    barrier.reset()
                }
            }.also { it.name = "probe-tp-$w" }
        }
        val cpu0 = Process.getElapsedCpuTime()
        val t0 = System.nanoTime()
        threads.forEach { it.start() }
        threads.forEach { it.join() }
        val wall = System.nanoTime() - t0
        val cpu = Process.getElapsedCpuTime() - cpu0
        firstError.get()?.let { throw it }
        var bad = 0
        for (w in 0 until workers) for (m in M.values()) {
            val want = expect(m) ?: continue
            val o = perWorker[w].getValue(m)
            // 這條執行緒可能沒拿到這顆模型的工作（全 0）：全 0 不算數
            if (o.all { a -> a.all { it == 0f } }) continue
            if (shaOf(o) != want) bad++
        }
        val med = M.values().associateWith { m ->
            val v = tasks.indices.filter { tasks[it] == m }.map { ms[it] }.sorted()
            if (v.isEmpty()) 0L else v[v.size / 2] / 1_000_000
        }
        return Throughput(wall, cpu, tasks.size, med, bad)
    }

    companion object {
        const val AOT_SIZE = 768
        const val MAX_REPEAT = 8

        /** 合併模式一頁用到的模型：翻譯找字、去字、夜讀找字、yolo、cseg。 */
        val ROUND = listOf(M.DBNET, M.AOT, M.DBNET, M.YOLO, M.CSEG)

        /** 兩兩同時跑。電腦（qemu）上驗過的組合先跑，還沒人試過的 AOT 組合放最後：萬一閃退，前面的資料已經留下。 */
        val PAIRS = listOf(
            M.DBNET to M.CSEG, M.DBNET to M.DBNET, M.DBNET to M.YOLO, M.YOLO to M.CSEG, M.YOLO to M.YOLO, M.CSEG to M.CSEG,
            M.AOT to M.YOLO, M.AOT to M.DBNET, M.AOT to M.CSEG, M.AOT to M.AOT,
        )

        fun shaOf(arrs: Array<FloatArray>): String {
            val md = MessageDigest.getInstance("SHA-256")
            val chunk = 1 shl 18
            val bb = ByteBuffer.allocate(chunk * 4)
            for (a in arrs) {
                var off = 0
                while (off < a.size) {
                    val n = minOf(chunk, a.size - off)
                    bb.clear()
                    bb.asFloatBuffer().put(a, off, n)
                    md.update(bb.array(), 0, n * 4)
                    off += n
                }
            }
            return md.digest().joinToString("") { "%02x".format(it) }
        }
    }
}
