package li.joye.yakuyomi.engine

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Typeface
import kotlinx.coroutines.runBlocking
import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assume.assumeTrue
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config
import java.io.File

/**
 * 桌面整頁測試（開發用，預設略過）：在 Robolectric 原生圖形下跑真正的 [Pipeline]（偵測→OCR→分群→去字→排版），
 * 譯文由 JSON 對照表提供（不打 LLM），輸出成品 PNG + 除錯疊圖 + 區域 JSON，拿來逐頁看去字／排版效果。
 *
 * 需要：主機版 libyakuyomi_ncnn.so（YAKU_NATIVE_DIR）與三顆模型（YAKU_HARNESS_MODELS）。
 *   YAKU_HARNESS_PAGES         頁面資料夾（jpg/png/webp）
 *   YAKU_HARNESS_OUT           輸出夾
 *   YAKU_HARNESS_TRANSLATIONS  {"原文": "譯文"} 對照表（可省略；沒有的原文記進 <out>/pending.json、以「؟」代替排版）
 *   YAKU_HARNESS_FONT          （選）字型檔，模擬裝置系統字型的 fallback（例：Noto Sans Arabic）
 */
@RunWith(RobolectricTestRunner::class)
@Config(sdk = [35])
class PageHarnessTest {

    private class MapTranslator(private val map: Map<String, String>, private val pending: MutableSet<String>) : Translator {
        override suspend fun translate(queries: List<String>): List<String> = queries.map { q ->
            map[q] ?: map[q.trim()] ?: run { pending += q; System.getenv("YAKU_HARNESS_PLACEHOLDER") ?: "؟" }
        }
    }

    @Test
    fun runPages() {
        val pagesDir = System.getenv("YAKU_HARNESS_PAGES")
        val modelsDir = System.getenv("YAKU_HARNESS_MODELS")
        assumeTrue("YAKU_HARNESS_PAGES / YAKU_HARNESS_MODELS 未設：略過", pagesDir != null && modelsDir != null)
        check(NcnnBackend.available) { "主機版 libyakuyomi_ncnn.so 未載入（設 YAKU_NATIVE_DIR）" }
        val out = File(System.getenv("YAKU_HARNESS_OUT") ?: "$pagesDir/out").apply { mkdirs() }
        val map = System.getenv("YAKU_HARNESS_TRANSLATIONS")?.let { f ->
            val o = JSONObject(File(f).readText())
            o.keys().asSequence().associateWith { o.getString(it) }
        } ?: emptyMap()
        val typeface = System.getenv("YAKU_HARNESS_FONT")?.let { Typeface.createFromFile(it) }
        val alphabet = File("src/main/assets/models/alphabet-all-v5.txt").readLines()
        val m = File(modelsDir!!)
        val cfg = EngineConfig(
            render = RenderConfig(
                orientation = if (System.getenv("YAKU_HARNESS_ORIENT") == "auto") TextOrientation.AUTO else TextOrientation.HORIZONTAL,
            ),
            inpainter = InpainterConfig(method = System.getenv("YAKU_HARNESS_INPAINT") ?: "aot"),
        )
        val pending = linkedSetOf<String>()
        val detector = Detector(File(m, "dbnet_detect.ncnn.param").path, cfg.detector)
        val ocr = Ocr(File(m, "ocr_48px_ctc.ncnn.param").path, alphabet, cfg.ocr)
        val inpainter = Inpainter(File(m, "mit_aot_fixed512.ncnn.param").path, cfg.inpainter)
        val pipeline = Pipeline(detector, ocr, MapTranslator(map, pending), inpainter, cfg, typeface)
        val dumpClean = System.getenv("YAKU_HARNESS_CLEAN") != null
        val only = System.getenv("YAKU_HARNESS_ONLY")?.split(',')?.toSet()
        val files = File(pagesDir!!).listFiles { f -> f.extension.lowercase() in setOf("jpg", "jpeg", "png", "webp") }!!
            .filter { only == null || it.nameWithoutExtension in only }
            .sortedBy { it.name }
        val summary = JSONArray()
        pipeline.use { p ->
            if (System.getenv("YAKU_HARNESS_SEAMS") != null) {
                seamRun(p, files, out)
                return@use
            }
            for (f in files) {
                val page = BitmapFactory.decodeFile(f.path).copy(Bitmap.Config.ARGB_8888, false)
                val t0 = System.currentTimeMillis()
                val r = runBlocking { p.translatePage(page) }
                val ms = System.currentTimeMillis() - t0
                val item = JSONObject().put("page", f.name).put("ms", ms)
                when (r) {
                    is PageResult.Translated -> {
                        save(r.page, File(out, "${f.nameWithoutExtension}.png"))
                        val regions = r.analysis?.regions.orEmpty()
                        save(debugOverlay(page, regions), File(out, "${f.nameWithoutExtension}.debug.png"))
                        item.put("result", "translated").put("regions", JSONArray(regions.map { reg ->
                            JSONObject()
                                .put("box", JSONArray(listOf(reg.x0, reg.y0, reg.x1, reg.y1).map { it.toInt() }))
                                .put("dir", reg.direction).put("onArt", reg.onArt)
                                .put("std", reg.dbgStd.toDouble()).put("lum", reg.dbgWhite.toDouble())
                                .put("src", reg.sourceText).put("dst", reg.translatedText)
                                .put("lineH", reg.lineHeight.toDouble())
                                .put("layout", JSONArray(reg.layoutBoxes.map { b -> JSONArray(b.map { it.toInt() }) }))
                        }))
                    }
                    is PageResult.Skipped -> item.put("result", "skipped").put("reason", r.reason)
                    is PageResult.Failed -> item.put("result", "failed").put("reason", r.reason)
                }
                if (dumpClean) {
                    // 去字中間結果（排版前）：看去字乾淨度與氣泡內部量測
                    val det = detector.detect(page)
                    runBlocking { ocr.recognize(page, det.lines) }
                    val regs = Grouping.mergeStacked(Grouping.group(det.lines)).filter { it.sourceText.isNotBlank() }
                    val clean = runBlocking { inpainter.inpaint(page, regs, det.textMask) }
                    save(debugOverlay(clean, regs), File(out, "${f.nameWithoutExtension}.clean.png"))
                }
                println("[harness] ${f.name} ${item.getString("result")} ${ms}ms")
                summary.put(item)
            }
        }
        File(out, "summary.json").writeText(summary.toString(1))
        File(out, "pending.json").writeText(JSONObject(pending.associateWith { "" }).toString(1))
    }

    /**
     * 模擬 app 的條漫流程（PageTranslator.translateChapter + translateSeams）：每頁跳過貼著相鄰頁那側的文字區，
     * 再把每個接縫拼條只翻跨縫區、切回兩頁。輸出各頁 `<name>.seam.png` 與整條拼接 `strip.png`。
     */
    private fun seamRun(p: Pipeline, files: List<File>, out: File) {
        val pages = files.map { BitmapFactory.decodeFile(it.path).copy(Bitmap.Config.ARGB_8888, true) }.toMutableList()
        val flags = ArrayList<Pair<Boolean, Boolean>>()
        val boxes = ArrayList<List<FloatArray>>()
        for (i in pages.indices) {
            val scope = RegionScope(skipTopEdge = i > 0, skipBottomEdge = i < pages.size - 1)
            when (val r = runBlocking { p.translatePage(pages[i], scope) }) {
                is PageResult.Translated -> { pages[i] = r.page; flags += r.stats.skippedTopEdge to r.stats.skippedBottomEdge; boxes += r.stats.skippedBoxes }
                is PageResult.Skipped -> { flags += r.stats.skippedTopEdge to r.stats.skippedBottomEdge; boxes += r.stats.skippedBoxes }
                is PageResult.Failed -> { flags += false to false; boxes += emptyList<FloatArray>() }
            }
            println("[harness] ${files[i].name} flags=${flags.last()}")
        }
        for (i in 0 until pages.size - 1) {
            if (!flags[i].second && !flags[i + 1].first) continue
            val a = pages[i]; val b = pages[i + 1]; val w = a.width
            val h1 = minOf(a.height, (w * 1.2f).toInt()); val h2 = minOf(b.height, (w * 1.2f).toInt())
            val band = Bitmap.createBitmap(w, h1 + h2, Bitmap.Config.ARGB_8888)
            Canvas(band).apply {
                drawBitmap(a, android.graphics.Rect(0, a.height - h1, w, a.height), android.graphics.Rect(0, 0, w, h1), null)
                drawBitmap(b, android.graphics.Rect(0, 0, w, h2), android.graphics.Rect(0, h1, w, h1 + h2), null)
            }
            val keep = boxes[i].filter { it[3] > a.height - h1 }.map { floatArrayOf(it[0], it[1] - (a.height - h1), it[2], it[3] - (a.height - h1)) } +
                boxes[i + 1].filter { it[1] < h2 }.map { floatArrayOf(it[0], it[1] + h1, it[2], it[3] + h1) }
            val r = runBlocking { p.translatePage(band, RegionScope(onlyTouchingY = h1, keepBoxes = keep)) }
            println("[harness] seam ${files[i].name}|${files[i + 1].name} -> ${r.javaClass.simpleName}")
            if (r is PageResult.Translated) {
                val oa = a.copy(Bitmap.Config.ARGB_8888, true); val ob = b.copy(Bitmap.Config.ARGB_8888, true)
                Canvas(oa).drawBitmap(r.page, android.graphics.Rect(0, 0, w, h1), android.graphics.Rect(0, a.height - h1, w, a.height), null)
                Canvas(ob).drawBitmap(r.page, android.graphics.Rect(0, h1, w, h1 + h2), android.graphics.Rect(0, 0, w, h2), null)
                pages[i] = oa; pages[i + 1] = ob
            }
        }
        files.forEachIndexed { i, f -> save(pages[i], File(out, "${f.nameWithoutExtension}.seam.png")) }
        val total = pages.sumOf { it.height }
        val strip = Bitmap.createBitmap(pages[0].width, total, Bitmap.Config.ARGB_8888)
        var y = 0f
        Canvas(strip).apply { for (pg in pages) { drawBitmap(pg, 0f, y, null); y += pg.height } }
        save(strip, File(out, "strip.png"))
    }

    private fun debugOverlay(page: Bitmap, regions: List<TextRegion>): Bitmap {
        val bmp = page.copy(Bitmap.Config.ARGB_8888, true)
        val c = Canvas(bmp)
        val box = Paint().apply { style = Paint.Style.STROKE; strokeWidth = 3f }
        val line = Paint().apply { style = Paint.Style.STROKE; strokeWidth = 1.5f; color = Color.BLUE }
        val label = Paint().apply { color = Color.MAGENTA; textSize = 22f; isAntiAlias = true }
        regions.forEachIndexed { i, r ->
            box.color = if (r.onArt) Color.RED else Color.GREEN
            c.drawRect(r.x0, r.y0, r.x1, r.y1, box)
            if (r.layoutBoxes.isEmpty()) for (ln in r.lines) {
                val q = ln.quad
                for (k in q.indices) c.drawLine(q[k].x, q[k].y, q[(k + 1) % q.size].x, q[(k + 1) % q.size].y, line)
            }
            c.drawText("#$i", r.x0, (r.y0 - 4).coerceAtLeast(20f), label)
            val lp = Paint().apply { style = Paint.Style.STROKE; strokeWidth = 1f; color = Color.rgb(255, 140, 0) }
            r.layoutBoxes.forEach { b -> c.drawRect(b[0], b[1], b[2], b[3], lp) }
        }
        return bmp
    }

    private fun save(b: Bitmap, f: File) = f.outputStream().use { b.compress(Bitmap.CompressFormat.PNG, 100, it) }
}
