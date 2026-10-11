package li.joye.yakuyomi.engine

import android.graphics.Bitmap
import android.graphics.Typeface
import android.util.Log
import kotlinx.coroutines.async
import kotlinx.coroutines.cancelAndJoin
import kotlinx.coroutines.coroutineScope

/**
 * 單頁翻譯結果（§11：成功才覆蓋+marker、略過不覆蓋、失敗不覆蓋待重試）。
 * 引擎只回結果，不碰檔案——覆蓋/marker/resume 由呼叫端（下載 worker）依此處理（§3、§12.6）。
 */
sealed interface PageResult {
    /** 成功：可覆蓋原檔 + 寫「已翻譯」marker。 */
    data class Translated(val page: Bitmap, val stats: PageStats, val analysis: PageAnalysis? = null) : PageResult

    /** 沒東西可翻（偵測不到字 / OCR 全空 / 譯文全被過濾）：保留原圖、標記略過、**不覆蓋**。 */
    data class Skipped(val reason: String, val stats: PageStats) : PageResult

    /** 出錯（網路/429 重試後仍失敗/例外）：保留原圖、**不標記**、之後可重試。 */
    data class Failed(val reason: String) : PageResult
}

/**
 * 重繪素材（給「最低成本切換去字方法」用）：seg 文字遮罩 + regions（含 quad/角度/onArt/源文/譯文）。
 * 原圖由呼叫端持有（translatePage 的輸入）、去字方法由呼叫端決定，故不在此。
 * 序列化/落地（含 mask 轉文字塞 json）由呼叫端（reader）負責。
 */
data class PageAnalysis(val mask: Bitmap, val regions: List<TextRegion>)

/** 逐階段計時與計數（除錯/效能用）。 */
data class PageStats(
    val lines: Int,
    val regions: Int,
    val kept: Int,
    val detectMs: Long,
    val ocrMs: Long,
    val translateMs: Long,
    val inpaintMs: Long,
    val renderMs: Long,
    val wallMs: Long = 0,   // 實際牆鐘時間（去字‖翻譯重疊 ⇒ 通常 < totalMs；省下的＝重疊掉的）
    val promptTokens: Int = 0,      // 本頁 LLM 請求的 prompt token（無 LLM/代理不回＝0）。供統計：用量只記、不計價。
    val completionTokens: Int = 0,  // 本頁 LLM 請求的 completion token。
    val skippedTopEdge: Boolean = false,    // 上緣有待接縫處理的字（跳過了貼邊區，或筆畫遮罩碰到上緣）
    val skippedBottomEdge: Boolean = false, // 下緣同上
    val skippedBoxes: List<FloatArray> = emptyList(), // 依 RegionScope 跳過的貼邊區 bbox（頁座標 x0,y0,x1,y1）
) {
    /** 各階段純計算時間之和（不含重疊修正）；實際耗時看 [wallMs]。 */
    val totalMs: Long get() = detectMs + ocrMs + translateMs + inpaintMs + renderMs
}

/**
 * 引擎主 pipeline：單頁 偵測→OCR→分群→翻譯→過濾→去字→排版。
 * 順序對齊 manga_translator.py 主流程（§5 順序＝第一層）；orchestration＝第二層。
 *
 * **§11 不變式焊進此處：永不用比原圖更糟的東西覆蓋。**
 *   - 偵測不到字 / OCR 全空 / 譯文全失敗 → [PageResult.Skipped]（保留原圖、不覆蓋；去字若已並發跑出來也丟棄）。
 *   - 單 block 翻譯失敗 → **去字後重貼 OCR 原文**（日文）——非「保留源圖」。讓去字不必等翻譯 ⇒ 去字(CPU)與翻譯(網路)並發重疊。
 *   - 任一階段拋例外（網路/429 重試後仍失敗等）→ [PageResult.Failed]（保留原圖、不覆蓋、可重試；丟棄已並發的去字）。
 *
 * **去字‖翻譯重疊**：兩者只依賴 OCR、互不爭資源（網路 vs CPU）⇒ 同時跑，[PageStats.wallMs] < [PageStats.totalMs]（省下重疊掉的）。
 *
 * 模型由呼叫端建好傳入；本類不碰檔案、不管跨頁批次與 resume。
 * **生命週期**：[close] 會收掉傳入的 detector/ocr/inpainter 的原生 session ——
 * 走 [Yakuyomi.create] 時這三顆由工廠建、歸本 pipeline 所有，`use { }` 即可。
 * 進階：若你注入「想重用、共享」的元件，請自己管生命週期、別呼叫本 [close]（否則會把共享元件一起關掉）。
 */
class Pipeline(
    private val detector: Detector,
    private val ocr: Ocr,
    private val translator: Translator?, // null＝不翻譯（純偵測/OCR 除錯用）
    private val inpainter: Inpainter,
    private val cfg: EngineConfig = EngineConfig(),
    private val typeface: Typeface? = null,
) : TranslationEngine {

    override suspend fun translatePage(page: Bitmap): PageResult = translatePage(page, RegionScope.ALL)

    override suspend fun translatePage(page: Bitmap, scope: RegionScope): PageResult = coroutineScope {
        val tWall = System.currentTimeMillis()
        EngineTrace.log("pipe.page.enter ${page.width}x${page.height}")
        // 偵測
        val tDet = System.currentTimeMillis()
        val detection = try {
            detector.detect(page)
        } catch (t: Throwable) {
            Log.e(TAG, "偵測失敗", t); return@coroutineScope PageResult.Failed("detect: ${t.message}")
        }
        val lines = detection.lines
        EngineTrace.log("pipe.detect.done lines=${lines.size}")
        val detectMs = System.currentTimeMillis() - tDet
        if (lines.isEmpty()) {
            return@coroutineScope PageResult.Skipped("偵測不到文字", PageStats(0, 0, 0, detectMs, 0, 0, 0, 0))
        }

        // OCR + 分群
        val tOcr = System.currentTimeMillis()
        EngineTrace.log("pipe.ocr.enter lines=${lines.size}")
        try {
            ocr.recognize(page, lines)
        } catch (t: Throwable) {
            Log.e(TAG, "OCR 失敗", t); return@coroutineScope PageResult.Failed("ocr: ${t.message}")
        }
        EngineTrace.log("pipe.ocr.exit")
        val regions = Grouping.mergeStacked(Grouping.group(lines))
        val ocrMs = System.currentTimeMillis() - tOcr
        // 去字集＝有 OCR 原文、且在 [scope] 內的區（空白＝疑似誤偵測，不去字、保畫面）。此集翻譯前就確定 ⇒ 去字可與翻譯並發。
        val nonBlank = regions.filter { it.sourceText.isNotBlank() }
        val textRegions = nonBlank.filter { scope.keeps(it, page.height) }
        // 跳過的貼邊區（含 OCR 讀不出的殘段）→ 交給接縫處理；筆畫遮罩碰到邊緣也算（被切掉的字殘段可能偵測不成行）
        val skippedRegions = if (scope.skipTopEdge || scope.skipBottomEdge) regions.filter { !scope.keeps(it, page.height) } else emptyList()
        val skippedTop = scope.skipTopEdge &&
            (skippedRegions.any { it.y0 < page.height / 2 } || maskTouchesRow(detection.textMask, 0))
        val skippedBottom = scope.skipBottomEdge &&
            (skippedRegions.any { it.y1 >= page.height / 2 } || maskTouchesRow(detection.textMask, page.height - 1))
        val skippedBoxes = skippedRegions.map { floatArrayOf(it.x0, it.y0, it.x1, it.y1) }
        if (textRegions.isEmpty()) {
            val why = if (nonBlank.isEmpty()) "OCR 全空" else "只有貼邊／範圍外的文字區"
            return@coroutineScope PageResult.Skipped(
                why,
                PageStats(lines.size, regions.size, 0, detectMs, ocrMs, 0, 0, 0, skippedTopEdge = skippedTop, skippedBottomEdge = skippedBottom, skippedBoxes = skippedBoxes),
            )
        }

        // ★ 去字（CPU）‖ 翻譯（網路）並發：兩者只依賴 OCR，可同時跑（網路等待時 CPU 去字、互不爭資源）。
        // §11 改：失敗區不再「保留源圖」，而是「去字後重貼 OCR 原文」(使用者拍板：重貼成本低、不需源圖狀態) ⇒ 去字與翻譯解耦。
        // 不限去字方法（boxfill/lama 皆可重疊）。boxfill(~0.5s)整段藏進翻譯(~2.7s)＝免費。
        var inpaintMs = 0L
        val inpaintJob = async {
            val t0 = System.currentTimeMillis()
            EngineTrace.log("pipe.inpaint.enter regions=${textRegions.size}")
            val r = inpainter.inpaint(page, textRegions, detection.textMask)
            EngineTrace.log("pipe.inpaint.exit")
            inpaintMs = System.currentTimeMillis() - t0
            r
        }

        var translateMs = 0L
        var promptTok = 0
        var completionTok = 0
        // per-call 診斷 metadata（區域變數、不讀 LlmTranslator 的共享單值欄位 → 跨頁併發各頁拿自己的、不 race）。
        var llmError: String? = null
        var llmRaw: String? = null
        if (translator != null) {
            val tTr = System.currentTimeMillis()
            EngineTrace.log("pipe.translate.enter n=${textRegions.size}")
            val cht = try {
                val llm = translator as? LlmTranslator
                if (llm != null) {
                    // 走 translateDetailed → per-call 拿 translations + usage + error + raw（跨頁併發安全）。
                    val r = llm.translateDetailed(textRegions.map { it.sourceText })
                    r.usage?.let { promptTok = it.promptTokens; completionTok = it.completionTokens }
                    llmError = r.error
                    llmRaw = r.raw
                    r.translations
                } else {
                    translator.translate(textRegions.map { it.sourceText })
                }
            } catch (t: Throwable) {
                Log.e(TAG, "翻譯失敗", t)
                inpaintJob.cancelAndJoin() // 翻譯掛 → 丟棄去字、留原圖（§11；native run 不可中斷，cancel 實為等它跑完再丟）
                return@coroutineScope PageResult.Failed("translate: ${t.message}")
            }
            textRegions.forEachIndexed { j, r -> r.translatedText = cht.getOrElse(j) { r.sourceText } }
            translateMs = System.currentTimeMillis() - tTr
            EngineTrace.log("pipe.translate.exit err=$llmError")
        } else {
            textRegions.forEach { it.translatedText = it.sourceText } // 無 key debug：排版原文
        }

        // 判定每區譯文有效性（空白/數字/regex/譯==原＝失敗）。整頁全失敗 → 留原圖（Skipped、丟棄去字）。
        val kept = if (translator != null) TextFilter.apply(textRegions, cfg.translator.filterText) else textRegions
        if (kept.isEmpty()) {
            val aligned = textRegions.count { it.translatedText.isNotBlank() && it.translatedText != it.sourceText }
            val dbg = textRegions.take(2).joinToString(" ‖ ") { "${it.sourceText.take(8)}→${it.translatedText.take(8)}" }
            Log.w(TAG, "全數過濾 對齊$aligned/${textRegions.size} err=$llmError 回應=$llmRaw")
            inpaintJob.cancelAndJoin()
            // §11 盲點修正：分辨「網路/格式軟失敗」vs「真的全不可譯」。
            // LlmTranslator 對網路/HTTP 例外是「catch + 回傳原文」（不丟例外）→ 全頁 translated==source → 落到這裡全數過濾。
            // 若一律回 Skipped(標記略過、算已處理)，網路失敗的頁會被當「已翻」、整章不變紅（正是此盲點）。改用 error 分流（per-call、不 race）：
            //  - error != null（例外〔網路/HTTP〕或部分解析）→ Failed：不標記、之後重試、整章變紅（呼叫端 drain 標 ERROR）。
            //  - error == null（LLM 正常全解析、但內容全被過濾，如整頁狀聲詞被原樣回 translated==source）→ Skipped：略過、不無限重試。
            return@coroutineScope if (llmError != null) {
                PageResult.Failed("全數過濾(LLM 失敗 $llmError)｜回應=${llmRaw?.take(80)}")
            } else {
                PageResult.Skipped(
                    "全數過濾 對齊$aligned/${textRegions.size}｜回應=$llmRaw｜$dbg",
                    PageStats(
                        lines.size, regions.size, 0, detectMs, ocrMs, translateMs, 0, 0,
                        promptTokens = promptTok, completionTokens = completionTok,
                        skippedTopEdge = skippedTop, skippedBottomEdge = skippedBottom, skippedBoxes = skippedBoxes,
                    ),
                )
            }
        }
        // 失敗的區（不在 kept）→ 譯文改回原文＝去字後重貼 OCR 日文（TextRegion 無 equals override→HashSet 走 identity）。
        val keptSet = kept.toHashSet()
        textRegions.forEach { if (it !in keptSet) it.translatedText = it.sourceText }

        // 等去字完成（多半已與翻譯重疊跑完）
        EngineTrace.log("pipe.inpaint.await")
        val cleaned = try {
            inpaintJob.await()
        } catch (t: Throwable) {
            Log.e(TAG, "去字失敗", t); return@coroutineScope PageResult.Failed("inpaint: ${t.message}")
        }

        // 沒譯成的區（狀聲詞、OCR 雜訊、LLM 回空／原樣）→ 把去字範圍還原成原圖、不重貼 OCR 原文
        // （重貼的是 OCR 讀錯的字＋預設字型，比原圖更糟；還原＝永不比原圖更糟）。去字已與翻譯並發，這裡只回貼像素。
        val failed = textRegions.filter { it !in keptSet }
        if (failed.isNotEmpty()) restoreOriginal(page, cleaned, failed, kept)

        // 排版（只排有譯文的區）
        val tRn = System.currentTimeMillis()
        EngineTrace.log("pipe.render.enter")
        val finalPage = Renderer.render(cleaned, kept, cfg.render, typeface)
        val renderMs = System.currentTimeMillis() - tRn
        EngineTrace.log("pipe.page.done")

        PageResult.Translated(
            finalPage,
            PageStats(
                lines.size, regions.size, kept.size, detectMs, ocrMs, translateMs, inpaintMs, renderMs,
                System.currentTimeMillis() - tWall, promptTok, completionTok,
                skippedTopEdge = skippedTop, skippedBottomEdge = skippedBottom, skippedBoxes = skippedBoxes,
            ),
            PageAnalysis(detection.textMask, textRegions),
        )
    }

    /** 把 [failed] 各區的去字範圍（bbox 外擴去字餘裕）還原成原圖像素；與有譯文的區重疊處不動。 */
    private fun restoreOriginal(page: Bitmap, cleaned: Bitmap, failed: List<TextRegion>, kept: List<TextRegion>) {
        val w = page.width; val h = page.height
        val pad = cfg.inpainter.bboxPad + (cfg.inpainter.maskDilate / 2f).toInt() + 2
        for (r in failed) {
            val x0 = (r.x0.toInt() - pad).coerceIn(0, w - 1); val y0 = (r.y0.toInt() - pad).coerceIn(0, h - 1)
            val x1 = (r.x1.toInt() + pad).coerceIn(x0 + 1, w); val y1 = (r.y1.toInt() + pad).coerceIn(y0 + 1, h)
            val bw = x1 - x0; val bh = y1 - y0
            val orig = IntArray(bw * bh); page.getPixels(orig, 0, bw, x0, y0, bw, bh)
            val cur = IntArray(bw * bh); cleaned.getPixels(cur, 0, bw, x0, y0, bw, bh)
            for (y in 0 until bh) for (x in 0 until bw) {
                val gx = (x0 + x).toFloat(); val gy = (y0 + y).toFloat()
                if (kept.none { gx >= it.x0 - pad && gx < it.x1 + pad && gy >= it.y0 - pad && gy < it.y1 + pad }) {
                    cur[y * bw + x] = orig[y * bw + x]
                }
            }
            cleaned.setPixels(cur, 0, bw, x0, y0, bw, bh)
        }
    }

    /** 筆畫遮罩在第 [y] 列（附近 3 列）有足夠的字（≥ 3 px）：被頁緣切掉的字殘段。 */
    private fun maskTouchesRow(mask: Bitmap, y: Int): Boolean {
        val w = mask.width
        val row = IntArray(w)
        var n = 0
        for (dy in 0 until 3) {
            val yy = if (y == 0) dy else y - dy
            if (yy !in 0 until mask.height) continue
            mask.getPixels(row, 0, w, 0, yy, w, 1)
            for (p in row) if ((p and 0xFF) > 127) n++
        }
        return n >= 3
    }

    /**
     * 單緒暖機：對 detector / OCR / 去字三個原生 session 各空跑一次推論，完成首次 lazy 初始化。
     * 建構後、放行跨頁併發前呼叫一次（見介面說明）。三者依序（單緒），best-effort（各自 catch）。
     */
    override fun warmUp() {
        EngineTrace.log("warmup.detector.enter")
        detector.warmUp()
        EngineTrace.log("warmup.ocr.enter")
        ocr.warmUp()
        EngineTrace.log("warmup.inpainter.enter")
        inpainter.warmUp()
        EngineTrace.log("warmup.done")
    }

    /** 釋放 detector/ocr/inpainter 的原生 ONNX session（見類別說明的生命週期注意事項）。 */
    override fun close() {
        runCatching { detector.close() }
        runCatching { ocr.close() }
        runCatching { inpainter.close() }
    }

    companion object {
        private const val TAG = "Pipeline"
    }
}
