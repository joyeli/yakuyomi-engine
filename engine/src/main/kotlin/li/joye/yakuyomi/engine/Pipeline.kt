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

    override suspend fun translatePage(page: Bitmap): PageResult = coroutineScope {
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
        val regions = Grouping.group(lines)
        val ocrMs = System.currentTimeMillis() - tOcr
        // 去字集＝有 OCR 原文的區（空白＝疑似誤偵測，不去字、保畫面）。此集翻譯前就確定 ⇒ 去字可與翻譯並發。
        val textRegions = regions.filter { it.sourceText.isNotBlank() }
        if (textRegions.isEmpty()) {
            return@coroutineScope PageResult.Skipped("OCR 全空", PageStats(lines.size, regions.size, 0, detectMs, ocrMs, 0, 0, 0))
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

        // 排版（全 textRegions：kept 貼譯文、失敗區貼原文）
        val tRn = System.currentTimeMillis()
        EngineTrace.log("pipe.render.enter")
        val finalPage = Renderer.render(cleaned, textRegions, cfg.render, typeface)
        val renderMs = System.currentTimeMillis() - tRn
        EngineTrace.log("pipe.page.done")

        PageResult.Translated(
            finalPage,
            PageStats(
                lines.size, regions.size, kept.size, detectMs, ocrMs, translateMs, inpaintMs, renderMs,
                System.currentTimeMillis() - tWall, promptTok, completionTok,
            ),
            PageAnalysis(detection.textMask, textRegions),
        )
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

    /** 偵測器的模型 `.param` 路徑（見 [TranslationEngine.detectorModelPath]）。 */
    override val detectorModelPath: String get() = detector.modelPath

    /**
     * 借這條 pipeline 的偵測器、帶 [cfg] 偵測一頁（`Detector.detect(page, cfg)`，見 [TranslationEngine.detectWith]）。
     * 不動 [translatePage] 用的 `cfg.detector`；[close] 之後不可呼叫。
     */
    @InternalEngineApi
    override fun detectWith(page: Bitmap, cfg: DetectorConfig): Detection = detector.detect(page, cfg)

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
