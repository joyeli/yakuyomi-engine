package li.joye.yakuyomi.engine

import android.graphics.Bitmap
import android.graphics.Typeface

/**
 * 引擎入口工廠：把「建三顆模型元件 + 組 [Pipeline]」收成一行，回傳可 `use { }` 的 [TranslationEngine]。
 *
 * 取代各消費端各自手動拼裝 + 各自記得 `close()` 的重複碼：
 * ```
 * val models = ModelSet.resolve(localModelFiles) ?: return   // 模型沒備齊 → 略過
 * Yakuyomi.create(models, alphabet, apiKey).use { engine ->
 *     for (bmp in pages) when (val r = engine.translatePage(bmp)) {
 *         is PageResult.Translated -> writeBack(r.page)
 *         is PageResult.Skipped    -> { /* 保留原圖 */ }
 *         is PageResult.Failed     -> { /* 保留原圖、可重試 */ }
 *     }
 * }
 * ```
 * **進階**（逐元件除錯，如 debug overlay）：可直接 new [Detector]/[Ocr]/[Inpainter]/[LlmTranslator] 再自組 [Pipeline]，
 * 但生命週期得自己管（這條工廠路徑才會幫你 close）。
 */
object Yakuyomi {
    /**
     * 建一個翻譯引擎。
     *
     * @param models   三顆模型的本機路徑（見 [ModelSet]；用 [ModelSet.resolve] 從檔名比對）。
     * @param alphabet OCR 字元表（48px CTC 解碼用；通常由引擎 assets 載入後傳入）。
     * @param apiKey   翻譯 LLM 的 API key；**null/空白＝不翻譯**（只跑偵測/OCR/去字，純除錯）。
     * @param config   引擎設定（全可調，預設見各 `*Config`）。
     * @param typeface 算繪字型；null＝系統預設 CJK。
     * @return 可 `use { }` 的 [TranslationEngine]；其 [TranslationEngine.close] 會釋放三顆模型的 native session。
     */
    fun create(
        models: ModelSet,
        alphabet: List<String>,
        apiKey: String?,
        config: EngineConfig = EngineConfig(),
        typeface: Typeface? = null,
    ): TranslationEngine {
        // 三顆模型（偵測／OCR／去字）全 NCNN（產品 arm64、NCNN 必在；ORT 已整個從引擎拔除、LaMa 退役）。
        check(NcnnBackend.available) { "NCNN 原生庫未載入（arm64 應可用）" }
        EngineTrace.log("create.detector")
        val detector = Detector(models.detectorNcnn ?: error("需 NCNN 偵測模型（.param）"), config.detector)
        EngineTrace.log("create.ocr")
        val ocr = Ocr(models.ocr, alphabet, config.ocr)
        // 去字兩門別（boxfill/aot）皆用同一顆 NCNN AOT 模型（boxfill 只平塗不跑它、但仍要載得起來）。
        EngineTrace.log("create.inpainter")
        val inpainter = Inpainter(models.aotInpainterNcnn ?: error("需 NCNN AOT 去字模型（.param）"), config.inpainter)
        val translator = apiKey?.takeIf { it.isNotBlank() }?.let { LlmTranslator(it, config.translator) }
        EngineTrace.log("create.done")
        return Pipeline(detector, ocr, translator, inpainter, config, typeface)
    }

    /** 這顆 CPU 有 fp16 storage/arithmetic（arm82 asimdhp）——決定 OCR 能不能用混合精度 param（見 [Ocr]、[OcrConfig.ncnnMixed]）。 */
    fun ncnnCpuSupportsFp16(): Boolean = NcnnBackend.cpuSupportsFp16

    /**
     * 診斷（sandbox 用）：對一頁跑一次偵測 → 同一批行框，逐個 [candidates]（OCR 模型路徑 + 設定）各建一個 [Ocr]
     * 做 OCR，回逐行讀取對照 + 各自的載入／recognize 耗時。用來真機 A/B NCNN 各精度組合（mixed／全 fp16／fp32，
     * 靠 [OcrCandidate.config] 的 ncnnMixed／ncnnFp16Storage／ncnnFp16Arith 切）。
     * 每個候選：暖跑（warmUp + 一輪不計時的 recognize，吃掉首次 forward 配置／JIT／冷啟）→ 正式計時一輪 → close。
     * 候選逐個開關（同時只有一顆 OCR 模型在記憶體）。
     */
    suspend fun ocrAbTest(
        detectorPath: String,
        alphabet: List<String>,
        page: Bitmap,
        candidates: List<OcrCandidate>,
        detectorConfig: DetectorConfig = DetectorConfig(),
    ): OcrAbResult {
        check(NcnnBackend.available) { "NCNN 原生庫未載入" }
        val detector = Detector(detectorPath, detectorConfig)
        try {
            val tDet = System.nanoTime()
            val det = detector.detect(page)
            val detectMs = (System.nanoTime() - tDet) / 1e6
            val clone = { det.lines.map { TextLine(it.quad, it.score) } } // recognize 就地寫 text → 每次跑用新副本
            val perCandidate = mutableListOf<List<String>>()
            val loadMs = mutableListOf<Double>()
            val recognizeMs = mutableListOf<Double>()
            val backends = mutableListOf<String>()
            for (c in candidates) {
                val tLoad = System.nanoTime()
                val ocr = Ocr(c.modelPath, alphabet, c.config)
                loadMs += (System.nanoTime() - tLoad) / 1e6
                backends += ocr.backend
                try {
                    ocr.warmUp()
                    ocr.recognize(page, clone())
                    val lines = clone()
                    val t0 = System.nanoTime()
                    ocr.recognize(page, lines)
                    recognizeMs += (System.nanoTime() - t0) / 1e6
                    perCandidate += lines.map { it.text }
                } finally {
                    runCatching { ocr.close() }
                }
            }
            val rows = det.lines.indices.map { i -> perCandidate.map { it[i] } }
            return OcrAbResult(candidates.map { it.label }, backends, rows, loadMs, recognizeMs, detectMs, det.lines.map { it.quad })
        } finally {
            runCatching { detector.close() }
        }
    }
}

/**
 * [Yakuyomi.ocrAbTest] 的一個候選：標籤 + OCR 模型路徑（NCNN `.param`；給原版或 `_mixed` 版皆可，實際載哪份由 [Ocr] 的
 * pickParam 依 config 決定）+ 設定（如 [OcrConfig.ncnnMixed]／[OcrConfig.ncnnFp16Storage]／[OcrConfig.ncnnFp16Arith]）。
 */
class OcrCandidate(val label: String, val modelPath: String, val config: OcrConfig = OcrConfig())

/**
 * [Yakuyomi.ocrAbTest] 結果：[labels]／[backends] 對應各候選；[rows] 逐行、每行是各候選讀出的文字（空＝低於信心門檻被丟）；
 * [loadMs]／[recognizeMs] 各候選的模型載入與正式一輪 recognize 耗時；[detectMs] 偵測耗時（只跑一次）；
 * [quads] 每行的偵測四邊形（與 [rows] 同序，給成果圖裁圖／畫框）。
 */
class OcrAbResult(
    val labels: List<String>,
    val backends: List<String>,
    val rows: List<List<String>>,
    val loadMs: List<Double>,
    val recognizeMs: List<Double>,
    val detectMs: Double,
    val quads: List<List<Pt>>,
)
