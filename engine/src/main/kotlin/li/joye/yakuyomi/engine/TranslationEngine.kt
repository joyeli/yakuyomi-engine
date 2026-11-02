package li.joye.yakuyomi.engine

import android.graphics.Bitmap

/**
 * 引擎對外主介面（CLAUDE.md §6「引擎解耦」、§13）。**建議用 [Yakuyomi.create] 取得實例**；唯一實作＝[Pipeline]。
 *
 * 回傳 [PageResult] 而非裸 Bitmap：要能表達 §11 的「略過（不覆蓋）／失敗（不覆蓋、可重試）／成功（可覆蓋+marker）」，
 * 裸 Bitmap 無法區分這三態。覆蓋原檔／寫 marker／page-level resume／跨頁批次併發＝呼叫端職責（§3）。
 *
 * **生命週期**：持有原生 ONNX session，用完要 [close] 釋放 native 記憶體。建議搭 `use { }`：
 * ```
 * Yakuyomi.create(models, alphabet, apiKey).use { engine ->
 *     when (val r = engine.translatePage(bitmap)) {
 *         is PageResult.Translated -> writeBack(r.page)
 *         is PageResult.Skipped    -> { /* 保留原圖、不覆蓋 */ }
 *         is PageResult.Failed     -> { /* 保留原圖、可重試 */ }
 *     }
 * }
 * ```
 * **執行緒**：[translatePage] 是 suspend、一次處理一頁，請在背景 dispatcher 呼叫；單一實例**非並發安全**（勿同實例同時翻多頁）。
 */
interface TranslationEngine : AutoCloseable {
    /**
     * 翻譯單頁。不會 recycle 輸入 [page]；成功時 [PageResult.Translated.page] 是**另一個新 bitmap**（非原物件）。
     * @param page 來源頁點陣圖（所有權仍屬呼叫端）。
     */
    suspend fun translatePage(page: Bitmap): PageResult

    /**
     * 單緒暖機：對每個原生 session（detector / OCR / 去字）各空跑一次推論，完成首次 lazy 初始化（權重預處理 / arena 配置等）。
     * **併發翻多頁前先呼叫一次**（單緒）——否則多頁同時打進「剛載好、還沒推論過」的冷 session 會撞首次初始化的 race → 原生 crash。
     * 建構後呼叫一次即可（見 fork `TranslationEngineService.ensureEngine`）；boxfill 去字不跑其 session、故不暖它。best-effort。
     */
    fun warmUp()

    /**
     * 偵測模型的 `.param` 路徑；null＝這個實作不借出偵測器（[detectWith] 不可用）。
     * 給想共用同一顆偵測器的呼叫端比對「是不是同一顆模型」（例如 Yakuyomi fork 的合併處理讓夜讀借用，見 [detectWith]）。
     */
    val detectorModelPath: String? get() = null

    /**
     * 用引擎那顆偵測器、但帶呼叫端給的 [cfg] 偵測一頁（`Detector.detect(page, cfg)`：Net 共用、參數各帶各的；不經 OCR／
     * 翻譯／去字、不碰 [translatePage] 的任何狀態）。可與 [translatePage] 同時呼叫——偵測器本來就給跨頁流水線多頁同時用。
     * 回傳的 [Detection.textMask] 歸呼叫端（用完 recycle）；[page] 所有權不變。
     *
     * **生命週期由呼叫端顧**：引擎 [close] 之後不可再呼叫（原生 Net 已釋放）。借用的呼叫端要自己確保借用期間引擎不會被關
     * （Yakuyomi fork 的 TranslationEngineService 用租約擋住關引擎的路徑）。
     * [detectorModelPath] 為 null 的實作呼叫會拋 [UnsupportedOperationException]。
     *
     * 內部 API（[InternalEngineApi]）：只給 Yakuyomi fork 的夜讀共用偵測器用，隨時可能改。
     */
    @InternalEngineApi
    fun detectWith(page: Bitmap, cfg: DetectorConfig): Detection =
        throw UnsupportedOperationException("這個 TranslationEngine 不借出偵測器")

    /** 釋放底層模型的原生資源（detector/ocr/inpainter 的 ONNX session）。呼叫後此實例不可再用。 */
    override fun close()
}
