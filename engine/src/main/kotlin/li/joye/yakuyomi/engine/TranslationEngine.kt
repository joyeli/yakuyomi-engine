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
     * 只處理 [scope] 涵蓋的文字區（其餘區不去字、不排版、不送翻譯，像素原樣）。條漫（長條切成多個圖檔）用：
     * 逐頁翻時跳過貼著上／下緣的區（多半被檔案邊界切成兩半），整章翻完再把相鄰兩頁接縫處拼成一條、只翻跨接縫的區。
     * 預設實作忽略 [scope]（舊實作照常可用）。
     */
    suspend fun translatePage(page: Bitmap, scope: RegionScope): PageResult = translatePage(page)

    /**
     * 單緒暖機：對每個原生 session（detector / OCR / 去字）各空跑一次推論，完成首次 lazy 初始化（權重預處理 / arena 配置等）。
     * **併發翻多頁前先呼叫一次**（單緒）——否則多頁同時打進「剛載好、還沒推論過」的冷 session 會撞首次初始化的 race → 原生 crash。
     * 建構後呼叫一次即可（見 fork `TranslationEngineService.ensureEngine`）；boxfill 去字不跑其 session、故不暖它。best-effort。
     */
    fun warmUp()

    /** 釋放底層模型的原生資源（detector/ocr/inpainter 的 ONNX session）。呼叫後此實例不可再用。 */
    override fun close()
}

/**
 * 文字區範圍（見 [TranslationEngine.translatePage]）。
 * @param skipTopEdge    跳過貼著（或離不到半行高）頁面上緣的區。
 * @param skipBottomEdge 跳過貼著頁面下緣的區。
 * @param onlyTouchingY  非 null：接縫模式——只留跨過這條水平線的區，或與 [keepBoxes] 重疊的區（逐頁時跳過的貼邊區，
 *   換算到拼條座標）。其餘區（多半是已經翻好的字）一律不動。
 */
data class RegionScope(
    val skipTopEdge: Boolean = false,
    val skipBottomEdge: Boolean = false,
    val onlyTouchingY: Int? = null,
    val keepBoxes: List<FloatArray> = emptyList(),
) {
    fun keeps(r: TextRegion, pageHeight: Int): Boolean {
        val near = maxOf(EDGE_TOL.toFloat(), r.lineHeight * 0.5f)
        if (skipTopEdge && r.y0 <= near) return false
        if (skipBottomEdge && r.y1 >= pageHeight - 1 - near) return false
        val y = onlyTouchingY ?: return true
        if (r.y0 < y - 1 && r.y1 > y + 1) return true
        return keepBoxes.any { b -> r.x0 < b[2] && r.x1 > b[0] && r.y0 < b[3] && r.y1 > b[1] }
    }

    companion object {
        val ALL = RegionScope()
        const val EDGE_TOL = 6
    }
}
