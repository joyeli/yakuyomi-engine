package li.joye.yakuyomi.engine

import android.graphics.Bitmap
import li.joye.yakuyomi.nightread.Gray
import li.joye.yakuyomi.nightread.Mask
import li.joye.yakuyomi.nightread.NightRead
import li.joye.yakuyomi.nightread.NightReadDebug
import li.joye.yakuyomi.nightread.NightReadInput
import li.joye.yakuyomi.nightread.NightReadParams
import li.joye.yakuyomi.nightread.NightReadStageTimer
import li.joye.yakuyomi.nightread.NightTier
import li.joye.yakuyomi.nightread.TextRegion as NrRegion
import java.util.concurrent.Executor
import kotlin.math.abs
import kotlin.math.floor
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

/**
 * 多顆人物分割器的聯集（yolo ∪ cseg 定案配方：只用 yolo 守護框違規 18→27，加上 cseg 更準；見 nightread README）。
 * segment 逐顆跑、結果就地 OR 進第一顆的陣列（省一份 w×h）；close 關全部。
 * 不另開緒：多緒 Net 在 NCNN 全域鎖下本來就串行；1 緒不進鎖的組只在夜讀讓路時用（一次一頁、刻意不吃核）。
 */
class UnionCharSegmenter(private val parts: List<CharSegmenter>) : CharSegmenter {

    init {
        require(parts.isNotEmpty()) { "UnionCharSegmenter 至少要一顆分割器" }
    }

    override fun segment(page: Bitmap): BooleanArray {
        val acc = parts[0].segment(page)
        for (k in 1 until parts.size) {
            val m = parts[k].segment(page)
            check(m.size == acc.size) { "分割器輸出尺寸不一致：${m.size} vs ${acc.size}" }
            for (i in acc.indices) if (m[i]) acc[i] = true
        }
        return acc
    }

    override fun warmUp() {
        parts.forEach { it.warmUp() }
    }

    override fun close() {
        // 一顆關失敗也要把其餘關掉（native handle 不能漏）
        parts.forEach { runCatching { it.close() } }
    }
}

/**
 * [NightReadRenderer.render]／[NightReadRenderer.renderTiers] 的耗時與縮圖紀錄（毫秒）。一條龍版與多檔版三段都填；
 * 分段版只填 [renderMs]。
 */
class NightReadStats {
    /** detector.detect 本身（不含分群、遮罩轉換——那些算進 [renderMs]）。 */
    var detectMs = 0L
    /** charSeg.segment（聯集配方＝各顆相加）。 */
    var maskMs = 0L
    /**
     * 灰階／彩度、分群 AABB、遮罩轉換、NightRead.render、輸出 Bitmap。多檔版＝分析＋各檔合成＋各檔轉 Bitmap，
     * **不含** sink 裡的時間（編碼由呼叫端自己量）。
     */
    var renderMs = 0L
    /** 頁超過 [NightReadRenderer.MAX_PIXELS] 時縮到的尺寸 (w, h)；沒縮＝null。縮圖本身的時間不計入三段。 */
    var scaledTo: Pair<Int, Int>? = null
    /**
     * [NightRead.render] 各段耗時（ms，依完成順序），只列 ≥ [STAGE_MIN_MS] 的段，例如 `separators=812 gutterBand=240`。
     * 段名＝nightread 的除錯回呼名，時間＝上一個回呼到這一個（同 ProfileTest 的「本段」）。只在傳了 stats 時量。
     */
    var stagesMs: String? = null
    /**
     * 多檔版（[NightReadRenderer.renderTiers]）每檔的合成耗時（ms，從該檔 keep 篩完、開始合成，到輸出 Bitmap 建好；不含
     * 分析、不含 sink），依傳入的檔位順序，例如三檔 `l1=310 l2=- l3=120`、產品兩檔 `l2=310 l3=120`；`-`＝與前一檔相同、
     * 沒合成（keep 去重）。keep 篩選記在 [stagesMs] 的 `lN.keep` 段，不在這裡；plain 判定（與檔位無關、只算一次）落在
     * 第一個用到它的檔（三檔＝L1、產品兩檔＝L2）的 `.keep` 段。合成了但逐像素與前一檔相同（輸出去重、交 null）的檔照樣
     * 有數字。單檔版不填。
     */
    var tierMs: String? = null

    companion object {
        const val STAGE_MIN_MS = 30L
    }
}

/**
 * 夜讀（頁面本身變暗的分區重繪）膠水：把引擎的偵測／人物分割輸出接成 nightread 函式庫要的 [NightReadInput]，
 * 回一張新的暗色頁 Bitmap。演算法全在 `li.joye.yakuyomi.nightread`（純 Kotlin、不碰 android.graphics），這裡只做
 * Bitmap ↔ 陣列、分群、縮圖與計時。從 sandbox 的 nightReadOnce 搬進引擎，好讓 fork 經 includeBuild 直接用。
 *
 * **輸入契約**（nightread README「格式要求」四條 + 文字區；踩了不會報錯、輸出默默變糟）：
 *  1. 灰階＝BT.601 `(R*299 + G*587 + B*114 + 500) / 1000`——管線所有門檻（白 235、墨 128…）都在這個空間量的。
 *  2. `seg`＝文字**區域**遮罩、非精確筆畫（核心填色拿它當種子，筆畫太瘦泡填不滿）。用 [Detection.textMask]（DBNet 第二輸出）正合。
 *  3. `seg` 要涵蓋**所有**文字（含 SFX／裝飾字，即使翻譯不處理）——這裡不濾任何區域、不看 OCR 結果。
 *  4. `charMask`＝分割模型**原輸出**（不收邊、不平滑；貼墨收邊與偽泡判定由管線自己做）。
 *  5. `regions`＝[Grouping.group] 後每個 [TextRegion] 的軸對齊 bbox（Float → Int、夾進頁面）。
 *  6. `inpaintMask`＝呼叫端給的譯後頁去字遮罩（翻譯素材 `<頁>.mask.png`）最近鄰對應到實際跑的尺寸；日文頁／沒有素材＝null
 *     （nightread 規則版本 4，只有「更多」看它）。
 *
 * **資源**：每頁在飛約 58 B/px 的 Java heap（重繪內部尖峰約 43 B/px ＋ 包裝層的 px/gray/chroma/seg/char；帶去字遮罩的頁
 * 再多約 1.1 B/px），7 MPx 頁要
 * 20 s 以上，所以 [render] 一條龍版把超過 [MAX_PIXELS] 的頁先等比縮到預算內再跑（偵測／分割／重繪都在縮圖上），
 * **輸出＝縮後尺寸**（夜讀是離線預算不是即時：真機一頁 6–25 s）。
 * **併發**：可以多頁並行——這裡與 nightread 函式庫都沒有共享可變狀態（函式庫已驗可重入：多緒 render 與單緒逐像素相同）。
 * 記憶體由呼叫端控管（每頁約 150 MB，512 MB heap 約只放得下 2 頁）；推論進不進 NCNN 全域鎖、持鎖前向的優先權與中止由
 * 模型組決定（[NcnnFlavor]、[NcnnLowPriorityHook]）；呼叫端要在每次推論前後插自己的檢查點時用 lambda 版 [render]。
 *
 * **頁內並行**（各重繪入口的 `parallel`，2026-10-06）：nightread 分析裡彼此獨立的分支（人物收邊平滑、場景曲線、灰圈證據、
 * 貼紙計畫）與「更多」的背景物件量測丟給這個 Executor 跑（`NightRead.renderTiers` 的同名參數；輸出逐位元相同；量測裡只有部分頁
 * 用得到的幾張，2026-10-07 起在合成時於重繪執行緒上、只在用得到的地方算）。**預設 null＝
 * 依序**：開了每頁的 heap 尖峰多 11–27 MB（2.6 MPx 頁，最多約 10.4 B/px，跟排程有關；2026-10-07 審查另一套量法的上緣），
 * 58 B/px 的估算要加到**至少 70**；行程 CPU 多約一成，換單頁牆鐘約 −22%（桌面 4 核）——多頁並行時核已經有人用、heap 預算也
 * 只夠兩頁，通常不划算；呼叫端在「這頁是唯一在飛的頁、核有空、heap 夠」時才給。分支還沒被池子開始跑時重繪執行緒自己跑（池子
 * 小或滿都不會乾等），所以池子可以多頁共用、大小自訂；分支在池子的執行緒上跑，那些執行緒的優先權歸呼叫端管（例如讓路時給一個
 * 「直接在呼叫執行緒跑」的 Executor）。重繪執行緒等分支時被中斷照樣做完（與依序版一樣不理會中斷，旗標回傳前補回去）。
 */
object NightReadRenderer {

    /** 頁面像素上限；超過就等比縮到 ≤ 此值（3.5 MPx ≈ 1500×2300，一般單頁掃圖不會碰到）。 */
    const val MAX_PIXELS = 3_500_000

    /**
     * 由 `.param` 路徑建人物分割器（`.bin`＝同名換副檔名，與 [Detector] 慣例同）。
     * 兩個都 null → null（夜讀模型沒下／BYOM 沒放，呼叫端關掉夜讀）；只有一個 → 單顆；否則 yolo ∪ cseg 聯集。
     * [flavor]＝兩顆共用的建法與推論路徑（見 [NcnnFlavor]）。建到一半失敗會把已開的那顆關掉再拋（native handle 不漏）。
     */
    fun charSegmenter(
        yoloParam: String?,
        csegParam: String?,
        flavor: NcnnFlavor = NcnnFlavor.DEFAULT,
    ): CharSegmenter? {
        val parts = ArrayList<CharSegmenter>(2)
        try {
            yoloParam?.let { parts += YoloSegSegmenter(it, binOf(it), flavor) }
            csegParam?.let { parts += CsegSegmenter(it, binOf(it), flavor) }
        } catch (t: Throwable) {
            parts.forEach { runCatching { it.close() } }
            throw t
        }
        return when (parts.size) {
            0 -> null
            1 -> parts[0]
            else -> UnionCharSegmenter(parts)
        }
    }

    /**
     * 一條龍：（必要時縮圖）→ [Detector.detect] → [CharSegmenter.segment] → [NightRead.render] → 新 ARGB_8888 Bitmap。
     * [page] 所有權不變（不 recycle）；內部產生的 [Detection.textMask] 與縮圖用完自己 recycle。
     * 回傳的 Bitmap 尺寸＝實際跑的尺寸（縮過就是縮後尺寸，見 [NightReadStats.scaledTo]）。
     *
     * [extraLines]：呼叫端另外知道的文字行（**[page] 座標**；縮圖時一併等比縮），與偵測結果聯集後才分群成文字區。
     * 用途＝譯後頁：翻譯素材裡的原文行框比譯文大、也涵蓋 DBNet 對短譯文抓不到的泡（「咦」「是的」），併進來
     * 既補偵測漏、又把「泡面積：字框長邊²」的分母拉大（見 nightread docs/DECISIONS「譯後頁的泡」）。
     * 只影響文字區（bbox）；筆畫遮罩仍是偵測器對這頁的輸出。
     *
     * [inpaintMask]：譯後頁的去字遮罩（翻譯素材 `.yakuyomi/<頁>.mask.png`；白＝去字區，看藍通道 > 127），給 nightread「更多」
     * 的孤島規則（規則版本 4：去字區旁的小塊不塗，`NightReadInput.inpaintMask`）。**蓋住整張 [page]**（同 [extraLines] 的
     * [page] 座標）：尺寸可以與 [page] 差幾個像素（降採樣解碼的取整），長寬比要一致（[maskFitsPage]，不一致＝拋
     * [IllegalArgumentException]、推論前就拋）；引擎以最近鄰把它對應到實際跑的尺寸（縮圖時跟頁一起縮，與研究端
     * `cv2.resize(INTER_NEAREST)` 同式，見 [resampleMask]）。null＝日文頁／沒有素材，與規則版本 3 相同；全空的遮罩也一樣。
     * 「標準」不看它。所有權不變（不 recycle），只在重繪前讀一次（逐列讀，不配整張 int 緩衝）。
     */
    fun render(
        page: Bitmap,
        detector: Detector,
        charSeg: CharSegmenter,
        params: NightReadParams = NightReadParams(),
        stats: NightReadStats? = null,
        extraLines: List<TextLine> = emptyList(),
        inpaintMask: Bitmap? = null,
        parallel: Executor? = null,
    ): Bitmap = checkNotNull(
        render(
            page, detect = detector::detect, segment = charSeg::segment, params = params, stats = stats,
            extraLines = extraLines, inpaintMask = inpaintMask, parallel = parallel,
        ),
    )

    /**
     * lambda 版一條龍（上面那版委派到這裡）：推論由呼叫端的 [detect]／[segment] 做，好讓呼叫端在每次推論前插自己的
     * 檢查點（例如 fork 夜讀的暫停／讓路）；縮圖、併 [extraLines]、對應 [inpaintMask]、回收 textMask 與縮圖的邏輯都留在這裡。
     * [detect]／[segment] 拋出的例外（含 [NcnnForwardAbortedException]）照樣往外拋，textMask 與縮圖在 finally 回收。
     * [inpaintMask] 見上面那版。
     *
     * [beforeRender]：推論做完、進入 Kotlin 重繪（一頁最貴的一段，數秒）之前呼叫一次。回 false＝放棄這頁 → 回 null，
     * finally 照樣回收 textMask 與縮圖（呼叫端用來在暫停／讓路時丟回待做、把 heap 放掉）。
     * [parallel]：頁內並行（見類別說明；null＝依序）。
     */
    fun render(
        page: Bitmap,
        detect: (Bitmap) -> Detection,
        segment: (Bitmap) -> BooleanArray,
        params: NightReadParams = NightReadParams(),
        stats: NightReadStats? = null,
        extraLines: List<TextLine> = emptyList(),
        inpaintMask: Bitmap? = null,
        beforeRender: () -> Boolean = { true },
        parallel: Executor? = null,
    ): Bitmap? = withInference(page, detect, segment, stats, extraLines, inpaintMask, beforeRender) { work, detection, chars ->
        render(work, detection, chars, params, stats, inpaintMask, parallel)
    }

    /**
     * 多檔一次產生：[tiers] 依給定順序（預設 [NightTier] 全部三檔 L1 → L2 → L3；fork 產品兩檔傳 `[L2, L3]`＝「標準」與
     * 「更多」，L3 含「更多」新規則 A2），參數＝`tier.apply(base)`；base 只帶亮度等非檔位參數。
     *
     * 縮圖、偵測、分割、[beforeRender] 都只做一次（同 lambda 版 [render]），nightread 的分析也只做一次
     * （[NightRead.renderTiers]）；每檔合成完就轉成 ARGB_8888 交給 [sink]，**sink 回傳後立刻 recycle**——同一時間只有一張
     * 輸出 Bitmap、整頁共用一份 px 緩衝。去重一律對「上一個交出的檔」（[tiers] 裡的前一檔，不是 L 編號的前一檔）：
     * **第一檔一定非 null**；之後某檔與前一檔逐位元相同時 sink 收到 null——合成鍵（keep 集合，加上「更多」的繪製開關）相同
     * （輸出必定相同）時 nightread 根本不合成；合成了、但逐像素跟前一檔一樣（keep 不同而成品相同，或「更多」keep 與「標準」
     * 相同、繪製開關沒改到任何像素）時這裡比對後也交 null、不轉 Bitmap——省一次無損編碼與寫檔，閱讀器切到這一檔也知道
     * 「沒有差異」。所以 **sink 不得留住 Bitmap**：要寫檔就在 sink 裡寫完。
     * 每檔交出的成品與單檔 `render(tier.apply(base))` 逐位元相同，跟 [tiers] 裡有沒有別檔無關（NightReadTiersTest 守
     * `[L2, L3]` 與三檔版的 L2／L3 相同）。
     *
     * 回傳 false＝[beforeRender] 回 false（暫停／讓路），一檔都沒合成、sink 一次都沒叫；true＝[tiers] 每檔都交過 sink
     * （依序、各一次）。[tiers] 不得為空（推論前就拋 [IllegalArgumentException]）。
     * [detect]／[segment]／[sink] 拋出的例外照樣往外拋，textMask、縮圖、當下那張 Bitmap 都在 finally 回收。
     * 輸出尺寸＝實際跑的尺寸（縮過就是縮後尺寸，見 [NightReadStats.scaledTo]），各檔一定同尺寸。
     * [stats]：detect／mask 同 [render]；[NightReadStats.renderMs] 不含 sink；[NightReadStats.tierMs] 每檔；
     * [NightReadStats.stagesMs] 的分析段照舊、各檔的段名加檔位前綴（`l2.paintSticker=45`）。
     * [inpaintMask] 同一條龍 [render]（只有「更多」看它）。[parallel]：頁內並行（見類別說明；null＝依序）。
     *
     * 記憶體：桌面 JVM 量最低可跑 heap（SerialGC、固定 young），多檔版與單檔 [render] 相同（nightread 共用分析的快取存
     * 1 bit/px），所以每頁 58 B/px 的估算照用；帶 [inpaintMask] 的頁多 1 B/px（轉成的布林遮罩）＋函式庫內部位元版 1/8 B/px
     * （遮罩 Bitmap 本身的像素在 native）。
     */
    fun renderTiers(
        page: Bitmap,
        detect: (Bitmap) -> Detection,
        segment: (Bitmap) -> BooleanArray,
        base: NightReadParams = NightReadParams(),
        stats: NightReadStats? = null,
        extraLines: List<TextLine> = emptyList(),
        inpaintMask: Bitmap? = null,
        beforeRender: () -> Boolean = { true },
        tiers: List<NightTier> = NightTier.entries,
        parallel: Executor? = null,
        sink: (NightTier, Bitmap?) -> Unit,
    ): Boolean {
        require(tiers.isNotEmpty()) { "renderTiers：至少要一檔" }
        return withInference(page, detect, segment, stats, extraLines, inpaintMask, beforeRender) { work, detection, chars ->
            renderTiersOn(work, detection, chars, base, tiers, stats, inpaintMask, parallel, sink)
        } != null
    }

    /** 多檔版的模型物件版（同一條龍 [render] 的關係）：沒有 beforeRender，所以 [tiers] 每檔一定都交過 [sink]。 */
    fun renderTiers(
        page: Bitmap,
        detector: Detector,
        charSeg: CharSegmenter,
        base: NightReadParams = NightReadParams(),
        stats: NightReadStats? = null,
        extraLines: List<TextLine> = emptyList(),
        inpaintMask: Bitmap? = null,
        tiers: List<NightTier> = NightTier.entries,
        parallel: Executor? = null,
        sink: (NightTier, Bitmap?) -> Unit,
    ) {
        renderTiers(page, detect = detector::detect, segment = charSeg::segment, base = base, stats = stats,
            extraLines = extraLines, inpaintMask = inpaintMask, tiers = tiers, parallel = parallel, sink = sink)
    }

    /**
     * 縮圖 → 偵測（併 [extraLines]）→ 分割 → [beforeRender] → [body]（在縮後的工作圖上）。[beforeRender] 回 false＝回 null。
     * textMask 與縮圖在 finally 回收（例外也一樣），[page] 所有權不變。lambda 版 [render] 與 [renderTiers] 共用。
     * [inpaintMask] 只在這裡驗與 [page] 對得上（推論前，免得白跑一頁才拋）；對應到工作圖尺寸在 [body] 裡（[toInput]）。
     */
    private fun <T : Any> withInference(
        page: Bitmap,
        detect: (Bitmap) -> Detection,
        segment: (Bitmap) -> BooleanArray,
        stats: NightReadStats?,
        extraLines: List<TextLine>,
        inpaintMask: Bitmap?,
        beforeRender: () -> Boolean,
        body: (Bitmap, Detection, BooleanArray) -> T,
    ): T? {
        inpaintMask?.let { requireFitsPage(it, page) }
        val work = scaleToBudget(page, stats)
        try {
            var t = System.nanoTime()
            val detected = detect(work)
            stats?.detectMs = (System.nanoTime() - t) / 1_000_000
            val detection = if (extraLines.isEmpty()) {
                detected
            } else {
                Detection(detected.lines + scaleLines(extraLines, work.width.toFloat() / page.width), detected.textMask)
            }
            try {
                t = System.nanoTime()
                val chars = segment(work)
                stats?.maskMs = (System.nanoTime() - t) / 1_000_000
                if (!beforeRender()) return null
                return body(work, detection, chars)
            } finally {
                detected.textMask.recycle()
            }
        } finally {
            if (work !== page) work.recycle()
        }
    }

    /**
     * 分段版：呼叫端已有**同尺寸**的 [detection]（[Detector.detect] 對 [page] 的結果）與 [charMask]（w×h、true＝人物）。
     * 不縮圖（素材尺寸已綁死在 page 上，要縮得在偵測前縮）、不 recycle [Detection.textMask]；只填 [NightReadStats.renderMs]。
     * [inpaintMask] 同一條龍 [render]（蓋住整張 [page]、最近鄰對應到 [page] 尺寸；對不上拋 [IllegalArgumentException]）。
     */
    fun render(
        page: Bitmap,
        detection: Detection,
        charMask: BooleanArray,
        params: NightReadParams = NightReadParams(),
        stats: NightReadStats? = null,
        inpaintMask: Bitmap? = null,
        parallel: Executor? = null,
    ): Bitmap {
        val t = System.nanoTime()
        inpaintMask?.let { requireFitsPage(it, page) }
        val px = IntArray(page.width * page.height)
        val input = toInput(page, detection, charMask, px, inpaintMask)

        // 分段計時：借 nightread 的除錯回呼記「上一段到這一段」的毫秒數；用只要段名的 NightReadStageTimer（函式庫不另掃整頁算
        // 遮罩計數，值只用段名），輸出不變。沒傳 stats 就不掛回呼。頁內並行時段的時間是重繪執行緒自己的牆鐘（並行分支算在等它的那一段）。
        val marks = if (stats != null) StringBuilder() else null
        var last = System.nanoTime()
        val debug: NightReadDebug? = marks?.let { sb ->
            object : NightReadStageTimer {
                override fun invoke(stage: String, value: Int) {
                    val now = System.nanoTime()
                    val d = (now - last) / 1_000_000
                    last = now
                    if (d >= NightReadStats.STAGE_MIN_MS) {
                        if (sb.isNotEmpty()) sb.append(' ')
                        sb.append(stage).append('=').append(d)
                    }
                }
            }
        }
        val res = NightRead.render(input, params, debug, parallel)
        stats?.stagesMs = marks?.toString()

        // 輸出：Gray 0..255 → 不透明灰 ARGB；重用 px 當輸出緩衝（省一份 w×h int）
        val bmp = toBitmap(res.out, px)
        stats?.renderMs = (System.nanoTime() - t) / 1_000_000
        return bmp
    }

    /**
     * 多檔版的重繪段（工作圖、偵測、人物遮罩已就緒）：一份 [NightReadInput]、一份 px 緩衝給各檔共用；去重在
     * [streamTiers]，這裡只轉 Bitmap、交 [sink]（交完就 recycle）與計時。計時見 [renderTiers]。
     */
    private fun renderTiersOn(
        page: Bitmap,
        detection: Detection,
        charMask: BooleanArray,
        base: NightReadParams,
        tiers: List<NightTier>,
        stats: NightReadStats?,
        inpaintMask: Bitmap?,
        parallel: Executor?,
        sink: (NightTier, Bitmap?) -> Unit,
    ) {
        val t = System.nanoTime()
        val px = IntArray(page.width * page.height)
        val input = toInput(page, detection, charMask, px, inpaintMask)

        // 計時（只在傳了 stats 時掛回呼）：分析段同單檔；nightread 每檔合成前送 ("tier", k)，之後的段名加檔位前綴，
        // 那一檔的 tierMs 從這裡量到 Bitmap 建好。sink 的時間從 renderMs 與分段裡扣掉。
        val marks = if (stats != null) StringBuilder() else null
        val tierMs = arrayOfNulls<Long>(tiers.size)
        var last = System.nanoTime()
        var tierStart = 0L
        var prefix = ""
        var sinkNs = 0L
        fun mark(name: String, now: Long) {
            val d = (now - last) / 1_000_000
            last = now
            if (marks != null && d >= NightReadStats.STAGE_MIN_MS) {
                if (marks.isNotEmpty()) marks.append(' ')
                marks.append(name).append('=').append(d)
            }
        }
        // 只要段名（NightReadStageTimer：函式庫不算遮罩計數；"tier" 照送檔位索引）
        val debug: NightReadDebug? = if (stats == null) null else object : NightReadStageTimer {
            override fun invoke(stage: String, value: Int) {
                val now = System.nanoTime()
                if (stage == "tier") {
                    prefix = tiers[value].key + "."
                    mark(prefix + "keep", now)          // 上一檔交出後到這一檔開始：篩 keep（第一個用到 plain 的檔含 plain 判定）
                    tierStart = now
                } else {
                    mark(prefix + stage, now)
                }
            }
        }

        streamTiers(input, tiers, base, px, debug, parallel) { k, emitted, composed ->
            if (!emitted) {
                val s0 = System.nanoTime()
                if (composed) tierMs[k] = (s0 - tierStart) / 1_000_000
                sink(tiers[k], null)
                val s1 = System.nanoTime()
                sinkNs += s1 - s0
                last = s1
            } else {
                // px 已由 streamTiers 寫好這一檔的 ARGB；createBitmap 會複製
                val bmp = Bitmap.createBitmap(px, page.width, page.height, Bitmap.Config.ARGB_8888)
                val s0 = System.nanoTime()
                tierMs[k] = (s0 - tierStart) / 1_000_000
                try {
                    sink(tiers[k], bmp)
                } finally {
                    bmp.recycle()
                }
                val s1 = System.nanoTime()
                sinkNs += s1 - s0               // 轉 Bitmap 算進合成；sink（編碼、寫檔）不算
                last = s1
            }
        }
        stats?.renderMs = (System.nanoTime() - t - sinkNs) / 1_000_000
        stats?.tierMs = tiers.indices.joinToString(" ") { "${tiers[it].key}=${tierMs[it] ?: "-"}" }
        stats?.stagesMs = marks?.toString()
    }

    /**
     * 多檔去重的核心（不碰 Bitmap，JVM 單元測試 NightReadTiersTest 直接打）：[tiers] 依給定順序套 `tier.apply(base)`
     * 交給 [NightRead.renderTiers]，每檔依序回呼 [out] 一次 `(k, emitted, composed)`，k＝[tiers] 裡的索引：
     *  - emitted＝true：與上一個交出的檔不同，[px] 已寫好這一檔的不透明灰 ARGB（呼叫端轉 Bitmap）。第 0 檔一定是這種。
     *  - emitted＝false：與上一個交出的檔逐像素相同，不交圖。composed＝false 是合成鍵相同（nightread 沒合成）；
     *    true 是合成了、但這裡比對後逐像素相同（輸出去重）。
     * [px]＝w×h 緩衝（呼叫端 [toInput] 用過的那份即可，第 0 檔前的內容不讀）；回呼之間它一直裝著最後交出那檔的 ARGB，
     * 比對靠它、不另配記憶體——所以呼叫端在 [out] 裡只能讀 [px]、不得改寫。
     */
    internal fun streamTiers(
        input: NightReadInput,
        tiers: List<NightTier>,
        base: NightReadParams,
        px: IntArray,
        debug: NightReadDebug?,
        parallel: Executor? = null,
        out: (k: Int, emitted: Boolean, composed: Boolean) -> Unit,
    ) {
        NightRead.renderTiers(input, tiers.map { it.apply(base) }, debug, parallel) { k, gray ->
            when {
                gray == null -> out(k, false, false)
                // 輸出去重：keep 不同、成品卻逐像素跟上一個交出的檔相同時也不交。第 0 檔一定非 null、一定寫進 px；
                // 被跳過的檔本來就等於再上一檔，所以 px 永遠是「上一個交出的檔」。比對不配記憶體、每檔幾 ms。
                k > 0 && sameAsLastEmitted(gray, px) -> out(k, false, true)
                else -> {
                    writeArgb(gray, px)
                    out(k, true, true)
                }
            }
        }
    }

    /**
     * Bitmap → [NightReadInput]（契約見類別說明）。[px] 是呼叫端的 w×h 緩衝：這裡 getPixels 進去算灰階與彩度，
     * 之後呼叫端拿它當輸出緩衝（[toBitmap]），整頁只配一份 w×h int。
     * [inpaintMask]（蓋住整頁、呼叫端已驗過 [requireFitsPage]）以最近鄰對應到 [page] 尺寸（[page] 是縮後的工作圖時就是跟著
     * 縮）→ `NightReadInput.inpaintMask`。
     */
    private fun toInput(
        page: Bitmap,
        detection: Detection,
        charMask: BooleanArray,
        px: IntArray,
        inpaintMask: Bitmap?,
    ): NightReadInput {
        val w = page.width
        val h = page.height
        require(charMask.size == w * h) { "charMask 尺寸 ${charMask.size} ≠ 頁面 ${w}×$h" }

        // 只 getPixels 一次，同一趟迴圈算灰階與彩度（省一次 w×h 掃描）
        page.getPixels(px, 0, w, 0, 0, w, h)
        val gray = Gray(w, h)
        val chroma = Gray(w, h)
        for (i in px.indices) {
            val p = px[i]
            val r = (p shr 16) and 0xFF
            val g = (p shr 8) and 0xFF
            val b = p and 0xFF
            // BT.601，與 cv2.imread(IMREAD_GRAYSCALE) 同式（契約第 1 條）
            gray.data[i] = (r * 299 + g * 587 + b * 114 + 500) / 1000
            chroma.data[i] = maxOf(r, g, b) - minOf(r, g, b)
        }

        // 文字區＝分群後的 AABB（契約第 5 條）；不濾空文字區——SFX／裝飾字也要進去（契約第 3 條）
        val regions = Grouping.group(detection.lines).map {
            NrRegion(
                it.x0.toInt().coerceIn(0, w), it.y0.toInt().coerceIn(0, h),
                it.x1.toInt().coerceIn(0, w), it.y1.toInt().coerceIn(0, h),
            )
        }
        val seg = maskFromBitmap(detection.textMask, w, h)
        val chars = Mask(w, h, charMask)
        val inpaint = inpaintMask?.let { maskFromBitmap(it, w, h) }
        return NightReadInput(gray, seg, regions, chars, chroma, inpaint)
    }

    /** [out] 與 [px]（[toBitmap] 上一次寫進去的灰 ARGB）是否逐像素相同；比的是 [toBitmap] 會寫的值（夾進 0..255）。 */
    private fun sameAsLastEmitted(out: Gray, px: IntArray): Boolean {
        val d = out.data
        for (i in px.indices) if ((px[i] and 0xFF) != d[i].coerceIn(0, 255)) return false
        return true
    }

    /** 輸出：Gray 0..255 → 不透明灰 ARGB 寫進 [px]（重用輸入時的緩衝）→ 新 ARGB_8888 Bitmap（createBitmap 會複製）。 */
    private fun toBitmap(out: Gray, px: IntArray): Bitmap {
        writeArgb(out, px)
        return Bitmap.createBitmap(px, out.w, out.h, Bitmap.Config.ARGB_8888)
    }

    /** Gray 0..255（夾進範圍）→ 不透明灰 ARGB 寫進 [px]。 */
    private fun writeArgb(out: Gray, px: IntArray) {
        val d = out.data
        for (i in px.indices) {
            val v = d[i].coerceIn(0, 255)
            px[i] = (0xFF shl 24) or (v shl 16) or (v shl 8) or v
        }
    }

    private fun binOf(param: String): String = param.removeSuffix(".param") + ".bin"

    /**
     * 像素數 > [MAX_PIXELS] 才縮：等比、雙線性（filter=true），邊長取 floor 保證縮後 ≤ 上限。
     * 縮的是「跑管線用的工作圖」，呼叫端的 [page] 不動；縮後尺寸記進 [stats]。
     * 沒縮時回傳 page 本身——呼叫端以 `!==` 判斷要不要 recycle（createScaledBitmap 同尺寸時也會回同一物件，別誤 recycle 原圖）。
     */
    private fun scaleToBudget(page: Bitmap, stats: NightReadStats?): Bitmap {
        val (nw, nh) = budgetSize(page.width, page.height) ?: return page
        stats?.scaledTo = nw to nh
        return Bitmap.createScaledBitmap(page, nw, nh, true)
    }

    /** [scaleToBudget] 的尺寸算術（JVM 可測）：像素數 ≤ [MAX_PIXELS]＝null（不縮）；否則等比、邊長 floor、至少 1。 */
    internal fun budgetSize(w: Int, h: Int): Pair<Int, Int>? {
        val n = w.toLong() * h
        if (n <= MAX_PIXELS) return null
        val s = sqrt(MAX_PIXELS.toDouble() / n)
        return max(1, floor(w * s).toInt()) to max(1, floor(h * s).toInt())
    }

    /** [extraLines]（頁座標）→ 工作圖座標：兩軸都乘 [s]（＝工作圖寬 ÷ 頁寬）；[s] ＝ 1 原樣回傳。 */
    internal fun scaleLines(lines: List<TextLine>, s: Float): List<TextLine> =
        if (s == 1f) lines else lines.map { l -> TextLine(l.quad.map { Pt(it.x * s, it.y * s) }, l.score) }

    /**
     * 遮罩 [mw]×[mh] 蓋不蓋得住頁 [pw]×[ph]：尺寸可以不同（降採樣解碼的取整、或給原尺寸的遮罩），長寬比要一致——交叉乘積差
     * ≤ 2 ×（四個邊長和），約等於頁邊錯位 ≤ 2 px。超過＝不是這頁的遮罩（拿錯頁、轉了方向、雙頁跨頁）。
     */
    internal fun maskFitsPage(mw: Int, mh: Int, pw: Int, ph: Int): Boolean =
        mw > 0 && mh > 0 && pw > 0 && ph > 0 &&
            abs(mw.toLong() * ph - mh.toLong() * pw) <= 2L * (mw.toLong() + mh + pw + ph)

    private fun requireFitsPage(mask: Bitmap, page: Bitmap) {
        require(maskFitsPage(mask.width, mask.height, page.width, page.height)) {
            "inpaintMask ${mask.width}×${mask.height} 與頁 ${page.width}×${page.height} 的長寬比對不上（不是這頁的去字遮罩？）"
        }
    }

    /**
     * 引擎二值遮罩 Bitmap（0xFFFFFFFF／0xFF000000，看藍通道 > 127）→ nightread [Mask]（w×h）：[resampleMask] 逐列讀，
     * 不配整張 int 緩衝。[Detector.detect] 的 textMask 是原圖尺寸（逐像素照搬）；去字遮罩縮圖時跟頁一起縮（最近鄰）。
     */
    private fun maskFromBitmap(bmp: Bitmap, w: Int, h: Int): Mask {
        val bw = bmp.width
        return resampleMask(bw, bmp.height, w, h) { y, row -> bmp.getPixels(row, 0, bw, 0, y, bw, 1) }
    }

    /**
     * 二值遮罩（[srcW]×[srcH] 的 ARGB，藍通道 > 127＝true）以最近鄰對應到 [w]×[h]，與 `cv2.resize(INTER_NEAREST)` 同式
     * （[nearestIndex]；研究端把翻譯素材的去字遮罩縮到頁面尺寸就是這樣）。同尺寸＝逐像素照搬。[readRow] 把來源第 y 列
     * （[srcW] 個 ARGB）填進給的緩衝；同一來源列只讀一次。
     */
    internal fun resampleMask(srcW: Int, srcH: Int, w: Int, h: Int, readRow: (y: Int, row: IntArray) -> Unit): Mask {
        val m = Mask(w, h)
        val xs = nearestIndex(srcW, w)
        val ys = nearestIndex(srcH, h)
        val row = IntArray(srcW)
        var loaded = -1
        for (y in 0 until h) {
            val sy = ys[y]
            if (sy != loaded) {
                readRow(sy, row)
                loaded = sy
            }
            val o = y * w
            for (x in 0 until w) m.data[o + x] = (row[xs[x]] and 0xFF) > 127
        }
        return m
    }

    /**
     * 最近鄰的來源索引，照抄 OpenCV `resize` 的 INTER_NEAREST：`ifx = 1 / (dst / src)`，`src_i = min(floor(i × ifx), src − 1)`
     * （不是 INTER_NEAREST_EXACT 的半像素中心）。同尺寸＝恆等。
     */
    internal fun nearestIndex(src: Int, dst: Int): IntArray {
        val inv = 1.0 / (dst.toDouble() / src)
        return IntArray(dst) { min(floor(it * inv).toInt(), src - 1) }
    }
}
