package li.joye.yakuyomi.engine

import android.graphics.Bitmap
import li.joye.yakuyomi.nightread.Gray
import li.joye.yakuyomi.nightread.Mask
import li.joye.yakuyomi.nightread.NightRead
import li.joye.yakuyomi.nightread.NightReadInput
import li.joye.yakuyomi.nightread.NightReadParams
import li.joye.yakuyomi.nightread.TextRegion as NrRegion
import kotlin.math.floor
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

/**
 * 多顆人物分割器的聯集（yolo ∪ cseg 定案配方：只用 yolo 守護框違規 18→27，加上 cseg 更準；見 nightread README）。
 * segment 逐顆跑、結果就地 OR 進第一顆的陣列（省一份 w×h）；close 關全部。
 * 各顆推論都在 NCNN 全域鎖下，逐顆串行本來就是實際行為，不另開緒。
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

/** [NightReadRenderer.render] 的耗時與縮圖紀錄（毫秒）。一條龍版三段都填；分段版只填 [renderMs]。 */
class NightReadStats {
    /** detector.detect 本身（不含分群、遮罩轉換——那些算進 [renderMs]）。 */
    var detectMs = 0L
    /** charSeg.segment（聯集配方＝各顆相加）。 */
    var maskMs = 0L
    /** 灰階／彩度、分群 AABB、遮罩轉換、NightRead.render、輸出 Bitmap。 */
    var renderMs = 0L
    /** 頁超過 [NightReadRenderer.MAX_PIXELS] 時縮到的尺寸 (w, h)；沒縮＝null。縮圖本身的時間不計入三段。 */
    var scaledTo: Pair<Int, Int>? = null
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
 *
 * **資源**：每像素約 40 B 工作記憶體（gray/chroma/seg/char 各 4–8 B + 管線中間遮罩），7 MPx 頁要 20 s 以上，
 * 所以 [render] 一條龍版把超過 [MAX_PIXELS] 的頁先等比縮到預算內再跑（偵測／分割／重繪都在縮圖上），
 * **輸出＝縮後尺寸**（夜讀是離線預算不是即時：真機一頁 6–25 s）。
 * **併發**：一次只跑一頁——由呼叫端保證，這裡不加鎖（NCNN 推論本就在全域鎖下串行；重繪的中間陣列兩頁同時撐會 OOM）。
 */
object NightReadRenderer {

    /** 頁面像素上限；超過就等比縮到 ≤ 此值（3.5 MPx ≈ 1500×2300，一般單頁掃圖不會碰到）。 */
    const val MAX_PIXELS = 3_500_000

    /**
     * 由 `.param` 路徑建人物分割器（`.bin`＝同名換副檔名，與 [Detector] 慣例同）。
     * 兩個都 null → null（夜讀模型沒下／BYOM 沒放，呼叫端關掉夜讀）；只有一個 → 單顆；否則 yolo ∪ cseg 聯集。
     * 建到一半失敗會把已開的那顆關掉再拋（native handle 不漏）。
     */
    fun charSegmenter(yoloParam: String?, csegParam: String?): CharSegmenter? {
        val parts = ArrayList<CharSegmenter>(2)
        try {
            yoloParam?.let { parts += YoloSegSegmenter(it, binOf(it)) }
            csegParam?.let { parts += CsegSegmenter(it, binOf(it)) }
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
     */
    fun render(
        page: Bitmap,
        detector: Detector,
        charSeg: CharSegmenter,
        params: NightReadParams = NightReadParams(),
        stats: NightReadStats? = null,
    ): Bitmap {
        val work = scaleToBudget(page, stats)
        try {
            var t = System.nanoTime()
            val detection = detector.detect(work)
            stats?.detectMs = (System.nanoTime() - t) / 1_000_000
            try {
                t = System.nanoTime()
                val chars = charSeg.segment(work)
                stats?.maskMs = (System.nanoTime() - t) / 1_000_000
                return render(work, detection, chars, params, stats)
            } finally {
                detection.textMask.recycle()
            }
        } finally {
            if (work !== page) work.recycle()
        }
    }

    /**
     * 分段版：呼叫端已有**同尺寸**的 [detection]（[Detector.detect] 對 [page] 的結果）與 [charMask]（w×h、true＝人物）。
     * 不縮圖（素材尺寸已綁死在 page 上，要縮得在偵測前縮）、不 recycle [Detection.textMask]；只填 [NightReadStats.renderMs]。
     */
    fun render(
        page: Bitmap,
        detection: Detection,
        charMask: BooleanArray,
        params: NightReadParams = NightReadParams(),
        stats: NightReadStats? = null,
    ): Bitmap {
        val w = page.width
        val h = page.height
        require(charMask.size == w * h) { "charMask 尺寸 ${charMask.size} ≠ 頁面 ${w}×$h" }
        val t = System.nanoTime()

        // 只 getPixels 一次，同一趟迴圈算灰階與彩度（省一次 w×h 掃描）
        val px = IntArray(w * h)
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

        val res = NightRead.render(NightReadInput(gray, seg, regions, chars, chroma), params)

        // 輸出：Gray 0..255 → 不透明灰 ARGB；重用 px 當輸出緩衝（省一份 w×h int）
        val out = res.out.data
        for (i in px.indices) {
            val v = out[i].coerceIn(0, 255)
            px[i] = (0xFF shl 24) or (v shl 16) or (v shl 8) or v
        }
        val bmp = Bitmap.createBitmap(px, w, h, Bitmap.Config.ARGB_8888)
        stats?.renderMs = (System.nanoTime() - t) / 1_000_000
        return bmp
    }

    private fun binOf(param: String): String = param.removeSuffix(".param") + ".bin"

    /**
     * 像素數 > [MAX_PIXELS] 才縮：等比、雙線性（filter=true），邊長取 floor 保證縮後 ≤ 上限。
     * 縮的是「跑管線用的工作圖」，呼叫端的 [page] 不動；縮後尺寸記進 [stats]。
     * 沒縮時回傳 page 本身——呼叫端以 `!==` 判斷要不要 recycle（createScaledBitmap 同尺寸時也會回同一物件，別誤 recycle 原圖）。
     */
    private fun scaleToBudget(page: Bitmap, stats: NightReadStats?): Bitmap {
        val n = page.width.toLong() * page.height
        if (n <= MAX_PIXELS) return page
        val s = sqrt(MAX_PIXELS.toDouble() / n)
        val nw = max(1, floor(page.width * s).toInt())
        val nh = max(1, floor(page.height * s).toInt())
        stats?.scaledTo = nw to nh
        return Bitmap.createScaledBitmap(page, nw, nh, true)
    }

    /**
     * 引擎二值遮罩 Bitmap（0xFFFFFFFF／0xFF000000）→ nightread [Mask]。[Detector.detect] 回的是原圖尺寸，
     * 但保留最近鄰放大這條後路（筆畫遮罩若哪天改回半解析度也不會默默錯位）。
     */
    private fun maskFromBitmap(bmp: Bitmap, w: Int, h: Int): Mask {
        val m = Mask(w, h)
        val bw = bmp.width
        val bh = bmp.height
        val px = IntArray(bw * bh)
        bmp.getPixels(px, 0, bw, 0, 0, bw, bh)
        if (bw == w && bh == h) {
            for (i in px.indices) m.data[i] = (px[i] and 0xFF) > 127
        } else {
            val sx = bw.toDouble() / w
            val sy = bh.toDouble() / h
            for (y in 0 until h) {
                val my = min(bh - 1, (y * sy).toInt())
                for (x in 0 until w) {
                    val mx = min(bw - 1, (x * sx).toInt())
                    m.data[y * w + x] = (px[my * bw + mx] and 0xFF) > 127
                }
            }
        }
        return m
    }
}
