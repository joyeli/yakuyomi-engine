package li.joye.yakuyomi.engine.nightread

/**
 * 夜讀移植用的純 Kotlin 影像原語——**語意逐一對齊 OpenCV**（`parity/nightread.py` 的 cv2 呼叫），
 * 不依賴 android.graphics ⇒ JVM 單元測試可直接跑、拿 Python 產的 fixture 逐位元比對。
 *
 * 設計約束：
 * - 影像一律 row-major、索引 `y * w + x`。
 * - [Mask]＝二值（true＝前景／非零）；[Gray]＝0..255 存 IntArray（避開 Byte 符號坑）；[FImg]＝float32。
 * - 邊界語意照 cv2 預設：`dilate` 邊界外視為 0（前景不外溢）、`erode` 邊界外視為前景
 *   （BORDER_CONSTANT + morphologyDefaultBorderValue ⇒ 影像邊緣不會被「外面的 0」吃掉）；
 *   高斯/均值模糊用 BORDER_REFLECT_101。
 * - 大核膨脹/腐蝕**不可**逐像素掃整個核（r=35 時 2.8M px × 5k ≈ 10¹⁰）：用「核逐列分解 ＋
 *   每列一維 run 膨脹（van Herk/Gil-Werman O(N)）＋ 列間位移 OR」＝ O(N·k)，且與 cv2 的
 *   任意核形狀**位元一致**（不是用距離場近似圓）。
 * - 結構元素形狀必須重現 `cv2.getStructuringElement(MORPH_ELLIPSE, (s, s))` 的**光柵化演算法**
 *   （cv2 不是理想圓：以 r=s/2 為半徑、逐列算 `dx = round(r * sqrt(1 - (dy/r)^2))` 的橫向跨距），
 *   否則 dilate 結果會跟 Python 差幾個像素、連鎖影響所有門檻量測。
 */
class Mask(val w: Int, val h: Int, val data: BooleanArray = BooleanArray(w * h)) {
    operator fun get(x: Int, y: Int): Boolean = data[y * w + x]
    operator fun set(x: Int, y: Int, v: Boolean) { data[y * w + x] = v }
    fun copy(): Mask = Mask(w, h, data.copyOf())
    fun count(): Int = data.count { it }
    fun any(): Boolean = data.any { it }
    infix fun and(o: Mask): Mask = Mask(w, h, BooleanArray(w * h) { data[it] && o.data[it] })
    infix fun or(o: Mask): Mask = Mask(w, h, BooleanArray(w * h) { data[it] || o.data[it] })
    fun not(): Mask = Mask(w, h, BooleanArray(w * h) { !data[it] })
    fun andNot(o: Mask): Mask = Mask(w, h, BooleanArray(w * h) { data[it] && !o.data[it] })
}

class Gray(val w: Int, val h: Int, val data: IntArray = IntArray(w * h)) {
    operator fun get(x: Int, y: Int): Int = data[y * w + x]
    operator fun set(x: Int, y: Int, v: Int) { data[y * w + x] = v }
    fun copy(): Gray = Gray(w, h, data.copyOf())
    /** `g >= th` */
    fun ge(th: Int): Mask = Mask(w, h, BooleanArray(w * h) { data[it] >= th })
    /** `g < th` */
    fun lt(th: Int): Mask = Mask(w, h, BooleanArray(w * h) { data[it] < th })
    fun toF(): FImg = FImg(w, h, FloatArray(w * h) { data[it].toFloat() })
}

class FImg(val w: Int, val h: Int, val data: FloatArray = FloatArray(w * h)) {
    operator fun get(x: Int, y: Int): Float = data[y * w + x]
    operator fun set(x: Int, y: Int, v: Float) { data[y * w + x] = v }
    fun copy(): FImg = FImg(w, h, data.copyOf())
}

/** 結構元素（`cv2.getStructuringElement`）：奇數尺寸、錨點置中。 */
class Kernel(val w: Int, val h: Int, val data: BooleanArray) {
    val ax: Int get() = w / 2
    val ay: Int get() = h / 2
}

/**
 * `cv2.connectedComponentsWithStats` 的結果：`labels` 與影像同尺寸（0＝背景）、其餘陣列長度 `n`
 *（含背景 0 號），欄位對應 CC_STAT_LEFT/TOP/WIDTH/HEIGHT/AREA。
 */
class CC(
    val n: Int,
    val labels: IntArray,
    val left: IntArray,
    val top: IntArray,
    val width: IntArray,
    val height: IntArray,
    val area: IntArray,
)

object Cv {
    // ── 結構元素 ─────────────────────────────────────────────────────
    /** `cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))`，size 奇數。形狀須與 cv2 光柵化位元一致。 */
    fun ellipse(size: Int): Kernel = TODO()

    /** `cv2.getStructuringElement(cv2.MORPH_RECT, (kw, kh))`（等同 `np.ones((kh, kw))`）。 */
    fun rect(kw: Int, kh: Int): Kernel = TODO()

    // ── 二值形態學（cv2.dilate / erode / morphologyEx）──────────────────
    fun dilate(m: Mask, k: Kernel, iterations: Int = 1): Mask = TODO()
    fun erode(m: Mask, k: Kernel, iterations: Int = 1): Mask = TODO()
    fun open(m: Mask, k: Kernel): Mask = dilate(erode(m, k), k)
    fun close(m: Mask, k: Kernel): Mask = erode(dilate(m, k), k)

    // ── 灰階形態學（只有 blackhat 會用：ink_line_mask 的 7×7 ellipse）──────
    fun dilateGray(g: Gray, k: Kernel): Gray = TODO()
    fun erodeGray(g: Gray, k: Kernel): Gray = TODO()
    /** `cv2.morphologyEx(g, MORPH_BLACKHAT, k)` ＝ close(g) − g（灰階，結果 ≥ 0）。 */
    fun blackhat(g: Gray, k: Kernel): Gray = TODO()

    // ── 連通元件 ─────────────────────────────────────────────────────
    /** `cv2.connectedComponentsWithStats(m, connectivity)`；標號順序須與 cv2 一致（掃描序、首見即編號）。 */
    fun ccStats(m: Mask, connectivity: Int = 8): CC = TODO()

    // ── 距離變換 ─────────────────────────────────────────────────────
    /**
     * `cv2.distanceTransform(m, DIST_L2, maskSize)`：每個**前景**像素到最近**背景**像素的距離、背景＝0。
     * 精確歐氏距離（Felzenszwalb 兩趟）即可——cv2 的 3×3/5×5 chamfer 是近似，差距 ≤ 數 %，
     * nightread 的所有距離門檻都有餘裕；parity 測試用容差比對。
     */
    fun distanceL2(m: Mask): FImg = TODO()

    // ── 濾波 ─────────────────────────────────────────────────────────
    /** `cv2.GaussianBlur(f, (0,0), sigma)`：核大小由 sigma 推（cv2 規則 `ksize = 2*ceil(3*sigma)+1`… 以 cv2 實際為準）、BORDER_REFLECT_101。 */
    fun gaussianBlur(f: FImg, sigma: Double): FImg = TODO()

    /** `cv2.blur(f, (k, k))`：均值濾波、BORDER_REFLECT_101。 */
    fun boxBlur(f: FImg, k: Int): FImg = TODO()

    // ── 縮放 ─────────────────────────────────────────────────────────
    /** `cv2.resize(f, (nw, nh), INTER_AREA)`（縮小用；整數倍時＝格平均）。 */
    fun resizeArea(f: FImg, nw: Int, nh: Int): FImg = TODO()

    /** `cv2.resize(m, (nw, nh), INTER_NEAREST)`（放大回原尺寸用）。 */
    fun resizeNearest(m: Mask, nw: Int, nh: Int): Mask = TODO()

    // ── 洞 / 輪廓 ────────────────────────────────────────────────────
    /**
     * 「1px 零邊框 + 從 (0,0) floodFill」的洞判定（`_hole_ink_ratio` / `sticker_metrics` 用）：
     * 回傳 **洞遮罩**＝非前景、且從影像外側（經 4-連通背景）到不了的像素。
     */
    fun holes(m: Mask): Mask = TODO()

    /**
     * `cv2.findContours(RETR_LIST, CHAIN_APPROX_NONE)` 後 `sum(arcLength(c, closed=True))`：
     * 前景所有輪廓（含內外）的**總周長**。arcLength 是折線長（對角步 √2、直步 1），
     * 用 Moore 鄰域追蹤重現；`sticker_metrics` 的 rough ＝ perim² / area。
     */
    fun totalContourLength(m: Mask): Double = TODO()

    /** `cv2.contourArea(quad)`（多邊形面積，shoelace 絕對值）。 */
    fun polygonArea(xs: FloatArray, ys: FloatArray): Double = TODO()
}
