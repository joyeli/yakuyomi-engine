package li.joye.yakuyomi.engine

import android.graphics.Bitmap
import android.util.Log
import kotlin.math.exp
import kotlin.math.floor
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt

/**
 * 人物語意分割（夜讀用）：把一頁的人物像素標出來，夜讀重繪據此「絕不塗錯」臉／手／白衣／白髮。
 *
 * 模型走 NCNN（與偵測、去字同一個後端、同一把鎖）。回傳與頁面同尺寸（row-major w×h）的布林遮罩，true＝人物。
 * 這是**模型原輸出**的聯集，貼墨收邊與平滑由夜讀管線負責。
 * 定案配方＝yolo ∪ cseg（[UnionCharSegmenter]，由 [NightReadRenderer.charSegmenter] 組）。
 */
interface CharSegmenter : AutoCloseable {
    fun segment(page: Bitmap): BooleanArray

    /**
     * 暖機：對 64×64 空白圖 segment 一次，讓 NCNN net 的首次 lazy 初始化在單緒做完（對照 [Detector.warmUp]）。
     * 之後才允許從並發的 coroutine 呼叫——多頁同時打進未初始化的 net 會原生 crash。失敗只記 log（暖機不是正式推論）。
     */
    fun warmUp() {
        val blank = Bitmap.createBitmap(64, 64, Bitmap.Config.ARGB_8888)
        try {
            segment(blank)
        } catch (t: Throwable) {
            Log.w("CharSegmenter", "人物分割暖機失敗：${t.message}")
        } finally {
            blank.recycle()
        }
    }
}

/**
 * CartoonSegmentation 的 RTMDet-Ins（`cartoonseg.ncnn.param/.bin`，pnnx 轉檔，fp16）。
 *
 * 前處理與後處理的規格在 `parity/export_cseg_ncnn.py`（切圖、blob 契約、numpy 參考實作），
 * 後處理由 [CsegPost] 移植、JVM 測試逐像素對 fixture。輸入固定長邊 640（訓練解析度、實測最佳）。
 *
 * **權重來源與授權**（2026-09 查證）：ONNX 權重取自 Hugging Face `Jakaline/CartoonSegmentationOnnx`（本專案再轉 NCNN）；
 * 上游 GitHub `CartoonSegmentation/CartoonSegmentation` **沒有 LICENSE 檔、README 也未寫授權**，訓練資料含 Manga109
 * → **再散布授權不明確**（不是 MIT、也不是 GPL）。本專案照散布（models.json role "charseg"）並附 attribution，
 * 聲明研究／非商業用途、權利人要求即下架、使用者亦可自行取得權重（BYOM 放 `models/`）。
 */
class CsegSegmenter(paramPath: String, binPath: String) : CharSegmenter {

    private var handle: Long = 0L

    init {
        check(NcnnBackend.available) { "NCNN 原生庫未載入，無法做人物分割" }
        handle = NcnnBackend.createNet(paramPath, binPath)
        check(handle != 0L) { "NCNN 人物分割模型載入失敗：$paramPath" }
    }

    override fun segment(page: Bitmap): BooleanArray {
        check(handle != 0L) { "CsegSegmenter 已關閉" }
        val w = page.width
        val h = page.height
        val px = IntArray(w * h)
        page.getPixels(px, 0, w, 0, 0, w, h)
        val pre = CsegPost.preprocess(px, w, h)
        val outs = CsegPost.allocOutputs()
        val rc = NcnnBackend.extract(handle, pre.chw, CsegPost.SIZE, CsegPost.SIZE, 3, CsegPost.OUT_NAMES, outs)
        check(rc == 0) { "NCNN 人物分割推論失敗 rc=$rc" }
        return CsegPost.unionMask(outs, pre.nw, pre.nh, w, h)
    }

    override fun close() {
        if (handle != 0L) {
            NcnnBackend.releaseNet(handle)
            handle = 0L
        }
    }
}

/**
 * cseg 的前／後處理，純 Kotlin（不碰 NCNN、不碰 android.graphics 的繪圖），JVM 可測。
 *
 * 規格＝`parity/export_cseg_ncnn.py`：
 *  - 先驗 (col×stride, row×stride)、offset 0、row-major 展平
 *  - 框：relu(reg)×stride 當 l,t,r,b 距離，夾到 [0,640]；score = sigmoid(cls) > 0.05；NMS IoU 0.6；最多 100
 *  - 遮罩：cat[相對座標(2), mask_feat(8)] → 1×1 動態卷積 10→8 relu → 8→8 relu → 8→1 → 雙線性 ×8 → sigmoid
 *  - 管線用法：score > 0.3、機率 > 0.5、裁 pad、最近鄰放回原尺寸、聯集
 */
object CsegPost {
    const val SIZE = 640
    val STRIDES = intArrayOf(8, 16, 32)
    const val SCORE_THR = 0.05f
    const val NMS_IOU = 0.6f
    const val MAX_PER_IMG = 100
    const val PIPELINE_SCORE = 0.3f
    const val MASK_THR = 0.5f
    const val K_PARAMS = 169          // weights 80+64+8, biases 8+8+1
    const val PROTO = 8
    private val MEAN = floatArrayOf(103.53f, 116.28f, 123.675f)   // BGR
    private val STD = floatArrayOf(57.375f, 57.12f, 58.395f)
    private const val PAD = 114

    /** blob 順序＝ONNX 切圖的輸出順序（cls×3、reg×3、kernel×3、mask_feat），pnnx 依序命名 out0..out9。 */
    val OUT_NAMES: Array<String> = Array(10) { "out$it" }

    fun allocOutputs(size: Int = SIZE): Array<FloatArray> {
        val hw = STRIDES.map { (size / it) * (size / it) }
        return arrayOf(
            FloatArray(hw[0]), FloatArray(hw[1]), FloatArray(hw[2]),
            FloatArray(4 * hw[0]), FloatArray(4 * hw[1]), FloatArray(4 * hw[2]),
            FloatArray(K_PARAMS * hw[0]), FloatArray(K_PARAMS * hw[1]), FloatArray(K_PARAMS * hw[2]),
            FloatArray(PROTO * hw[0]),
        )
    }

    class Pre(val chw: FloatArray, val nw: Int, val nh: Int)

    /**
     * ARGB 像素 → in0：等比縮到長邊 [size]（面積平均，比照 cv2 INTER_AREA）、右下角 pad 114、BGR、(x−mean)/std。
     */
    fun preprocess(px: IntArray, w: Int, h: Int, size: Int = SIZE): Pre {
        val s = size.toDouble() / max(w, h)
        val nw = (w * s).roundToInt()
        val nh = (h * s).roundToInt()
        val chw = FloatArray(3 * size * size)
        val plane = size * size
        for (c in 0 until 3) chw.fill((PAD - MEAN[c]) / STD[c], c * plane, (c + 1) * plane)
        // 面積平均縮放：每個目標像素 = 來源框 [x·w/nw, (x+1)·w/nw) × [y·h/nh, (y+1)·h/nh) 的加權平均
        val sx = w.toDouble() / nw
        val sy = h.toDouble() / nh
        val acc = DoubleArray(3)
        for (y in 0 until nh) {
            val y0 = y * sy
            val y1 = (y + 1) * sy
            val iy0 = floor(y0).toInt()
            val iy1 = min(h, kotlin.math.ceil(y1).toInt())
            for (x in 0 until nw) {
                val x0 = x * sx
                val x1 = (x + 1) * sx
                val ix0 = floor(x0).toInt()
                val ix1 = min(w, kotlin.math.ceil(x1).toInt())
                acc[0] = 0.0; acc[1] = 0.0; acc[2] = 0.0
                var wsum = 0.0
                for (yy in iy0 until iy1) {
                    val wy = min(y1, yy + 1.0) - max(y0, yy.toDouble())
                    if (wy <= 0) continue
                    val row = yy * w
                    for (xx in ix0 until ix1) {
                        val wx = min(x1, xx + 1.0) - max(x0, xx.toDouble())
                        if (wx <= 0) continue
                        val wgt = wx * wy
                        val p = px[row + xx]
                        acc[2] += ((p shr 16) and 0xFF) * wgt   // R → BGR 的第 2 通道
                        acc[1] += ((p shr 8) and 0xFF) * wgt
                        acc[0] += (p and 0xFF) * wgt
                        wsum += wgt
                    }
                }
                val idx = y * size + x
                for (c in 0 until 3) {
                    val v = (acc[c] / wsum).roundToInt().coerceIn(0, 255)   // cv2 縮完是 uint8，先取整再正規化
                    chw[c * plane + idx] = (v - MEAN[c]) / STD[c]
                }
            }
        }
        return Pre(chw, nw, nh)
    }

    /** NMS 後的一個實例（640 座標）。[level]/[index] 指回原始輸出，遮罩要用時再算。 */
    class Det(val x1: Float, val y1: Float, val x2: Float, val y2: Float, val score: Float, val level: Int, val index: Int)

    private fun sigmoid(v: Float): Float = 1f / (1f + exp(-v))

    /** 解碼 + 門檻 + NMS。回傳依分數遞減、最多 [maxPerImg] 個。 */
    fun decode(
        outs: Array<FloatArray>, size: Int = SIZE,
        scoreThr: Float = SCORE_THR, iouThr: Float = NMS_IOU, maxPerImg: Int = MAX_PER_IMG,
    ): List<Det> {
        val cand = ArrayList<Det>()
        for (lv in STRIDES.indices) {
            val stride = STRIDES[lv]
            val fw = size / stride
            val fh = size / stride
            val hw = fw * fh
            val cls = outs[lv]
            val reg = outs[3 + lv]
            for (row in 0 until fh) {
                for (col in 0 until fw) {
                    val idx = row * fw + col
                    val score = sigmoid(cls[idx])
                    if (score <= scoreThr) continue
                    val px = (col * stride).toFloat()
                    val py = (row * stride).toFloat()
                    val l = max(reg[idx], 0f) * stride
                    val t = max(reg[hw + idx], 0f) * stride
                    val r = max(reg[2 * hw + idx], 0f) * stride
                    val b = max(reg[3 * hw + idx], 0f) * stride
                    cand.add(
                        Det(
                            (px - l).coerceIn(0f, size.toFloat()), (py - t).coerceIn(0f, size.toFloat()),
                            (px + r).coerceIn(0f, size.toFloat()), (py + b).coerceIn(0f, size.toFloat()),
                            score, lv, idx,
                        ),
                    )
                }
            }
        }
        if (cand.isEmpty()) return emptyList()
        // 穩定排序（分數相同保留展平序，與 numpy argsort(kind=stable) 一致）
        val order = cand.indices.sortedWith(compareByDescending<Int> { cand[it].score }.thenBy { it })
        val keep = ArrayList<Det>()
        val alive = BooleanArray(cand.size) { true }
        for (oi in order.indices) {
            val i = order[oi]
            if (!alive[i]) continue
            val a = cand[i]
            keep.add(a)
            if (keep.size >= maxPerImg) break
            val areaA = (a.x2 - a.x1) * (a.y2 - a.y1)
            for (oj in oi + 1 until order.size) {
                val j = order[oj]
                if (!alive[j]) continue
                val b = cand[j]
                val iw = min(a.x2, b.x2) - max(a.x1, b.x1)
                val ih = min(a.y2, b.y2) - max(a.y1, b.y1)
                if (iw <= 0f || ih <= 0f) continue
                val inter = iw * ih
                val areaB = (b.x2 - b.x1) * (b.y2 - b.y1)
                val iou = inter / max(areaA + areaB - inter, 1e-9f)
                if (iou > iouThr) alive[j] = false
            }
        }
        return keep
    }

    /**
     * 一個實例的遮罩機率（[size]×[size]）：相對座標 + mask_feat 過三層動態 1×1 卷積，雙線性 ×8，sigmoid。
     */
    fun maskOf(det: Det, outs: Array<FloatArray>, size: Int = SIZE): FloatArray {
        val stride = STRIDES[det.level]
        val fw = size / stride
        val hw = fw * fw
        val ker = outs[6 + det.level]
        val k = FloatArray(K_PARAMS) { ker[it * hw + det.index] }
        val mf = outs[9]
        val mw = size / STRIDES[0]
        val mh = mw
        val mhw = mw * mh
        val px = ((det.index % fw) * stride).toFloat()
        val py = ((det.index / fw) * stride).toFloat()
        val denom = stride * 8f
        // 層 1：10 → 8（輸入＝相對座標 2 通道 + 原型 8 通道）
        val h1 = FloatArray(8 * mhw)
        for (o in 0 until 8) {
            val wBase = o * 10
            val bias = k[152 + o]
            for (j in 0 until mhw) {
                val gx = ((j % mw) * STRIDES[0]).toFloat()
                val gy = ((j / mw) * STRIDES[0]).toFloat()
                var v = bias + k[wBase] * ((px - gx) / denom) + k[wBase + 1] * ((py - gy) / denom)
                for (c in 0 until PROTO) v += k[wBase + 2 + c] * mf[c * mhw + j]
                h1[o * mhw + j] = max(v, 0f)
            }
        }
        // 層 2：8 → 8
        val h2 = FloatArray(8 * mhw)
        for (o in 0 until 8) {
            val wBase = 80 + o * 8
            val bias = k[160 + o]
            for (j in 0 until mhw) {
                var v = bias
                for (c in 0 until 8) v += k[wBase + c] * h1[c * mhw + j]
                h2[o * mhw + j] = max(v, 0f)
            }
        }
        // 層 3：8 → 1
        val logit = FloatArray(mhw)
        for (j in 0 until mhw) {
            var v = k[168]
            for (c in 0 until 8) v += k[144 + c] * h2[c * mhw + j]
            logit[j] = v
        }
        // 雙線性 ×8（align_corners=False：來源座標 = (dst+0.5)/8 − 0.5）→ sigmoid
        val out = FloatArray(size * size)
        val sc = size / mw
        val ys0 = IntArray(size)
        val ys1 = IntArray(size)
        val wys = FloatArray(size)
        for (d in 0 until size) {
            val srcF = ((d + 0.5f) / sc - 0.5f).coerceIn(0f, (mh - 1).toFloat())
            val i0 = floor(srcF).toInt()
            ys0[d] = i0
            ys1[d] = min(i0 + 1, mh - 1)
            wys[d] = srcF - i0
        }
        for (y in 0 until size) {
            val r0 = ys0[y] * mw
            val r1 = ys1[y] * mw
            val wy = wys[y]
            val base = y * size
            for (x in 0 until size) {
                val x0 = ys0[x]     // 正方形：x 的取樣表與 y 相同
                val x1 = ys1[x]
                val wx = wys[x]
                val top = logit[r0 + x0] * (1 - wx) + logit[r0 + x1] * wx
                val bot = logit[r1 + x0] * (1 - wx) + logit[r1 + x1] * wx
                out[base + x] = sigmoid(top * (1 - wy) + bot * wy)
            }
        }
        return out
    }

    /**
     * 管線用法：score > 0.3 的實例、機率 > 0.5 在 640 座標取聯集，裁掉 pad（只留 [nw]×[nh]），
     * 最近鄰放回原尺寸 [w]×[h]。最近鄰取樣對每張遮罩取的是同一個來源像素，所以「先聯集再放大」
     * 與 Python 的「逐張放大再聯集」逐位元相同。
     */
    fun unionMask(outs: Array<FloatArray>, nw: Int, nh: Int, w: Int, h: Int, size: Int = SIZE): BooleanArray {
        val dets = decode(outs, size)
        val u = BooleanArray(size * size)
        var any = false
        for (d in dets) {
            if (d.score <= PIPELINE_SCORE) continue
            val m = maskOf(d, outs, size)
            for (i in u.indices) if (m[i] > MASK_THR) u[i] = true
            any = true
        }
        val out = BooleanArray(w * h)
        if (!any) return out
        // cv2 INTER_NEAREST：src = floor(dst × src_size / dst_size)
        val xs = IntArray(w) { min(nw - 1, floor(it * (nw.toDouble() / w)).toInt()) }
        for (y in 0 until h) {
            val sy = min(nh - 1, floor(y * (nh.toDouble() / h)).toInt())
            val srow = sy * size
            val drow = y * w
            for (x in 0 until w) out[drow + x] = u[srow + xs[x]]
        }
        return out
    }
}

/**
 * yoloseg（YOLO11-seg，`manga_seg_s.ncnn.param/.bin`，ultralytics `format=ncnn` 匯出，fp16）。
 * 規格在 `parity/export_yoloseg_ncnn.py`；後處理由 [YoloSegPost] 移植，JVM 測試逐像素對 fixture。
 *
 * **權重來源與授權**（2026-09 查證）：權重取自 Hugging Face `anonimkaq4/manga-page-element-segmentation`
 * （模型卡 license: other；以 MangaSeg／Manga109-s 標註訓練）。架構 YOLO11 為 Ultralytics **AGPL-3.0**（與本專案
 * GPL-3.0 相容——GPLv3 §13——但相容不等於沒有義務）。模型卡要求：再散布或商用前自行確認 MangaSeg、Manga109-s、
 * Ultralytics 三方授權、標註 "Copyrighted by Minshan Xie"、引用 CVPR 2025 MangaSeg 論文 → **再散布授權不明確**。
 * 本專案照散布（models.json role "charseg"）並附上述 attribution，聲明研究／非商業用途、權利人要求即下架、
 * 使用者亦可自行取得權重（BYOM 放 `models/`）。
 */
class YoloSegSegmenter(paramPath: String, binPath: String) : CharSegmenter {

    private var handle: Long = 0L

    init {
        check(NcnnBackend.available) { "NCNN 原生庫未載入，無法做人物分割" }
        handle = NcnnBackend.createNet(paramPath, binPath)
        check(handle != 0L) { "NCNN yoloseg 模型載入失敗：$paramPath" }
    }

    override fun segment(page: Bitmap): BooleanArray {
        check(handle != 0L) { "YoloSegSegmenter 已關閉" }
        val w = page.width
        val h = page.height
        val px = IntArray(w * h)
        page.getPixels(px, 0, w, 0, 0, w, h)
        val pre = YoloSegPost.preprocess(px, w, h)
        val outs = YoloSegPost.allocOutputs()
        val rc = NcnnBackend.extract(handle, pre.chw, YoloSegPost.SIZE, YoloSegPost.SIZE, 3, YoloSegPost.OUT_NAMES, outs)
        check(rc == 0) { "NCNN yoloseg 推論失敗 rc=$rc" }
        return YoloSegPost.unionMask(outs[0], outs[1], pre, w, h)
    }

    override fun close() {
        if (handle != 0L) {
            NcnnBackend.releaseNet(handle)
            handle = 0L
        }
    }
}

/**
 * yoloseg 的前／後處理，純 Kotlin、JVM 可測。規格＝`parity/export_yoloseg_ncnn.py`（＝research 的 run_yoloseg_onnx）：
 *  - letterbox：等比縮到長邊 1024（雙線性，比照 cv2 INTER_LINEAR）、置中 pad 114、RGB、/255
 *  - 只取 character 類（索引 2）且分數 > 0.25；NMS IoU 0.45（貪婪、分數遞減）
 *  - 遮罩：sigmoid(係數·prototypes) [256²] → 裁到 bbox → 雙線性到 1024 → 去 letterbox → 雙線性到原尺寸 → > 0.5 → 聯集
 * 兩段雙線性只在框的支撐區內算（框外的值恆為 0，> 0.5 不成立），與整張算逐位元相同。
 */
object YoloSegPost {
    const val SIZE = 1024
    const val CONF = 0.25f
    const val IOU = 0.45f
    const val CHAR_CLASS = 2
    const val MASK_THR = 0.5f
    const val NC = 3
    const val NPROTO = 32
    const val ANCHORS = 21504       // 1024 輸入：(128² + 64² + 32²)
    const val MW = 256
    private const val PAD = 114
    val OUT_NAMES: Array<String> = arrayOf("out0", "out1")

    fun allocOutputs(): Array<FloatArray> = arrayOf(FloatArray((4 + NC + NPROTO) * ANCHORS), FloatArray(NPROTO * MW * MW))

    class Pre(val chw: FloatArray, val nw: Int, val nh: Int, val top: Int, val left: Int)

    /** cv2.resize INTER_LINEAR 的取樣座標：半像素中心，夾在 [0, src−1]。 */
    private fun lerpIdx(dst: Int, scale: Double, srcN: Int, i0: IntArray, i1: IntArray, f: FloatArray) {
        var sx = (dst + 0.5) * scale - 0.5
        if (sx < 0) sx = 0.0
        val a = min(floor(sx).toInt(), srcN - 1)
        i0[dst] = a
        i1[dst] = min(a + 1, srcN - 1)
        f[dst] = (sx - a).toFloat().coerceIn(0f, 1f)
    }

    fun preprocess(px: IntArray, w: Int, h: Int): Pre {
        val r = min(SIZE.toDouble() / h, SIZE.toDouble() / w)
        val nh = (h * r).roundToInt()
        val nw = (w * r).roundToInt()
        val top = (SIZE - nh) / 2
        val left = (SIZE - nw) / 2
        val plane = SIZE * SIZE
        val chw = FloatArray(3 * plane) { PAD / 255f }
        val x0 = IntArray(nw); val x1 = IntArray(nw); val fx = FloatArray(nw)
        val y0 = IntArray(nh); val y1 = IntArray(nh); val fy = FloatArray(nh)
        for (x in 0 until nw) lerpIdx(x, w.toDouble() / nw, w, x0, x1, fx)
        for (y in 0 until nh) lerpIdx(y, h.toDouble() / nh, h, y0, y1, fy)
        for (y in 0 until nh) {
            val ra = y0[y] * w
            val rb = y1[y] * w
            val wy = fy[y]
            val dst = (top + y) * SIZE + left
            for (x in 0 until nw) {
                val pa = px[ra + x0[x]]; val pb = px[ra + x1[x]]
                val pc = px[rb + x0[x]]; val pd = px[rb + x1[x]]
                val wx = fx[x]
                for (c in 0 until 3) {
                    val sh = 16 - 8 * c      // R,G,B
                    val v = ((pa shr sh) and 0xFF) * (1 - wx) * (1 - wy) + ((pb shr sh) and 0xFF) * wx * (1 - wy) +
                        ((pc shr sh) and 0xFF) * (1 - wx) * wy + ((pd shr sh) and 0xFF) * wx * wy
                    chw[c * plane + dst + x] = (v + 0.5f).toInt().coerceIn(0, 255) / 255f   // cv2 縮完是 uint8
                }
            }
        }
        return Pre(chw, nw, nh, top, left)
    }

    class Det(val cx: Float, val cy: Float, val bw: Float, val bh: Float, val score: Float, val index: Int)

    /** 篩 character 類 + NMS。[o0] 排列 [39][ANCHORS]。 */
    fun decode(o0: FloatArray): List<Det> {
        val n = ANCHORS
        val cand = ArrayList<Det>()
        for (i in 0 until n) {
            var best = 0
            var bestS = -1f
            for (c in 0 until NC) {
                val v = o0[(4 + c) * n + i]
                if (v > bestS) { bestS = v; best = c }
            }
            if (best == CHAR_CLASS && bestS > CONF) cand.add(Det(o0[i], o0[n + i], o0[2 * n + i], o0[3 * n + i], bestS, i))
        }
        if (cand.isEmpty()) return emptyList()
        val order = cand.indices.sortedWith(compareByDescending<Int> { cand[it].score }.thenBy { it })
        val alive = BooleanArray(cand.size) { true }
        val keep = ArrayList<Det>()
        for (oi in order.indices) {
            val i = order[oi]
            if (!alive[i]) continue
            val a = cand[i]
            keep.add(a)
            val ax1 = a.cx - a.bw / 2; val ay1 = a.cy - a.bh / 2; val ax2 = ax1 + a.bw; val ay2 = ay1 + a.bh
            for (oj in oi + 1 until order.size) {
                val j = order[oj]
                if (!alive[j]) continue
                val b = cand[j]
                val bx1 = b.cx - b.bw / 2; val by1 = b.cy - b.bh / 2
                val iw = max(0f, min(ax2, bx1 + b.bw) - max(ax1, bx1))
                val ih = max(0f, min(ay2, by1 + b.bh) - max(ay1, by1))
                val inter = iw * ih
                val iou = inter / max(a.bw * a.bh + b.bw * b.bh - inter, 1e-9f)
                if (iou > IOU) alive[j] = false
            }
        }
        return keep
    }

    /** 全部實例的聯集遮罩（原尺寸）。 */
    fun unionMask(o0: FloatArray, o1: FloatArray, pre: Pre, w: Int, h: Int): BooleanArray {
        val out = BooleanArray(w * h)
        val dets = decode(o0)
        if (dets.isEmpty()) return out
        val n = ANCHORS
        val mw = MW
        val mhw = mw * mw
        val m = FloatArray(mhw)
        // 第二段（sub → 原尺寸）的取樣表；sub 的座標＝1024 座標減 letterbox 偏移
        val sx0 = IntArray(w); val sx1 = IntArray(w); val sfx = FloatArray(w)
        val sy0 = IntArray(h); val sy1 = IntArray(h); val sfy = FloatArray(h)
        for (x in 0 until w) lerpIdx(x, pre.nw.toDouble() / w, pre.nw, sx0, sx1, sfx)
        for (y in 0 until h) lerpIdx(y, pre.nh.toDouble() / h, pre.nh, sy0, sy1, sfy)
        // 第一段（prototype → 1024）的取樣表
        val px0 = IntArray(SIZE); val px1 = IntArray(SIZE); val pfx = FloatArray(SIZE)
        for (d in 0 until SIZE) lerpIdx(d, mw.toDouble() / SIZE, mw, px0, px1, pfx)
        val full = FloatArray(SIZE * SIZE)        // 只在支撐區內寫、用完清回 0
        for (d in dets) {
            // sigmoid(係數 · prototypes)，裁到 bbox（crop_mask）
            val x1 = ((d.cx - d.bw / 2) * mw / SIZE)
            val x2 = ((d.cx + d.bw / 2) * mw / SIZE)
            val y1 = ((d.cy - d.bh / 2) * mw / SIZE)
            val y2 = ((d.cy + d.bh / 2) * mw / SIZE)
            val cx0 = max(0, x1.toInt()); val cx1 = min(mw, kotlin.math.ceil(x2.toDouble()).toInt())
            val cy0 = max(0, y1.toInt()); val cy1 = min(mw, kotlin.math.ceil(y2.toDouble()).toInt())
            if (cx1 <= cx0 || cy1 <= cy0) continue
            java.util.Arrays.fill(m, 0f)
            for (yy in cy0 until cy1) for (xx in cx0 until cx1) {
                val j = yy * mw + xx
                var v = 0f
                for (c in 0 until NPROTO) v += o0[(4 + NC + c) * n + d.index] * o1[c * mhw + j]
                m[j] = 1f / (1f + exp(-v))
            }
            // 第一段：只有取樣到裁切區的 1024 像素會非零 → 支撐區 = 來源落在 [cx0−1, cx1] 的目標像素
            val fxA = max(0, floor((cx0 - 1.0) * SIZE / mw).toInt()); val fxB = min(SIZE, kotlin.math.ceil((cx1 + 1.0) * SIZE / mw).toInt())
            val fyA = max(0, floor((cy0 - 1.0) * SIZE / mw).toInt()); val fyB = min(SIZE, kotlin.math.ceil((cy1 + 1.0) * SIZE / mw).toInt())
            for (fy in fyA until fyB) {
                val ra = px0[fy] * mw; val rb = px1[fy] * mw; val wy = pfx[fy]
                val row = fy * SIZE
                for (fx in fxA until fxB) {
                    val a = px0[fx]; val b = px1[fx]; val wx = pfx[fx]
                    full[row + fx] = m[ra + a] * (1 - wx) * (1 - wy) + m[ra + b] * wx * (1 - wy) +
                        m[rb + a] * (1 - wx) * wy + m[rb + b] * wx * wy
                }
            }
            // 第二段：去 letterbox 後放到原尺寸，只算支撐區對應的原圖範圍
            val oxA = max(0, floor((fxA - pre.left - 1.0) * w / pre.nw).toInt()); val oxB = min(w, kotlin.math.ceil((fxB - pre.left + 1.0) * w / pre.nw).toInt())
            val oyA = max(0, floor((fyA - pre.top - 1.0) * h / pre.nh).toInt()); val oyB = min(h, kotlin.math.ceil((fyB - pre.top + 1.0) * h / pre.nh).toInt())
            for (oy in oyA until oyB) {
                val ra = (sy0[oy] + pre.top) * SIZE + pre.left; val rb = (sy1[oy] + pre.top) * SIZE + pre.left; val wy = sfy[oy]
                val drow = oy * w
                for (ox in oxA until oxB) {
                    val a = sx0[ox]; val b = sx1[ox]; val wx = sfx[ox]
                    val v = full[ra + a] * (1 - wx) * (1 - wy) + full[ra + b] * wx * (1 - wy) +
                        full[rb + a] * (1 - wx) * wy + full[rb + b] * wx * wy
                    if (v > MASK_THR) out[drow + ox] = true
                }
            }
            // 清掉這個實例的支撐區（下一個實例的 full 要從 0 開始）
            for (fy in fyA until fyB) java.util.Arrays.fill(full, fy * SIZE + fxA, fy * SIZE + fxB, 0f)
        }
        return out
    }
}
