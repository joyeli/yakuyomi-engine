package li.joye.yakuyomi.engine

import android.graphics.Bitmap
import kotlin.math.exp
import kotlin.math.floor
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt

/**
 * 人物語意分割（夜讀用）：把一頁的人物像素標出來，夜讀重繪據此「絕不塗錯」臉／手／白衣／白髮。
 *
 * 模型走 NCNN（與偵測、去字同一個後端、同一把鎖）。回傳與頁面同尺寸的布林遮罩，true＝人物。
 * 這是**模型原輸出**的聯集，貼墨收邊與平滑由夜讀管線負責。
 */
interface CharSegmenter : AutoCloseable {
    fun segment(page: Bitmap): BooleanArray
}

/**
 * CartoonSegmentation 的 RTMDet-Ins（`cartoonseg.ncnn.param/.bin`，pnnx 轉檔，fp16）。
 *
 * 前處理與後處理的規格在 `parity/export_cseg_ncnn.py`（切圖、blob 契約、numpy 參考實作），
 * 後處理由 [CsegPost] 移植、JVM 測試逐像素對 fixture。輸入固定長邊 640（訓練解析度、實測最佳）。
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
