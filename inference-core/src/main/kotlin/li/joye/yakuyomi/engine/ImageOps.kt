package li.joye.yakuyomi.engine

import android.graphics.Bitmap
import android.graphics.Canvas
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt

/**
 * 前處理 helper（CLAUDE.md §5 ImageOps：第三層，重寫）。
 */
internal object ImageOps {

    /** NCNN 偵測前處理結果：裸 NCHW FloatArray ＋ 縮放比例（原圖座標 ×ratio→模型空間；反算 ÷ratio）＋ 模型輸入 w/h（DBNet resize_aspect＝矩形）。 */
    class DetectorInputArray(val chw: FloatArray, val ratio: Float, val w: Int, val h: Int)

    /**
     * DBNet（m-i-t default 偵測器）前處理：resize_aspect_ratio（矩形、pad 到 MULT=256 倍數）+ 正規化 /127.5-1（[-1,1]，非 ctd /255）+ 可選銳利化。
     * ported from manga_translator/detection/default.py:_infer + default_utils/imgproc.py:resize_aspect_ratio @ d5a3eee
     * ★★ 為何 resize_aspect 而非正方形 letterbox：**ncnn 對正方形 832-992 尺寸帶有 heap corruption bug（malloc crash，x86+arm64 同）**。
     *   pad 到 256 倍數 → 輸入維度永遠是 256 倍數（768/1024/1280）、永不落 crash 帶。這也正是 m-i-t 這樣設計的原因。
     *   long 邊縮到 [size]、按比例縮 short 邊，兩邊各 pad 右/下到 256 倍數（origin 左上）；ratio=size/max(H,W)，座標反算 ÷ratio。
     */
    fun detectorChwDbnet(page: Bitmap, size: Int, sharpen: Boolean): DetectorInputArray {
        val mult = 256
        val ratio = size.toFloat() / max(page.width, page.height)   // target_ratio（long 邊縮到 size）
        val tw = (page.width * ratio).roundToInt().coerceAtLeast(1)
        val th = (page.height * ratio).roundToInt().coerceAtLeast(1)
        val inW = tw + (mult - tw % mult) % mult                    // pad 右到 256 倍數
        val inH = th + (mult - th % mult) % mult                    // pad 下到 256 倍數

        val scaled = Bitmap.createScaledBitmap(page, tw, th, true)
        val canvas = Bitmap.createBitmap(inW, inH, Bitmap.Config.ARGB_8888) // pad 區＝黑(RGB0)
        Canvas(canvas).drawBitmap(scaled, 0f, 0f, null)

        val area = inW * inH
        val pixels = IntArray(area)
        canvas.getPixels(pixels, 0, inW, 0, 0, inW, inH)
        if (sharpen) unsharp(pixels, inW, tw, th, 1.5f)             // 只銳化有效區 [0:th,0:tw]

        val chw = FloatArray(3 * area)
        for (i in 0 until area) {
            val p = pixels[i]
            chw[i] = (((p shr 16) and 0xFF) / 127.5f) - 1f          // R → plane 0
            chw[area + i] = (((p shr 8) and 0xFF) / 127.5f) - 1f     // G → plane 1
            chw[2 * area + i] = ((p and 0xFF) / 127.5f) - 1f         // B → plane 2
        }
        if (scaled !== page) scaled.recycle()
        canvas.recycle()
        return DetectorInputArray(chw, ratio, inW, inH)
    }

    // unsharp mask（separable Gaussian σ≈2 的 9-tap + amount，比照桌面 sharp960.py 的 GaussianBlur+addWeighted）。
    // 只處理 letterbox 有效區 [0:nh,0:nw]（黑邊不動）；per-channel（R/G/B），alpha 保留。
    private val GK = floatArrayOf(0.02763f, 0.06630f, 0.12383f, 0.18018f, 0.20416f, 0.18018f, 0.12383f, 0.06630f, 0.02763f)

    private fun unsharp(px: IntArray, stride: Int, nw: Int, nh: Int, amount: Float) {
        val rad = 4
        fun channel(shift: Int) {
            val orig = FloatArray(nw * nh)
            for (y in 0 until nh) for (x in 0 until nw) orig[y * nw + x] = ((px[y * stride + x] shr shift) and 0xFF).toFloat()
            val tmp = FloatArray(nw * nh)   // 水平 pass
            for (y in 0 until nh) for (x in 0 until nw) {
                var s = 0f
                for (k in -rad..rad) s += orig[y * nw + (x + k).coerceIn(0, nw - 1)] * GK[k + rad]
                tmp[y * nw + x] = s
            }
            val blur = FloatArray(nw * nh)  // 垂直 pass
            for (y in 0 until nh) for (x in 0 until nw) {
                var s = 0f
                for (k in -rad..rad) s += tmp[(y + k).coerceIn(0, nh - 1) * nw + x] * GK[k + rad]
                blur[y * nw + x] = s
            }
            for (y in 0 until nh) for (x in 0 until nw) {   // out = orig + amount*(orig-blur)
                val i = y * nw + x
                val v = (orig[i] + amount * (orig[i] - blur[i])).roundToInt().coerceIn(0, 255)
                val di = y * stride + x
                px[di] = (px[di] and (0xFF shl shift).inv()) or (v shl shift)
            }
        }
        channel(16); channel(8); channel(0)
    }
}
