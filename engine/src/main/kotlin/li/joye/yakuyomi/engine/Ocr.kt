package li.joye.yakuyomi.engine

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Matrix
import android.graphics.Paint
import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.async
import kotlinx.coroutines.awaitAll
import kotlinx.coroutines.coroutineScope
import kotlinx.coroutines.sync.Semaphore
import kotlinx.coroutines.sync.withPermit
import kotlin.math.abs
import kotlin.math.ceil
import kotlin.math.floor
import kotlin.math.hypot
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt

/**
 * 48px CTC OCR（NCNN 後端）。
 *
 * ported from manga_translator/ocr/model_48px_ctc.py (+ ocr/common.py, utils/generic.py) @ d5a3eee
 *   裁切：sortPnts 定直/橫書 + get_transformed_region（findHomography→warpPerspective→48px 條，直書轉90°）
 *         此處用 Android Matrix.setPolyToPoly 取代 cv2 透視（§6）。
 *   前處理：(x-127.5)/127.5、NCHW、RGB。
 *   解碼：greedy CTC（blank=0、收合重複+去blank）→ 查字典。
 *   ignore_bubble（cfg.ignoreBubble，ported from utils/bubble.py）：跳過彩色/非氣泡 SFX 類文字。
 *   顏色 head 不採用（彩底太雜）；文字色改由 [Renderer] 取去字後背景亮度判黑/白。
 *
 * 推論只有 NCNN 一條路（[modelPath] 必須是 `.param`，parity/export_ocr_ncnn.py 轉出；ORT `.onnx` 路徑 2026-09-26 連同
 * ONNX Runtime 一起從引擎拔除——三顆模型全 NCNN）：
 *   · 寬度牆的繞法＝正弦位置編碼不烤進圖、當第二個輸入每條現算（[sinusoidalPe]，T=floor(W/4)−1），
 *     argmax／log_softmax 在 JNI 算完只回 idx+logp（[NcnnBackend.ocrCtc]），Kotlin 只做 CTC 收合（[ctcCollapse]）。
 *   · 精度＝混合精度優先：`<name>_mixed.ncnn.param`（backbone fp16、transformer+char_pred fp32，靠 per-layer featmask
 *     + 手插 Cast 層）與原 `<name>.ncnn.param` 共用同一份 `.bin`。[pickParam] 只在 fp16 storage 開且 CPU 有 asimdhp 時
 *     載 mixed，否則退回原 param（依 [OcrConfig.ncnnFp16Storage]／[OcrConfig.ncnnFp16Arith] 跑全 fp16 或 fp32）。
 *     真機 9 頁 242 行 A/B（真值 ORT fp32）：mixed 241、fp32 242、全 fp16 219（小假名讀錯）；mixed 比舊 ORT int8 快 ~23%。
 *   · 並發模式下 Net 以 1 緒建、逐條 forward 不進 ncnn 全域鎖（原因見 [NcnnBackend.ocrCtc]）。
 */
class Ocr(
    modelPath: String,
    private val dictionary: List<String>,
    private val cfg: OcrConfig = OcrConfig(),
) : AutoCloseable {

    /** 實際生效的精度組合："NCNN"（原 param：全 fp16／fp32 依 cfg）或 "NCNN-mixed"（無 adb 時由呼叫端寫進 log 確認）。 */
    val backend: String
    private var ncnnHandle: Long = 0L
    // 並發模式：每行單緒（Net num_threads=1）、靠 N 行並發填核；序列模式：單行用滿 NUM_THREADS。
    private val threads = if (cfg.concurrent) 1 else NUM_THREADS

    init {
        require(modelPath.endsWith(".param")) {
            "OCR 模型只支援 NCNN `.param`（同名 `.ncnn.bin` 需在旁；ORT `.onnx` 路徑已拔除）：$modelPath"
        }
        check(NcnnBackend.available) { "NCNN 原生庫未載入，無法 OCR" }
        val (param, mixed) = pickParam(modelPath)
        // 兩份 param 共用同一份 .bin：不論給的是原版還是 _mixed，bin 都是「原版 param 去掉 .param 加 .bin」（與偵測／去字
        // 同一條命名規則：`<name>.ncnn.param` ↔ `<name>.ncnn.bin`）。路徑載入＝native 記憶體、不佔 JVM heap。
        val bin = baseParam(modelPath).removeSuffix(".param") + ".bin"
        if (!mixed && !java.io.File(param).exists()) {
            // ModelSet 允許只有 mixed 版在場；但 mixed 的 Cast 層要 fp16 storage + asimdhp 才正確，條件不成立又沒原版可退
            // → 與其載成垃圾（不 crash、OCR 全錯）不如明講。
            error("缺 OCR 原版 param $param：只有 mixed 版、但目前不能用 mixed（ncnnMixed=${cfg.ncnnMixed} fp16Storage=${cfg.ncnnFp16Storage} cpuFp16=${NcnnBackend.cpuSupportsFp16}）")
        }
        ncnnHandle = NcnnBackend.createNetEx(param, bin, threads, cfg.ncnnFp16Storage, cfg.ncnnFp16Arith)
        check(ncnnHandle != 0L) { "NCNN OCR 模型載入失敗：$param（bin=$bin）" }
        backend = if (mixed) "NCNN-mixed" else "NCNN"
        // 進診斷紀錄（fork 的 TraceLog 收 EngineTrace）：無 adb 也能確認產品實際載的是哪份 param／精度組合
        EngineTrace.log("ocr.load $backend ${java.io.File(param).name} threads=$threads fp16=${cfg.ncnnFp16Storage}/${cfg.ncnnFp16Arith} cpuFp16=${NcnnBackend.cpuSupportsFp16}")
        Log.i(TAG, "NCNN ocr loaded $param (threads=$threads fp16 storage=${cfg.ncnnFp16Storage} arith=${cfg.ncnnFp16Arith} cpuFp16=${NcnnBackend.cpuSupportsFp16} mixed=$mixed)")
    }

    /** 給的是 `_mixed` 就還原成原版 param 路徑，否則原樣。 */
    private fun baseParam(given: String): String =
        if (given.endsWith(MIXED_SUFFIX)) given.removeSuffix(MIXED_SUFFIX) + ".ncnn.param" else given

    /**
     * 決定實際載哪份 param：`<name>.ncnn.param`（全域精度）或 `<name>_mixed.ncnn.param`（backbone fp16、transformer fp32，
     * parity/export_ocr_ncnn.py write_mixed_param）。兩份共用 `<name>.ncnn.bin`。
     * mixed 的前提＝[OcrConfig.ncnnMixed] 且 fp16 storage 開且 CPU 有 asimdhp：mixed param 裡的 Cast 層宣告「進來的是
     * fp16」，任一條件不成立時 conv 吐的是 fp32、Cast 會靜默讀成垃圾（不 crash、OCR 全錯）→ 一律退回原 param。
     * 給的是哪一份都行（sandbox A/B 會直接指 mixed）；回 (param 路徑, 是否 mixed)。
     */
    private fun pickParam(given: String): Pair<String, Boolean> {
        val base = baseParam(given)
        val mixedPath = base.removeSuffix(".ncnn.param").removeSuffix(".param") + MIXED_SUFFIX
        val canMixed = cfg.ncnnMixed && cfg.ncnnFp16Storage && NcnnBackend.cpuSupportsFp16
        val useMixed = canMixed && java.io.File(mixedPath).exists()
        if (given.endsWith(MIXED_SUFFIX) && !useMixed) {
            Log.w(TAG, "指定 mixed param 但條件不符（mixed=${cfg.ncnnMixed} fp16Storage=${cfg.ncnnFp16Storage} cpuFp16=${NcnnBackend.cpuSupportsFp16}），退回 $base")
        }
        return if (useMixed) mixedPath to true else base to false
    }

    /**
     * 對每條文字行做 OCR，就地填入 direction 與 text。每條右側加 [PAD_MARGIN] 白邊（見 [stripToChw]）讓 CTC 不截尾字。
     * [OcrConfig.concurrent]＝true：多行並發（小圖塊吃不滿 intra-op→改單緒、並發填核，見 init）；false：逐行序列（現狀）。
     * 批次 padding 已否決（寬度差→padding 浪費）；此處是「並發」（零 padding），與批次不同。
     */
    suspend fun recognize(
        page: Bitmap,
        lines: List<TextLine>,
        bicubic: Boolean = cfg.useBicubic, // 裁切縮放內插法：true=手刻 bicubic（救小假名漏讀）、false=Canvas bilinear（現行）
    ): Unit = coroutineScope {
        if (cfg.concurrent && lines.size > 1) {
            val sem = Semaphore(cfg.concurrency.coerceAtLeast(1))
            lines.map { line ->
                async(Dispatchers.Default) { sem.withPermit { recognizeOne(page, line, bicubic) } }
            }.awaitAll()
        } else {
            for (line in lines) recognizeOne(page, line, bicubic)
        }
    }

    /**
     * 暖機：對空白 strip 跑一次 OCR，讓 NCNN 首次 forward 的一次性配置（blob／workspace allocator、共用 PE 表 lazy 建表）
     * 先在單緒做完。併發翻多頁前先呼叫一次；strip 內容不重要（只為觸發一次 forward）。
     */
    fun warmUp() {
        val strip = Bitmap.createBitmap(160, cfg.textHeight, Bitmap.Config.ARGB_8888)
        try {
            infer(stripToChw(strip))
        } catch (t: Throwable) {
            Log.w(TAG, "OCR 暖機失敗：${t.message}")
        } finally {
            strip.recycle()
        }
    }

    /** 單行 OCR：裁切→前處理→CTC→填 text。thread-safe：只寫自己的 line、JNI forward 可並發（1 緒 Net 不進全域鎖）、其餘皆 local/唯讀。 */
    private fun recognizeOne(page: Bitmap, line: TextLine, bicubic: Boolean) {
        // ★ 先外擴、再 sortPnts：sortPnts 定的點序是 warp 要的，擴完才排才不會亂序（擴張本身不改直/橫書判定）。
        val quad = if (cfg.stripPad > 0) expandQuad(line.quad, cfg.stripPad, page.width, page.height) else line.quad
        val (ordered, isV) = sortPnts(quad)
        line.direction = if (isV) "v" else "h"
        val strip = transformedRegion(page, ordered, isV, cfg.textHeight, bicubic) ?: return
        if (cfg.ignoreBubble in 1..50 && isIgnore(strip, cfg.ignoreBubble)) {
            strip.recycle()  // 彩色/非氣泡 SFX 類文字 → 跳過
            return
        }
        try {
            val (text, prob) = infer(stripToChw(strip))
            if (prob >= cfg.minProb) line.text = text  // 低信心誤讀 → 丟
        } catch (t: Throwable) {
            Log.w(TAG, "OCR 單行失敗：${t.message}")
        } finally {
            strip.recycle()
        }
    }

    /** 前處理好的一條：[chw]=[3,h,w]。 */
    private class Chw(val chw: FloatArray, val w: Int, val h: Int)

    /**
     * 一條 → (text, prob)。T=floor(W/4)−1（backbone 下採樣，parity 逐寬驗過），PE 用共用正弦表的前 T 列；JNI 回每時步
     * argmax+logp，這裡收合。T 超過表長（W>8196px 的字條）＝模型自己的 max_len 也裝不下 → 當空讀。
     */
    private fun infer(x: Chw): Pair<String, Float> {
        val t = x.w / 4 - 1
        if (t < 1) return "" to 0f
        if (t > PE_ROWS) {
            Log.w(TAG, "OCR 字條太寬（W=${x.w} → T=$t > $PE_ROWS），略過")
            return "" to 0f
        }
        val idx = IntArray(t)
        val logp = FloatArray(t)
        val rc = NcnnBackend.ocrCtc(ncnnHandle, x.chw, x.w, x.h, PE, t, idx, logp, serialize = threads > 1)
        check(rc == t) { "NCNN OCR 失敗 rc=$rc（W=${x.w} T=$t）" }
        return ctcCollapse(idx, logp, t, dictionary)
    }

    /**
     * quad 四邊各外擴 [pad] px 供 OCR 裁切（[OcrConfig.stripPad]）：minAreaRect → w,h 各 +2*pad → corners → clamp 進圖內。
     * **只回新的點、不動 [TextLine.quad]** ⇒ 偵測框與去字遮罩（走 seg 筆畫）完全不受影響。
     * 對齊桌面 exp_pad.py:expand_quad（cv2.minAreaRect + boxPoints）。救「框太瘦把字切掉 → CTC 空讀 → 留原文不翻」。
     */
    private fun expandQuad(pts: List<Pt>, pad: Int, w: Int, h: Int): List<Pt> {
        val rect = Geometry.minAreaRect(pts) ?: return pts
        return rect.expand(pad.toFloat()).corners().map {
            Pt(it.x.coerceIn(0f, (w - 1).toFloat()), it.y.coerceIn(0f, (h - 1).toFloat()))
        }
    }

    /** 對齊 generic.py:sort_pnts —— 回傳排序後 4 點與是否直書。 */
    private fun sortPnts(quad: List<Pt>): Pair<List<Pt>, Boolean> {
        val n = quad.size
        var best0 = 0
        var best1 = 0
        // 16 對向量、取長度排序的第 8、10 名（長邊）
        val norms = FloatArray(n * n)
        for (i in 0 until n) for (j in 0 until n) {
            val vx = quad[i].x - quad[j].x
            val vy = quad[i].y - quad[j].y
            norms[i * n + j] = hypot(vx.toDouble(), vy.toDouble()).toFloat()
        }
        val order = (0 until n * n).sortedBy { norms[it] }
        best0 = order[8]
        best1 = order[10]
        var l0x = quad[best0 / n].x - quad[best0 % n].x
        var l0y = quad[best0 / n].y - quad[best0 % n].y
        val l1x = quad[best1 / n].x - quad[best1 % n].x
        val l1y = quad[best1 / n].y - quad[best1 % n].y
        if (l0x * l1x + l0y * l1y < 0f) { l0x = -l0x; l0y = -l0y }
        val sx = abs((l0x + l1x) / 2f)
        val sy = abs((l0y + l1y) / 2f)
        val isV = sx <= sy

        return if (isV) {
            val byY = quad.sortedBy { it.y }
            val first2 = listOf(byY[0], byY[1]).sortedBy { it.x }
            val last2 = listOf(byY[2], byY[3]).sortedBy { it.x }
            Pair(listOf(first2[0], first2[1], last2[1], last2[0]), true)
        } else {
            val byX = quad.sortedBy { it.x }
            val ls = listOf(byX[0], byX[1]).sortedBy { it.y } // 左：上、下
            val rs = listOf(byX[2], byX[3]).sortedBy { it.y } // 右：上、下
            Pair(listOf(ls[0], rs[0], rs[1], ls[1]), false)
        }
    }

    /** 對齊 generic.py:Quadrilateral.get_transformed_region。[bicubic]=true 時 warp 用手刻 bicubic 取樣（見 [bicubicWarp]）。 */
    private fun transformedRegion(page: Bitmap, pts: List<Pt>, isV: Boolean, th: Int, bicubic: Boolean): Bitmap? {
        // structure 邊中點
        val p1x = (pts[0].x + pts[1].x) / 2f; val p1y = (pts[0].y + pts[1].y) / 2f
        val p2x = (pts[2].x + pts[3].x) / 2f; val p2y = (pts[2].y + pts[3].y) / 2f
        val p3x = (pts[1].x + pts[2].x) / 2f; val p3y = (pts[1].y + pts[2].y) / 2f
        val p4x = (pts[3].x + pts[0].x) / 2f; val p4y = (pts[3].y + pts[0].y) / 2f
        val vLen = hypot((p2x - p1x).toDouble(), (p2y - p1y).toDouble()).toFloat()
        val hLen = hypot((p4x - p3x).toDouble(), (p4y - p3y).toDouble()).toFloat()
        val ratio = vLen / max(hLen, 1e-6f)

        val w: Int
        val h: Int
        if (!isV) {
            h = max(th, 2)
            w = max((th / ratio).roundToInt(), 2)
        } else {
            w = max(th, 2)
            h = max((th * ratio).roundToInt(), 2)
        }

        val src = floatArrayOf(
            pts[0].x, pts[0].y, pts[1].x, pts[1].y,
            pts[2].x, pts[2].y, pts[3].x, pts[3].y,
        )
        val dst = floatArrayOf(
            0f, 0f, (w - 1).toFloat(), 0f,
            (w - 1).toFloat(), (h - 1).toFloat(), 0f, (h - 1).toFloat(),
        )
        val m = Matrix()
        if (!m.setPolyToPoly(src, 0, dst, 0, 4)) return null

        // bicubic：手刻 perspective bicubic 取樣（Android Canvas 只有 bilinear）；bilinear 把 ~30px 直行上採樣到 48px
        // 時糊掉小假名 → OCR 漏字（連句尾否定漏→意思相反）。parity 實測 bicubic 517字/100行 vs bilinear 486/96。
        val region = if (bicubic) {
            bicubicWarp(page, m, w, h, pts) ?: return null
        } else {
            Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
                .also { Canvas(it).drawBitmap(page, m, Paint(Paint.FILTER_BITMAP_FLAG)) }
        }
        if (!isV) return region

        // 直書：cv2.ROTATE_90_COUNTERCLOCKWISE → 高度變 48
        val rot = Matrix().apply { postRotate(-90f) }
        // filter=false（無損 transpose）：精確 90° 旋轉不需內插；filter=true 會在 warp 的 bilinear 之上再疊一層
        // bilinear 模糊、把小假名糊掉 → OCR 漏字。旋轉本身無縮放，關掉 filter 不失真且更快（見 stripToTensor 銳化）。
        val rotated = Bitmap.createBitmap(region, 0, 0, w, h, rot, false)
        region.recycle()
        return rotated
    }

    // ── 手刻 perspective bicubic warp（a=-0.75 cubic convolution）：逐位對齊 parity 驗證腳本 verify_handcubic(a=-0.75)
    //    → 517 字 / 100 行過信心門檻（vs 現行 bilinear 486/96），救回被 bilinear 縮放糊掉的小假名（含句尾否定→意思相反）。
    //    Android Canvas 無 bicubic 故手刻。效能：strip 小（平均 ~13K px）、每頁額外 ~12M 乘加 ≈ +1~4% OCR 時間（parity 估）。
    private fun bicubicWarp(page: Bitmap, forward: Matrix, w: Int, h: Int, pts: List<Pt>): Bitmap? {
        val inv = Matrix()
        if (!forward.invert(inv)) return null
        val iv = FloatArray(9)
        inv.getValues(iv) // 3x3：sx=(iv0*dx+iv1*dy+iv2)/(iv6*dx+iv7*dy+iv8)、sy 同理（Android Matrix 透視同 cv2）
        // 只 crop quad 來源包圍盒（±2 容 4-tap）→ 一次 getPixels、省記憶體
        val pw = page.width
        val ph = page.height
        val x0 = (floor(pts.minOf { it.x }) - 2f).toInt().coerceIn(0, pw - 1)
        val y0 = (floor(pts.minOf { it.y }) - 2f).toInt().coerceIn(0, ph - 1)
        val x1 = (ceil(pts.maxOf { it.x }) + 2f).toInt().coerceIn(0, pw - 1)
        val y1 = (ceil(pts.maxOf { it.y }) + 2f).toInt().coerceIn(0, ph - 1)
        val cw = x1 - x0 + 1
        val ch = y1 - y0 + 1
        if (cw < 2 || ch < 2) return null
        val src = IntArray(cw * ch)
        page.getPixels(src, 0, cw, x0, y0, cw, ch)
        val out = IntArray(w * h)
        for (dy in 0 until h) {
            for (dx in 0 until w) {
                val den = iv[6] * dx + iv[7] * dy + iv[8]
                val sx = (iv[0] * dx + iv[1] * dy + iv[2]) / den - x0
                val sy = (iv[3] * dx + iv[4] * dy + iv[5]) / den - y0
                out[dy * w + dx] = bicubicSample(src, cw, ch, sx, sy)
            }
        }
        return Bitmap.createBitmap(out, w, h, Bitmap.Config.ARGB_8888)
    }

    /** 對 [px]（cw×ch，ARGB）在 (sx,sy) 做 4×4 bicubic 取樣（a=-0.75、邊界 clamp、per-channel）。 */
    private fun bicubicSample(px: IntArray, cw: Int, ch: Int, sx: Float, sy: Float): Int {
        val a = -0.75f
        val fx = floor(sx).toInt()
        val fy = floor(sy).toInt()
        var r = 0f
        var g = 0f
        var b = 0f
        for (my in -1..2) {
            val wy = cubicW(sy - (fy + my), a)
            val rowBase = (fy + my).coerceIn(0, ch - 1) * cw
            var rr = 0f
            var gg = 0f
            var bb = 0f
            for (mx in -1..2) {
                val wx = cubicW(sx - (fx + mx), a)
                val p = px[rowBase + (fx + mx).coerceIn(0, cw - 1)]
                rr += wx * ((p shr 16) and 0xFF)
                gg += wx * ((p shr 8) and 0xFF)
                bb += wx * (p and 0xFF)
            }
            r += wy * rr
            g += wy * gg
            b += wy * bb
        }
        return (0xFF shl 24) or
            (r.roundToInt().coerceIn(0, 255) shl 16) or
            (g.roundToInt().coerceIn(0, 255) shl 8) or
            b.roundToInt().coerceIn(0, 255)
    }

    /** cubic convolution kernel（Horner 展開）：|t|≤1 與 1<|t|<2 兩段，其餘 0。 */
    private fun cubicW(t: Float, a: Float): Float {
        val x = abs(t)
        return when {
            x <= 1f -> ((a + 2f) * x - (a + 3f)) * x * x + 1f
            x < 2f -> ((a * x - 5f * a) * x + 8f * a) * x - 4f * a
            else -> 0f
        }
    }

    /**
     * Unsharp mask（原地銳化 [px]＝strip 的 ARGB 像素、[w]×[h]）：抵銷 warp 把 ~30px 寬直行上採樣到 48px 時的模糊。
     * 那層模糊會把小假名（に/い/ど/ん）糊成一團 → CTC 收合掉 → 整句漏字（真機把「なれない者など」讀成「者」）。
     *
     * 做法＝per-channel separable Gaussian(σ≈1.2、5-tap) 求模糊，再 `orig + AMOUNT*(orig-blur)`、clamp 0..255。
     * parity 實測（此前處理 + 銳化 amount=1.6）：整章可讀字數 255→431、過信心門檻(minProb 0.5)行數 60→90（/101）；
     * 對本就讀對(p≈1.0)的乾淨行無副作用。移植自 `parity` 驗證腳本 verify_myunsharp.py:my_unsharp（逐位對齊）。
     * 銳化因子與內插法皆由 parity benchmark 定（bilinear+unsharp 取 bicubic+unsharp 約 93% 效益、Android 無 bicubic 故選此）。
     */
    private fun unsharp(px: IntArray, w: Int, h: Int) {
        val n = w * h
        if (n == 0) return
        val amount = 1.6f
        val k0 = 0.3434f
        val k1 = 0.2428f
        val k2 = 0.0855f // Gaussian σ≈1.2 正規化 5-tap（中心/±1/±2）
        for (shift in intArrayOf(16, 8, 0)) { // R、G、B 各自銳化（文字多為灰階、但彩色 SFX 也照顧）
            val ch = FloatArray(n)
            for (i in 0 until n) ch[i] = ((px[i] shr shift) and 0xFF).toFloat()
            val tmp = FloatArray(n)
            for (y in 0 until h) { // 水平模糊（邊界 clamp 複製）
                val row = y * w
                for (x in 0 until w) {
                    tmp[row + x] = k2 * ch[row + max(0, x - 2)] + k1 * ch[row + max(0, x - 1)] +
                        k0 * ch[row + x] + k1 * ch[row + min(w - 1, x + 1)] + k2 * ch[row + min(w - 1, x + 2)]
                }
            }
            for (y in 0 until h) { // 垂直模糊 + 銳化寫回 px（僅動本 channel 的 8 bit、保留其餘）
                val row = y * w
                val rm2 = max(0, y - 2) * w
                val rm1 = max(0, y - 1) * w
                val rp1 = min(h - 1, y + 1) * w
                val rp2 = min(h - 1, y + 2) * w
                for (x in 0 until w) {
                    val blur = k2 * tmp[rm2 + x] + k1 * tmp[rm1 + x] + k0 * tmp[row + x] +
                        k1 * tmp[rp1 + x] + k2 * tmp[rp2 + x]
                    val orig = ch[row + x]
                    val v = (orig + amount * (orig - blur)).coerceIn(0f, 255f).toInt()
                    val i = row + x
                    px[i] = (px[i] and (0xFF shl shift).inv()) or (v shl shift)
                }
            }
        }
    }

    private fun stripToChw(strip: Bitmap): Chw {
        val sw = strip.width
        val h = strip.height
        val w = sw + PAD_MARGIN // 右側白邊：避免 CTC 截掉尾字（坂→坂本、ねえね→ねえねえ）
        val px = IntArray(sw * h)
        strip.getPixels(px, 0, sw, 0, 0, sw, h)
        if (cfg.ocrUnsharp) unsharp(px, sw, h) // 銳化：抵銷縮放模糊、救回漏讀小假名（見 [unsharp]、OcrConfig.ocrUnsharp）
        val area = h * w
        val chw = FloatArray(3 * area) { 1f } // 白底（(255-127.5)/127.5=1.0），右邊維持白
        for (y in 0 until h) {
            val inRow = y * sw
            val outRow = y * w
            for (x in 0 until sw) {
                val p = px[inRow + x]
                chw[outRow + x] = (((p shr 16) and 0xFF) - 127.5f) / 127.5f
                chw[area + outRow + x] = (((p shr 8) and 0xFF) - 127.5f) / 127.5f
                chw[2 * area + outRow + x] = ((p and 0xFF) - 127.5f) / 127.5f
            }
        }
        return Chw(chw, w, h)
    }

    /** SFX/非氣泡文字判定（ported from utils/bubble.py:is_ignore @ d5a3eee）：邊框混色（非乾淨氣泡底）或彩色 → 跳過。 */
    private fun isIgnore(strip: Bitmap, ignoreBubble: Int): Boolean {
        val w = strip.width
        val h = strip.height
        if (w < 4 || h < 4) return false
        val px = IntArray(w * h)
        strip.getPixels(px, 0, w, 0, 0, w, h)
        fun dark(i: Int): Boolean {
            val p = px[i]
            return 0.299f * ((p shr 16) and 0xFF) + 0.587f * ((p shr 8) and 0xFF) + 0.114f * (p and 0xFF) < 127f
        }
        var black = 0
        var total = 0
        for (y in 0..1) for (x in 0 until w) { if (dark(y * w + x)) black++; total++ }
        for (y in h - 2 until h) for (x in 0 until w) { if (dark(y * w + x)) black++; total++ }
        for (y in 2 until h - 2) for (x in 0..1) { if (dark(y * w + x)) black++; total++ }
        for (y in 2 until h - 2) for (x in w - 2 until w) { if (dark(y * w + x)) black++; total++ }
        val ratio = if (total > 0) black.toDouble() / total * 100 else 0.0
        if (ratio >= ignoreBubble && ratio <= 100 - ignoreBubble) return true
        return checkColor(px)
    }

    /** 彩色文字判定（ported from utils/bubble.py:check_color）：>10 個非灰階像素 → True。 */
    private fun checkColor(px: IntArray): Boolean {
        var n = 0
        for (p in px) {
            val r = (p shr 16) and 0xFF
            val g = (p shr 8) and 0xFF
            val b = p and 0xFF
            val gray = 0.299 * r + 0.587 * g + 0.114 * b
            val dr = r - gray
            val dg = g - gray
            val db = b - gray
            if (dr * dr + dg * dg + db * db > 100) { n++; if (n > 10) return true }
        }
        return false
    }

    override fun close() {
        if (ncnnHandle != 0L) {
            NcnnBackend.releaseNet(ncnnHandle)
            ncnnHandle = 0L
        }
    }

    companion object {
        private const val TAG = "Ocr"
        private const val NUM_THREADS = 4
        private const val BLANK = 0
        private const val PAD_MARGIN = 16  // 每條右側白邊：讓 CTC 有 context、不截尾字
        private const val D_MODEL = 320   // transformer 寬度（PE 列寬）
        private const val PE_ROWS = 2048  // 上游 PositionalEncoding(max_len=2048)；共用表一次算好（2.6MB）
        private const val MIXED_SUFFIX = "_mixed.ncnn.param"

        /** 共用的正弦位置表（[PE_ROWS]×[D_MODEL]，模型第二個輸入 `in1` 的來源），第一次用到才算。 */
        private val PE: FloatArray by lazy { sinusoidalPe(PE_ROWS) }

        /**
         * 正弦位置編碼（model_48px_ctc.py:PositionalEncoding 逐式照抄）：pe[t,2i]=sin(t·div_i)、pe[t,2i+1]=cos(t·div_i)、
         * div_i=exp(2i·(−ln 10000/320))。回 [rows×320] row-major。parity fixture（ocr/strip0_pe.bin）逐值對過。
         */
        internal fun sinusoidalPe(rows: Int): FloatArray {
            val pe = FloatArray(rows * D_MODEL)
            val half = D_MODEL / 2
            val div = DoubleArray(half) { i -> Math.exp((2 * i) * (-Math.log(10000.0) / D_MODEL)) }
            for (t in 0 until rows) {
                val base = t * D_MODEL
                for (i in 0 until half) {
                    val a = t * div[i]
                    pe[base + 2 * i] = Math.sin(a).toFloat()
                    pe[base + 2 * i + 1] = Math.cos(a).toFloat()
                }
            }
            return pe
        }

        /**
         * greedy CTC 收合（對齊 decode_ctc_top1）：[idx]=每時步 argmax、[logp]=該時步 top-1 log_softmax；
         * blank=0、與前一時步相同者收掉，留下的字查 [dictionary]（`<SP>`→空白）；prob＝exp(留下字的 logp 平均)。
         * idx／logp 由 JNI（[NcnnBackend.ocrCtc]：逐時步 argmax + top-1 log_softmax）算；JVM parity 測試拿 fixture 的逐時步值直接餵這裡。
         */
        internal fun ctcCollapse(idx: IntArray, logp: FloatArray, t: Int, dictionary: List<String>): Pair<String, Float> {
            val sb = StringBuilder()
            var last = BLANK
            var logpSum = 0.0
            var nChars = 0
            for (ti in 0 until t) {
                val best = idx[ti]
                if (best != last && best != BLANK) {
                    val ch = dictionary[best]
                    sb.append(if (ch == "<SP>") " " else ch)
                    logpSum += logp[ti]
                    nChars++
                }
                last = best
            }
            val prob = if (nChars > 0) Math.exp(logpSum / nChars).toFloat() else 0f
            return sb.toString() to prob
        }
    }
}
