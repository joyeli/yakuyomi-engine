package li.joye.yakuyomi.engine

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.util.Log
import kotlin.math.abs
import kotlin.math.roundToInt
import kotlinx.coroutines.coroutineScope

/**
 * 去字（text removal）。真機 A/B 定案＝兩門別，皆純 NCNN（`.param`/`.bin`）：
 *  - **boxfill（快速去字）**：逐區把去字遮罩像素平塗成背景色。瞬間、不跑模型；乾淨白泡完美、壓畫面是平色塊。
 *  - **aot（AI 去字·預設）**：AOT-GAN（m-i-t 漫畫權重）整頁一次縮到 [InpainterConfig.tileSize]（768）重建背景、全區重建。
 *    全卷積 → 任意尺寸；768＝畫質/記憶體/藏在翻譯下（§8 去字‖翻譯重疊）的甜蜜點。
 *
 * AOT I/O 契約（對齊 parity/inpaint_parity.py）：img∈[-1,1] 且洞歸零（m-i-t `img*(1-mask)`）、mask∈{0,1}(1=擦)、輸出∈[-1,1]。
 * LaMa（整頁縮 512 必糊）與 AOT 逐格（原生解析度·CPU 太貴）皆已退役移除；GPU/Vulkan 實測算不對 AOT-GAN（見 memory ncnn-vulkan-fp16）。
 */
class Inpainter(
    modelPath: String,
    private val cfg: InpainterConfig = InpainterConfig(),
) : AutoCloseable {

    private var ncnnHandle: Long = 0L
    /** 實際生效的後端；無 adb 時由呼叫端寫進 log/圖確認。 */
    val ep: String = "NCNN-CPU"

    init {
        check(modelPath.endsWith(".param")) { "去字需 NCNN `.param` 模型（AOT-GAN）：$modelPath" }
        check(NcnnBackend.available) { "NCNN 原生庫未載入，無法去字" }
        val bin = modelPath.removeSuffix(".param") + ".bin"
        ncnnHandle = NcnnBackend.createNet(modelPath, bin)
        check(ncnnHandle != 0L) { "NCNN AOT 模型載入失敗：$modelPath" }
    }

    suspend fun inpaint(page: Bitmap, regions: List<TextRegion>, textMask: Bitmap): Bitmap = coroutineScope {
        val w = page.width
        val h = page.height
        val result = page.copy(Bitmap.Config.ARGB_8888, true)
        // seg 細筆畫遮罩 ∩ 保留區 bbox（外擴 bboxPad）再膨脹：只動筆畫、限制在翻譯過的區（SFX/未譯留原圖，§11）。
        val maskPx = buildSegMask(regions, textMask, w, h)

        if (cfg.method == "boxfill") {
            // 逐區平塗背景色：白泡乾淨無殘留、忙碌區是平色塊（要品質用 aot）。
            val px = IntArray(w * h); result.getPixels(px, 0, w, 0, 0, w, h)
            val tightPx = IntArray(w * h); textMask.getPixels(tightPx, 0, w, 0, 0, w, h)
            // 單色／漸層底（任何顏色）同 aot 路徑：行框＋光暈塗背景模型、量氣泡內部供排版；其餘才平塗平均色。
            val flatModels = ArrayList<Pair<TextRegion, BgModel>>()
            for (r in regions) {
                val s = bgStats(px, tightPx, r, w, h)
                r.onArt = false; r.dbgStd = s.std; r.dbgWhite = s.meanLum // dbg 值給 sandbox 去背比較標框
                val model = chooseModel(s, fitPlane(px, tightPx, r, w, h), r)
                if (model != null) {
                    fillSmooth(page, result, px, tightPx, r, model, w, h)
                    flatModels += r to model
                } else {
                    flatFill(result, maskPx, r, s.color, cfg.bboxPad, w, h)
                }
            }
            if (flatModels.isNotEmpty()) {
                val cleanPx = IntArray(w * h); result.getPixels(cleanPx, 0, w, 0, 0, w, h)
                val holes = ArrayList<Int>()
                for ((r, model) in flatModels) {
                    holes.clear()
                    r.layoutBoxes = bubbleInterior(cleanPx, r, model, w, h, holes)
                    for (i in holes) cleanPx[i] = model.at(i % w, i / w)
                    if (holes.isNotEmpty()) result.setPixels(cleanPx, 0, w, 0, 0, w, h)
                }
            }
            return@coroutineScope result
        }

        // aot（預設）：
        //  1. 背景近乎單色的區（乾淨白泡、色框、純色底，任何顏色）→ 只把文字行框（略外擴）平塗成背景眾數色：瞬間、無殘影、
        //     不碰氣泡外框。onArt=false → Renderer 依背景亮度選黑／白字。
        //  2. 其餘（壓在畫面上）→ AOT-GAN 重建；onArt=true → 黑字粗白邊。
        //     一般頁整頁一次（縮到 tileSize）；長條頁（條漫）整頁縮會把字壓扁成橫紋 → 改在各區周圍取「短邊見方」視窗跑。
        val px = IntArray(w * h); page.getPixels(px, 0, w, 0, 0, w, h)
        val tightPx = IntArray(w * h); textMask.getPixels(tightPx, 0, w, 0, 0, w, h)
        val busy = ArrayList<TextRegion>()
        val flat = ArrayList<Pair<TextRegion, BgModel>>()
        for (r in regions) {
            val s = bgStats(px, tightPx, r, w, h)
            r.dbgStd = s.std; r.dbgWhite = s.meanLum
            // 背景模型（單色或平滑漸層，任何顏色）；都不像 → 壓在畫面上走 AOT
            val model = chooseModel(s, fitPlane(px, tightPx, r, w, h), r)
            if (model != null) {
                r.onArt = false
                fillSmooth(page, result, px, tightPx, r, model, w, h)
                flat += r to model
            } else {
                r.onArt = true
                busy += r
            }
        }
        // 氣泡內部在「去字後」量：字已塗成背景色，內部是一整片單色／漸層（原圖上量會被漏網的筆畫切斷）。
        if (flat.isNotEmpty()) {
            val cleanPx = IntArray(w * h); result.getPixels(cleanPx, 0, w, 0, 0, w, h)
            val holes = ArrayList<Int>()
            for ((r, model) in flat) {
                holes.clear()
                r.layoutBoxes = bubbleInterior(cleanPx, r, model, w, h, holes)
                for (i in holes) cleanPx[i] = model.at(i % w, i / w)
                if (holes.isNotEmpty()) result.setPixels(cleanPx, 0, w, 0, 0, w, h)
            }
        }
        if (busy.isEmpty()) return@coroutineScope result
        val busyMask = if (busy.size == regions.size) maskPx else buildSegMask(busy, textMask, w, h)
        val maskBmp = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
        maskBmp.setPixels(busyMask, 0, w, 0, 0, w, h)
        if (maxOf(w, h) > TALL_RATIO * minOf(w, h)) {
            inpaintWindows(page, result, busy, busyMask, maskBmp, w, h)
        } else {
            runWholeAot(page, maskBmp, w, h)?.let { compositePixels(result, busyMask, it) }
        }
        maskBmp.recycle()
        result
    }

    /** 背景色模型：每個通道 c = k + gx·x + gy·y（單色底＝gx＝gy＝0；漸層底＝平面擬合）。 */
    private class BgModel(val k: DoubleArray, val gx: DoubleArray, val gy: DoubleArray, val rms: Double) {
        fun at(x: Int, y: Int): Int {
            val r = (k[0] + gx[0] * x + gy[0] * y).roundToInt().coerceIn(0, 255)
            val g = (k[1] + gx[1] * x + gy[1] * y).roundToInt().coerceIn(0, 255)
            val b = (k[2] + gx[2] * x + gy[2] * y).roundToInt().coerceIn(0, 255)
            return Color.rgb(r, g, b)
        }
        companion object {
            fun constant(c: Int) = BgModel(
                doubleArrayOf(((c shr 16) and 0xFF).toDouble(), ((c shr 8) and 0xFF).toDouble(), (c and 0xFF).toDouble()),
                DoubleArray(3), DoubleArray(3), 2.0,
            )
        }
    }

    /**
     * 選背景模型：
     *  - 外圍平面擬合成功（真正的背景，量在字的光暈之外）→ 用它；若它幾乎是單色、且行框內眾數色與它相近 → 改用眾數的精確色
     *    （平均值會差 1～2 階，塗出來看得到一塊色差）。
     *  - 擬合到的顏色接近筆畫亮度（取樣跑進黑環、深色外框）→ 視為擬合失敗。
     *  - 擬合失敗但行框內眾數色佔多數（氣泡緊貼著字、外圍取樣跑到泡外）→ 用眾數色。
     *  - 都不成立 → null（壓在畫面上）。
     */
    private fun chooseModel(s: BgStat, planeIn: BgModel?, r: TextRegion): BgModel? {
        var plane = planeIn
        // 背景不可能跟字同色（字會看不見）：擬合到的若接近筆畫亮度＝取樣跑進了深色外框／黑環 → 不採用
        if (plane != null && s.textLum >= 0f) {
            val e = plane.at(((r.x0 + r.x1) / 2).toInt(), ((r.y0 + r.y1) / 2).toInt())
            val el = (299 * ((e shr 16) and 0xFF) + 587 * ((e shr 8) and 0xFF) + 114 * (e and 0xFF)) / 1000f
            if (abs(el - s.textLum) < MIN_TEXT_CONTRAST) plane = null
        }
        if (plane != null) {
            val span = maxOf(r.x1 - r.x0, r.y1 - r.y0).toDouble()
            val nearlyFlat = (0..2).all { abs(plane.gx[it]) * span < 4 && abs(plane.gy[it]) * span < 4 }
            if (nearlyFlat && s.modeFrac >= FLAT_FRAC) {
                val e = plane.at(((r.x0 + r.x1) / 2).toInt(), ((r.y0 + r.y1) / 2).toInt())
                if (residual(s.modeColor, BgModel.constant(e), 0, 0) <= 8.0) return BgModel.constant(s.modeColor)
            }
            return plane
        }
        return if (s.modeFrac >= FLAT_FRAC) BgModel.constant(s.modeColor) else null
    }

    /**
     * 平滑漸層底：在文字外圍一圈（bbox 外擴 0.55～1.4 行高、去掉筆畫；內圈可能落在字的光暈裡，靠剔除離群處理）取樣，逐通道擬合平面 c = k + gx·x + gy·y，
     * 剔除離群點（外框、畫面細節）重擬兩次。內群夠多（≥ [PLANE_INLIER]）、殘差夠小（≤ [PLANE_RMS]）、且上下左右都有內群
     * （背景真的包住文字）才算平滑底；否則回 null（交給 AOT）。
     */
    private fun fitPlane(px: IntArray, tightPx: IntArray, r: TextRegion, w: Int, h: Int): BgModel? {
        val lh = r.lineHeight.coerceAtLeast(8f)
        val i0 = lh * 0.55f; val o0 = lh * 1.4f
        val ox0 = (r.x0 - o0).toInt().coerceIn(0, w - 1); val ox1 = (r.x1 + o0).toInt().coerceIn(ox0 + 1, w)
        val oy0 = (r.y0 - o0).toInt().coerceIn(0, h - 1); val oy1 = (r.y1 + o0).toInt().coerceIn(oy0 + 1, h)
        val ix0 = r.x0 - i0; val ix1 = r.x1 + i0; val iy0 = r.y0 - i0; val iy1 = r.y1 + i0
        // 只取「與文字底色平滑相連」的外圍像素：從字間空白出發往外長，相鄰亮度差 ≤ [CONNECT_STEP] 才走得過去。
        // 氣泡外框是陡變 → 擋住（小泡外的畫面不會被當背景）；字的光暈是緩變 → 走得過（光暈外的漸層底才取得到）。
        val bw = ox1 - ox0; val bh = oy1 - oy0
        fun lumAt(x: Int, y: Int): Int { val v = px[y * w + x]; return (299 * ((v shr 16) and 0xFF) + 587 * ((v shr 8) and 0xFF) + 114 * (v and 0xFF)) / 1000 }
        val reach = BooleanArray(bw * bh)
        val stack = IntArray(bw * bh); var sp = 0
        val sx0 = r.x0.toInt().coerceIn(ox0, ox1 - 1); val sx1 = r.x1.toInt().coerceIn(sx0 + 1, ox1)
        val sy0 = r.y0.toInt().coerceIn(oy0, oy1 - 1); val sy1 = r.y1.toInt().coerceIn(sy0 + 1, oy1)
        for (yy in sy0 until sy1) for (xx in sx0 until sx1) {
            if ((tightPx[yy * w + xx] and 0xFF) > 127) continue
            val li = (yy - oy0) * bw + (xx - ox0)
            if (!reach[li]) { reach[li] = true; stack[sp++] = li }
        }
        while (sp > 0) {
            val li = stack[--sp]; val lx = li % bw; val ly = li / bw
            val l0 = lumAt(ox0 + lx, oy0 + ly)
            for (n in intArrayOf(if (lx > 0) li - 1 else -1, if (lx < bw - 1) li + 1 else -1, if (ly > 0) li - bw else -1, if (ly < bh - 1) li + bw else -1)) {
                if (n < 0 || reach[n]) continue
                val nx = ox0 + n % bw; val ny = oy0 + n / bw
                if ((tightPx[ny * w + nx] and 0xFF) > 127) continue
                if (abs(lumAt(nx, ny) - l0) > CONNECT_STEP) continue
                reach[n] = true; stack[sp++] = n
            }
        }
        val xs = ArrayList<Int>(); val ys = ArrayList<Int>()
        val step = if ((ox1 - ox0) * (oy1 - oy0) > 200_000) 3 else 2
        var y = oy0
        while (y < oy1) {
            var x = ox0
            while (x < ox1) {
                if (!(x >= ix0 && x < ix1 && y >= iy0 && y < iy1) && reach[(y - oy0) * bw + (x - ox0)]) { xs += x; ys += y }
                x += step
            }
            y += step
        }
        if (xs.size < 40) return null
        var inl = BooleanArray(xs.size) { true }
        var model: BgModel? = null
        repeat(3) { iter ->
            val m = solvePlane(px, xs, ys, inl, w) ?: return null
            // 殘差（三通道最大值）→ 依中位數門檻剔除離群
            val res = DoubleArray(xs.size) { i -> residual(px[ys[i] * w + xs[i]], m, xs[i], ys[i]) }
            val sorted = res.sorted()
            val med = sorted[sorted.size / 2]
            val cut = maxOf(PLANE_RMS * 1.5, med * 3.0)
            inl = BooleanArray(xs.size) { res[it] <= cut }
            var ss = 0.0; var n = 0
            for (i in res.indices) if (inl[i]) { ss += res[i] * res[i]; n++ }
            model = BgModel(m.k, m.gx, m.gy, if (n > 0) kotlin.math.sqrt(ss / n) else 99.0)
            if (iter == 2 && n < xs.size * PLANE_INLIER) return null
        }
        val mm = model ?: return null
        if (mm.rms > PLANE_RMS) return null
        // 文字四周都要有背景內群（否則多半是文字壓在畫面邊上）
        var up = 0; var dn = 0; var lf = 0; var rt = 0
        for (i in xs.indices) if (inl[i]) {
            if (ys[i] < r.y0) up++; if (ys[i] > r.y1) dn++; if (xs[i] < r.x0) lf++; if (xs[i] > r.x1) rt++
        }
        val need = 6
        if (listOf(up, dn, lf, rt).count { it >= need } < 3) return null
        return mm
    }

    private fun residual(p: Int, m: BgModel, x: Int, y: Int): Double {
        val e = m.at(x, y)
        return maxOf(
            abs(((p shr 16) and 0xFF) - ((e shr 16) and 0xFF)),
            abs(((p shr 8) and 0xFF) - ((e shr 8) and 0xFF)),
            abs((p and 0xFF) - (e and 0xFF)),
        ).toDouble()
    }

    /** 最小平方擬合平面（三通道共用設計矩陣 [1, x, y]）；只用 [use] 為 true 的樣本。 */
    private fun solvePlane(px: IntArray, xs: List<Int>, ys: List<Int>, use: BooleanArray, w: Int): BgModel? {
        var n = 0.0; var sx = 0.0; var sy = 0.0; var sxx = 0.0; var syy = 0.0; var sxy = 0.0
        val sc = DoubleArray(3); val sxc = DoubleArray(3); val syc = DoubleArray(3)
        for (i in xs.indices) {
            if (!use[i]) continue
            val x = xs[i].toDouble(); val y = ys[i].toDouble()
            val p = px[ys[i] * w + xs[i]]
            val c = doubleArrayOf(((p shr 16) and 0xFF).toDouble(), ((p shr 8) and 0xFF).toDouble(), (p and 0xFF).toDouble())
            n++; sx += x; sy += y; sxx += x * x; syy += y * y; sxy += x * y
            for (k in 0..2) { sc[k] += c[k]; sxc[k] += x * c[k]; syc[k] += y * c[k] }
        }
        if (n < 10) return null
        // 解 [[n,sx,sy],[sx,sxx,sxy],[sy,sxy,syy]]·[k,gx,gy] = [sc,sxc,syc]（Cramer）
        val det = n * (sxx * syy - sxy * sxy) - sx * (sx * syy - sxy * sy) + sy * (sx * sxy - sxx * sy)
        if (abs(det) < 1e-6) return null
        val k = DoubleArray(3); val gx = DoubleArray(3); val gy = DoubleArray(3)
        for (c in 0..2) {
            val a = sc[c]; val b = sxc[c]; val d = syc[c]
            k[c] = (a * (sxx * syy - sxy * sxy) - sx * (b * syy - sxy * d) + sy * (b * sxy - sxx * d)) / det
            gx[c] = (n * (b * syy - sxy * d) - a * (sx * syy - sxy * sy) + sy * (sx * d - b * sy)) / det
            gy[c] = (n * (sxx * d - b * sxy) - sx * (sx * d - b * sy) + a * (sx * sxy - sxx * sy)) / det
        }
        return BgModel(k, gx, gy, 0.0)
    }

    /**
     * 平滑底區去字：
     *  1. 每個文字行框（四邊形，各邊外擴行高×[QUAD_GROW]＋2px）＝必塗區。
     *  2. 從必塗區往外長「字的光暈」：與背景模型差、且方向與字相反（黑字的白暈／白字的黑暈）的像素，最多外擴約半行高。
     *     方向相反才長 ⇒ 深色字旁的深色外框不會被吃掉。
     *  3. 上述像素一律塗成背景模型在該點的顏色（單色或漸層）。
     */
    private fun fillSmooth(page: Bitmap, result: Bitmap, px: IntArray, tightPx: IntArray, region: TextRegion, model: BgModel, w: Int, h: Int) {
        val lh = region.lineHeight.coerceAtLeast(8f)
        val ext = (lh * 0.9f + 6f).toInt()
        val ax0 = (region.x0.toInt() - ext).coerceIn(0, w - 1); val ax1 = (region.x1.toInt() + ext).coerceIn(ax0 + 1, w)
        val ay0 = (region.y0.toInt() - ext).coerceIn(0, h - 1); val ay1 = (region.y1.toInt() + ext).coerceIn(ay0 + 1, h)
        val aw = ax1 - ax0; val ah = ay1 - ay0
        // 1. 行框（外擴）光柵化到區域遮罩
        val qBmp = Bitmap.createBitmap(aw, ah, Bitmap.Config.ARGB_8888)
        val c = Canvas(qBmp)
        val p = Paint().apply { color = Color.WHITE; style = Paint.Style.FILL; isAntiAlias = false }
        for (line in region.lines) {
            val q = line.quad
            if (q.size < 4) continue
            val cx = q.sumOf { it.x.toDouble() }.toFloat() / q.size
            val cy = q.sumOf { it.y.toDouble() }.toFloat() / q.size
            // 沿四邊形自身兩軸外擴（u＝q0→q1、v＝q1→q2），長條行的上下緣也確實外推
            val ux = q[1].x - q[0].x; val uy = q[1].y - q[0].y
            val vx = q[2].x - q[1].x; val vy = q[2].y - q[1].y
            val ul = kotlin.math.hypot(ux.toDouble(), uy.toDouble()).toFloat().coerceAtLeast(1f)
            val vl = kotlin.math.hypot(vx.toDouble(), vy.toDouble()).toFloat().coerceAtLeast(1f)
            val grow = minOf(ul, vl) * QUAD_GROW + 2f
            val path = android.graphics.Path()
            q.forEachIndexed { i, pt ->
                val dx = pt.x - cx; val dy = pt.y - cy
                val su = if (dx * ux + dy * uy >= 0) 1f else -1f
                val sv = if (dx * vx + dy * vy >= 0) 1f else -1f
                val x = pt.x + grow * (su * ux / ul + sv * vx / vl) - ax0
                val y = pt.y + grow * (su * uy / ul + sv * vy / vl) - ay0
                if (i == 0) path.moveTo(x, y) else path.lineTo(x, y)
            }
            path.close()
            c.drawPath(path, p)
        }
        val qm = IntArray(aw * ah); qBmp.getPixels(qm, 0, aw, 0, 0, aw, ah); qBmp.recycle()
        val fill = BooleanArray(aw * ah) { (qm[it] and 0xFF) > 127 }
        // 2. 字是比背景深還是淺（筆畫亮度中位數 vs 背景模型）
        fun lum(v: Int) = 0.299 * ((v shr 16) and 0xFF) + 0.587 * ((v shr 8) and 0xFF) + 0.114 * (v and 0xFF)
        val tl = ArrayList<Double>(); var bgSum = 0.0; var bgN = 0
        for (yy in 0 until ah) for (xx in 0 until aw) {
            if (!fill[yy * aw + xx]) continue
            val gx = ax0 + xx; val gy = ay0 + yy
            if ((tightPx[gy * w + gx] and 0xFF) > 127) tl += lum(px[gy * w + gx]) else { bgSum += lum(model.at(gx, gy)); bgN++ }
        }
        // 字比背景深或淺：看筆畫亮度的兩端（10%／90% 分位）哪一端離背景遠（遮罩偏寬，中位數不可靠）
        val textDark = if (tl.isEmpty() || bgN == 0) {
            true
        } else {
            val st = tl.sorted(); val bgL = bgSum / bgN
            abs(st[st.size / 10] - bgL) >= abs(st[st.size * 9 / 10] - bgL)
        }
        val tol = maxOf(HALO_TOL, model.rms * 2.0)
        val maxD = (lh * 0.8f).toInt() + 4
        val dist = IntArray(aw * ah) { if (fill[it]) 0 else -1 }
        val queue = IntArray(aw * ah); var qh = 0; var qt = 0
        for (i in fill.indices) if (fill[i]) queue[qt++] = i
        while (qh < qt) {
            val i = queue[qh++]; val d = dist[i]
            if (d >= maxD) continue
            val xx = i % aw; val yy = i / aw
            for (n in intArrayOf(if (xx > 0) i - 1 else -1, if (xx < aw - 1) i + 1 else -1, if (yy > 0) i - aw else -1, if (yy < ah - 1) i + aw else -1)) {
                if (n < 0 || dist[n] >= 0) continue
                val gx = ax0 + n % aw; val gy = ay0 + n / aw
                val dev = lum(px[gy * w + gx]) - lum(model.at(gx, gy))
                if ((textDark && dev > tol) || (!textDark && dev < -tol)) { dist[n] = d + 1; fill[n] = true; queue[qt++] = n }
            }
        }
        // 3. 塗成背景模型色；外緣 [FEATHER] px 漸變混合（光暈尾巴、JPEG 色階不留一圈細邊）
        val cur = IntArray(aw * ah); result.getPixels(cur, 0, aw, ax0, ay0, aw, ah)
        for (i in fill.indices) if (fill[i]) cur[i] = model.at(ax0 + i % aw, ay0 + i / aw)
        val ring = IntArray(aw * ah) { if (fill[it]) 0 else -1 }
        qh = 0; qt = 0
        for (i in fill.indices) if (fill[i]) queue[qt++] = i
        while (qh < qt) {
            val i = queue[qh++]; val d = ring[i]
            if (d >= FEATHER) continue
            val xx = i % aw; val yy = i / aw
            for (n in intArrayOf(if (xx > 0) i - 1 else -1, if (xx < aw - 1) i + 1 else -1, if (yy > 0) i - aw else -1, if (yy < ah - 1) i + aw else -1)) {
                if (n < 0 || ring[n] >= 0) continue
                ring[n] = d + 1; queue[qt++] = n
                val gx = ax0 + n % aw; val gy = ay0 + n / aw
                // 只混「接近背景」的像素（外框等深色結構原樣保留）
                val o = cur[n]; val e = model.at(gx, gy)
                if (residual(o, BgModel.constant(e), 0, 0) > 4 * HALO_TOL + model.rms * 4) continue
                val t = (d + 1).toFloat() / (FEATHER + 1) // 越外越接近原圖
                cur[n] = Color.rgb(
                    (((o shr 16) and 0xFF) * t + ((e shr 16) and 0xFF) * (1 - t)).roundToInt(),
                    (((o shr 8) and 0xFF) * t + ((e shr 8) and 0xFF) * (1 - t)).roundToInt(),
                    ((o and 0xFF) * t + (e and 0xFF) * (1 - t)).roundToInt(),
                )
            }
        }
        result.setPixels(cur, 0, aw, ax0, ay0, aw, ah)
    }

    /**
     * 量單色底區的「氣泡內部」並回傳幾個內接矩形候選（由扁到高）：
     *  1. 從文字中心附近洪水填充「背景色」像素（[px] 是去字後的圖），範圍限在 bbox 外擴上限內（旁白落在整片白底時不會無限大）。
     *  2. 以文字中心為軸，對不同半高 k 取「上下各 k 列都連通」的最寬對稱矩形。
     */
    private fun bubbleInterior(px: IntArray, r: TextRegion, model: BgModel, w: Int, h: Int, holes: MutableList<Int>): List<FloatArray> {
        val lh = r.lineHeight.coerceAtLeast(8f)
        val tx0 = r.x0.toInt().coerceIn(0, w - 1); val tx1 = r.x1.toInt().coerceIn(tx0 + 1, w)
        val ty0 = r.y0.toInt().coerceIn(0, h - 1); val ty1 = r.y1.toInt().coerceIn(ty0 + 1, h)
        val gx = ((tx1 - tx0) * 0.8f + lh * 3f).toInt(); val gy = ((ty1 - ty0) * 1.0f + lh * 3f).toInt()
        val lx0 = (tx0 - gx).coerceAtLeast(0); val lx1 = (tx1 + gx).coerceAtMost(w)
        val ly0 = (ty0 - gy).coerceAtLeast(0); val ly1 = (ty1 + gy).coerceAtMost(h)
        val lw = lx1 - lx0; val lhh = ly1 - ly0
        fun ok(x: Int, y: Int): Boolean {
            val p = px[y * w + x]
            val e = model.at(x, y)
            return abs(((p shr 16) and 0xFF) - ((e shr 16) and 0xFF)) <= BUBBLE_TOL &&
                abs(((p shr 8) and 0xFF) - ((e shr 8) and 0xFF)) <= BUBBLE_TOL &&
                abs((p and 0xFF) - (e and 0xFF)) <= BUBBLE_TOL
        }
        // 1. 洪水填充（區域內座標）
        val inside = BooleanArray(lw * lhh)
        val stack = IntArray(lw * lhh); var sp = 0
        for (ln in r.lines) {
            val sx = (ln.quad.sumOf { it.x.toDouble() } / ln.quad.size).toInt().coerceIn(lx0, lx1 - 1)
            val sy = (ln.quad.sumOf { it.y.toDouble() } / ln.quad.size).toInt().coerceIn(ly0, ly1 - 1)
            val li = (sy - ly0) * lw + (sx - lx0)
            if (!inside[li] && ok(sx, sy)) { inside[li] = true; stack[sp++] = li }
        }
        while (sp > 0) {
            val li = stack[--sp]; val x = li % lw; val y = li / lw
            if (x > 0) { val n = li - 1; if (!inside[n] && ok(lx0 + x - 1, ly0 + y)) { inside[n] = true; stack[sp++] = n } }
            if (x < lw - 1) { val n = li + 1; if (!inside[n] && ok(lx0 + x + 1, ly0 + y)) { inside[n] = true; stack[sp++] = n } }
            if (y > 0) { val n = li - lw; if (!inside[n] && ok(lx0 + x, ly0 + y - 1)) { inside[n] = true; stack[sp++] = n } }
            if (y < lhh - 1) { val n = li + lw; if (!inside[n] && ok(lx0 + x, ly0 + y + 1)) { inside[n] = true; stack[sp++] = n } }
        }
        // 1b. 補洞：完全被內部包圍、靠近文字、面積小的非背景塊（行框沒蓋到的「……」、字的殘渣）→ 記進 holes 由呼叫端塗成背景。
        //     與外界相連的（氣泡外框、外面的畫）從限制框邊緣洪水填充得到，一律不動。
        // 只在「封閉」的氣泡內補洞：內部碰到限制框邊緣＝開放底（旁白在整片白頁、或漏出氣泡），那時被包圍的可能是別的氣泡外框。
        var closed = true
        for (x in 0 until lw) if (inside[x] || inside[(lhh - 1) * lw + x]) { closed = false; break }
        if (closed) for (y in 0 until lhh) if (inside[y * lw] || inside[y * lw + lw - 1]) { closed = false; break }
        val outside = BooleanArray(lw * lhh)
        sp = 0
        fun pushOut(li: Int) { if (!inside[li] && !outside[li]) { outside[li] = true; stack[sp++] = li } }
        for (x in 0 until lw) { pushOut(x); pushOut((lhh - 1) * lw + x) }
        for (y in 0 until lhh) { pushOut(y * lw); pushOut(y * lw + lw - 1) }
        while (sp > 0) {
            val li = stack[--sp]; val x = li % lw; val y = li / lw
            if (x > 0) pushOut(li - 1)
            if (x < lw - 1) pushOut(li + 1)
            if (y > 0) pushOut(li - lw)
            if (y < lhh - 1) pushOut(li + lw)
        }
        val seen = BooleanArray(lw * lhh)
        val maxHole = (lh * lh * 2f).toInt()
        val nx0 = (tx0 - lh * 1.5f).toInt(); val nx1 = (tx1 + lh * 1.5f).toInt()
        val ny0 = (ty0 - lh * 1.5f).toInt(); val ny1 = (ty1 + lh * 1.5f).toInt()
        val comp = ArrayList<Int>()
        for (start in 0 until lw * lhh) {
            if (!closed) break
            if (inside[start] || outside[start] || seen[start]) continue
            comp.clear(); seen[start] = true; stack[0] = start; sp = 1
            var near = true
            var cx0 = lw; var cx1 = 0; var cy0 = lhh; var cy1 = 0
            while (sp > 0) {
                val li = stack[--sp]; comp += li
                val x = li % lw; val y = li / lw
                cx0 = minOf(cx0, x); cx1 = maxOf(cx1, x); cy0 = minOf(cy0, y); cy1 = maxOf(cy1, y)
                val gx = lx0 + x; val gy = ly0 + y
                if (gx < nx0 || gx >= nx1 || gy < ny0 || gy >= ny1) near = false
                for (n in intArrayOf(if (x > 0) li - 1 else -1, if (x < lw - 1) li + 1 else -1, if (y > 0) li - lw else -1, if (y < lhh - 1) li + lw else -1)) {
                    if (n >= 0 && !inside[n] && !outside[n] && !seen[n]) { seen[n] = true; stack[sp++] = n }
                }
            }
            // 小塊（長寬都 ≤ 一行字高）：點點、殘渣；外框、圖案等大塊不動
            if (near && comp.size <= maxHole && cx1 - cx0 < lh && cy1 - cy0 < lh) {
                for (li in comp) { holes += (ly0 + li / lw) * w + lx0 + li % lw; inside[li] = true }
            }
        }
        // 2. 以文字中心 (cx, cy) 為軸，每列向左右延伸到第一個非內部像素
        val cx = ((tx0 + tx1) / 2 - lx0).coerceIn(0, lw - 1)
        val cy = ((ty0 + ty1) / 2 - ly0).coerceIn(0, lhh - 1)
        val left = IntArray(lhh) { -1 }; val right = IntArray(lhh) { -1 }
        for (y in 0 until lhh) {
            if (!inside[y * lw + cx]) continue
            var a = cx; while (a > 0 && inside[y * lw + a - 1]) a--
            var b = cx; while (b < lw - 1 && inside[y * lw + b + 1]) b++
            left[y] = a; right[y] = b + 1
        }
        val out = ArrayList<FloatArray>()
        var l = 0; var rr = lw
        var k = 0
        val minK = (lh * 0.5f).toInt().coerceAtLeast(3)   // 最矮候選＝約一行高（扁寬框放短句）
        val step = (lh / 6f).toInt().coerceAtLeast(2)
        while (cy - k >= 0 && cy + k < lhh && left[cy - k] >= 0 && left[cy + k] >= 0) {
            l = maxOf(l, left[cy - k], left[cy + k]); rr = minOf(rr, right[cy - k], right[cy + k])
            if (rr - l < lh) break
            // 不必對稱於文字中心（框可偏向氣泡較寬的一側），但中心離原文中心不超過 [off]——開放白底的旁白才不會漂走
            if (k >= minK && (k - minK) % step == 0) {
                var a = l; var b = rr
                val off = (tx1 - tx0) * 0.15f + lh
                val mid = (a + b) / 2f
                if (mid > cx + off) b = (2 * (cx + off) - a).toInt() else if (mid < cx - off) a = (2 * (cx - off) - b).toInt()
                if (b - a >= lh) out += floatArrayOf((lx0 + a).toFloat(), (ly0 + cy - k).toFloat(), (lx0 + b).toFloat(), (ly0 + cy + k + 1).toFloat())
            }
            k++
        }
        if (out.isEmpty()) out += floatArrayOf(r.x0, r.y0, r.x1, r.y1)
        return out
    }

    /** 長條頁的 AOT：各區周圍取「短邊×短邊」視窗（區比短邊大才放大）跑 AOT，只把該區（bbox 外擴）內的遮罩像素貼回；同視窗涵蓋的區合併推論。 */
    private fun inpaintWindows(
        page: Bitmap, result: Bitmap, busy: List<TextRegion>, maskPx: IntArray, maskBmp: Bitmap, w: Int, h: Int,
    ) {
        val pad = cfg.bboxPad + (cfg.maskDilate / 2f).roundToInt() + 1
        val side = minOf(w, h)
        class Win(val x0: Int, val y0: Int, val ww: Int, val wh: Int, val members: MutableList<TextRegion> = ArrayList())
        val wins = ArrayList<Win>()
        for (r in busy.sortedBy { it.y0 }) {
            val rx0 = (r.x0.toInt() - pad).coerceAtLeast(0); val ry0 = (r.y0.toInt() - pad).coerceAtLeast(0)
            val rx1 = (r.x1.toInt() + pad).coerceAtMost(w); val ry1 = (r.y1.toInt() + pad).coerceAtMost(h)
            val host = wins.firstOrNull { rx0 >= it.x0 && ry0 >= it.y0 && rx1 <= it.x0 + it.ww && ry1 <= it.y0 + it.wh }
            if (host != null) { host.members += r; continue }
            val ww = maxOf(side, rx1 - rx0 + 2 * pad).coerceAtMost(w)
            val wh = maxOf(side, ry1 - ry0 + 2 * pad).coerceAtMost(h)
            val cx = (rx0 + rx1) / 2; val cy = (ry0 + ry1) / 2
            val x0 = (cx - ww / 2).coerceIn(0, w - ww); val y0 = (cy - wh / 2).coerceIn(0, h - wh)
            wins += Win(x0, y0, ww, wh).also { it.members += r }
        }
        for (win in wins) {
            val cropPage = Bitmap.createBitmap(page, win.x0, win.y0, win.ww, win.wh)
            val cropMask = Bitmap.createBitmap(maskBmp, win.x0, win.y0, win.ww, win.wh)
            try {
                val o = runWholeAot(cropPage, cropMask, win.ww, win.wh) ?: continue
                val cur = IntArray(win.ww * win.wh)
                result.getPixels(cur, 0, win.ww, win.x0, win.y0, win.ww, win.wh)
                for (r in win.members) {
                    val ax0 = ((r.x0.toInt() - pad) - win.x0).coerceIn(0, win.ww)
                    val ay0 = ((r.y0.toInt() - pad) - win.y0).coerceIn(0, win.wh)
                    val ax1 = ((r.x1.toInt() + pad) - win.x0).coerceIn(0, win.ww)
                    val ay1 = ((r.y1.toInt() + pad) - win.y0).coerceIn(0, win.wh)
                    for (y in ay0 until ay1) {
                        val mrow = (win.y0 + y) * w + win.x0
                        val row = y * win.ww
                        for (x in ax0 until ax1) if ((maskPx[mrow + x] and 0xFF) > 127) cur[row + x] = o.px[row + x]
                    }
                }
                result.setPixels(cur, 0, win.ww, win.x0, win.y0, win.ww, win.wh)
            } finally {
                if (cropPage !== page) cropPage.recycle()
                if (cropMask !== maskBmp) cropMask.recycle()
            }
        }
    }

    /**
     * 暖機：aot 方法對空白小圖跑一次 NCNN AOT session（首次 lazy 初始化在單緒完成）。
     * boxfill 只平塗、不跑該 session → 無需暖（也不會冷撞）。併發翻多頁前先呼叫一次。
     */
    fun warmUp() {
        if (cfg.method == "boxfill") return
        val w = 64
        val h = 64
        val page = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
        val mask = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
        try {
            runWholeAot(page, mask, w, h)
        } catch (t: Throwable) {
            Log.w(TAG, "去字暖機失敗：${t.message}")
        } finally {
            page.recycle()
            mask.recycle()
        }
    }

    /** 去字遮罩 Bitmap（白＝要去字）。給重繪素材/視覺化用；與 inpaint 同一份 seg 細遮罩。 */
    fun buildMask(page: Bitmap, regions: List<TextRegion>, textMask: Bitmap): Bitmap {
        val w = page.width; val h = page.height
        val maskPx = buildSegMask(regions, textMask, w, h)
        return Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888).apply { setPixels(maskPx, 0, w, 0, 0, w, h) }
    }

    /**
     * seg 細遮罩＝seg 細筆畫 ∩ 已保留區的「區域 bbox 矩形（外擴 bboxPad）」（allow），再膨脹。回傳 ARGB 像素（白＝要去字）。
     * ★ 用 bbox 矩形(非緊的文字行框)＝涵蓋漢字旁的注音假名；pad 再外擴涵蓋貼 bbox 邊界的假名（桌面 auto_diag.py 實證）。
     */
    private fun buildSegMask(regions: List<TextRegion>, textMask: Bitmap, w: Int, h: Int): IntArray {
        val pad = cfg.bboxPad
        val allow = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
        Canvas(allow).apply {
            drawColor(Color.BLACK)
            val p = Paint().apply { color = Color.WHITE; style = Paint.Style.FILL }
            for (region in regions) drawRect(region.x0 - pad, region.y0 - pad, region.x1 + pad, region.y1 + pad, p)
        }
        val mask = IntArray(w * h)
        allow.getPixels(mask, 0, w, 0, 0, w, h)
        allow.recycle()
        val seg = IntArray(w * h)
        textMask.getPixels(seg, 0, w, 0, 0, w, h)
        for (i in mask.indices) {
            mask[i] = if ((mask[i] and 0xFF) > 127 && (seg[i] and 0xFF) > 127) Color.WHITE else Color.BLACK
        }
        dilate(mask, w, h, (cfg.maskDilate / 2f).roundToInt().coerceAtLeast(1))
        return mask
    }

    /** 二值遮罩可分離膨脹（先橫後縱 max-filter），radius 像素。覆蓋筆畫抗鋸齒邊緣、給去字餘裕。 */
    private fun dilate(px: IntArray, w: Int, h: Int, radius: Int) {
        if (radius <= 0) return
        val tmp = IntArray(px.size)
        for (y in 0 until h) {
            val row = y * w
            for (x in 0 until w) {
                var on = false; var k = -radius
                while (k <= radius) { val xx = x + k; if (xx in 0 until w && (px[row + xx] and 0xFF) > 127) { on = true; break }; k++ }
                tmp[row + x] = if (on) Color.WHITE else Color.BLACK
            }
        }
        for (x in 0 until w) {
            for (y in 0 until h) {
                var on = false; var k = -radius
                while (k <= radius) { val yy = y + k; if (yy in 0 until h && (tmp[yy * w + x] and 0xFF) > 127) { on = true; break }; k++ }
                px[y * w + x] = if (on) Color.WHITE else Color.BLACK
            }
        }
    }

    /** [modeColor]／[modeFrac]：背景最常見的顏色（5-bit 量化分箱的平均）與落在該色 ±[MODE_TOL] 內的背景像素比例。 */
    private class BgStat(
        val meanLum: Float, val std: Float, val color: Int, val modeColor: Int = color, val modeFrac: Float = 1f,
        /** 筆畫（tight 遮罩）亮度中位數；-1＝量不到。 */
        val textLum: Float = -1f,
    )

    /**
     * 區 bbox 內「非文字(背景)」像素的亮度均值+std+平均色。tightPx＝未膨脹 textMask（量得到筆畫間的白）。
     * boxfill 用 [BgStat.color] 平塗；std/meanLum 只給 sandbox 去背比較標框（對照用）。用行框多邊形局部遮罩避開軸對齊 bbox 角落雜訊。
     */
    private fun bgStats(px: IntArray, tightPx: IntArray, region: TextRegion, w: Int, h: Int): BgStat {
        val x0 = region.x0.toInt().coerceIn(0, w - 1)
        val y0 = region.y0.toInt().coerceIn(0, h - 1)
        val x1 = region.x1.toInt().coerceIn(x0 + 1, w)
        val y1 = region.y1.toInt().coerceIn(y0 + 1, h)
        val bw = x1 - x0; val bh = y1 - y0
        val qmBmp = Bitmap.createBitmap(bw, bh, Bitmap.Config.ARGB_8888)
        Canvas(qmBmp).apply {
            drawColor(Color.BLACK)
            val p = Paint().apply { color = Color.WHITE; style = Paint.Style.FILL }
            for (line in region.lines) {
                val q = line.quad
                if (q.size < 4) continue
                val path = android.graphics.Path().apply {
                    moveTo(q[0].x - x0, q[0].y - y0)
                    for (i in 1..3) lineTo(q[i].x - x0, q[i].y - y0)
                    close()
                }
                drawPath(path, p)
            }
        }
        val qm = IntArray(bw * bh)
        qmBmp.getPixels(qm, 0, bw, 0, 0, bw, bh)
        qmBmp.recycle()
        var n = 0; var sl = 0.0; var sl2 = 0.0; var sr = 0L; var sg = 0L; var sb = 0L
        for (y in 0 until bh) {
            for (x in 0 until bw) {
                if ((qm[y * bw + x] and 0xFF) <= 127) continue
                val gi = (y0 + y) * w + (x0 + x)
                if ((tightPx[gi] and 0xFF) > 127) continue // 文字像素
                val p = px[gi]
                val r = (p shr 16) and 0xFF; val g = (p shr 8) and 0xFF; val b = p and 0xFF
                val lum = 0.299 * r + 0.587 * g + 0.114 * b
                sl += lum; sl2 += lum * lum; sr += r; sg += g; sb += b; n++
            }
        }
        if (n < 16) return BgStat(255f, 0f, Color.WHITE)
        val mean = sl / n
        val std = kotlin.math.sqrt((sl2 / n - mean * mean).coerceAtLeast(0.0))
        // 眾數色：抗鋸齒字緣（不在 tight 遮罩內）會把均值拉灰；平塗改用最常見的顏色，並量它佔背景的比例判斷是否單色底。
        val bins = IntArray(32 * 32 * 32)
        for (y in 0 until bh) for (x in 0 until bw) {
            if ((qm[y * bw + x] and 0xFF) <= 127) continue
            val gi = (y0 + y) * w + (x0 + x)
            if ((tightPx[gi] and 0xFF) > 127) continue
            val p = px[gi]
            bins[(((p shr 19) and 31) shl 10) or (((p shr 11) and 31) shl 5) or ((p shr 3) and 31)]++
        }
        var best = 0
        for (i in bins.indices) if (bins[i] > bins[best]) best = i
        val br = ((best shr 10) and 31) * 8 + 4; val bg = ((best shr 5) and 31) * 8 + 4; val bb = (best and 31) * 8 + 4
        var near = 0
        val exact = HashMap<Int, Int>()
        for (y in 0 until bh) for (x in 0 until bw) {
            if ((qm[y * bw + x] and 0xFF) <= 127) continue
            val gi = (y0 + y) * w + (x0 + x)
            if ((tightPx[gi] and 0xFF) > 127) continue
            val p = px[gi]
            val r = (p shr 16) and 0xFF; val g = (p shr 8) and 0xFF; val b = p and 0xFF
            if (abs(r - br) <= MODE_TOL && abs(g - bg) <= MODE_TOL && abs(b - bb) <= MODE_TOL) {
                near++; exact.merge(p or (0xFF shl 24), 1, Int::plus)
            }
        }
        // 平塗色＝眾數色箱內「最常出現的精確色」（平均會被字緣抗鋸齒拉灰，填出來看得到一塊色差）
        val modeColor = exact.maxByOrNull { it.value }?.key ?: Color.rgb(br, bg, bb)
        // 筆畫亮度（中位數）：給 chooseModel 排除「跟字一樣顏色」的背景模型
        val tl = ArrayList<Int>()
        for (y in 0 until bh) for (x in 0 until bw) {
            if ((qm[y * bw + x] and 0xFF) <= 127) continue
            val gi = (y0 + y) * w + (x0 + x)
            if ((tightPx[gi] and 0xFF) <= 127) continue
            val p = px[gi]
            tl += (299 * ((p shr 16) and 0xFF) + 587 * ((p shr 8) and 0xFF) + 114 * (p and 0xFF)) / 1000
        }
        tl.sort()
        // 筆畫遮罩偏寬、含大量字旁背景 → 中位數會被拉向背景色；字色取離眾數色較遠的那一端（10% 或 90% 分位）
        val ml = (299 * ((modeColor shr 16) and 0xFF) + 587 * ((modeColor shr 8) and 0xFF) + 114 * (modeColor and 0xFF)) / 1000
        val textLum = if (tl.isEmpty()) {
            -1f
        } else {
            val lo = tl[tl.size / 10]; val hi = tl[tl.size * 9 / 10]
            (if (abs(lo - ml) >= abs(hi - ml)) lo else hi).toFloat()
        }
        return BgStat(
            mean.toFloat(), std.toFloat(), Color.rgb((sr / n).toInt(), (sg / n).toInt(), (sb / n).toInt()),
            modeColor, near.toFloat() / n, textLum,
        )
    }

    /** 白泡去字＝把區域 bbox(外擴 pad)內的去字遮罩像素直接平塗成背景色。均勻白泡保證無殘留。 */
    private fun flatFill(result: Bitmap, maskPx: IntArray, region: TextRegion, color: Int, pad: Int, w: Int, h: Int) {
        val x0 = (region.x0.toInt() - pad).coerceIn(0, w - 1)
        val y0 = (region.y0.toInt() - pad).coerceIn(0, h - 1)
        val x1 = (region.x1.toInt() + pad).coerceIn(x0 + 1, w)
        val y1 = (region.y1.toInt() + pad).coerceIn(y0 + 1, h)
        val bw = x1 - x0; val bh = y1 - y0
        val sub = IntArray(bw * bh)
        result.getPixels(sub, 0, bw, x0, y0, bw, bh)
        for (y in 0 until bh) {
            val mrow = (y0 + y) * w + x0
            val row = y * bw
            for (x in 0 until bw) if ((maskPx[mrow + x] and 0xFF) > 127) sub[row + x] = color
        }
        result.setPixels(sub, 0, bw, x0, y0, bw, bh)
    }

    private class WinOut(val x0: Int, val y0: Int, val ww: Int, val wh: Int, val px: IntArray)

    /**
     * 整頁跑一次 NCNN AOT-GAN：整張縮到方形 [cfg.tileSize] → 推論 → 放回原尺寸。只讀 page/maskBmp ⇒ 併發安全。
     * NCNN net 固定方形輸入、同尺寸復用安全（跨尺寸 reuse 會崩，故一律整頁 tileSize）。
     */
    private fun runWholeAot(page: Bitmap, maskBmp: Bitmap, w: Int, h: Int): WinOut? {
        val t = cfg.tileSize
        val imgScaled = Bitmap.createScaledBitmap(page, t, t, true)
        val maskScaled = Bitmap.createScaledBitmap(maskBmp, t, t, false)
        return try {
            val imgChw = aotImageChw(imgScaled, maskScaled, t)
            val maskArr = maskArr(maskScaled, t)
            val out = FloatArray(3 * t * t)
            val rc = NcnnBackend.inpaintAot(ncnnHandle, imgChw, maskArr, t, out)
            check(rc == 0) { "NCNN AOT 推論失敗 rc=$rc" }
            val resScaled = aotArrToBitmap(out, t)
            val resWin = Bitmap.createScaledBitmap(resScaled, w, h, true)
            val px = IntArray(w * h)
            resWin.getPixels(px, 0, w, 0, 0, w, h)
            if (resWin !== resScaled) resScaled.recycle()
            resWin.recycle()
            WinOut(0, 0, w, h, px)
        } catch (t2: Throwable) {
            Log.w(TAG, "AOT 去字失敗：${t2.message}"); null
        } finally {
            if (imgScaled !== page) imgScaled.recycle()
            if (maskScaled !== maskBmp) maskScaled.recycle()
        }
    }

    /** AOT 影像的裸 NCHW 陣列 [3*n*n]：RGB→[-1,1]，遮罩處歸零（m-i-t `img*(1-mask)`）。 */
    private fun aotImageChw(imgBmp: Bitmap, maskBmp: Bitmap, n: Int): FloatArray {
        val area = n * n
        val px = IntArray(area); imgBmp.getPixels(px, 0, n, 0, 0, n, n)
        val mp = IntArray(area); maskBmp.getPixels(mp, 0, n, 0, 0, n, n)
        val chw = FloatArray(3 * area)
        for (i in 0 until area) {
            if ((mp[i] and 0xFF) > 127) continue // 洞＝0
            val p = px[i]
            chw[i] = ((p shr 16) and 0xFF) / 127.5f - 1f
            chw[area + i] = ((p shr 8) and 0xFF) / 127.5f - 1f
            chw[2 * area + i] = (p and 0xFF) / 127.5f - 1f
        }
        return chw
    }

    /** 遮罩的裸陣列 [n*n]（1=擦）。 */
    private fun maskArr(bmp: Bitmap, n: Int): FloatArray {
        val px = IntArray(n * n); bmp.getPixels(px, 0, n, 0, 0, n, n)
        val m = FloatArray(n * n)
        for (i in px.indices) m[i] = if ((px[i] and 0xFF) > 127) 1f else 0f
        return m
    }

    /** AOT 輸出陣列 [3*n*n]∈[-1,1] → Bitmap（(x+1)*127.5）。 */
    private fun aotArrToBitmap(arr: FloatArray, n: Int): Bitmap {
        val area = n * n
        val px = IntArray(area)
        for (i in 0 until area) {
            val r = ((arr[i] + 1f) * 127.5f).toInt().coerceIn(0, 255)
            val g = ((arr[area + i] + 1f) * 127.5f).toInt().coerceIn(0, 255)
            val b = ((arr[2 * area + i] + 1f) * 127.5f).toInt().coerceIn(0, 255)
            px[i] = Color.rgb(r, g, b)
        }
        return Bitmap.createBitmap(px, n, n, Bitmap.Config.ARGB_8888)
    }

    /** 把 AOT 輸出貼回 result，只換遮罩內像素（序列呼叫、寫入安全）。 */
    private fun compositePixels(result: Bitmap, maskPx: IntArray, o: WinOut) {
        val w = result.width
        val cur = IntArray(o.ww * o.wh)
        result.getPixels(cur, 0, o.ww, o.x0, o.y0, o.ww, o.wh)
        for (y in 0 until o.wh) {
            val maskRow = (o.y0 + y) * w + o.x0
            val row = y * o.ww
            for (x in 0 until o.ww) if ((maskPx[maskRow + x] and 0xFF) > 127) cur[row + x] = o.px[row + x]
        }
        result.setPixels(cur, 0, o.ww, o.x0, o.y0, o.ww, o.wh)
    }

    override fun close() {
        if (ncnnHandle != 0L) { NcnnBackend.releaseNet(ncnnHandle); ncnnHandle = 0L }
    }

    companion object {
        private const val TAG = "Inpainter"
        /** 長邊 > 短邊×此值 視為長條頁（條漫），壓畫面區改走視窗去字（見 [inpaintWindows]）。一般單頁約 1.4、跨頁約 1/1.4。 */
        private const val TALL_RATIO = 1.8f
        /** 單色底平塗時行框外擴＝行高×此值（＋2px）。 */
        private const val QUAD_GROW = 0.12f
        /** 區背景有此比例以上的像素落在眾數色 ±[MODE_TOL] 內 → 視為單色底（乾淨白泡／色框），平塗即可、不跑 AOT。 */
        private const val FLAT_FRAC = 0.85f
        private const val MODE_TOL = 16
        /** 量氣泡內部時「背景色」的容差（JPEG 雜訊）。 */
        private const val BUBBLE_TOL = 22
        /** 平滑漸層底的判定：平面擬合內群比例下限、內群殘差上限。 */
        private const val PLANE_INLIER = 0.7
        private const val PLANE_RMS = 7.0
        /** 光暈判定：與背景模型的亮度差下限。 */
        private const val HALO_TOL = 3.0
        private const val FEATHER = 3
        /** 背景連通：相鄰像素亮度差上限（JPEG 雜訊、光暈緩變走得過；外框、泡緣陡變擋住）。 */
        private const val CONNECT_STEP = 10
        /** 背景模型與筆畫的最小亮度差（低於此＝模型其實量到了跟字同色的東西）。 */
        private const val MIN_TEXT_CONTRAST = 60f
    }
}
