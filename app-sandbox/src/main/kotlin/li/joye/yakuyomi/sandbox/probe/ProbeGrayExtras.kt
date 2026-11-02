package li.joye.yakuyomi.sandbox.probe

import li.joye.yakuyomi.nightread.Cv
import li.joye.yakuyomi.nightread.FImg
import li.joye.yakuyomi.nightread.Gray
import li.joye.yakuyomi.nightread.Kernel
import li.joye.yakuyomi.nightread.Mask
import li.joye.yakuyomi.nightread.NightGrayWorkload
import li.joye.yakuyomi.nightread.NightNativeGray
import li.joye.yakuyomi.nightread.NightReadInput
import li.joye.yakuyomi.sandbox.probe.ProbeBinExtras.Case

/**
 * 探針步驟 c3b.x 的案例：第 3b 階搬了、但 [NightGrayWorkload] 步驟表沒涵蓋的參數與尺寸，手機上只比位元。
 *  - 每頁從 4 張公開頁裁出邊界尺寸（寬 1／2／63／64／65／127／128／129 × 高 1／2／3／7／64／65 裡挑 10 組）：灰階形態學（cv2 版、不限 8 位元版）、
 *    blackhat（含不是每列都含錨點的核，走 cv2 那條路）、medianBlur8 1／3／5／7、Canny（三組門檻、Q16 右移 16、不限 8 位元右移 7）、
 *    gaussianBlur（σ 0、0.3、0.7、2、4）與 boxBlur（0、1、2、3、15），浮點灰階與 0／1 遮罩兩種輸入；
 *  - 每頁整頁：步驟表沒有的 σ 與核長的可分離濾波、medianBlur8 3／5、原始灰階的 Canny、兩種 blackhat；
 *  - 跟頁無關：繪圖（粗細 −1–17、端點在影像外與遠在外、一次畫多段、多邊形頂點在外與退化）、numpy 相容（maxFilterWrapRows、npArgsort、
 *    以及第 1 階就搬了、這一階才有 JNI 入口的 npSum、eigh2、gaussW）。
 *
 * **會受 FMA 影響的案例只標整頁的可分離濾波**（[Case.fma]、[Filter.fma]；[NightProbeGrayJvmTest] 在電腦上用 Math.fma 版的 Kotlin 逐一核對，
 * 標了的每一組合成 FMA 都會不同）。裁出來的小塊有的是常數（全白、全 0 的遮罩），合成 FMA 也可能一樣，所以不標、不算進「會受 FMA 影響」的組數，
 * 它們是比邊界與迴圈的寫法。σ 0（直接複製）與核長 0／1（乘 1、加 0 都是精確的）不受影響，也不標。
 * Kotlin 裡 internal 的 gaussAt、gaussQ16Sparse、MedianRows 探針呼叫不到：手機上只由步驟表的 SPARSE 族比，其餘只有電腦與模擬晶片驗過。
 *
 * 純 Kotlin、不碰 android.*；建案例不會呼叫原生（只有 [Case.native] 會）。[NightProbeGrayJvmTest] 在電腦上跑每個
 * [Case.kotlin]，確認 Kotlin 那半不會丟例外（兩邊都丟例外在手機上算相同，那種案例比了等於沒比）。
 */
internal object ProbeGrayExtras {
    /** 會出現的運算（摘要與 JVM 測試用）。 */
    val OPS = listOf(
        "dilateGray", "erodeGray", "dilateGrayI", "erodeGrayI", "blackhat", "medianBlur8", "canny", "gaussianBlur", "boxBlur",
        "line", "lines", "fillPoly", "maxFilterWrapRows", "npArgsort", "npSum", "eigh2", "gaussW",
    )

    /** 有 Float 乘加的運算（JVM 測試用 Math.fma 版核對的就是這兩種）。 */
    val FILTER_OPS = setOf("gaussianBlur", "boxBlur")

    /** 從頁裡裁出的邊界尺寸（寬 to 高）。 */
    val EDGE_SIZES = listOf(1 to 1, 1 to 65, 65 to 1, 2 to 3, 63 to 7, 64 to 64, 65 to 2, 127 to 65, 128 to 3, 129 to 64)

    /** 畫圖用的空白影像尺寸（寬 to 高）。 */
    val DRAW_SIZES = listOf(1 to 1, 2 to 3, 64 to 64, 129 to 7, 200 to 65, 1351 to 37)

    private fun kernels(): List<Pair<String, Kernel>> = listOf(
        "e3" to Cv.ellipse(3), "e11" to Cv.ellipse(11), "r2x4" to Cv.rect(2, 4), "r1x5" to Cv.rect(1, 5),
        // 不是每列都含錨點那一欄：blackhat 走 erodeGray(dilateGray) 那條路
        "k9x3" to Kernel(9, 3, BooleanArray(27) { it % 9 >= 6 }),
    )

    private fun crop(g: Gray, x: Int, y: Int, w: Int, h: Int): Gray {
        val d = IntArray(w * h)
        for (j in 0 until h) System.arraycopy(g.data, (y + j) * g.w + x, d, j * w, w)
        return Gray(w, h, d)
    }

    private fun maskF(g: Gray): FImg = FImg(g.w, g.h, FloatArray(g.data.size) { if (g.data[it] < 128) 1f else 0f })

    /** 不限 8 位元的版本（Q16 那種量級、有負值）：I 版形態學與 blackhat 的 cv2 路徑、Canny 的右移。 */
    private fun wide(g: Gray): Gray = Gray(g.w, g.h, IntArray(g.data.size) { g.data[it] * 65536 - 40000 })

    /** 一個可分離濾波的案例（[op]＝gaussianBlur 用 [sigma]、boxBlur 用 [k]）。整理成資料，JVM 測試才能用 Math.fma 版核對 [fma] 標得對不對。 */
    class Filter(val op: String, val tag: String, val f: FImg, val sigma: Double, val k: Int, val fma: Boolean) {
        fun kotlin(): FImg = if (op == "gaussianBlur") Cv.gaussianBlur(f, sigma) else Cv.boxBlur(f, k)
        fun native(): Any = if (op == "gaussianBlur") NightNativeGray.gaussianBlur(f, sigma) else NightNativeGray.boxBlur(f, k)
        fun case(): Case = Case(op, tag, this::kotlin, this::native, fma)
    }

    /** 邊界尺寸的裁切（標籤 to 灰階）：每個尺寸換一個位置裁（頁的中段，偏一點避免每塊都是同一處）。 */
    private fun crops(name: String, gray: Gray): List<Pair<String, Gray>> = EDGE_SIZES.mapIndexed { i, (w, h) ->
        val x = minOf(gray.w - w, gray.w * (3 + i % 4) / 10)
        val y = minOf(gray.h - h, gray.h * (2 + i % 5) / 10)
        "$name/${w}x$h" to crop(gray, x, y, w, h)
    }

    /**
     * 一頁的可分離濾波案例。裁切：σ 0／0.3／0.7／2／4、核長 0／1／2／3／15，浮點灰階與 0／1 遮罩，都不標 FMA。
     * 整頁：步驟表是 inkF σ0.7、halfF σ4、視窗 boxBlur 15，這裡補其他 σ 與核長（產品用到的範圍），標 FMA；另加兩組不受影響的對照（σ 0 直接複製、
     * 核長 1 乘 1），不標。
     */
    fun filters(name: String, gray: Gray): List<Filter> {
        val out = ArrayList<Filter>()
        for ((tag, g) in crops(name, gray)) {
            for ((fn, f) in listOf("grayF" to g.toF(), "maskF" to maskF(g))) {
                for (s in listOf(0.0, 0.3, 0.7, 2.0, 4.0)) out += Filter("gaussianBlur", "$tag/$fn/s$s", f, s, 0, fma = false)
                for (k in listOf(0, 1, 2, 3, 15)) out += Filter("boxBlur", "$tag/$fn/$k", f, 0.0, k, fma = false)
            }
        }
        val grayF = gray.toF()
        val inkF = maskF(gray)
        val halfF = Cv.resizeArea(grayF, maxOf(1, grayF.w / 2), maxOf(1, grayF.h / 2))
        val full = "$name/full"
        for ((fn, f, s) in listOf(Triple("grayF", grayF, 0.7), Triple("grayF", grayF, 1.5), Triple("grayF", grayF, 3.0), Triple("inkF", inkF, 2.0), Triple("halfF", halfF, 1.0))) {
            out += Filter("gaussianBlur", "$full/$fn/s$s", f, s, 0, fma = true)
        }
        for ((fn, f, k) in listOf(Triple("grayF", grayF, 3), Triple("inkF", inkF, 15), Triple("halfF", halfF, 5))) {
            out += Filter("boxBlur", "$full/$fn/$k", f, 0.0, k, fma = true)
        }
        out += Filter("gaussianBlur", "$full/halfF/s0.0", halfF, 0.0, 0, fma = false)
        out += Filter("boxBlur", "$full/halfF/1", halfF, 0.0, 1, fma = false)
        return out
    }

    /** 一頁的案例。輸入先在這裡算好（Kotlin，不計時），閉包裡只剩要比的那一次呼叫。 */
    fun pageCases(name: String, input: NightReadInput): List<Case> {
        val gray = input.gray
        val out = ArrayList<Case>()
        val ks = kernels()
        for ((tag, g) in crops(name, gray)) {
            val w = g.w
            val h = g.h
            for ((kn, k) in ks) {
                out += Case("dilateGray", "$tag/$kn", { Cv.dilateGray(g, k) }, { NightNativeGray.dilateGray(g, k) })
                out += Case("erodeGray", "$tag/$kn", { Cv.erodeGray(g, k) }, { NightNativeGray.erodeGray(g, k) })
                out += Case("blackhat", "$tag/$kn", { Cv.blackhat(g, k) }, { NightNativeGray.blackhat(g, k) })
            }
            val wg = wide(g)
            for ((kn, k) in ks.take(3)) {
                out += Case("dilateGrayI", "$tag/wide/$kn", { Cv.dilateGrayI(wg, k) }, { NightNativeGray.dilateGrayI(wg, k) })
                out += Case("erodeGrayI", "$tag/wide/$kn", { Cv.erodeGrayI(wg, k) }, { NightNativeGray.erodeGrayI(wg, k) })
                out += Case("blackhat", "$tag/wide/$kn", { Cv.blackhat(wg, k) }, { NightNativeGray.blackhat(wg, k) })
            }
            for (m in listOf(1, 3, 5, 7)) out += Case("medianBlur8", "$tag/$m", { Cv.medianBlur8(g, m) }, { NightNativeGray.medianBlur8(g, m) })
            for ((lo, hi) in listOf(15 to 40, 0 to 1, 50 to 150)) {
                out += Case("canny", "$tag/0/$lo-$hi", { Cv.canny(g.data, w, h, 0, lo, hi) }, { NightNativeGray.canny(g.data, w, h, 0, lo, hi) })
            }
            val q = Cv.gaussQ16(g, 2.0)
            out += Case("canny", "$tag/q16/15-40", { Cv.canny(q.data, w, h, 16, 15, 40) }, { NightNativeGray.canny(q.data, w, h, 16, 15, 40) })
            out += Case("canny", "$tag/wide/100-3000", { Cv.canny(wg.data, w, h, 7, 100, 3000) }, { NightNativeGray.canny(wg.data, w, h, 7, 100, 3000) })
        }
        filters(name, gray).mapTo(out) { it.case() }
        // 整頁：步驟表是 medianBlur8 7、Q16 的 Canny、e7 的 blackhat
        val full = "$name/full"
        for (m in listOf(3, 5)) out += Case("medianBlur8", "$full/$m", { Cv.medianBlur8(gray, m) }, { NightNativeGray.medianBlur8(gray, m) })
        out += Case("canny", "$full/0/15-40", { Cv.canny(gray.data, gray.w, gray.h, 0, 15, 40) }, { NightNativeGray.canny(gray.data, gray.w, gray.h, 0, 15, 40) })
        for ((kn, k) in ks.filter { it.first == "e3" || it.first == "k9x3" }) {
            out += Case("blackhat", "$full/$kn", { Cv.blackhat(gray, k) }, { NightNativeGray.blackhat(gray, k) })
        }
        return out
    }

    /** 跟頁無關的案例：繪圖、numpy 相容（固定種子）。 */
    fun globalCases(): List<Case> {
        val out = ArrayList<Case>()
        val rnd = java.util.Random(0x3b)
        fun rint(lo: Int, hi: Int): Int = lo + (rnd.nextDouble() * (hi.toLong() - lo)).toInt()   // [lo, hi)
        for ((w, h) in DRAW_SIZES) {
            val m = Mask(w, h)
            val segs = IntArray(5 * 12)
            for (t in 0 until 12) {
                val th = listOf(-1, 0, 1, 2, 3, 7, 17)[t % 7]
                // 很遠的端點只給細線：粗線從 ymin 逐列走到 ymax（頁外的列也走），兩邊都慢
                val far = when (t % 4) { 0 -> 1; 1 -> 3; 2 -> 100; else -> if (th <= 1) 1 shl 20 else 100 }
                val x0 = rint(-w * far - 2, w * far + 3)
                val y0 = rint(-h * far - 2, h * far + 3)
                val x1 = if (t == 5) x0 else rint(-w * far - 2, w * far + 3)
                val y1 = if (t == 5) y0 else rint(-h * far - 2, h * far + 3)
                intArrayOf(x0, y0, x1, y1, th).copyInto(segs, 5 * t)
                out += Case("line", "${w}x$h/#$t/th$th", { m.copy().also { Cv.line(it, x0, y0, x1, y1, th) } }, { NightNativeGray.line(m, x0, y0, x1, y1, th) })
            }
            out += Case("lines", "${w}x$h/12", {
                m.copy().also { c -> for (t in 0 until 12) Cv.line(c, segs[5 * t], segs[5 * t + 1], segs[5 * t + 2], segs[5 * t + 3], segs[5 * t + 4]) }
            }, { NightNativeGray.lines(m, segs) })
            for (t in 0 until 6) {
                val n = 1 + rint(0, 8)
                val xs = IntArray(n) { rint(-w, 2 * w + 1) }
                val ys = IntArray(n) { if (t == 2) minOf(3, h - 1) else rint(-h, 2 * h + 1) }   // t＝2：全部同一列（水平邊）
                out += Case("fillPoly", "${w}x$h/#$t/n$n", { m.copy().also { Cv.fillPoly(it, xs, ys) } }, { NightNativeGray.fillPoly(m, xs, ys) })
            }
            val rx = intArrayOf(0, w - 1, w - 1, 0)
            val ry = intArrayOf(0, 0, h - 1, h - 1)
            out += Case("fillPoly", "${w}x$h/rect", { m.copy().also { Cv.fillPoly(it, rx, ry) } }, { NightNativeGray.fillPoly(m, rx, ry) })
        }
        for ((rows, cols) in listOf(1 to 1, 1 to 9, 9 to 1, 2 to 63, 7 to 64, 3 to 65, 5 to 127, 4 to 128, 6 to 129, 360 to 41)) {
            for ((nt, nr) in listOf(0 to 0, 1 to 2, 3 to 6, 20 to 1, 0 to 70)) {
                val s = IntArray(rows * cols) { if (rint(0, 4) == 0) rint(0, 1000) else 0 }
                out += Case("maxFilterWrapRows", "${rows}x$cols/$nt,$nr", { Cv.maxFilterWrapRows(s, rows, cols, nt, nr) }, { NightNativeGray.maxFilterWrapRows(s, rows, cols, nt, nr) })
            }
        }
        for (t in 0 until 120) {
            val n = if (t < 40) t else rint(1, 6000)
            val v = IntArray(n) { if (t % 3 == 0) rint(0, 5) else rnd.nextInt() }
            out += Case("npArgsort", "#$t/n$n", { Cv.npArgsort(v) }, { NightNativeGray.npArgsort(v) })
            val d = DoubleArray(n) { (rnd.nextDouble() - 0.5) * Math.pow(10.0, rint(-5, 6).toDouble()) }
            out += Case("npSum", "#$t/all", { Cv.npSum(d) }, { NightNativeGray.npSum(d) })
            if (n > 2) out += Case("npSum", "#$t/1,${n - 2}", { Cv.npSum(d, 1, n - 2) }, { NightNativeGray.npSum(d, 1, n - 2) })
            out += Case("npSum", "#$t/$n,0", { Cv.npSum(d, n, 0) }, { NightNativeGray.npSum(d, n, 0) })
        }
        val special = doubleArrayOf(0.0, -0.0, 1.0, -1.0, 1e-300, 1e300, 4.9e-324, 3.0, 1e-16)
        for (a in special) for (b in special) for (c in special) out += Case("eigh2", "$a,$b,$c", { Cv.eigh2(a, b, c) }, { NightNativeGray.eigh2(a, b, c) })
        repeat(300) { t ->
            val a = (rnd.nextDouble() - 0.5) * 1e4
            val b = (rnd.nextDouble() - 0.5) * 1e4
            val c = (rnd.nextDouble() - 0.5) * 1e4
            out += Case("eigh2", "#$t", { Cv.eigh2(a, b, c) }, { NightNativeGray.eigh2(a, b, c) })
        }
        for (s in listOf(0.3, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 10.0)) out += Case("gaussW", "s$s", { Cv.gaussW(s) }, { NightNativeGray.gaussW(s) })
        return out
    }

    /** 兩個結果逐位元相同（Double 與 DoubleArray 比原始位元；其餘交給 [NightGrayWorkload.same]）。 */
    fun same(a: Any, b: Any): Boolean = when {
        a is DoubleArray && b is DoubleArray -> a.size == b.size && a.indices.all { a[it].toRawBits() == b[it].toRawBits() }
        a is Double && b is Double -> a.toRawBits() == b.toRawBits()
        else -> NightGrayWorkload.same(a, b)
    }
}
