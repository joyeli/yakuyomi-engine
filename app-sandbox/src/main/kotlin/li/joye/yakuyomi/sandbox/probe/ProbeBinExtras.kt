package li.joye.yakuyomi.sandbox.probe

import li.joye.yakuyomi.nightread.Cv
import li.joye.yakuyomi.nightread.FImg
import li.joye.yakuyomi.nightread.Kernel
import li.joye.yakuyomi.nightread.NightBinWorkload
import li.joye.yakuyomi.nightread.NightNativeBin
import li.joye.yakuyomi.nightread.NightPrimWorkload
import li.joye.yakuyomi.nightread.NightReadInput

/**
 * 探針步驟 c3a.x 的案例：第 3a 階搬了、但 [NightBinWorkload] 步驟表沒有的原語，手機上只比位元。
 *  - 一般核與線核的 dilate／erode（含偶數高、偶數寬的核：一般核與線核取樣差一列的那處），兩種視窗＋整頁一組；
 *  - resizeArea／resizeBilinear（灰階的浮點版與視窗的 chamfer 距離，後者有小數）：縮小與同尺寸，含不是整數倍的比例；resizeBilinear
 *    另從縮成一半的圖放大回來（產品就是這樣用它）。resizeNearestGray；
 *  - ellipse 0–201（有 sqrt）、rect 0–9 × 0–9；
 *  - polygonArea，固定種子，0–16 個頂點輪流。
 * **會受 FMA 影響的只有 [FMA_SENSITIVE]**（resizeArea、resizeBilinear：權重有小數的乘加，編成 fmadd 結果就會變；整數倍縮小的權重是 1，
 * 那幾組不受影響，所以另加了不是整數倍的比例）。polygonArea 與 ellipse 不受 FMA 影響：兩個 Float 轉 Double 相乘一定是精確值
 * （24 位 × 24 位 ≤ 53 位），合成 fma(a, b, −c·d) 跟分開算一樣只捨入一次；ellipse 的 r·r、dy·dy 是精確的整數，其餘乘法後面沒有加法。
 * 它們留著是比迴圈、捨入與 sqrt 的寫法。手機上「沒有 FMA」的證據是這幾組加上步驟表（另有 CI 掃反組譯）。
 * 輸入照步驟表的準備步驟從 4 張頁導出。Kotlin 裡 internal 的小工具（anyBits、bitRuns、reflect101、Ring.has／set）探針呼叫不到，
 * 只有電腦與模擬晶片驗過。
 *
 * 純 Kotlin、不碰 android.*；建案例不會呼叫原生（只有 [Case.native] 會）。[NightProbeBinJvmTest] 在電腦上跑每個 [Case.kotlin]，
 * 確認 Kotlin 那半不會丟例外（兩邊都丟例外在手機上算相同，那種案例比了等於沒比）。
 */
internal object ProbeBinExtras {
    /** 一組比對。[fma]＝這一組在電腦上核對過合成 FMA 會改結果（只有第 3b 階的 [ProbeGrayExtras] 逐組標；第 3a 階照運算算，見 [FMA_SENSITIVE]）。 */
    class Case(val op: String, val tag: String, val kotlin: () -> Any, val native: () -> Any, val fma: Boolean = false)

    /** 多邊形數（0–16 個頂點輪流，每種 200 個）。 */
    const val POLYGON_CASES = 3400

    /** 會出現的運算（摘要與 JVM 測試用）。 */
    val OPS = listOf("ellipse", "rect", "polygonArea", "dilate", "erode", "resizeArea", "resizeBilinear", "resizeNearestGray")

    /** 結果會因 FMA 合成而變的運算（紀錄 BIN_EXTRA 的 fma 欄、關卡的 fmaCases）。 */
    val FMA_SENSITIVE = setOf("resizeArea", "resizeBilinear")

    /** 跟頁無關的案例：ellipse、rect、polygonArea。 */
    fun globalCases(): List<Case> {
        val out = ArrayList<Case>()
        for (s in 0..201) out += Case("ellipse", "$s", { Cv.ellipse(s) }, { NightNativeBin.ellipse(s) })
        for (kw in 0..9) for (kh in 0..9) out += Case("rect", "${kw}x$kh", { Cv.rect(kw, kh) }, { NightNativeBin.rect(kw, kh) })
        val rnd = java.util.Random(0x3a)
        for (t in 0 until POLYGON_CASES) {
            val n = t % 17
            val xs = FloatArray(n) { rnd.nextFloat() * 2000f }
            val ys = FloatArray(n) { rnd.nextFloat() * 2000f }
            out += Case("polygonArea", "#$t", { Cv.polygonArea(xs, ys) }, { NightNativeBin.polygonArea(xs, ys) })
        }
        return out
    }

    private fun kernels(): List<Pair<String, Kernel>> = listOf(
        "e3" to Cv.ellipse(3), "e7" to Cv.ellipse(7), "r4x2" to Cv.rect(4, 2), "r2x4" to Cv.rect(2, 4),
        "r1x4" to Cv.rect(1, 4), "r5x1" to Cv.rect(5, 1), "r3x3" to Cv.rect(3, 3),
    )

    /** 一頁的案例。輸入先照步驟表的準備步驟算好（Kotlin，不計時），閉包裡只剩要比的那一次呼叫。 */
    fun pageCases(name: String, input: NightReadInput): List<Case> {
        val gray = input.gray
        val regs = NightPrimWorkload.Regs(gray, input.seg, input.charMask)
        for (st in NightBinWorkload.program(gray.w, gray.h)) if (st.family == null) NightBinWorkload.runPrep(st, regs)
        val out = ArrayList<Case>()
        for (mn in listOf("whiteWin", "inkWin")) {
            val m = regs.mask(mn)
            for ((kn, k) in kernels()) {
                for (n in if (kn == "e3") listOf(1, 2) else listOf(1)) {
                    val tag = "$name/$mn/$kn/$n"
                    out += Case("dilate", tag, { Cv.dilate(m, k, n) }, { NightNativeBin.dilate(m, k, n) })
                    out += Case("erode", tag, { Cv.erode(m, k, n) }, { NightNativeBin.erode(m, k, n) })
                }
            }
        }
        val ink = regs.mask("ink")
        val e5 = Cv.ellipse(5)
        out += Case("dilate", "$name/ink/e5/1", { Cv.dilate(ink, e5) }, { NightNativeBin.dilate(ink, e5) })
        out += Case("erode", "$name/ink/e5/1", { Cv.erode(ink, e5) }, { NightNativeBin.erode(ink, e5) })
        val grayF = regs.map.getValue("grayF") as FImg
        val chamfer = Cv.distanceChamfer(regs.mask("whiteWin"), 5)
        for ((fn, f) in listOf("grayF" to grayF, "chamferWin" to chamfer)) {
            val down = listOf(
                f.w / 2 to f.h / 2, f.w / 3 to f.h / 3 + 1, f.w * 3 / 5 to f.h * 3 / 5, f.w to f.h,
                // 不是整數倍的比例：權重有小數（FMA 會改結果的那種乘加）
                f.w * 5 / 7 to f.h * 5 / 7, f.w * 2 / 3 + 1 to f.h * 7 / 9, f.w * 4 / 9 + 1 to f.h * 5 / 11 + 1,
            )
            for ((nw, nh) in down) {
                val tag = "$name/$fn/${nw}x$nh"
                out += Case("resizeArea", tag, { Cv.resizeArea(f, nw, nh) }, { NightNativeBin.resizeArea(f, nw, nh) })
                out += Case("resizeBilinear", tag, { Cv.resizeBilinear(f, nw, nh) }, { NightNativeBin.resizeBilinear(f, nw, nh) })
            }
            // 放大：從縮成一半的圖（Kotlin 先算好，不計時）放大回整頁與兩個不是整數倍的尺寸
            val half = Cv.resizeArea(f, f.w / 2, f.h / 2)
            for ((nw, nh) in listOf(f.w to f.h, f.w * 5 / 7 to f.h * 5 / 7, half.w * 3 / 2 + 1 to half.h * 4 / 3)) {
                out += Case("resizeBilinear", "$name/${fn}Half/${nw}x$nh", { Cv.resizeBilinear(half, nw, nh) }, { NightNativeBin.resizeBilinear(half, nw, nh) })
            }
        }
        for ((nw, nh) in listOf(gray.w / 2 to gray.h / 2, gray.w / 3 + 1 to gray.h / 3 + 1, gray.w to gray.h)) {
            out += Case("resizeNearestGray", "$name/gray/${nw}x$nh", { Cv.resizeNearestGray(gray, nw, nh) }, { NightNativeBin.resizeNearestGray(gray, nw, nh) })
        }
        return out
    }

    /** 兩個結果逐位元相同（核另外比；其餘交給 [NightBinWorkload.same]）。 */
    fun same(a: Any, b: Any): Boolean =
        if (a is Kernel && b is Kernel) a.w == b.w && a.h == b.h && a.data.contentEquals(b.data) else NightBinWorkload.same(a, b)
}
