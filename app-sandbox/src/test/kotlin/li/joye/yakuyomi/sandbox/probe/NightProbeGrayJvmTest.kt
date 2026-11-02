package li.joye.yakuyomi.sandbox.probe

import li.joye.yakuyomi.nightread.Cv
import li.joye.yakuyomi.nightread.FImg
import li.joye.yakuyomi.nightread.NightGrayWorkload
import li.joye.yakuyomi.nightread.NightPrimWorkload
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.File

/**
 * 探針 (c) 的第 3b 階步驟（c3b、c3b.x）在電腦上能做的部分：C++ 只在手機上跑（Windows 的 Gradle 載不了原生庫），這裡確認
 *  - 探針自己解碼的 4 張頁餵得進 [NightGrayWorkload] 的步驟表、關卡要比滿的步數算得對、每族都有計入權重的步驟；
 *  - 「會受 FMA 影響」的標記是真的：步驟表的 fmaSensitive 與補充案例的 fma，在這 4 張頁上用 Math.fma 版的 Kotlin 逐一核對
 *    （標了的合成 FMA 一定會不同、沒標的濾波步驟一定一樣），手機摘要 6a／6b／6d 的 FMA 組數才當得了證據；
 *  - 補充案例的 Kotlin 那半都不丟例外（兩邊都丟在手機上算相同，比了等於沒比）、標籤唯一、運算齊全。
 */
class NightProbeGrayJvmTest {
    private val dir = File("../yakuyomi-nightread/nightread/src/test/resources/page")

    private fun load(name: String): ByteArray? = File(dir, name).takeIf { it.isFile }?.readBytes()

    @Test
    fun grayStepTableOnProbePagesAndFmaFlags() {
        for (page in NightProbeRunner.PRIM_PAGES) {
            val input = GuardDigest.input(::load, page)
            val w = input.gray.w
            val h = input.gray.h
            val regs = NightPrimWorkload.Regs(input.gray, input.seg, input.charMask)
            val steps = NightGrayWorkload.program(w, h)
            var prims = 0
            var fma = 0
            for (st in steps) {
                if (NightGrayWorkload.runPrep(st, regs)) continue
                prims++
                if (st.fmaSensitive) fma++
                if (st.op != "gaussianBlur" && st.op != "boxBlur") {
                    assertFalse("$page ${st.id} 沒有浮點乘加，不該標 FMA 敏感", st.fmaSensitive)
                    continue
                }
                val f = regs.map.getValue(st.inp) as FImg
                val d = if (st.op == "gaussianBlur") FmaEmu.diff(Cv.gaussianBlur(f, st.args[0] / 10.0), FmaEmu.gaussian(f, st.args[0] / 10.0))
                else FmaEmu.diff(Cv.boxBlur(f, st.args[0]), FmaEmu.box(f, st.args[0]))
                println("c3b $page ${st.id}：合成 FMA 會不同的像素 $d")
                assertEquals("$page ${st.id} 的 fmaSensitive 標記（合成 FMA 不同的像素 $d）", d > 0, st.fmaSensitive)
            }
            assertEquals(page, NightProbeRunner.graySteps(w, h), prims)
            assertEquals(page, NightProbeRunner.grayFmaSteps(w, h), fma)
            assertTrue("$page：要有 FMA 敏感的步驟", fma > 0)
            val weighted = steps.filter { it.weight > 0 }.mapNotNull { it.family }.toSet()
            assertEquals("$page：每族都要有計入權重的步驟（不然那族的倍數是 NaN）", NightGrayWorkload.Family.values().toSet(), weighted)
            assertEquals("$page：步驟 id 不重複", steps.size, steps.map { it.id }.toSet().size)
        }
    }

    @Test
    fun extraFilterFmaFlagsAreReal() {
        var flagged = 0
        var cropSensitive = 0
        var cropTotal = 0
        for (page in NightProbeRunner.PRIM_PAGES) {
            val input = GuardDigest.input(::load, page)
            val filters = ProbeGrayExtras.filters(page, input.gray)
            for (fl in filters) {
                val want = fl.kotlin()
                val emu = if (fl.op == "gaussianBlur") FmaEmu.gaussian(fl.f, fl.sigma) else FmaEmu.box(fl.f, fl.k)
                val d = FmaEmu.diff(want, emu)
                when {
                    fl.fma -> {
                        flagged++
                        assertTrue("${fl.op} ${fl.tag} 標了 FMA，合成 FMA 卻一樣", d > 0)
                    }
                    "/full/" in fl.tag -> assertEquals("${fl.op} ${fl.tag} 是不受影響的對照（σ 0／核長 1），合成 FMA 卻不同", 0, d)
                    else -> {
                        cropTotal++
                        if (d > 0) cropSensitive++
                    }
                }
            }
            val cases = ProbeGrayExtras.pageCases(page, input)
            assertEquals("$page：案例裡標 FMA 的就是整頁濾波那幾組", filters.count { it.fma }, cases.count { it.fma })
            assertTrue("$page：只有濾波會標 FMA", cases.filter { it.fma }.all { it.op in ProbeGrayExtras.FILTER_OPS })
        }
        assertTrue("要有標 FMA 的案例", flagged > 0)
        println("c3b.x：標 FMA 的 $flagged 組合成 FMA 全部會不同；裁切的 $cropTotal 組裡 $cropSensitive 組也會（不標、不計）")
    }

    @Test
    fun extraCasesRunInKotlinWithoutThrowing() {
        val cases = ProbeGrayExtras.globalCases() + NightProbeRunner.PRIM_PAGES.flatMap { ProbeGrayExtras.pageCases(it, GuardDigest.input(::load, it)) }
        val thrown = cases.mapNotNull { c -> runCatching { c.kotlin() }.exceptionOrNull()?.let { "${c.op} ${c.tag}: $it" } }
        assertEquals("Kotlin 那半丟了例外：${thrown.take(5)}", 0, thrown.size)
        assertEquals(ProbeGrayExtras.OPS.toSet(), cases.map { it.op }.toSet())
        assertEquals("案例的標籤要唯一（紀錄裡才找得到是哪一個）", cases.size, cases.map { "${it.op}/${it.tag}" }.toSet().size)
        println("c3b.x：${cases.size} 組（其中標 FMA ${cases.count { it.fma }}；${cases.groupingBy { it.op }.eachCount()}）")
    }

    /**
     * Cv 的可分離濾波照抄，但每一個 Float 乘加都合成 FMA（Math.fma）：C++ 沒關掉合成時會算出的東西（同 nightread 的 CppGrayParityTest）。
     * 高斯核取 Cv 私有的 gaussianKernel（反射），反射邊界在這裡自己寫一份（Cv.reflect101 是 internal）。
     */
    private object FmaEmu {
        private val gaussianKernel = Cv::class.java.getDeclaredMethod("gaussianKernel", Int::class.javaPrimitiveType, Double::class.javaPrimitiveType)
            .also { it.isAccessible = true }

        private fun reflect101(i: Int, n: Int): Int {
            if (n == 1) return 0
            var x = i
            while (x < 0 || x >= n) {
                if (x < 0) x = -x
                if (x >= n) x = 2 * (n - 1) - x
            }
            return x
        }

        private fun sep(f: FImg, k: DoubleArray): FImg {
            val w = f.w
            val h = f.h
            val r = k.size / 2
            val kf = FloatArray(k.size) { k[it].toFloat() }
            val mid = FloatArray(w * h)
            for (y in 0 until h) {
                val base = y * w
                for (x in 0 until minOf(r, w)) {
                    var acc = 0f
                    for (t in kf.indices) acc = Math.fma(kf[t], f.data[base + reflect101(x + t - r, w)], acc)
                    mid[base + x] = acc
                }
                val hiX = w - r
                for (x in r until hiX) {
                    var acc = 0f
                    var idx = base + x - r
                    for (t in kf.indices) { acc = Math.fma(kf[t], f.data[idx], acc); idx++ }
                    mid[base + x] = acc
                }
                for (x in maxOf(r, hiX) until w) {
                    var acc = 0f
                    for (t in kf.indices) acc = Math.fma(kf[t], f.data[base + reflect101(x + t - r, w)], acc)
                    mid[base + x] = acc
                }
            }
            val out = FloatArray(w * h)
            for (y in 0 until h) {
                val obase = y * w
                if (y < r || y >= h - r) {
                    for (x in 0 until w) {
                        var acc = 0f
                        for (t in kf.indices) acc = Math.fma(kf[t], mid[reflect101(y + t - r, h) * w + x], acc)
                        out[obase + x] = acc
                    }
                } else {
                    for (t in kf.indices) {
                        val c = kf[t]
                        val sbase = (y + t - r) * w
                        if (t == 0) for (x in 0 until w) out[obase + x] = c * mid[sbase + x]
                        else for (x in 0 until w) out[obase + x] = Math.fma(c, mid[sbase + x], out[obase + x])
                    }
                }
            }
            return FImg(w, h, out)
        }

        fun gaussian(f: FImg, sigma: Double): FImg {
            if (sigma <= 0.0) return f.copy()
            var ks = Math.round(sigma * 4.0 * 2.0 + 1.0).toInt() or 1
            if (ks < 3) ks = 3
            return sep(f, gaussianKernel.invoke(Cv, ks, sigma) as DoubleArray)
        }

        fun box(f: FImg, k: Int): FImg = sep(f, DoubleArray(k) { 1.0 / k })

        fun diff(a: FImg, b: FImg): Int {
            assertEquals(a.data.size, b.data.size)
            var n = 0
            for (i in a.data.indices) if (a.data[i].toRawBits() != b.data[i].toRawBits()) n++
            return n
        }
    }
}
