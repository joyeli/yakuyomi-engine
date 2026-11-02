package li.joye.yakuyomi.sandbox.probe

import li.joye.yakuyomi.nightread.NightBinWorkload
import li.joye.yakuyomi.nightread.NightPrimWorkload
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.File

/**
 * 探針 (c) 的第 3a 階步驟（c3a、c3a.x）在電腦上能做的部分：C++ 只在手機上跑（Windows 的 Gradle 載不了原生庫），這裡只確認
 * 頁有打包進 assets、探針自己解碼的頁餵得進 [NightBinWorkload] 的步驟表（Kotlin 那半）、關卡要比滿的步數算得對、
 * 補充案例的 Kotlin 那半都不丟例外（兩邊都丟在手機上算相同，比了等於沒比）。
 */
class NightProbeBinJvmTest {
    private val dir = File("../yakuyomi-nightread/nightread/src/test/resources/page")

    private fun load(name: String): ByteArray? = File(dir, name).takeIf { it.isFile }?.readBytes()

    @Test
    fun primPagesArePackaged() {
        for (p in NightProbeRunner.PRIM_PAGES) assertTrue("$p 沒打包進 assets（不在守門頁清單）", p in GuardDigest.PAGES)
    }

    @Test
    fun binStepTableRunsOnProbePagesInKotlin() {
        for (page in NightProbeRunner.PRIM_PAGES) {
            val input = GuardDigest.input(::load, page)
            val regs = NightPrimWorkload.Regs(input.gray, input.seg, input.charMask)
            val steps = NightBinWorkload.program(input.gray.w, input.gray.h)
            var prims = 0
            for (st in steps) {
                if (NightBinWorkload.runPrep(st, regs)) continue
                NightBinWorkload.runKotlin(st, regs)
                prims++
            }
            assertEquals(page, NightProbeRunner.binSteps(input.gray.w, input.gray.h), prims)
            val weighted = steps.filter { it.weight > 0 }.mapNotNull { it.family }.toSet()
            assertEquals("$page：每族都要有計入權重的步驟（不然那族的倍數是 NaN）", NightBinWorkload.Family.values().toSet(), weighted)
        }
    }

    @Test
    fun extraCasesRunInKotlinWithoutThrowing() {
        val cases = ProbeBinExtras.globalCases() + NightProbeRunner.PRIM_PAGES.flatMap { ProbeBinExtras.pageCases(it, GuardDigest.input(::load, it)) }
        val thrown = cases.mapNotNull { c -> runCatching { c.kotlin() }.exceptionOrNull()?.let { "${c.op} ${c.tag}: $it" } }
        assertEquals("Kotlin 那半丟了例外：${thrown.take(5)}", 0, thrown.size)
        assertEquals(ProbeBinExtras.OPS.toSet(), cases.map { it.op }.toSet())
        assertEquals("案例的標籤要唯一（紀錄裡才找得到是哪一個）", cases.size, cases.map { "${it.op}/${it.tag}" }.toSet().size)
        println("c3a.x：${cases.size} 組（${cases.groupingBy { it.op }.eachCount()}）")
    }
}
