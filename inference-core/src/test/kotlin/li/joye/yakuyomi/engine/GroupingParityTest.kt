package li.joye.yakuyomi.engine

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import kotlin.math.abs
import kotlin.math.roundToInt

/**
 * 跨語言 parity：Kotlin Grouping 必須與已驗證的 parity/mit_grouping.py 在相同偵測輸入下給相同區域（§7、§12 規則 4）。
 * fixture（行 quads + 期望區域 bbox）由 parity/emit_grouping_fixture.py 對 test/raw 真圖產生。
 */
class GroupingParityTest {

    @Test
    fun matchesMitGrouping() {
        for (page in GroupingFixture.pages) {
            val lines = page.lines.map { TextLine(it, 1f) }
            val regions = Grouping.group(lines)

            assertEquals("${page.name}：區域數", page.regions.size, regions.size)

            val cmp = compareBy<IntArray>({ it[0] }, { it[1] }, { it[2] }, { it[3] })
            val got = regions.map {
                intArrayOf(it.x0.toInt(), it.y0.toInt(), it.x1.toInt(), it.y1.toInt(), it.angle.roundToInt())
            }.sortedWith(cmp)
            val exp = page.regions.sortedWith(cmp)

            for (i in exp.indices) {
                for (k in 0..3) {
                    assertTrue(
                        "${page.name} 區域[$i] 座標$k：期望 ${exp[i][k]} 得 ${got[i][k]}",
                        abs(exp[i][k] - got[i][k]) <= 2,
                    )
                }
                assertTrue(
                    "${page.name} 區域[$i] 角度：期望 ${exp[i][4]} 得 ${got[i][4]}",
                    abs(exp[i][4] - got[i][4]) <= 1,
                )
            }
        }
    }
}
