package li.joye.yakuyomi.sandbox.probe

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

/** 關卡不能在沒比到東西時回報通過。 */
class ProbeRecordsJvmTest {
    private fun pair(th: Int, page: String, a: String, b: String, mismatches: Int = 0, compared: Int = 4, noExpect: Int = 0) =
        "PAIR\tpage=$page\ta=$a\tb=$b\tthreads=$th\tmismatches=$mismatches\tcompared=$compared\tnoExpect=$noExpect"

    private fun tp(mode: String, wall: Long = 1000, mismatches: Int = 0, compared: Int = 4, noExpect: Int = 0) =
        "TP\tmode=$mode\twallMs=$wall\tmismatches=$mismatches\tcompared=$compared\tnoExpect=$noExpect"

    @Test fun verdictNeedsAllCompared() {
        assertEquals("true", ProbeRecords.verdict(0, 60, 60))
        assertEquals("incomplete", ProbeRecords.verdict(0, 59, 60))
        assertEquals("incomplete", ProbeRecords.verdict(0, 0, 60))
        assertEquals("false", ProbeRecords.verdict(1, 60, 60))
        assertEquals("false", ProbeRecords.verdict(1, 3, 60)) // 有不同就是沒過，不管比了幾組
    }

    @Test fun noPairsIsNotPass() {
        val g = ProbeRecords.pairs(listOf("FAIL\tstep=d.pair.3.a.jpg.AOT+DBNET\terror=x"))
        assertEquals(0, g.pairs)
        assertEquals("incomplete", ProbeRecords.verdict(g.mismatches, g.compared, 60))
    }

    @Test fun pairWithoutBaselineIsNotCompared() {
        val g = ProbeRecords.pairs(listOf(pair(3, "p", "AOT", "DBNET", compared = 0, noExpect = 6), pair(3, "p", "AOT", "CSEG", compared = 3, noExpect = 2), pair(3, "p", "YOLO", "CSEG")))
        assertEquals(3, g.pairs)
        assertEquals(1, g.compared)
        assertEquals("incomplete", ProbeRecords.verdict(g.mismatches, g.compared, 3))
    }

    @Test fun duplicatedPairCountsOnce() {
        val l = pair(3, "p", "AOT", "DBNET")
        val g = ProbeRecords.pairs(listOf(l, l, pair(6, "p", "AOT", "DBNET")))
        assertEquals(2, g.pairs)
        assertEquals(2, g.compared)
    }

    @Test fun throughputCountsModes() {
        val g = ProbeRecords.throughput(listOf(tp("A1"), tp("B1", compared = 0, noExpect = 4), tp("C", mismatches = 1)))
        assertEquals(3, g.wallMs.size)
        assertEquals(2, g.comparedModes)
        assertEquals(1, g.mismatches)
        assertEquals(4, g.noExpect)
        assertEquals("false", ProbeRecords.verdict(g.mismatches, g.comparedModes, 5))
        assertEquals("incomplete", ProbeRecords.verdict(0, 2, 5))
    }

    @Test fun tornLineIsHarmless() {
        val g = ProbeRecords.throughput(listOf("TP\tmode=A1\twallMs=", "TP\t"))
        assertEquals(0, g.wallMs.size)
        assertEquals(0, ProbeRecords.pairs(listOf("PAIR\tpage=p")).compared)
    }

    @Test fun foregroundCountsFromFileAcrossResume() {
        val lines = listOf(
            "STEP\tid=a\ttitle=x",
            "ENV\tat=begin\tstep=a\timportance=100",
            "STEP\tid=c\ttitle=x",
            "ENV\tat=begin\tstep=c\timportance=100",
            "WARN\twhat=${ProbeRecords.LEFT_FOREGROUND}",
            "ENV\tat=end\tstep=c\timportance=100",
            "RESUME\tdone=3",
            "STEP\tid=d.pair.3.x\ttitle=x",
            "ENV\tat=begin\tstep=d.pair.3.x\timportance=125",
            "WARN\twhat=不在最上層，這段的時間不能算\tstep=d.pair.3.x",
        )
        val fg = ProbeRecords.foreground(lines)
        assertEquals(1, fg.left)
        assertEquals(setOf("c", "d.pair.3.x"), fg.steps)
    }

    @Test fun failedStepsDistinct() {
        assertEquals(
            listOf("d.page.a", "c"),
            ProbeRecords.failedSteps(listOf("FAIL\tstep=d.page.a\terror=x", "FAIL\tstep=c", "FAIL\tstep=d.page.a", "STEP\tid=z")),
        )
    }

    @Test fun tombstonesDoNotOverwriteEachOther() {
        val f = "1102_120000_nightprobe.txt"
        val a = ProbeRecords.tombstoneName(f, "d.pair.3.p.jpg.AOT+DBNET", "1102_120100")
        val b = ProbeRecords.tombstoneName(f, "d.pair.3.p.jpg.AOT+CSEG", "1102_120200")
        val c = ProbeRecords.tombstoneName(f, null, "1102_120300")
        assertEquals(3, setOf(a, b, c).size)
        assertFalse(a.contains('+'))
        for (n in listOf(a, b, c, f)) assertTrue(n, ProbeRecords.belongsToReport(n, f))
        assertFalse(ProbeRecords.belongsToReport("1102_120000_nightprobe_exit.pb", f))
        assertFalse(ProbeRecords.belongsToReport(ProbeRecords.tombstoneName("1102_130000_nightprobe.txt", null, "t"), f))
        assertFalse(ProbeRecords.belongsToReport("other.txt", f))
    }
}
