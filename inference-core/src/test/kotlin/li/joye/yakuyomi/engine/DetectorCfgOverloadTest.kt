package li.joye.yakuyomi.engine

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import kotlin.math.abs

/**
 * `Detector.detect(page, cfg)` 多載的後處理段（[Detector.linesFromLogits]）：門檻、unclip、minSide 全看**傳進來的** cfg。
 * 前向要原生、拿不到 Bitmap，JVM 只能打這段——它是 [Detector.detect] 兩個多載與 [Detector.postprocess] 共用的那一段，
 * 「cfg 從哪裡來」只在呼叫端決定。合成 logits：背景 −10（sigmoid ≈ 0），中間一塊 [BLOCK_LOGIT]（sigmoid ≈ 0.881）。
 */
class DetectorCfgOverloadTest {

    private val w = 64
    private val h = 32

    /** 2 通道 db（ch0 有塊、ch1 不用）。塊＝x∈[10,40)、y∈[10,20)。 */
    private fun logits(): FloatArray {
        val db = FloatArray(2 * w * h) { -10f }
        for (y in 10 until 20) for (x in 10 until 40) db[y * w + x] = BLOCK_LOGIT
        return db
    }

    private fun run(cfg: DetectorConfig, ratio: Float = 1f, origW: Int = w, origH: Int = h): List<TextLine> =
        Detector.linesFromLogits(logits(), w, h, ratio, origW, origH, cfg)

    private fun width(q: List<Pt>): Float = q.maxOf { it.x } - q.minOf { it.x }

    @Test
    fun defaultCfgFindsTheBlock() {
        val lines = run(DetectorConfig())
        assertEquals(1, lines.size)
        // score＝連通元件平均機率＝sigmoid(BLOCK_LOGIT)
        assertEquals(SIGMOID_BLOCK, lines[0].score, 1e-5f)
    }

    @Test
    fun boxThresholdComesFromTheGivenCfg() {
        // 0.881 < 0.9 → 整塊被分數門檻濾掉；預設 0.7 會留（上一個測試）
        assertEquals(0, run(DetectorConfig(dbBoxThreshold = 0.9f)).size)
    }

    @Test
    fun binThresholdComesFromTheGivenCfg() {
        // 機率 0.881 沒有超過 0.9 → 一個像素都不算前景
        assertEquals(0, run(DetectorConfig(dbBinThreshold = 0.9f)).size)
    }

    @Test
    fun minSideComesFromTheGivenCfg() {
        // 塊短邊 10 格，unclip 之前就比 minSide 小 → 丟
        assertEquals(0, run(DetectorConfig(minSide = 50f)).size)
        assertEquals(1, run(DetectorConfig(minSide = 5f)).size)
    }

    @Test
    fun unclipRatioComesFromTheGivenCfg() {
        val tight = run(DetectorConfig(dbUnclipRatio = 1f), origW = 1000, origH = 1000)
        val loose = run(DetectorConfig(dbUnclipRatio = 3f), origW = 1000, origH = 1000)
        assertEquals(1, tight.size)
        assertEquals(1, loose.size)
        assertTrue("unclip 大的框要比較寬", width(loose[0].quad) > width(tight[0].quad) + 1f)
    }

    @Test
    fun coordinatesAreDividedByRatioAndClampedToThePage() {
        val base = run(DetectorConfig(), ratio = 1f, origW = 1000, origH = 1000)[0].quad
        val half = run(DetectorConfig(), ratio = 0.5f, origW = 1000, origH = 1000)[0].quad
        base.zip(half).forEach { (a, b) ->
            assertEquals(a.x * 2f, b.x, 1e-3f)
            assertEquals(a.y * 2f, b.y, 1e-3f)
        }
        // 頁比網格小 → 夾進頁面
        val clamped = run(DetectorConfig(dbUnclipRatio = 3f), origW = 30, origH = 15)[0].quad
        assertTrue(clamped.all { it.x in 0f..30f && it.y in 0f..15f })
    }

    @Test
    fun sameCfgSameLines() {
        // 帶等值的兩份 cfg（一份預設建構、一份逐欄照抄）結果逐值相同：多載不看 cfg 物件身分
        val a = run(DetectorConfig())
        val d = DetectorConfig()
        val b = run(
            DetectorConfig(
                minSide = d.minSide,
                segThreshold = d.segThreshold,
                dbnetInputSize = d.dbnetInputSize,
                detectUnsharp = d.detectUnsharp,
                dbBinThreshold = d.dbBinThreshold,
                dbBoxThreshold = d.dbBoxThreshold,
                dbUnclipRatio = d.dbUnclipRatio,
            ),
        )
        assertEquals(a.size, b.size)
        a.zip(b).forEach { (x, y) ->
            assertEquals(x.score, y.score, 0f)
            x.quad.zip(y.quad).forEach { (p, q) -> assertTrue(abs(p.x - q.x) == 0f && abs(p.y - q.y) == 0f) }
        }
    }

    private companion object {
        const val BLOCK_LOGIT = 2f
        val SIGMOID_BLOCK = 1f / (1f + kotlin.math.exp(-BLOCK_LOGIT))
    }
}
