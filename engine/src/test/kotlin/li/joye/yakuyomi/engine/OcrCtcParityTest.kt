package li.joye.yakuyomi.engine

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * NCNN OCR 路徑裡「Kotlin 自己算」的兩塊對 parity fixture（`parity/export_ocr_ncnn.py --fixture`）：
 *  · [Ocr.sinusoidalPe]：正弦位置表 vs numpy `make_pe`（ocr/strip0_pe.bin，T×320 float32）——PE 是模型輸入，錯一列就整條讀錯。
 *  · [Ocr.ctcCollapse]：拿 fixture 的逐時步 idx/logp（ORT fp32 算的）收合 → 文字要等於 numpy `ctc_decode`、prob 對上。
 * 推論本身（JNI）桌面 JVM 跑不了，那段由 export 腳本的 30 條字條 + 寬度掃描驗（30/30 逐行相同）。
 */
class OcrCtcParityTest {

    private fun res(name: String) = javaClass.classLoader!!.getResourceAsStream("ocr/$name")!!

    private fun meta(): Map<String, String> = res("strip0_meta.txt").bufferedReader().readLines()
        .filter { it.isNotBlank() }.associate { it.substringBefore(' ') to it.substringAfter(' ') }

    @Test
    fun sinusoidalPeMatchesNumpy() {
        val t = meta()["t"]!!.toInt()
        val bytes = res("strip0_pe.bin").readBytes()
        assertEquals("fixture PE 大小", t * 320 * 4, bytes.size)
        val ref = FloatArray(t * 320).also { ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer().get(it) }
        val got = Ocr.sinusoidalPe(t)
        var maxDiff = 0f
        for (i in ref.indices) maxDiff = maxOf(maxDiff, Math.abs(ref[i] - got[i]))
        println("PE parity：T=$t max|Δ|=$maxDiff")
        assertTrue("PE 與 numpy 差太多 $maxDiff", maxDiff <= 1e-5f)
    }

    @Test
    fun ctcCollapseMatchesNumpy() {
        val m = meta()
        val t = m["t"]!!.toInt()
        val idx = m["idx"]!!.trim().split(' ').map { it.toInt() }.toIntArray()
        val logp = m["logp"]!!.trim().split(' ').map { it.toFloat() }.toFloatArray()
        assertEquals(t, idx.size)
        val alphabet = File("src/main/assets/models/alphabet-all-v5.txt").readLines()
        val (text, prob) = Ocr.ctcCollapse(idx, logp, t, alphabet)
        println("CTC 收合：「$text」 p=$prob（期望「${m["text"]}」）")
        assertEquals(m["text"], text)
        // prob＝exp(留下字的 logp 平均)：用 fixture 自己重算一次對照
        var last = 0; var sum = 0.0; var n = 0
        for (i in 0 until t) { if (idx[i] != last && idx[i] != 0) { sum += logp[i]; n++ }; last = idx[i] }
        assertEquals(Math.exp(sum / n).toFloat(), prob, 1e-5f)
    }
}
