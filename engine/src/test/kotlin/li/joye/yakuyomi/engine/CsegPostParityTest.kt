package li.joye.yakuyomi.engine

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.DataInputStream
import javax.imageio.ImageIO

/**
 * cseg 後處理的 parity：拿 `parity/export_cseg_ncnn.py --fixture` 存下的十個原始頭輸出（fp16），
 * 跑 [CsegPost] 的解碼／NMS／動態卷積遮罩／聯集，對 numpy 參考實作的期望遮罩逐像素比。
 * numpy 那份已證明與完整 ONNX（圖內 NMS）逐像素相同（12 頁機率差 0.000），所以這裡對上＝對上上游。
 */
class CsegPostParityTest {

    private fun res(name: String) = javaClass.classLoader!!.getResourceAsStream("charseg/$name")!!

    /** IEEE 半精度 → float（JVM 17 沒有 Float.float16ToFloat）。 */
    private fun halfToFloat(h: Int): Float {
        val s = (h shr 15) and 1
        val e = (h shr 10) and 0x1F
        val f = h and 0x3FF
        val bits = when (e) {
            0 -> if (f == 0) s shl 31 else {   // subnormal
                var m = f
                var ex = 1
                while (m and 0x400 == 0) { m = m shl 1; ex-- }
                (s shl 31) or ((ex + 127 - 15) shl 23) or ((m and 0x3FF) shl 13)
            }
            0x1F -> (s shl 31) or 0x7F800000 or (f shl 13)
            else -> (s shl 31) or ((e + 127 - 15) shl 23) or (f shl 13)
        }
        return java.lang.Float.intBitsToFloat(bits)
    }

    @Test
    fun unionMaskMatchesNumpyReference() {
        val page = "ch34_011"
        val meta = res("${page}_meta.txt").bufferedReader().readLines()
        fun metaInt(key: String) = meta.first { it.startsWith("$key ") }.substringAfter(' ').trim().toInt()
        val w = metaInt("w")
        val h = metaInt("h")
        val nw = metaInt("nw")
        val nh = metaInt("nh")
        val shapes = meta.first { it.startsWith("shapes ") }.substringAfter(' ').split(';')
            .map { s -> s.split('x').map { it.toInt() } }
        val expectedDets = meta.filter { it.startsWith("det ") }.size

        val outs = CsegPost.allocOutputs()
        DataInputStream(res("${page}_raw.bin").buffered()).use { din ->
            for ((i, shape) in shapes.withIndex()) {
                val n = shape.reduce { a, b -> a * b }
                assertEquals("blob $i 大小", outs[i].size, n)
                for (j in 0 until n) {
                    val lo = din.read()
                    val hi = din.read()
                    outs[i][j] = halfToFloat((hi shl 8) or lo)   // little-endian fp16
                }
            }
        }

        val dets = CsegPost.decode(outs)
        assertEquals("NMS 後實例數", expectedDets, dets.size)

        val got = CsegPost.unionMask(outs, nw, nh, w, h)
        val img = ImageIO.read(res("${page}_expected.png"))
        assertEquals(w, img.width)
        assertEquals(h, img.height)
        var inter = 0
        var union = 0
        var diff = 0
        val raster = img.raster
        for (y in 0 until h) for (x in 0 until w) {
            val e = raster.getSample(x, y, 0) > 127
            val g = got[y * w + x]
            if (e && g) inter++
            if (e || g) union++
            if (e != g) diff++
        }
        val iou = inter.toDouble() / union
        println("cseg 後處理 parity：$page ${w}×$h 實例 ${dets.size} 聯集 IoU=${"%.5f".format(iou)} 不同像素 $diff")
        // 門檻 0.5 附近的像素 sigmoid/exp 實作差異可能翻面；容 0.1% 像素
        assertTrue("聯集遮罩 IoU $iou 太低", iou >= 0.999)
        assertTrue("不同像素 $diff 太多", diff <= w * h / 1000)
    }
}
