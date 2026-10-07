package li.joye.yakuyomi.engine

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.DataInputStream
import javax.imageio.ImageIO

/**
 * yoloseg 後處理的 parity：拿 `parity/export_yoloseg_ncnn.py --fixture` 存下的 out0/out1（fp16），
 * 跑 [YoloSegPost] 的篩選／NMS／係數×prototype／兩段雙線性／聯集，對 numpy 參考（＝research 的
 * run_yoloseg_onnx，桌面守護框數字的來源）逐像素比。
 */
class YoloSegPostParityTest {

    private fun res(name: String) = javaClass.classLoader!!.getResourceAsStream("charseg/$name")!!

    private fun halfToFloat(h: Int): Float {
        val s = (h shr 15) and 1
        val e = (h shr 10) and 0x1F
        val f = h and 0x3FF
        val bits = when (e) {
            0 -> if (f == 0) s shl 31 else {
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
        val meta = res("${page}_yolo_meta.txt").bufferedReader().readLines()
        fun metaInt(key: String) = meta.first { it.startsWith("$key ") }.substringAfter(' ').trim().toInt()
        val w = metaInt("w")
        val h = metaInt("h")
        val pre = YoloSegPost.Pre(FloatArray(0), metaInt("nw"), metaInt("nh"), metaInt("top"), metaInt("left"))
        val expectedInstances = metaInt("instances")

        val outs = YoloSegPost.allocOutputs()
        DataInputStream(res("${page}_yolo_raw.bin").buffered()).use { din ->
            for (o in outs) {
                for (j in o.indices) {
                    val lo = din.read()
                    val hi = din.read()
                    o[j] = halfToFloat((hi shl 8) or lo)
                }
            }
        }

        val dets = YoloSegPost.decode(outs[0])
        assertEquals("NMS 後實例數", expectedInstances, dets.size)

        val got = YoloSegPost.unionMask(outs[0], outs[1], pre, w, h)
        val img = ImageIO.read(res("${page}_yolo_expected.png"))
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
        println("yoloseg 後處理 parity：$page ${w}×$h 實例 ${dets.size} 聯集 IoU=${"%.5f".format(iou)} 不同像素 $diff")
        assertTrue("聯集遮罩 IoU $iou 太低", iou >= 0.999)
        assertTrue("不同像素 $diff 太多", diff <= w * h / 1000)
    }
}
