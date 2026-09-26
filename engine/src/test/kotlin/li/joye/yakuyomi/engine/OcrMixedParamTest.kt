package li.joye.yakuyomi.engine

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * OCR 混合精度 param（`ocr_48px_ctc_mixed.ncnn.param`，parity/export_ocr_ncnn.py write_mixed_param）的結構守門。
 *
 * 規則來自 ncnn 的 featmask 語意（net.cpp get_masked_option／convert_layout）：`31=` 只管**這層自己**怎麼算，
 * **不管進來的 blob**——masked 層（fp32）若直接吃到未 mask 層（fp16）的輸出，Net 不會 cast，Permute 會把 fp16 當 fp32
 * 寫兩倍量 → 堆積溢出 → 真機在別的執行緒的卷積裡 SIGSEGV（2026-09-21 第一版就是這樣）。所以：
 *  1. 每個 masked 層的每個輸入，都必須來自 masked 層、`in1`（PE，由 Extractor 直接塞 fp32）或 `Cast`（fp16→fp32）。
 *  2. Cast 必須是 `0=2 1=1`、本身**不**mask（它要靠 Cast_arm 的 support_fp16_storage 收到原生 fp16）。
 *  3. mask 值含 bit0/1/2（fp16 arith、fp16 storage/packed、bf16）。
 *  4. 表頭層數／blob 數與內容一致；與原版 param 只差 Cast 那一層與 mask 尾巴（權重順序不變 → .bin 共用）。
 */
class OcrMixedParamTest {

    private fun res(name: String) = javaClass.classLoader!!.getResourceAsStream("ocr/$name")!!.bufferedReader().readLines()

    private class L(val type: String, val name: String, val ins: List<String>, val outs: List<String>, val params: Map<String, String>)

    private fun parse(lines: List<String>): Pair<IntArray, List<L>> {
        val header = lines[1].trim().split(Regex("\\s+")).map { it.toInt() }.toIntArray()
        val layers = lines.drop(2).filter { it.isNotBlank() }.map { line ->
            val f = line.trim().split(Regex("\\s+"))
            val nIn = f[2].toInt()
            val nOut = f[3].toInt()
            val ins = f.subList(4, 4 + nIn)
            val outs = f.subList(4 + nIn, 4 + nIn + nOut)
            val params = f.drop(4 + nIn + nOut).associate { kv -> kv.substringBefore('=') to kv.substringAfter('=') }
            L(f[0], f[1], ins, outs, params)
        }
        return header to layers
    }

    @Test
    fun maskedLayersOnlyConsumeFp32Producers() {
        val (header, layers) = parse(res("ocr_48px_ctc_mixed.ncnn.param"))
        assertEquals("表頭層數", layers.size, header[0])
        val blobs = layers.flatMap { it.outs }.toSet()
        assertEquals("表頭 blob 數", blobs.size, header[1])

        val producer = HashMap<String, L>()
        for (l in layers) for (o in l.outs) producer[o] = l
        val masked = layers.filter { "31" in it.params }
        assertTrue("應有 masked 層", masked.isNotEmpty())
        for (l in masked) {
            val m = l.params["31"]!!.toInt()
            assertEquals("${l.name} 的 mask 應含 fp16 arith/storage/bf16 三個位元", 7, m and 7)
            for (b in l.ins) {
                val p = producer[b] ?: error("${l.name} 的輸入 $b 沒有 producer")
                val ok = "31" in p.params || p.type == "Cast" || (p.type == "Input" && p.name == "in1")
                assertTrue("masked 層 ${l.name}(${l.type}) 吃到未 mask 的 ${p.name}(${p.type}) 輸出 $b：Net 不會 cast、會把 fp16 當 fp32", ok)
            }
        }
        val casts = layers.filter { it.type == "Cast" }
        assertEquals("恰一個 Cast（backbone → transformer 邊界）", 1, casts.size)
        val cast = casts[0]
        assertEquals("Cast 應 fp16→fp32", "2", cast.params["0"])
        assertEquals("Cast 應 fp16→fp32", "1", cast.params["1"])
        assertTrue("Cast 本身不能 mask（要靠 Cast_arm 收原生 fp16）", "31" !in cast.params)
        assertTrue("Cast 的輸入應來自卷積", producer[cast.ins[0]]!!.type == "Convolution")
        // in0 那側（backbone）全部不 mask
        val convs = layers.filter { it.type == "Convolution" }
        assertTrue("backbone 卷積不該 mask", convs.none { "31" in it.params })
        println("mixed param：${layers.size} 層、masked ${masked.size} 層、Cast=${cast.name}（${cast.ins[0]} → ${cast.outs[0]}）")
    }

    @Test
    fun mixedMatchesBaseExceptCastAndMask() {
        val (_, base) = parse(res("ocr_48px_ctc.ncnn.param"))
        val (_, mixed) = parse(res("ocr_48px_ctc_mixed.ncnn.param"))
        val mixedNoCast = mixed.filter { it.type != "Cast" }
        assertEquals("除 Cast 外層數相同", base.size, mixedNoCast.size)
        for ((a, b) in base.zip(mixedNoCast)) {
            assertEquals("層型別順序要一樣（.bin 才能共用）", a.type, b.type)
            assertEquals(a.name, b.name)
            val pa = a.params
            val pb = b.params.filterKeys { it != "31" }
            assertEquals("${a.name} 除 31= 外參數應相同", pa, pb)
        }
    }
}
