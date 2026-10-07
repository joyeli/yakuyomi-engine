package li.joye.yakuyomi.engine

import li.joye.yakuyomi.nightread.Gray
import li.joye.yakuyomi.nightread.Mask
import li.joye.yakuyomi.nightread.NightReadInput
import li.joye.yakuyomi.nightread.NightReadParams
import li.joye.yakuyomi.nightread.NightTier
import li.joye.yakuyomi.nightread.TextRegion
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.File
import javax.imageio.ImageIO
import kotlin.math.abs

/**
 * 譯後頁的去字遮罩（nightread 規則版本 4 的 `NightReadInput.inpaintMask`）經 [NightReadRenderer] 進函式庫的那一段。
 * Bitmap 在 JVM 單元測試裡不能用，所以打的是 Bitmap 以外的純函式：[NightReadRenderer.resampleMask]（Bitmap 版
 * maskFromBitmap 就是它加逐列 getPixels）、[NightReadRenderer.nearestIndex]、[NightReadRenderer.budgetSize]、
 * [NightReadRenderer.scaleLines]、[NightReadRenderer.maskFitsPage]，再用 [NightReadRenderer.streamTiers] 把結果交給函式庫。
 *  - 尺寸與方向：從 2× 解析度的來源對應回頁尺寸要逐像素等於原遮罩；交給函式庫的成品要等於函式庫拿原遮罩的成品，
 *    左右翻過的遮罩成品不同（方向錯了測得出來）、不給遮罩也不同（遮罩真的有用到）；「標準」三種情況都相同。
 *  - 縮圖：頁超過 MAX_PIXELS 時遮罩跟頁一起縮，對到的位置與 extraLines 縮後的字框一致（±1 px）。
 * fixture＝nightread 函式庫的合成守門頁 syn_v4（帶去字遮罩 syn_v4_inpaint.png；engine 的巢狀 submodule）。
 */
class NightReadInpaintMaskTest {

    private val dir = File("../yakuyomi-nightread/nightread/src/test/resources/page")

    private fun file(name: String): File = File(dir, name).also {
        check(it.isFile) { "缺 fixture：${it.path}（engine 的 yakuyomi-nightread submodule 沒 checkout？）" }
    }

    private fun readGray(name: String): Gray {
        val img = ImageIO.read(file(name))
        val g = Gray(img.width, img.height)
        val raster = img.raster
        for (y in 0 until img.height) for (x in 0 until img.width) g.data[y * img.width + x] = raster.getSample(x, y, 0)
        return g
    }

    private fun readMask(name: String): Mask {
        val g = readGray(name)
        return Mask(g.w, g.h, BooleanArray(g.data.size) { g.data[it] > 127 })
    }

    private fun readRegions(name: String): List<TextRegion> =
        file(name).readText().lineSequence().filter { it.isNotBlank() }.map {
            val p = it.trim().split(" ").map(String::toInt)
            TextRegion(p[0], p[1], p[2], p[3])
        }.toList()

    private fun input(page: String) = NightReadInput(
        gray = readGray("${page}_gray.png"),
        seg = readMask("${page}_seg.png"),
        regions = readRegions("${page}_regions.txt"),
        charMask = readMask("${page}_char.png"),
        chroma = readGray("${page}_chroma.png"),
    )

    private fun argb(on: Boolean): Int = if (on) WHITE else BLACK

    /** 產品兩檔（標準、更多）各自要顯示的灰階（沒交圖＝與上一檔相同）。 */
    private fun productTiers(inp: NightReadInput): List<ByteArray> {
        val tiers = listOf(NightTier.L2, NightTier.L3)
        val px = IntArray(inp.gray.w * inp.gray.h)
        val out = ArrayList<ByteArray>(tiers.size)
        NightReadRenderer.streamTiers(inp, tiers, NightReadParams(), px, debug = null) { _, emitted, _ ->
            out += if (emitted) ByteArray(px.size) { (px[it] and 0xFF).toByte() } else out.last()
        }
        assertEquals(tiers.size, out.size)
        return out
    }

    private fun bbox(m: Mask): IntArray {
        var x0 = m.w
        var y0 = m.h
        var x1 = 0
        var y1 = 0
        for (y in 0 until m.h) for (x in 0 until m.w) {
            if (m[x, y]) {
                x0 = minOf(x0, x)
                y0 = minOf(y0, y)
                x1 = maxOf(x1, x + 1)
                y1 = maxOf(y1, y + 1)
            }
        }
        return intArrayOf(x0, y0, x1, y1)
    }

    /** 期望值＝`cv2.resize(np.arange(src)[None], (dst, 1), interpolation=cv2.INTER_NEAREST)`（2026-10-06 研究端 python 跑的）。 */
    @Test
    fun nearestIndexIsCv2InterNearest() {
        assertArrayEquals(intArrayOf(0, 1, 2, 3, 4), NightReadRenderer.nearestIndex(5, 5))
        // 縮：ifx = 2.5
        assertArrayEquals(intArrayOf(0, 2, 5, 7), NightReadRenderer.nearestIndex(10, 4))
        // 放：ifx = 0.4（floor，不是半像素中心）
        assertArrayEquals(intArrayOf(0, 0, 0, 1, 1, 2, 2, 2, 3, 3), NightReadRenderer.nearestIndex(4, 10))
        // 多一像素（ifx = 0.8）：重複的是開頭那格，不是最後一格
        assertArrayEquals(intArrayOf(0, 0, 1, 2, 3), NightReadRenderer.nearestIndex(4, 5))
        // 降採樣解碼的取整（遮罩 1200 → 頁 1201）：cv2 前十個 0,0,1,…,8、最後三個 1197,1198,1199
        val r = NightReadRenderer.nearestIndex(1200, 1201)
        assertArrayEquals(intArrayOf(0, 0, 1, 2, 3, 4, 5, 6, 7, 8), r.copyOfRange(0, 10))
        assertArrayEquals(intArrayOf(1197, 1198, 1199), r.copyOfRange(1198, 1201))
    }

    @Test
    fun maskReachesLibraryAtPageScaleAndOrientation() {
        val ref = readMask("syn_v4_inpaint.png")
        assertTrue("syn_v4 的去字遮罩不該是空的", ref.data.any { it })
        val base = input("syn_v4")
        assertEquals("遮罩與頁同尺寸", base.gray.w to base.gray.h, ref.w to ref.h)

        // 來源是 2× 解析度（每像素複製成 2×2，白＝去字區）：對應回頁尺寸要逐像素等於原遮罩
        val sw = ref.w * 2
        val sh = ref.h * 2
        val reads = HashSet<Int>()
        val mapped = NightReadRenderer.resampleMask(sw, sh, ref.w, ref.h) { y, row ->
            assertTrue("來源列 $y 讀了兩次", reads.add(y))
            for (x in 0 until sw) row[x] = argb(ref[x / 2, y / 2])
        }
        assertEquals(ref.w to ref.h, mapped.w to mapped.h)
        assertArrayEquals("2× 來源縮回頁尺寸要逐像素等於原遮罩", ref.data, mapped.data)
        // 同尺寸＝逐像素照搬
        val same = NightReadRenderer.resampleMask(ref.w, ref.h, ref.w, ref.h) { y, row ->
            for (x in 0 until ref.w) row[x] = argb(ref[x, y])
        }
        assertArrayEquals(ref.data, same.data)

        val withRef = productTiers(base.copy(inpaintMask = ref))
        val withMapped = productTiers(base.copy(inpaintMask = mapped))
        val without = productTiers(base)
        val flipped = Mask(ref.w, ref.h, BooleanArray(ref.data.size) { ref[ref.w - 1 - it % ref.w, it / ref.w] })
        val withFlipped = productTiers(base.copy(inpaintMask = flipped))

        assertArrayEquals("標準：經引擎的遮罩與函式庫自己讀的相同", withRef[0], withMapped[0])
        assertArrayEquals("更多：經引擎的遮罩與函式庫自己讀的相同", withRef[1], withMapped[1])
        assertFalse("更多：不給遮罩要不同（遮罩真的進了函式庫）", withRef[1].contentEquals(without[1]))
        assertFalse("更多：左右翻過的遮罩要不同（方向錯了測得出來）", withRef[1].contentEquals(withFlipped[1]))
        assertArrayEquals("標準不看去字遮罩（沒給）", withRef[0], without[0])
        assertArrayEquals("標準不看去字遮罩（翻過）", withRef[0], withFlipped[0])
    }

    @Test
    fun downscaledPageKeepsMaskAlignedWithExtraLines() {
        // 7.68 MPx 頁 → 引擎縮到 ≤ MAX_PIXELS；字框（extraLines，頁座標）與去字遮罩（頁尺寸）要縮到同一個位置
        val pw = 2400
        val ph = 3200
        val size = NightReadRenderer.budgetSize(pw, ph)
        assertNotNull("超過上限要縮", size)
        val (nw, nh) = size!!
        assertTrue(nw.toLong() * nh <= NightReadRenderer.MAX_PIXELS)
        assertNull("沒超過上限不縮", NightReadRenderer.budgetSize(1351, 1920))

        // 左下的一塊（不對稱：上下、左右翻了都對不上）
        val x0 = 300
        val y0 = 2000
        val x1 = 900
        val y1 = 2600
        val mapped = NightReadRenderer.resampleMask(pw, ph, nw, nh) { y, row ->
            for (x in 0 until pw) row[x] = argb(x in x0 until x1 && y in y0 until y1)
        }
        assertEquals(nw to nh, mapped.w to mapped.h)
        val line = TextLine(listOf(Pt(x0.toFloat(), y0.toFloat()), Pt(x1.toFloat(), y0.toFloat()),
            Pt(x1.toFloat(), y1.toFloat()), Pt(x0.toFloat(), y1.toFloat())), 1f)
        val q = NightReadRenderer.scaleLines(listOf(line), nw.toFloat() / pw).single().quad
        val b = bbox(mapped)
        val want = floatArrayOf(q[0].x, q[0].y, q[2].x, q[2].y)
        for (i in 0 until 4) {
            assertTrue("縮後遮罩的 bbox ${b.toList()} 與縮後字框 ${want.toList()} 差 > 1 px", abs(b[i] - want[i]) <= 1f)
        }

        // 降採樣解碼的取整（頁 JPEG 進位 1201×1601、遮罩 PNG 捨去 1200×1600）：對得上、位置差 ≤ 1 px
        assertTrue(NightReadRenderer.maskFitsPage(1200, 1600, 1201, 1601))
        val m2 = NightReadRenderer.resampleMask(1200, 1600, 1201, 1601) { y, row ->
            for (x in 0 until 1200) row[x] = argb(x in 150 until 450 && y in 1000 until 1300)
        }
        val b2 = bbox(m2)
        val want2 = intArrayOf(150, 1000, 450, 1300)
        for (i in 0 until 4) assertTrue("取整後 bbox ${b2.toList()}", abs(b2[i] - want2[i]) <= 1)
    }

    @Test
    fun maskFitsPageRejectsOtherPages() {
        assertTrue(NightReadRenderer.maskFitsPage(1000, 1500, 1000, 1500))
        // 原尺寸遮罩配 inSampleSize＝2 解出來的頁
        assertTrue(NightReadRenderer.maskFitsPage(2401, 3201, 1201, 1601))
        // 縮到工作圖尺寸（與 budgetSize 同算術）
        val (nw, nh) = NightReadRenderer.budgetSize(2400, 3200)!!
        assertTrue(NightReadRenderer.maskFitsPage(2400, 3200, nw, nh))

        assertFalse("轉了方向", NightReadRenderer.maskFitsPage(1500, 1000, 1000, 1500))
        assertFalse("雙頁跨頁", NightReadRenderer.maskFitsPage(2000, 1500, 1000, 1500))
        assertFalse("長寬比差 1.3%（別頁）", NightReadRenderer.maskFitsPage(1000, 1520, 1000, 1500))
        assertFalse("空遮罩", NightReadRenderer.maskFitsPage(0, 0, 1000, 1500))
    }

    private companion object {
        const val WHITE = -0x1 // 0xFFFFFFFF
        const val BLACK = -0x1000000 // 0xFF000000
    }
}
