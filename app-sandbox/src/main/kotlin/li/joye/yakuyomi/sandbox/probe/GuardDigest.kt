package li.joye.yakuyomi.sandbox.probe

import li.joye.yakuyomi.nightread.Gray
import li.joye.yakuyomi.nightread.Mask
import li.joye.yakuyomi.nightread.NightRead
import li.joye.yakuyomi.nightread.NightReadInput
import li.joye.yakuyomi.nightread.NightReadParams
import li.joye.yakuyomi.nightread.NightTier
import li.joye.yakuyomi.nightread.TextRegion
import java.security.MessageDigest

/**
 * 規則版本守門摘要（nightread 的 RulesVersionGuardTest.productOutputMatchesRulesVersion）在這台機器上跑一次：8 張 fixture 頁、產品兩檔
 *（標準＝L2、更多＝L3，[NightRead.renderTiers]、預設亮度）。計算方式逐行照那個測試，電腦上的答案是 [EXPECTED]。
 *
 * 純 Kotlin、不碰 android.*：[GuardDigestJvmTest] 在電腦上跑同一段，確認探針本身（含 [GrayPng]）算得出那個答案。
 */
internal object GuardDigest {
    /** 規則版本 4 的摘要（RulesVersionGuardTest.digests[4]）。 */
    const val EXPECTED = "31f2ed4d88e7d85fc8314e958ff2c6483d85d884427115f24c12efa196b32e85"
    val PAGES = listOf("ch34_011", "demo01", "demo02", "demo05", "demo06", "ch34_015", "syn_more", "syn_v4")

    /** 每頁要的 fixture（打包成 assets 的就是這些）。 */
    fun fixtureFiles(): List<String> = PAGES.flatMap { p ->
        listOf("${p}_gray.png", "${p}_seg.png", "${p}_char.png", "${p}_chroma.png", "${p}_regions.txt")
    } + "syn_v4_inpaint.png"

    private fun gray(load: (String) -> ByteArray?, name: String): Gray {
        val img = GrayPng.decode(load(name) ?: error("缺 fixture：$name"))
        return Gray(img.w, img.h, img.samples)
    }

    fun mask(load: (String) -> ByteArray?, name: String): Mask {
        val g = gray(load, name)
        return Mask(g.w, g.h, BooleanArray(g.data.size) { g.data[it] > 127 })
    }

    fun grayOf(load: (String) -> ByteArray?, name: String): Gray = gray(load, name)

    private fun regions(load: (String) -> ByteArray?, name: String): List<TextRegion> =
        String(load(name) ?: error("缺 fixture：$name"), Charsets.UTF_8)
            .lineSequence().filter { it.isNotBlank() }.map {
                val p = it.trim().split(" ").map(String::toInt)
                TextRegion(p[0], p[1], p[2], p[3])
            }.toList()

    fun input(load: (String) -> ByteArray?, page: String) = NightReadInput(
        gray = gray(load, "${page}_gray.png"),
        seg = mask(load, "${page}_seg.png"),
        regions = regions(load, "${page}_regions.txt"),
        charMask = mask(load, "${page}_char.png"),
        chroma = gray(load, "${page}_chroma.png"),
        inpaintMask = load("${page}_inpaint.png")?.let { mask(load, "${page}_inpaint.png") },
    )

    private fun hex(b: ByteArray) = b.joinToString("") { "%02x".format(it) }

    /** 回整份摘要；[onPage]：頁名、該頁摘要、毫秒。 */
    fun run(load: (String) -> ByteArray?, onPage: (String, String, Long) -> Unit): String {
        val md = MessageDigest.getInstance("SHA-256")
        val base = NightReadParams()
        val tiers = listOf(NightTier.L2, NightTier.L3)
        for (page in PAGES) {
            val t0 = System.nanoTime()
            val pageMd = MessageDigest.getInstance("SHA-256")
            NightRead.renderTiers(input(load, page), tiers.map { it.apply(base) }) { k, g ->
                val head = "$page/${tiers[k].key}:".toByteArray()
                md.update(head)
                pageMd.update(head)
                if (g == null) {
                    val eq = "=".toByteArray()
                    md.update(eq)
                    pageMd.update(eq)
                } else {
                    val bytes = ByteArray(g.data.size) { g.data[it].toByte() }
                    val size = "${g.w}x${g.h}".toByteArray()
                    md.update(size)
                    md.update(bytes)
                    pageMd.update(size)
                    pageMd.update(bytes)
                }
            }
            onPage(page, hex(pageMd.digest()), (System.nanoTime() - t0) / 1_000_000)
        }
        return hex(md.digest())
    }
}
