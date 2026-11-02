package li.joye.yakuyomi.sandbox.probe

import org.junit.Assert.assertEquals
import org.junit.Test
import java.io.File

/**
 * 探針的守門摘要在電腦上要算出規則版本的答案：手機上對不上時，才能排除是探針（PNG 解碼、餵法）寫錯。
 * 頁在 nightread submodule 的測試資源（Gradle 跑單元測試的工作目錄是模組目錄）。
 */
class GuardDigestJvmTest {
    private val dir = File("../yakuyomi-nightread/nightread/src/test/resources/page")

    private fun load(name: String): ByteArray? = File(dir, name).takeIf { it.isFile }?.readBytes()

    @Test
    fun digestMatchesRulesVersion() {
        val digest = GuardDigest.run(::load) { page, sha, ms -> println("$page $sha ${ms}ms") }
        assertEquals(GuardDigest.EXPECTED, digest)
    }

    @Test
    fun allFixtureFilesExist() {
        for (f in GuardDigest.fixtureFiles()) assertEquals("缺 $f", true, File(dir, f).isFile)
    }
}
