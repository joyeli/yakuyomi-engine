package li.joye.yakuyomi.engine

import org.junit.Assert.assertEquals
import org.junit.Assert.assertSame
import org.junit.Assert.assertTrue
import org.junit.Assert.fail
import org.junit.Test

/**
 * [NightReadRenderer.handTier]（一檔輸出交給 sink 時誰負責 recycle；不碰 Bitmap）：
 *  - 一般版（[NightReadRenderer.renderTiers]）：sink 回傳後、sink 拋例外時都由引擎 recycle，各一次，而且在 sink 之後。
 *  - 取得所有權版（[NightReadRenderer.renderTiersOwning]，S5）：sink 被呼叫起就歸 sink，引擎一次都不 recycle（含 sink 拋例外）；
 *    sink 可以把它交給別的執行緒、之後才由那邊 recycle。
 * sink 收到的就是交出的那個物件；例外原樣往外拋。
 */
class NightTierHandoffTest {

    private class Out(val id: Int) {
        var recycled = 0
    }

    private val events = ArrayList<String>()

    private fun recycle(o: Out) {
        o.recycled++
        events += "recycle${o.id}"
    }

    @Test
    fun plainSinkRecyclesAfterReturn() {
        val out = Out(1)
        NightReadRenderer.handTier(out, sinkOwns = false, recycle = ::recycle) {
            assertSame(out, it)
            assertEquals("sink 期間還沒回收", 0, it.recycled)
            events += "sink${it.id}"
        }
        assertEquals(1, out.recycled)
        assertEquals(listOf("sink1", "recycle1"), events)
    }

    @Test
    fun plainSinkRecyclesWhenSinkThrows() {
        val out = Out(2)
        try {
            NightReadRenderer.handTier(out, sinkOwns = false, recycle = ::recycle) {
                events += "sink${it.id}"
                throw IllegalStateException("寫檔失敗")
            }
            fail("sink 的例外要往外拋")
        } catch (e: IllegalStateException) {
            assertEquals("寫檔失敗", e.message)
        }
        assertEquals(1, out.recycled)
        assertEquals(listOf("sink2", "recycle2"), events)
    }

    @Test
    fun owningSinkIsNeverRecycledByEngine() {
        val out = Out(3)
        var kept: Out? = null
        NightReadRenderer.handTier(out, sinkOwns = true, recycle = ::recycle) {
            assertSame(out, it)
            kept = it // 交給「編碼執行緒」：回傳後還要用
        }
        assertEquals("引擎不回收交出去的那張", 0, out.recycled)
        assertSame(out, kept)
        // 之後由接手的那邊回收
        recycle(kept!!)
        assertEquals(1, out.recycled)
        assertEquals(listOf("recycle3"), events)
    }

    @Test
    fun owningSinkIsNotRecycledWhenSinkThrows() {
        val out = Out(4)
        try {
            NightReadRenderer.handTier(out, sinkOwns = true, recycle = ::recycle) {
                events += "sink${it.id}"
                throw IllegalArgumentException("交不出去")
            }
            fail("sink 的例外要往外拋")
        } catch (e: IllegalArgumentException) {
            assertEquals("交不出去", e.message)
        }
        assertEquals("拋例外也歸 sink，引擎不碰", 0, out.recycled)
        assertEquals(listOf("sink4"), events)
    }

    @Test
    fun errorsPassThroughUnchanged() {
        // OOM 這類 Error 也原樣拋（一般版照樣回收）
        val out = Out(5)
        var caught: Throwable? = null
        try {
            NightReadRenderer.handTier(out, sinkOwns = false, recycle = ::recycle) { throw OutOfMemoryError("壓縮") }
        } catch (t: Throwable) {
            caught = t
        }
        assertTrue(caught is OutOfMemoryError)
        assertEquals(1, out.recycled)
    }
}
