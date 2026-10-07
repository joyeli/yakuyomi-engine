package li.joye.yakuyomi.engine

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Assert.fail
import org.junit.Test
import java.util.Collections
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicInteger
import kotlin.concurrent.thread

/**
 * 低優先權（夜讀）上鎖路徑的掛鉤順序（[NcnnBackend.forward]，前向換成假的 block，不碰原生）。
 *
 * 守的是優先權倒置的窗口：[NcnnLowPriorityHook.aroundLockedForward] 要在取全域鎖**之前**進、放鎖**之後**出——
 * 夜讀在那裡把 nice 拉回 0，拿到鎖才進掛鉤的話，取鎖到拉回之間、還原到放鎖之間都是 nice 9 持鎖。
 */
class NcnnForwardHookTest {

    /**
     * 記下每個回呼當下有沒有持鎖；[abortOnCall] 為第幾次 shouldAbort 回 true（1 起算，0＝永不）。
     * [events] 可跨執行緒寫（翻譯執行緒也記進同一份，看先後）；[abortAsked] 每問一次 shouldAbort 倒數一次。
     */
    private open class RecordingHook(
        private val abortOnCall: Int = 0,
        val abortAsked: CountDownLatch = CountDownLatch(0),
    ) : NcnnLowPriorityHook {
        val events: MutableList<String> = Collections.synchronizedList(ArrayList())
        private val abortCalls = AtomicInteger()

        override fun shouldAbort(): Boolean {
            val n = abortCalls.incrementAndGet()
            events += "abort? locked=${NcnnBackend.holdsNcnnLock()}"
            abortAsked.countDown()
            return n == abortOnCall
        }

        override fun <T> aroundLockedForward(block: () -> T): T {
            events += "enter locked=${NcnnBackend.holdsNcnnLock()}"
            try {
                return block()
            } finally {
                events += "exit locked=${NcnnBackend.holdsNcnnLock()}"
            }
        }
    }

    @Test
    fun hookBracketsTheWholeLockHold() {
        val hook = RecordingHook()
        val r = NcnnBackend.forward(serialize = true, lowPriority = true, hook = hook) {
            hook.events += "forward locked=${NcnnBackend.holdsNcnnLock()}"
            42
        }
        assertEquals(42, r)
        assertEquals(
            listOf(
                "abort? locked=false", // 前向開始前
                "enter locked=false", // 取鎖之前就進掛鉤
                "abort? locked=true", // 拿到鎖後正式開跑前再問一次
                "forward locked=true",
                "exit locked=false", // 放鎖之後才出掛鉤
            ),
            hook.events,
        )
        assertFalse(NcnnBackend.holdsNcnnLock())
    }

    @Test
    fun abortAfterTakingTheLockLeavesTheHookAfterReleasingIt() {
        val hook = RecordingHook(abortOnCall = 2)
        try {
            NcnnBackend.forward(serialize = true, lowPriority = true, hook = hook) {
                fail("放棄了就不該進前向")
            }
            fail("應拋 NcnnForwardAbortedException")
        } catch (e: NcnnForwardAbortedException) {
            // 預期
        }
        assertEquals(
            listOf("abort? locked=false", "enter locked=false", "abort? locked=true", "exit locked=false"),
            hook.events,
        )
    }

    /**
     * 翻譯執行緒持鎖期間（hiWaiters＝1）在另一條執行緒跑 [night]；先等 [hook] 被問過 [minAborts] 次放棄（開始前一次＋閘外輪詢），
     * 確認這段沒進掛鉤，才讓翻譯放鎖。回傳 [night] 拋出的例外（沒有＝null）。
     */
    private fun whileTranslationHoldsTheLock(
        hook: RecordingHook,
        minAborts: Int,
        night: () -> Unit,
    ): Throwable? {
        val inside = CountDownLatch(1)
        val release = CountDownLatch(1)
        val translation = thread(name = "translation") {
            NcnnBackend.forward(serialize = true, lowPriority = false, hook = null) {
                inside.countDown()
                release.await(10, TimeUnit.SECONDS)
            }
        }
        assertTrue("翻譯沒拿到鎖", inside.await(5, TimeUnit.SECONDS))
        var failure: Throwable? = null
        val nightThread = thread(name = "night") {
            try {
                night()
            } catch (t: Throwable) {
                failure = t
            }
        }
        // 開始前問一次＋等翻譯期間至少再問 minAborts−1 次（每 LOW_PRIORITY_POLL_MS 一次）；這段夜讀都不能進掛鉤
        val asked = hook.abortAsked.await(5, TimeUnit.SECONDS)
        val enteredWhileTranslating = hook.events.any { it.startsWith("enter") }
        release.countDown()
        translation.join(5_000)
        nightThread.join(5_000)
        assertFalse("夜讀執行緒沒結束", nightThread.isAlive)
        assertTrue("等翻譯期間沒問放棄（至少 $minAborts 次）", asked)
        assertFalse("翻譯持鎖時夜讀進了掛鉤", enteredWhileTranslating)
        return failure
    }

    @Test
    fun waitingForTranslationStaysOutsideTheHook() {
        val hook = RecordingHook(abortAsked = CountDownLatch(3))
        var forwards = 0
        val failure = whileTranslationHoldsTheLock(hook, minAborts = 3) {
            NcnnBackend.forward(serialize = true, lowPriority = true, hook = hook) {
                forwards++
                hook.events += "forward locked=${NcnnBackend.holdsNcnnLock()}"
            }
        }
        assertEquals(null, failure)
        assertEquals(1, forwards)
        // 進掛鉤之前全是閘外、沒持鎖時問的放棄；之後才是正常的一趟
        val firstEnter = hook.events.indexOf("enter locked=false")
        assertTrue("至少問了 3 次放棄才進掛鉤：${hook.events}", firstEnter >= 3)
        assertTrue(hook.events.subList(0, firstEnter).all { it == "abort? locked=false" })
        assertEquals(
            listOf("enter locked=false", "abort? locked=true", "forward locked=true", "exit locked=false"),
            hook.events.subList(firstEnter, hook.events.size),
        )
    }

    @Test
    fun abortWhileWaitingForTranslationNeverEntersTheHook() {
        // 第 2 次問（等翻譯的第一次輪詢）就放棄：不進掛鉤、不跑前向
        val hook = RecordingHook(abortOnCall = 2, abortAsked = CountDownLatch(2))
        val failure = whileTranslationHoldsTheLock(hook, minAborts = 2) {
            NcnnBackend.forward(serialize = true, lowPriority = true, hook = hook) { fail("放棄了就不該進前向") }
        }
        assertTrue("應拋 NcnnForwardAbortedException，拿到 $failure", failure is NcnnForwardAbortedException)
        assertEquals(listOf("abort? locked=false", "abort? locked=false"), hook.events)
    }

    /**
     * 拿到鎖時翻譯又來了（RetryGate）：在掛鉤裡放鎖、出掛鉤，回翻譯優先閘等翻譯跑完，再進一次掛鉤才前向。
     *
     * 要讓「夜讀拿到鎖、檢查 hiWaiters 之前」翻譯剛好登記，第一次進掛鉤時先（重入地）拿住全域鎖——借一次不帶掛鉤的夜讀前向——
     * 再起翻譯執行緒：它登記等鎖（hiWaiters＝1）後卡在鎖外（BLOCKED），這時才讓夜讀的持鎖段跑，必定看到翻譯在等。
     */
    @Test
    fun translationArrivingAtTheLockSendsNightBackToTheGate() {
        var translation: Thread? = null
        val hook = object : RecordingHook() {
            private var entries = 0

            override fun <T> aroundLockedForward(block: () -> T): T {
                entries++
                if (entries > 1) return super.aroundLockedForward(block)
                return super.aroundLockedForward {
                    NcnnBackend.forward(serialize = true, lowPriority = true, hook = null) {
                        val t = thread(name = "translation") {
                            NcnnBackend.forward(serialize = true, lowPriority = false, hook = null) {
                                events += "translation locked=${NcnnBackend.holdsNcnnLock()}"
                            }
                        }
                        translation = t
                        val deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(5)
                        while (t.state != Thread.State.BLOCKED) {
                            if (System.nanoTime() > deadline) fail("翻譯沒卡在鎖外：${t.state}")
                            Thread.sleep(1)
                        }
                        block() // 重入取鎖 → 看到翻譯在等 → 放掉、讓它先
                    }
                }
            }
        }
        var forwards = 0
        NcnnBackend.forward(serialize = true, lowPriority = true, hook = hook) {
            forwards++
            hook.events += "forward locked=${NcnnBackend.holdsNcnnLock()}"
        }
        translation!!.join(5_000)
        assertEquals(1, forwards)
        val ev = hook.events.toList()
        // 第一趟：進掛鉤、拿到鎖就讓（沒問放棄、沒前向）、出掛鉤時已放鎖
        assertEquals(listOf("abort? locked=false", "enter locked=false"), ev.subList(0, 2))
        val firstExit = ev.indexOf("exit locked=false")
        assertTrue(ev.subList(2, firstExit).all { it.startsWith("translation") })
        // 翻譯恰好跑一次，在夜讀第二次進掛鉤之前
        val secondEnter = ev.lastIndexOf("enter locked=false")
        assertTrue("夜讀沒有第二次進掛鉤：$ev", secondEnter > firstExit)
        assertEquals(listOf("translation locked=true"), ev.filter { it.startsWith("translation") })
        assertTrue(ev.indexOf("translation locked=true") < secondEnter)
        // 兩趟之間只可能是閘外、沒持鎖時問的放棄；第二趟正常前向
        assertTrue(
            ev.subList(firstExit + 1, secondEnter).all { it == "abort? locked=false" || it.startsWith("translation") },
        )
        assertEquals(
            listOf("enter locked=false", "abort? locked=true", "forward locked=true", "exit locked=false"),
            ev.subList(secondEnter, ev.size),
        )
        assertFalse(NcnnBackend.holdsNcnnLock())
    }

    @Test
    fun unserializedAndUnhookedPathsDoNotEnterTheHook() {
        // 不進鎖（FREE 組）：只在開始前問一次放棄，不進掛鉤
        val hook = RecordingHook()
        assertEquals(7, NcnnBackend.forward(serialize = false, lowPriority = true, hook = hook) { 7 })
        assertEquals(listOf("abort? locked=false"), hook.events)
        // 翻譯（沒有掛鉤）：照樣持鎖跑
        assertTrue(NcnnBackend.forward(serialize = true, lowPriority = false, hook = null) { NcnnBackend.holdsNcnnLock() })
        // 夜讀但沒帶掛鉤：同樣持鎖跑
        assertTrue(NcnnBackend.forward(serialize = true, lowPriority = true, hook = null) { NcnnBackend.holdsNcnnLock() })
    }
}
