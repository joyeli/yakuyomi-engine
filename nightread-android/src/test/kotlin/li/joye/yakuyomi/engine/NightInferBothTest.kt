package li.joye.yakuyomi.engine

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotEquals
import org.junit.Assert.assertSame
import org.junit.Assert.assertTrue
import org.junit.Assert.fail
import org.junit.Test
import java.util.Collections
import java.util.concurrent.CountDownLatch
import java.util.concurrent.Executor
import java.util.concurrent.RejectedExecutionException
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicInteger

/**
 * [NightReadRenderer.inferBoth]（S11：夜讀的偵測與人物分割同時跑；不碰 Bitmap、不碰原生）。
 * 守：真的同時跑、分支沒開跑時呼叫執行緒自己跑且只跑一次、任一邊拋例外都先等另一邊結束並回收偵測結果、例外型別與優先、
 * 中斷旗標、依序版（null）的順序與清理。
 */
class NightInferBothTest {

    private class Det(val id: Int) {
        var discarded = 0
    }

    private val events: MutableList<String> = Collections.synchronizedList(ArrayList())

    private fun discard(d: Det) {
        d.discarded++
        events += "discard"
    }

    @Test
    fun runsBothAtTheSameTime() {
        // 偵測等分割開跑、分割等偵測開跑：依序執行會卡死（逾時＝失敗）
        val detStarted = CountDownLatch(1)
        val segStarted = CountDownLatch(1)
        val threads = Collections.synchronizedSet(HashSet<Thread>())
        val (d, s) = NightReadRenderer.inferBoth(
            detect = {
                threads += Thread.currentThread()
                detStarted.countDown()
                check(segStarted.await(5, TimeUnit.SECONDS)) { "分割沒有同時開跑" }
                Det(1)
            },
            segment = {
                threads += Thread.currentThread()
                segStarted.countDown()
                check(detStarted.await(5, TimeUnit.SECONDS)) { "偵測沒有同時開跑" }
                "chars"
            },
            branch = NightReadRenderer.NEW_THREAD_BRANCH,
            discard = ::discard,
        )
        assertEquals(1, d.id)
        assertEquals("chars", s)
        assertEquals(0, d.discarded)
        assertEquals("兩邊在不同執行緒上", 2, threads.size)
        assertTrue("偵測在呼叫執行緒上", Thread.currentThread() in threads)
    }

    @Test
    fun branchThatNeverStartsIsRunByTheCaller() {
        // 收下但永遠不跑的 Executor：偵測做完後呼叫執行緒自己跑分割
        val parked = ArrayList<Runnable>()
        val runs = AtomicInteger()
        val (_, s) = NightReadRenderer.inferBoth(
            detect = { Det(1) },
            segment = {
                runs.incrementAndGet()
                Thread.currentThread()
            },
            branch = Executor { parked += it },
            discard = ::discard,
        )
        assertSame(Thread.currentThread(), s)
        assertEquals(1, runs.get())
        // 池子之後才輪到它：已被搶走，不會再跑一次
        parked.forEach { it.run() }
        assertEquals(1, runs.get())
    }

    @Test
    fun rejectingExecutorFallsBackToTheCaller() {
        val (d, s) = NightReadRenderer.inferBoth(
            detect = { Det(2) },
            segment = { Thread.currentThread() },
            branch = Executor { throw RejectedExecutionException("滿了") },
            discard = ::discard,
        )
        assertEquals(2, d.id)
        assertSame(Thread.currentThread(), s)
    }

    @Test
    fun detectFailureWaitsForSegmentThenRethrows() {
        val segDone = CountDownLatch(1)
        val boom = IllegalStateException("偵測壞了")
        try {
            NightReadRenderer.inferBoth<Det, String>(
                detect = { throw boom },
                segment = {
                    Thread.sleep(200)
                    events += "segment done"
                    segDone.countDown()
                    "chars"
                },
                branch = NightReadRenderer.NEW_THREAD_BRANCH,
                discard = ::discard,
            )
            fail("應該拋出")
        } catch (t: IllegalStateException) {
            assertSame(boom, t)
        }
        // 拋出時分割已經結束（不留一個前向在背景跑）；沒有偵測結果可回收
        assertEquals(0L, segDone.count)
        assertEquals(listOf("segment done"), events.toList())
    }

    @Test
    fun segmentFailureDiscardsTheDetectionThenRethrowsWithItsType() {
        val det = Det(3)
        val oom = OutOfMemoryError("分割 OOM")
        try {
            NightReadRenderer.inferBoth<Det, String>(
                detect = {
                    Thread.sleep(100)
                    det
                },
                segment = { throw oom },
                branch = NightReadRenderer.NEW_THREAD_BRANCH,
                discard = ::discard,
            )
            fail("應該拋出")
        } catch (t: OutOfMemoryError) {
            assertSame("例外型別原樣保留（OOM 照樣是 OOM）", oom, t)
        }
        assertEquals("偵測結果回收一次", 1, det.discarded)
    }

    @Test
    fun bothFailDetectWinsSegmentIsSuppressed() {
        val detErr = NcnnForwardAbortedException("偵測前放棄")
        val segErr = NcnnForwardAbortedException("分割前放棄")
        try {
            NightReadRenderer.inferBoth<Det, String>(
                detect = { throw detErr },
                segment = { throw segErr },
                branch = NightReadRenderer.NEW_THREAD_BRANCH,
                discard = ::discard,
            )
            fail("應該拋出")
        } catch (t: NcnnForwardAbortedException) {
            assertSame(detErr, t)
            assertEquals(listOf<Throwable>(segErr), t.suppressed.toList())
        }
    }

    @Test
    fun interruptDoesNotCutTheWaitShortAndIsRestored() {
        val caller = Thread.currentThread()
        val segStarted = CountDownLatch(1)
        val segRelease = CountDownLatch(1)
        val segFinished = AtomicInteger()
        val result = NightReadRenderer.inferBoth(
            detect = {
                // 偵測做完、開始等分割之前把自己標成中斷
                caller.interrupt()
                Det(4)
            },
            segment = {
                segStarted.countDown()
                check(segRelease.await(5, TimeUnit.SECONDS))
                segFinished.incrementAndGet()
                "chars"
            },
            branch = Executor { r ->
                Thread(r).start()
                // 確定分割已在分支上開跑（呼叫執行緒搶不到），再讓它晚一點才放行：呼叫執行緒等它時帶著中斷旗標
                check(segStarted.await(5, TimeUnit.SECONDS))
                Thread {
                    Thread.sleep(150)
                    segRelease.countDown()
                }.start()
            },
            discard = ::discard,
        )
        assertEquals("chars", result.second)
        assertEquals(1, segFinished.get())
        assertTrue("中斷旗標補回去", Thread.interrupted()) // 同時清掉，免得影響其他測試
    }

    @Test
    fun nullBranchIsSequentialOnTheCaller() {
        val order = ArrayList<String>()
        val (_, s) = NightReadRenderer.inferBoth(
            detect = {
                order += "detect"
                Det(5)
            },
            segment = {
                order += "segment"
                Thread.currentThread()
            },
            branch = null,
            discard = ::discard,
        )
        assertEquals(listOf("detect", "segment"), order)
        assertSame(Thread.currentThread(), s)
    }

    @Test
    fun nullBranchSegmentFailureDiscardsDetection() {
        val det = Det(6)
        try {
            NightReadRenderer.inferBoth<Det, String>(
                detect = { det },
                segment = { throw IllegalArgumentException("x") },
                branch = null,
                discard = ::discard,
            )
            fail("應該拋出")
        } catch (_: IllegalArgumentException) {
        }
        assertEquals(1, det.discarded)
    }

    @Test
    fun nullBranchDetectFailureNeverRunsSegment() {
        var segRan = false
        try {
            NightReadRenderer.inferBoth<Det, String>(
                detect = { throw IllegalStateException("x") },
                segment = {
                    segRan = true
                    "chars"
                },
                branch = null,
                discard = ::discard,
            )
            fail("應該拋出")
        } catch (_: IllegalStateException) {
        }
        assertFalse(segRan)
    }

    @Test
    fun defaultBranchThreadIsNamedAndDaemon() {
        var name = ""
        var daemon = false
        NightReadRenderer.inferBoth(
            detect = {
                Thread.sleep(100)
                Det(7)
            },
            segment = {
                name = Thread.currentThread().name
                daemon = Thread.currentThread().isDaemon
                "chars"
            },
            branch = NightReadRenderer.NEW_THREAD_BRANCH,
            discard = ::discard,
        )
        // 分支可能被呼叫執行緒搶走（執行緒起得慢），所以只在真的跑在分支上時檢查
        if (name != Thread.currentThread().name) {
            assertEquals(NightReadRenderer.INFERENCE_THREAD_NAME, name)
            assertTrue(daemon)
        }
        assertNotEquals("", name)
    }
}
