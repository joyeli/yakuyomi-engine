package li.joye.yakuyomi.engine

import android.util.Log
import java.util.concurrent.TimeUnit
import java.util.concurrent.locks.ReentrantLock
import kotlin.concurrent.withLock

/**
 * SimpleOMP 常駐 worker 的執行緒名（＝建池那條專用執行緒的名字，worker 由它 pthread_create、繼承 comm）。
 * 診斷用：fork 的夜讀效能探針掃 `/proc/self/task/<tid>/comm` 找這個名字，彙總 worker 的 nice（見 [NcnnBackend] 的建池說明）。
 */
const val NCNN_POOL_THREAD_NAME = "yaku-ncnn-pool"

/**
 * 一組 NCNN Net 的建法與推論路徑。翻譯全用 [DEFAULT]；夜讀依當下是否要讓路，在 [NIGHT_LOCKED]／[NIGHT_FREE] 之間換組。
 *
 * **緒數只能在載模型時定**：卷積的 winograd／gemm 路徑用載入時的緒數（ncnn `convolution_arm.cpp`），載入後改
 * `opt.num_threads` 不生效 → 要換緒數就得換一組 Net。
 *
 * @property numThreads > 0＝以此緒數建（createNetEx）；0＝ncnn 預設（`get_physical_big_cpu_count()`，大核數）。
 * @property serialize true＝推論進全域鎖（見 [NcnnBackend] 的 ncnnLock）；false 只允許 [numThreads] == 1——SimpleOMP 對
 *   1 緒的 parallel region 走 inline、不碰共用 task queue（同 OCR 並發模式，見 [NcnnBackend.ocrCtc]）。
 * @property lowPriority true＝夜讀：進鎖前若有翻譯在等鎖或持鎖就先讓（翻譯優先閘，見 [NcnnBackend]）；只在 [serialize] 時有意義。
 * @property hook 低優先權推論的呼叫端掛鉤（中止、持鎖前向的優先權調整，見 [NcnnLowPriorityHook]）；只能配 [lowPriority]。
 */
data class NcnnFlavor(
    val numThreads: Int = 0,
    val serialize: Boolean = true,
    val lowPriority: Boolean = false,
    val hook: NcnnLowPriorityHook? = null,
) {
    init {
        require(serialize || numThreads == 1) { "不進鎖的 Net 只能 1 緒（numThreads=$numThreads）" }
        require(hook == null || lowPriority) { "hook 只給低優先權（夜讀）的 Net" }
    }

    companion object {
        /** 翻譯：ncnn 預設緒數、進鎖、一般優先。 */
        val DEFAULT = NcnnFlavor()

        /** 夜讀（沒有翻譯在跑）：ncnn 預設緒數、進鎖，但翻譯一來就讓它先。 */
        val NIGHT_LOCKED = NcnnFlavor(lowPriority = true)

        /** 夜讀（翻譯在跑、讓路）：1 緒、不進鎖——從不擋翻譯的偵測／去字。 */
        val NIGHT_FREE = NcnnFlavor(numThreads = 1, serialize = false, lowPriority = true)
    }
}

/**
 * 低優先權（夜讀）推論的呼叫端掛鉤，一律在**發起推論的那條執行緒**上被呼叫（同一組 Net 可能被多頁共用，
 * 實作要靠執行緒區域狀態分辨是哪一頁）。翻譯的 Net 不帶掛鉤，行為不變。
 */
interface NcnnLowPriorityHook {
    /**
     * 要不要放棄這次推論：每次前向開始前問一次；上鎖路徑在等翻譯讓出鎖期間約每 [NcnnBackend.LOW_PRIORITY_POLL_MS]
     * 再問、拿到鎖後正式開跑前再問一次（這一次在 [aroundLockedForward] 裡、持鎖時問）。true → 不進原生、拋
     * [NcnnForwardAbortedException]（呼叫端丟回待做）。
     */
    fun shouldAbort(): Boolean

    /**
     * 包住**持有全域鎖的整段**：從取全域鎖之前到放開之後（等監視器、拿到鎖後的翻譯複查與 [shouldAbort]、原生前向本身），
     * 不含翻譯優先閘的等待（等翻譯讓出鎖）與 Kotlin 前後處理。用途例如低優先權執行緒暫時拉回一般優先權，免得它拿著鎖
     * 慢慢算、翻譯在鎖外乾等——Java 監視器沒辦法在拿到鎖的那一刻才調優先權，所以要在取鎖前就調好、放鎖後才還原，
     * 持鎖期間才不會有任何一段是低優先權；等監視器是阻塞、不吃 CPU，蓋進來沒有代價。
     *
     * 必須恰好呼叫 [block] 一次，並原樣回傳它的結果或讓它的例外往外拋。拿到鎖時翻譯剛好又來了，[block] 會放掉鎖、
     * 回一個引擎內部的標記（引擎出來後回到翻譯優先閘重等），所以一次推論可能進出本掛鉤不只一次。
     */
    fun <T> aroundLockedForward(block: () -> T): T
}

/** [NcnnLowPriorityHook.shouldAbort] 要求放棄：這次推論沒有進原生、沒有任何輸出。 */
class NcnnForwardAbortedException(message: String = "低優先權推論被呼叫端放棄") : RuntimeException(message)

/**
 * NCNN 推論後端（去字 + 偵測 + 人物分割 + OCR）。
 *
 * handle-based：`createNet` 載入 pnnx 轉出的 .param/.bin 回一個 native handle（ncnn::Net*），pipeline 每頁復用、
 * 收工 `releaseNet`。模型換到手機 CPU 的 NEON/Winograd → 偵測實測比 ORT-XNNPACK 快 ~3.7×（見 memory litert-gpu-blocked）。
 * OCR 原本卡在「transformer 位置編碼被 trace 烤死在轉換寬度」的寬度牆，2026-09-21 改把 PE 當第二個輸入餵進去
 * （parity/export_ocr_ncnn.py）後也搬過來；[ocrCtc] 的並發規則見其註解。
 *
 * **SimpleOMP worker 以 nice 0 出生**：ncnn 的多緒 runtime 是 SimpleOMP（整個行程一個常駐池、`cpu_count−1` 條 worker），
 * 由**第一個**進 `__kmpc_fork_call` 的執行緒建立（`pthread_once`；載模型的 create_pipeline 也會進），worker 的 nice、cgroup、
 * 執行緒名都從建立者繼承、之後永不調整。若第一個呼叫者是低優先權的執行緒，整個行程之後的推論（含翻譯）都跑在低優先權的
 * worker 上。所以 [createNet]／[createNetEx] 第一次被呼叫前，先在一條明確設成 nice 0 的專用執行緒（[NCNN_POOL_THREAD_NAME]）
 * 上跑一次 2 緒 ReLU 把池建起來、join 等它做完（[ensureThreadPool]）。
 *
 * 不屬公開 API：在 :inference-core，給 :engine（偵測以外的 OCR／去字）與 :nightread-android（人物分割）用，所以標
 * [InternalEngineApi]（原本是 internal，拆模組後跨不了模組）。
 */
@InternalEngineApi
object NcnnBackend {
    private const val TAG = "NcnnBackend"

    /** 原生庫是否載得起來（缺 .so / 非 arm64 → false；三顆模型全 NCNN、沒有備援，呼叫端只能報錯）。 */
    val available: Boolean = try {
        System.loadLibrary("yakuyomi_ncnn")
        true
    } catch (t: Throwable) {
        false
    }

    // ★ external 名＝JNI 符號（Java_li_joye_yakuyomi_engine_NcnnBackend_<名>）；改名要連 ncnn_jni.cpp 一起改，否則 UnsatisfiedLinkError。
    private external fun createNetNative(paramPath: String, binPath: String): Long

    private external fun createNetExNative(
        paramPath: String,
        binPath: String,
        numThreads: Int,
        fp16Storage: Boolean,
        fp16Arith: Boolean,
    ): Long

    private external fun initThreadPoolNative(): Int

    /** SimpleOMP 常駐池已由 nice 0 的專用執行緒建好（見 [ensureThreadPool]）；只在成功時設 true。 */
    @Volatile
    private var threadPoolReady = false
    private val threadPoolLock = Any()

    /**
     * 確保 SimpleOMP 常駐池由 nice 0 的專用執行緒建好（見類別說明）：開 [NCNN_POOL_THREAD_NAME]、先
     * `setThreadPriority(DEFAULT)`（Java 執行緒會繼承父執行緒的優先權）、JNI 跑一次 2 緒 ReLU（一定進 fork_call）、join。
     * 同一把鎖：第二個呼叫者會等池建好才返回。
     *
     * - **join 不可中斷**：呼叫端帶著中斷旗標（例如 `runInterruptible`、executor 被 cancel(true)）也照樣等建池做完、
     *   事後補回中斷旗標，不讓載模型直接拋 InterruptedException。
     * - **只記住成功**：建池失敗（rc≠0）記 WARN、下次 [createNet] 再試一次——否則池會改由第一個真正推論／載模型的執行緒
     *   建立，那若是 nice 9 的夜讀 worker，整個行程之後的推論（含翻譯）都跑在低優先權的 worker 上。
     *   池已建好時重試無害（`pthread_once`，只多跑一次 ReLU）。
     */
    private fun ensureThreadPool() {
        if (threadPoolReady || !available) return
        synchronized(threadPoolLock) {
            if (threadPoolReady) return
            var rc = Int.MIN_VALUE
            val t = Thread({
                runCatching { android.os.Process.setThreadPriority(android.os.Process.THREAD_PRIORITY_DEFAULT) }
                rc = runCatching { initThreadPoolNative() }.getOrDefault(Int.MIN_VALUE)
            }, NCNN_POOL_THREAD_NAME)
            t.priority = Thread.NORM_PRIORITY
            t.start()
            var interrupted = false
            while (t.isAlive) {
                try {
                    t.join()
                } catch (e: InterruptedException) {
                    interrupted = true
                }
            }
            if (interrupted) Thread.currentThread().interrupt()
            EngineTrace.log("ncnn.pool init rc=$rc")
            if (rc == 0) {
                threadPoolReady = true
            } else {
                Log.w(TAG, "SimpleOMP 常駐池建立失敗 rc=$rc（下次載模型再試）")
            }
        }
    }

    /** 載入 .param/.bin，回 native handle（純 CPU；NEON/Winograd、ncnn 預設緒數）；0=失敗。 */
    fun createNet(paramPath: String, binPath: String): Long {
        ensureThreadPool()
        return createNetNative(paramPath, binPath)
    }

    /**
     * [createNet] 加選項：numThreads（>0 才設；OCR 並發模式設 1）、fp16Storage／fp16Arith 分開控制半精度
     * （storage=權重與中間值存 fp16、arith=用 fp16 指令算；transformer 若半精度算術讀錯字，可只留 storage）。
     */
    fun createNetEx(paramPath: String, binPath: String, numThreads: Int, fp16Storage: Boolean, fp16Arith: Boolean): Long {
        ensureThreadPool()
        return createNetExNative(paramPath, binPath, numThreads, fp16Storage, fp16Arith)
    }

    /** 依 [flavor] 建 Net：指定緒數走 [createNetEx]（fp16 全開＝與 [createNet] 只差緒數），否則 [createNet]。 */
    fun createNet(paramPath: String, binPath: String, flavor: NcnnFlavor): Long =
        if (flavor.numThreads > 0) {
            createNetEx(paramPath, binPath, flavor.numThreads, fp16Storage = true, fp16Arith = true)
        } else {
            createNet(paramPath, binPath)
        }

    external fun releaseNet(handle: Long)

    private external fun cpuSupportsFp16Native(): Boolean

    /** 這顆 CPU 有 fp16 storage/arithmetic（arm82 asimdhp）；OCR 混合精度 param 只在此為 true 時才能用（見 :engine 的 `Ocr`）。 */
    val cpuSupportsFp16: Boolean by lazy { available && runCatching { cpuSupportsFp16Native() }.getOrDefault(false) }

    /**
     * ★ 全域鎖：序列化多緒 Net 的原生推論（偵測 + 去字 + 人物分割）。
     *
     * 起因是 libomp 年代：多個 app 執行緒同時進 ncnn forward、各開 OpenMP parallel region → `__kmp_abort_process` 直接
     * abort 行程（真機 tombstone，2026-07-14）。現在的 runtime 是 SimpleOMP（見 memory ncnn-simpleomp-crash-fix），
     * 多緒 Net 仍保守持鎖：共用一個 task queue 的多個並發 master 沒驗過，且 CPU 本來也無法真的同時跑兩份多緒前向。
     * **不進鎖的前提**＝Net 以 1 緒建（[NcnnFlavor.serialize]＝false）：SimpleOMP 對 num_threads==1 的 parallel region 走
     * inline（`__kmpc_fork_call`：不碰共用 task queue、全域初始化用 pthread_once），可與持鎖中的多緒前向同時跑——OCR 並發模式
     * （[ocrCtc]）與夜讀讓路時的模型組（[NcnnFlavor.NIGHT_FREE]）走這條。
     *
     * **翻譯優先閘**：夜讀的上鎖呼叫（lowPriority）不能擋翻譯。翻譯端從開始等鎖到放鎖期間 [hiWaiters] > 0；夜讀進鎖前先等它
     * 歸零、拿到鎖後再看一次（翻譯剛好又來就放掉重等）→ 翻譯最多只等夜讀當下正在跑的那一個前向。Java 監視器沒有優先權繼承，
     * 單靠降夜讀執行緒的 nice 會把翻譯卡在低優先權持鎖者後面（優先權倒置）——所以持鎖的整段（取鎖前到放鎖後）另外交給
     * [NcnnLowPriorityHook.aroundLockedForward]（夜讀在那裡暫時拉回一般優先權）。夜讀帶掛鉤時，等翻譯期間每
     * [LOW_PRIORITY_POLL_MS] 問一次 [NcnnLowPriorityHook.shouldAbort]，暫停／讓路時不必等到翻譯空檔。
     */
    private val ncnnLock = Any()

    /** 目前執行緒是否持有 [ncnnLock]（JVM 測試用：驗證掛鉤包住整段持鎖）。 */
    internal fun holdsNcnnLock(): Boolean = Thread.holdsLock(ncnnLock)

    /** 帶掛鉤的上鎖路徑：拿到鎖時翻譯又來了 → 放掉鎖、出掛鉤後回到翻譯優先閘重等（見 [forward]）。 */
    private object RetryGate

    /** 翻譯端（非 lowPriority）正在等 [ncnnLock] 或持有它的呼叫數。在 [gateLock] 下改、在 [ncnnLock] 內無鎖讀。 */
    @Volatile
    private var hiWaiters = 0
    private val gateLock = ReentrantLock()
    private val noHiWaiters = gateLock.newCondition()

    /** 夜讀帶掛鉤時，等翻譯讓出鎖期間多久問一次 [NcnnLowPriorityHook.shouldAbort]。 */
    const val LOW_PRIORITY_POLL_MS = 50L

    private fun <T> forward(flavor: NcnnFlavor, block: () -> T): T =
        forward(flavor.serialize, flavor.lowPriority, flavor.hook, block)

    /**
     * 依 [serialize]／[lowPriority] 跑一次原生推論（見 [ncnnLock] 的翻譯優先閘）；[hook] 見 [NcnnLowPriorityHook]。
     * internal 供 JVM 測試（[block] 換成假的前向）；正式呼叫一律經帶 [NcnnFlavor] 的多載。
     */
    internal fun <T> forward(serialize: Boolean, lowPriority: Boolean, hook: NcnnLowPriorityHook?, block: () -> T): T {
        if (hook != null && hook.shouldAbort()) throw NcnnForwardAbortedException()
        if (!serialize) return block()
        if (!lowPriority) {
            gateLock.withLock { hiWaiters++ }
            try {
                return synchronized(ncnnLock) { block() }
            } finally {
                gateLock.withLock {
                    hiWaiters--
                    if (hiWaiters == 0) noHiWaiters.signalAll()
                }
            }
        }
        while (true) {
            // 翻譯在等鎖或持鎖 → 先讓。帶掛鉤就限時等、回來問要不要放棄（掛鉤是呼叫端的程式，不在 gateLock 內呼叫）
            val clear = gateLock.withLock {
                if (hiWaiters > 0) {
                    if (hook == null) {
                        noHiWaiters.awaitUninterruptibly()
                    } else {
                        try {
                            noHiWaiters.await(LOW_PRIORITY_POLL_MS, TimeUnit.MILLISECONDS)
                        } catch (e: InterruptedException) {
                            Thread.currentThread().interrupt()
                            throw NcnnForwardAbortedException("等翻譯讓出鎖時被中斷")
                        }
                    }
                }
                hiWaiters == 0
            }
            if (!clear) {
                if (hook != null && hook.shouldAbort()) throw NcnnForwardAbortedException()
                continue
            }
            if (hook == null) {
                synchronized(ncnnLock) {
                    // 拿到鎖時翻譯又來了 → 放掉、讓它先
                    if (hiWaiters == 0) return block()
                }
                continue
            }
            // ★ 掛鉤包住整段持鎖（取鎖前進、放鎖後出，見 [NcnnLowPriorityHook.aroundLockedForward]）：拿到鎖才進掛鉤的話，
            // 取鎖到拉回優先權之間、還原優先權到放鎖之間都是低優先權持鎖，被翻譯那邊的 CPU 工作搶走時翻譯反而在鎖外等它。
            val r = hook.aroundLockedForward<Any?> {
                synchronized(ncnnLock) {
                    if (hiWaiters != 0) {
                        RetryGate // 拿到鎖時翻譯又來了 → 放掉、讓它先
                    } else {
                        // 等 ncnnLock 期間（別頁的夜讀前向）可能已經要放棄了；鎖序＝ncnnLock → 呼叫端的鎖（掛鉤不會回頭拿 ncnnLock）
                        if (hook.shouldAbort()) throw NcnnForwardAbortedException()
                        block()
                    }
                }
            }
            if (r !== RetryGate) {
                @Suppress("UNCHECKED_CAST")
                return r as T
            }
        }
    }

    private external fun detectDbnetNative(handle: Long, chw: FloatArray, inW: Int, inH: Int, db: FloatArray, mask: FloatArray): Int

    private external fun inpaintAotNative(handle: Long, img: FloatArray, mask: FloatArray, s: Int, out: FloatArray): Int

    private external fun extractNative(handle: Long, chw: FloatArray, inW: Int, inH: Int, inC: Int, outNames: Array<String>, outs: Array<FloatArray>): Int

    private external fun ocrCtcNative(handle: Long, chw: FloatArray, w: Int, h: Int, pe: FloatArray, t: Int, idx: IntArray, logp: FloatArray): Int

    /**
     * DBNet 偵測（矩形 resize_aspect 輸入，繞開正方形 832-992 crash 帶）：chw=[3,inH,inW] → db 填 [2*inW*inH]（raw logits 2ch 全解析）、
     * mask 填 [(inW/2)*(inH/2)]（已 sigmoid 半解析）。回 mask.h（>0=OK）。[flavor]＝進不進鎖、優先權與掛鉤（見 [NcnnFlavor]）。
     */
    fun detectDbnet(
        handle: Long,
        chw: FloatArray,
        inW: Int,
        inH: Int,
        db: FloatArray,
        mask: FloatArray,
        flavor: NcnnFlavor = NcnnFlavor.DEFAULT,
    ): Int {
        EngineTrace.log("ncnn.detectDbnet.enter ${inW}x$inH")
        return forward(flavor) {
            EngineTrace.log("ncnn.detectDbnet.call ${inW}x$inH")
            val rc = detectDbnetNative(handle, chw, inW, inH, db, mask)
            EngineTrace.log("ncnn.detectDbnet.exit rc=$rc")
            rc
        }
    }

    /** 去字 AOT：img=NCHW[3,s,s]（[-1,1] holes-zeroed）+ mask=[s*s] → out 填 [3*s*s]（[-1,1]）。回 0=OK。序列化（見 [ncnnLock]）。 */
    fun inpaintAot(handle: Long, img: FloatArray, mask: FloatArray, s: Int, out: FloatArray): Int {
        EngineTrace.log("ncnn.inpaint.enter s=$s")
        return forward(NcnnFlavor.DEFAULT) {
            EngineTrace.log("ncnn.inpaint.call s=$s")
            val rc = inpaintAotNative(handle, img, mask, s, out)
            EngineTrace.log("ncnn.inpaint.exit rc=$rc")
            rc
        }
    }

    /**
     * 通用抽取（後處理在 Kotlin 的模型，如人物分割）：chw=[inC,inH,inW] 進 in0，依 [outNames] 抽出各 blob、
     * 逐 channel 複製進 [outs]（每個陣列大小要等於該 blob 的 w×h×c）。回 0=OK、-2 大小不合、-3 抽取失敗。
     * [flavor]＝進不進鎖、優先權與掛鉤（見 [NcnnFlavor]）。
     */
    fun extract(
        handle: Long,
        chw: FloatArray,
        inW: Int,
        inH: Int,
        inC: Int,
        outNames: Array<String>,
        outs: Array<FloatArray>,
        flavor: NcnnFlavor = NcnnFlavor.DEFAULT,
    ): Int {
        EngineTrace.log("ncnn.extract.enter ${inW}x$inH x$inC → ${outNames.size} blobs")
        return forward(flavor) {
            val rc = extractNative(handle, chw, inW, inH, inC, outNames, outs)
            EngineTrace.log("ncnn.extract.exit rc=$rc")
            rc
        }
    }

    /**
     * 48px CTC OCR 單條：chw=[3,h,w]（h=48）+ pe=正弦位置表（至少 t×320，只讀前 t 列）→ JNI 內算完 argmax 與
     * top-1 log_softmax，填 idx[t]、logp[t]。回 t（>0=OK）；負值見 ncnn_jni.cpp。
     *
     * [serialize]=false 時**不進 [ncnnLock]**：OCR 的 Net 以 num_threads=1 建（:engine 的 `Ocr` 並發模式），SimpleOMP 對
     * num_threads==1 的 parallel region 走 inline（simpleomp.cpp `__kmpc_fork_call`：不碰共用 task queue、全域初始化
     * 用 pthread_once）→ 多條 strip 可同時 forward，也不會與持鎖中的偵測/去字（走 task queue）互撞。這是 OCR 保住
     * 「8 行並發快 46%」的前提。非並發模式（num_threads>1）照舊序列化。
     */
    fun ocrCtc(handle: Long, chw: FloatArray, w: Int, h: Int, pe: FloatArray, t: Int, idx: IntArray, logp: FloatArray, serialize: Boolean): Int =
        forward(serialize, lowPriority = false, hook = null) { ocrCtcNative(handle, chw, w, h, pe, t, idx, logp) }
}
