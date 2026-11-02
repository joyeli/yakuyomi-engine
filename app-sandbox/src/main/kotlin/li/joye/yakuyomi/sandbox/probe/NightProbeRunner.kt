package li.joye.yakuyomi.sandbox.probe

import android.content.Context
import android.os.Process
import androidx.documentfile.provider.DocumentFile
import li.joye.yakuyomi.engine.NcnnBackend
import li.joye.yakuyomi.nightread.NightBinWorkload
import li.joye.yakuyomi.nightread.NightGrayWorkload
import li.joye.yakuyomi.nightread.NightNative
import li.joye.yakuyomi.nightread.NightNativePrims
import li.joye.yakuyomi.nightread.NightNumericProbe
import li.joye.yakuyomi.nightread.NightPrimWorkload
import li.joye.yakuyomi.nightread.NightRead
import li.joye.yakuyomi.nightread.NightReadInput
import java.io.File

/**
 * 夜讀 C++ 探針本體，跑在一條普通的工作執行緒上（預設堆疊、預設優先權，跟之後產品呼叫 C++ 的條件一樣）：
 *  (a) 8 頁守門摘要（Kotlin）；
 *  (b) 數值核心：C++ 對 ART 的 StrictMath、捨入、BigDecimal、常數表；
 *  (c) 影像原語：同行程差分＋計時（先暖機）。第 2 階的 5 族（步驟 c）、第 3a 階的二值原語（步驟 c3a，[NightBinWorkload]）、
 *      第 3b 階的灰階／濾波／繪圖／numpy 原語（步驟 c3b，[NightGrayWorkload]），以及步驟表沒涵蓋的第 3a、3b 階案例只比位元
 *      （步驟 c3a.x、c3b.x）；
 *  (d) 模型：緒數矩陣、兩兩同時跑、吞吐。
 *
 * 每一步先記「正在跑」再動手：閃退時下次開 app 看得出是哪一步（[ProbeRecovery]）。一步丟例外（含 OOM）只記 FAIL，繼續下一步。
 * 只回報，不下結論：關卡的判斷在紀錄最後的 SUMMARY 列出來給人看。
 */
class NightProbeRunner(
    private val ctx: Context,
    private val tree: DocumentFile,
    private val report: ProbeReport,
    private val state: ProbeState,
    private val buildTag: String,
    private val status: (String) -> Unit,
) {
    private val env = ProbeEnv(ctx)
    private val t0 = System.nanoTime()

    /** 這一輪以前就有的紀錄（接著跑時，矩陣的預期 sha 從這裡讀回來）。 */
    private val expected = HashMap<String, String>()

    private var nativeLoaded = false

    private fun secs() = "%.1f".format((System.nanoTime() - t0) / 1e9)

    private fun envLine(tag: String, step: String) {
        report.line("ENV", "at" to tag, "step" to step, "t" to secs(), *env.snapshot())
        if (env.importance() != 100) {
            report.line("WARN", "what" to "不在最上層，這段的時間不能算（這台手機只給 5 顆核心）", "step" to step, "importance" to env.importance(), "cpus" to env.cpusAllowed())
        }
    }

    private fun fail(id: String, e: Throwable) {
        report.line("FAIL", "step" to id, "error" to "${e.javaClass.name}: ${e.message}", "stack" to e.stackTrace.take(6).joinToString(" < "))
    }

    private fun begin(id: String, title: String, limitMin: Int) {
        status(title)
        state.inProgress = id
        report.line("STEP", "id" to id, "title" to title)
        envLine("begin", id)
        current = Current(id, System.nanoTime() + limitMin * 60_000_000_000L)
    }

    /** 跑一步：做過或閃退過的跳過；先記「正在跑」，例外不外傳。 */
    private fun step(id: String, title: String, limitMin: Int = 20, body: () -> Unit) {
        if (id in state.crashed) {
            report.line("SKIP", "step" to id, "why" to "上次在這一步閃退")
            return
        }
        if (id in state.done) return
        begin(id, title, limitMin)
        try {
            body()
            report.line("STEP_OK", "id" to id)
        } catch (e: Throwable) {
            fail(id, e)
        }
        current = null
        envLine("end", id)
        state.addDone(id)
        state.inProgress = null
        report.publish(force = !id.startsWith("d.pair."))   // 組合很多、資料夾可能是雲端；閃退時下次開 app 會補一份
    }

    /**
     * 準備步驟（載 .so、複製模型、頁面輸入、建 Net 與暖機）：一樣先記「正在跑」、一樣有看門狗，所以閃退或卡死都查得到是這裡。
     * 結果留在記憶體，接著跑也要重做，所以不看 done；閃退過就不再試（不然每次接著跑都死在同一處）。
     * 失敗或閃退過回 null，靠它的步驟由呼叫端記 SKIP。
     */
    private fun <T : Any> prep(id: String, title: String, limitMin: Int = 10, body: () -> T): T? {
        if (id in state.crashed) {
            report.line("SKIP", "step" to id, "why" to "上次在這一步閃退")
            return null
        }
        begin(id, title, limitMin)
        var out: T? = null
        try {
            out = body()
            report.line("STEP_OK", "id" to id)
        } catch (e: Throwable) {
            fail(id, e)
        }
        current = null
        envLine("end", id)
        state.inProgress = null
        report.publish()
        return out
    }

    private class Current(val id: String, val deadlineNs: Long)

    @Volatile
    private var current: Current? = null

    /**
     * 卡死也不能無聲無息（兩個多緒前向同時跑是第一次在手機上試）：一步超過時限就把所有 Java 執行緒的堆疊寫進紀錄、那一步記成閃退、結束行程。
     * 使用者重開按「接著跑」會跳過它。
     */
    private fun startWatchdog() {
        Thread({
            while (ProbeSession.running) {
                Thread.sleep(5000)
                val c = current ?: continue
                if (System.nanoTime() < c.deadlineNs) continue
                report.line("HANG", "step" to c.id, *env.snapshot())
                for ((t, st) in Thread.getAllStackTraces()) {
                    if (t.name.startsWith("probe-") || t.name == "night-probe") {
                        report.line("HANG_THREAD", "name" to t.name, "state" to t.state, "stack" to st.take(8).joinToString(" < "))
                    }
                }
                state.addCrashed(c.id)
                state.selfKill = c.id
                state.inProgress = null
                report.publish()
                status("「${c.id}」卡住了（已寫進紀錄），app 會自己關閉。重新打開後按「接著跑」。")
                Thread.sleep(4000)
                Process.killProcess(Process.myPid())
            }
        }, "probe-watchdog").apply { isDaemon = true }.start()
    }

    /** 熱了就等一下再量：狀態 ≥ 中等時等到回落，最多 [maxSec] 秒；等過就記下來。 */
    private fun cooldown(maxSec: Int = 150) {
        val t = System.nanoTime()
        var waited = false
        while (env.thermalStatus() >= 2 && (System.nanoTime() - t) / 1_000_000_000 < maxSec) {
            if (!waited) status("手機有點熱，等它降溫（最多 $maxSec 秒）…")
            waited = true
            Thread.sleep(5000)
        }
        if (waited) report.line("COOLDOWN", "waitedSec" to (System.nanoTime() - t) / 1_000_000_000, *env.snapshot())
    }

    fun run() {
        Process.setThreadPriority(Process.THREAD_PRIORITY_DEFAULT)
        readExpected()
        startWatchdog()
        if (state.done.isEmpty() && state.crashed.isEmpty()) {
            report.line("BEGIN", "format" to FORMAT, "thread" to Thread.currentThread().name, "priority" to Process.getThreadPriority(Process.myTid()))
            env.metaLines(report, buildTag)
            if (env.debuggable()) report.line("WARN", "what" to "這是可除錯的 APK：ART 比正式版慢很多，C++ 的倍數會被高估，數字不能當關卡依據")
            report.publish()
        } else {
            // 接著跑的可能是舊版探針開的紀錄：這一段之後的列照這一版的格式
            report.line("RESUME", "done" to state.done.size, "crashed" to state.crashed.joinToString(","), "format" to FORMAT, "buildTag" to buildTag)
        }
        // 載入唯一的 .so：夜讀的 native 方法要它載入後才找得到。JNI_OnLoad 死掉也要留得下紀錄，而且 (a) 不需要它，照跑。
        nativeLoaded = prep("init", "載入原生庫") {
            val so = NcnnBackend.available
            report.line(
                "META",
                "soLoaded" to so, "nightNative" to NightNative.available(),
                "numericBridge" to NightNumericProbe.available(),
                "rulesVersion" to NightRead.RULES_VERSION, "abi" to NightNative.ABI_VERSION,
                "workload" to NightPrimWorkload.VERSION, "binWorkload" to NightBinWorkload.VERSION, "grayWorkload" to NightGrayWorkload.VERSION,
            )
            so
        } == true
        cooldown()
        step("a", "(a) 8 頁守門摘要（約 2 分鐘）") { stepGuard() }
        step("b", "(b) 數值核心對照（約 1–2 分鐘）") { stepNumeric() }
        cooldown()
        step("c", "(c) 5 族影像原語：比位元、比時間（約 2 分鐘）") { stepPrims() }
        cooldown()
        step("c3a", "(c) 第 3a 階影像原語（二值、形態學、距離）：比位元、比時間（約 2–3 分鐘）") { stepBin() }
        step("c3a.x", "(c) 第 3a 階補充：步驟表沒有的原語，只比位元（約 30 秒）") { stepBinExtra() }
        cooldown()
        step("c3b", "(c) 第 3b 階影像原語（灰階、濾波、繪圖、numpy）：比位元、比時間（約 2–3 分鐘）") { stepGray() }
        step("c3b.x", "(c) 第 3b 階補充：步驟表沒涵蓋的參數與尺寸，只比位元（約 1 分鐘）") { stepGrayExtra() }
        cooldown()
        // 不管 (d) 發生什麼事都要有摘要：(a)(b)(c) 的關卡數字才不會只剩散落的 GATE 列
        var modelsReady = true
        try {
            modelsReady = stepModels()
        } catch (e: Throwable) {
            fail("d", e)
        }
        current = null
        state.inProgress = null
        summary(modelsReady)
        state.finished = modelsReady   // 缺模型不算跑完：換資料夾後按「接著跑」，(a)(b)(c) 不用重做
        report.publish()
        status(
            if (modelsReady) {
                "完成（這一輪 ${secs()} 秒）。摘要在下面，紀錄：${report.fileName}。請按「分享紀錄」。"
            } else {
                "(d) 的模型沒準備好（看紀錄的 FAIL 列）。換資料夾後按「接著跑」，做完的步驟不會重做。"
            },
        )
    }

    // ── (a) ─────────────────────────────────────────────────────────────────

    private fun fixtures(): (String) -> ByteArray? = { name ->
        try {
            ctx.assets.open("nightprobe/page/$name").use { it.readBytes() }
        } catch (e: java.io.IOException) {
            null
        }
    }

    private fun stepGuard() {
        val digest = GuardDigest.run(fixtures()) { page, sha, ms ->
            report.line("GUARD", "page" to page, "sha16" to sha.take(16), "ms" to ms)
            report.publish(force = false)
        }
        report.line("GATE", "key" to "kotlin_digest", "pass" to (digest == GuardDigest.EXPECTED), "digest" to digest, "expected" to GuardDigest.EXPECTED)
    }

    // ── (b) ─────────────────────────────────────────────────────────────────

    private fun stepNumeric() {
        if (!NightNative.available() || !NightNumericProbe.available()) {
            report.line("GATE", "key" to "numeric", "pass" to false, "why" to "這個 .so 沒有數值自測橋（要用 -Pyakuyomi.nightProbe=true 建置）或 C++ 版本不符")
            return
        }
        val threads = minOf(4, Runtime.getRuntime().availableProcessors())
        var cases = 0L
        var diffs = 0L
        val rec = { r: NightNumericProbe.Result ->
            cases += r.cases
            diffs += r.diffs
            report.line("NUM", "name" to r.name, "cases" to r.cases, "diffs" to r.diffs, "throws" to r.throws, "ms" to r.millis, "note" to r.note)
        }
        NightNumericProbe.runScalars(NUMERIC_CASES, threads, rec)
        report.publish(force = false)
        NightNumericProbe.runArrays(NUMERIC_ARRAY_SCALE, rec)
        report.publish(force = false)
        NightNumericProbe.runTables(threads, 1, rec)
        report.line("GATE", "key" to "numeric", "pass" to (diffs == 0L && cases > 0L), "cases" to cases, "diffs" to diffs)
    }

    // ── (c) ─────────────────────────────────────────────────────────────────

    private class PrimPage(val name: String, val input: NightReadInput)

    private fun primPages(): List<PrimPage> {
        val load = fixtures()
        return PRIM_PAGES.map { PrimPage(it, GuardDigest.input(load, it)) }
    }

    /** 一步的結果。第 2 階（[NightPrimWorkload]）與第 3a 階（[NightBinWorkload]）的 StepResult 是不同的類別，都轉成這個樣子寫紀錄。 */
    private class DiffRow(
        val id: String,
        val family: String,
        val weight: Double,
        val kotlinNs: Long,
        val nativeNs: Long,
        val identical: Boolean,
        val peakBytes: Long,
        val fma: Boolean = false,
    )

    /** 一族（[family] null＝合計）的加權總和，取自各自工作量的 summarize。 */
    private class DiffSum(val family: String?, val kotlinNs: Double, val nativeNs: Double, val identical: Boolean) {
        val ratio: Double get() = kotlinNs / nativeNs
    }

    /**
     * 一組原語寫的紀錄種類：第 2 階沿用 WARM／PRIM／PRIM_PAGE／PRIM_TOTAL（跟 nightprobe-2 一樣），第 3a 階用 BIN_*，第 3b 階用 GRAY_*。
     * [label] 接在暖機警告後面；[fma]＝逐步的列多一欄 fma（這一步合成 FMA 會不會改結果，第 3b 階才有）。
     */
    private class DiffKinds(val warm: String, val step: String, val page: String, val total: String, val label: String, val fma: Boolean = false)

    /** [different]＝正式量時有任一次不同的步驟；[warmDifferent]＝暖機時不同的「頁/步驟」。 */
    private class DiffOutcome(val rows: List<DiffRow>, val sums: List<DiffSum>, val different: List<String>, val warmDifferent: Set<String>)

    /**
     * 一組原語在 [pages] 上的同行程差分與計時：先暖機（ART 的 JIT 要幾輪才穩定；每輪都記，連兩輪 Kotlin 時間變化 < 3% 就停，最少 3 輪、最多 8 輪），
     * 再每頁 [PRIM_REPEAT] 次、取中位數，逐步、逐頁、合計寫紀錄。
     * 暖機也比位元：只偶爾才不同的那種差異（重用記憶體、讀到未初始化資料）只在這裡出現一次也算沒過，所以不能只看最後那輪。
     */
    private fun <R> diffSet(
        kinds: DiffKinds,
        pages: List<PrimPage>,
        differential: (NightReadInput, Int, ((R) -> Unit)?) -> List<R>,
        row: (R) -> DiffRow,
        summarize: (List<R>) -> List<DiffSum>,
    ): DiffOutcome {
        val warmDifferent = LinkedHashSet<String>()
        var prev = Long.MAX_VALUE
        var stable = 0
        for (pass in 1..8) {
            val rs = pages.flatMap { pg ->
                differential(pg.input, 1, null).map(row).onEach { r ->
                    if (!r.identical) {
                        warmDifferent += "${pg.name}/${r.id}"
                        report.line("WARN", "what" to "暖機時 C++ 與 Kotlin 位元不同${kinds.label}", "page" to pg.name, "step" to r.id, "pass" to pass)
                    }
                }
            }
            val kt = rs.sumOf { it.kotlinNs } / 1_000_000
            val cc = rs.sumOf { it.nativeNs } / 1_000_000
            report.line(kinds.warm, "pass" to pass, "kotlinMs" to kt, "cppMs" to cc, "identical" to rs.all { it.identical })
            stable = if (prev != Long.MAX_VALUE && Math.abs(kt - prev) < prev * 0.03) stable + 1 else 0
            prev = kt
            if (pass >= 3 && stable >= 1) break
        }
        report.publish(force = false)
        val raw = ArrayList<R>()
        for (pg in pages) {
            val rs = differential(pg.input, PRIM_REPEAT) { res ->
                val r = row(res)
                val cols = arrayOf<Pair<String, Any?>>(
                    "page" to pg.name, "step" to r.id, "family" to r.family, "weight" to r.weight,
                    "kotlinUs" to r.kotlinNs / 1000, "cppUs" to r.nativeNs / 1000, "identical" to r.identical, "peakKB" to r.peakBytes / 1024,
                )
                if (kinds.fma) report.line(kinds.step, *cols, "fma" to r.fma) else report.line(kinds.step, *cols)
            }
            raw += rs
            for (s in summarize(rs)) {
                report.line(
                    kinds.page, "page" to pg.name, "family" to (s.family ?: "ALL"), "kotlinMs" to "%.1f".format(s.kotlinNs / 1e6),
                    "cppMs" to "%.1f".format(s.nativeNs / 1e6), "ratio" to "%.3f".format(s.ratio), "identical" to s.identical,
                )
            }
            report.publish(force = false)
        }
        val sums = summarize(raw)
        for (s in sums) {
            report.line(
                kinds.total, "family" to (s.family ?: "ALL"), "kotlinMs" to "%.1f".format(s.kotlinNs / 1e6),
                "cppMs" to "%.1f".format(s.nativeNs / 1e6), "ratio" to "%.3f".format(s.ratio), "identical" to s.identical,
            )
        }
        val rows = raw.map(row)
        return DiffOutcome(rows, sums, rows.filter { !it.identical }.map { it.id }.distinct(), warmDifferent)
    }

    private fun perFamily(sums: List<DiffSum>) = sums.dropLast(1).joinToString(",") { "${it.family}=${"%.3f".format(it.ratio)}" }

    /** 第 2 階的 5 族（[NightPrimWorkload]）：關卡＝全部相同＋合計至少 1.5 倍。 */
    private fun stepPrims() {
        if (!NightNative.available()) {
            report.line("GATE", "key" to "families_identical", "pass" to false, "why" to "C++ 不可用（版本不符或 .so 沒載入）")
            return
        }
        val o = diffSet(
            PRIM_KINDS, primPages(),
            { p: NightReadInput, rep: Int, cb: ((NightPrimWorkload.StepResult) -> Unit)? -> NightPrimWorkload.differential(p.gray, p.seg, p.charMask, rep, cb) },
            { r -> DiffRow(r.step.id, r.step.family.toString(), r.step.weight, r.kotlinNs, r.nativeNs, r.identical, r.nativePeakBytes) },
            { rs -> NightPrimWorkload.summarize(rs).map { DiffSum(it.family?.name, it.kotlinWeightedNs, it.nativeWeightedNs, it.identical) } },
        )
        report.line(
            "GATE", "key" to "families_identical", "pass" to (o.different.isEmpty() && o.warmDifferent.isEmpty()), "steps" to o.rows.size,
            "different" to o.different.joinToString(","), "warmDifferent" to o.warmDifferent.joinToString(","),
        )
        val total = o.sums.last()
        report.line(
            "GATE", "key" to "combined_ratio", "pass" to (total.ratio >= 1.5), "ratio" to "%.3f".format(total.ratio), "need" to 1.5,
            "perFamily" to perFamily(o.sums),
            "note" to "權重＝一頁產品兩檔的實際呼叫次數（6 頁中位數，NightPrimWorkload 檔頭）；遮罩是頁面導出的代理；C++ 時間不含 JNI 複製",
        )
    }

    /**
     * 第 3a 階的二值原語（[NightBinWorkload]：MORPH、LINE、DIST、MISC、BITS）。這一階的關卡只有「全部相同」（要比滿整張步驟表）；
     * 倍數只做參考：權重 0 的步驟（巢狀在其他原語裡的、一頁 0–1 次的）只比位元、不計入合計。
     */
    private fun stepBin() {
        if (!NightNative.available()) {
            report.line("GATE", "key" to "bin_identical", "pass" to false, "why" to "C++ 不可用（版本不符或 .so 沒載入）")
            return
        }
        val pages = primPages()
        val want = pages.sumOf { pg -> binSteps(pg.input.gray.w, pg.input.gray.h) }
        val o = diffSet(
            BIN_KINDS, pages,
            { p: NightReadInput, rep: Int, cb: ((NightBinWorkload.StepResult) -> Unit)? -> NightBinWorkload.differential(p.gray, p.seg, p.charMask, rep, cb) },
            { r -> DiffRow(r.step.id, r.step.family.toString(), r.step.weight, r.kotlinNs, r.nativeNs, r.identical, r.nativePeakBytes) },
            { rs -> NightBinWorkload.summarize(rs).map { DiffSum(it.family?.name, it.kotlinWeightedNs, it.nativeWeightedNs, it.identical) } },
        )
        report.line(
            "GATE", "key" to "bin_identical", "pass" to ProbeRecords.verdict(o.different.size + o.warmDifferent.size, o.rows.size, want),
            "steps" to o.rows.size, "want" to want, "different" to o.different.joinToString(","), "warmDifferent" to o.warmDifferent.joinToString(","),
            "workload" to NightBinWorkload.VERSION,
        )
        val total = o.sums.last()
        report.line(
            "GATE", "key" to "bin_ratio", "ratio" to "%.3f".format(total.ratio), "perFamily" to perFamily(o.sums),
            "note" to "只做參考、不是關卡；權重＝一頁產品兩檔的實際呼叫次數（6 頁中位數，NightBinWorkload 檔頭），巢狀與一頁 0–1 次的權重 0；遮罩是頁面導出的代理；C++ 時間不含 JNI 複製",
        )
    }

    /**
     * 第 3b 階的灰階、濾波、繪圖、numpy 原語（[NightGrayWorkload]：GMORPH、FILTER、EDGE、DRAW、SPARSE、NUMPY）。關卡同第 3a 階：只有「全部相同」
     * （要比滿整張步驟表）；倍數只做參考。步驟表標了 fmaSensitive 的步驟（可分離濾波）合成 FMA 會改結果，所以這幾步相同＝手機上的 C++
     * 沒把 Float 乘加合成 FMA（CI 另外掃反組譯）：關卡列出比到幾步、哪幾步。
     */
    private fun stepGray() {
        if (!NightNative.available()) {
            report.line("GATE", "key" to "gray_identical", "pass" to false, "why" to "C++ 不可用（版本不符或 .so 沒載入）")
            return
        }
        val pages = primPages()
        val want = pages.sumOf { pg -> graySteps(pg.input.gray.w, pg.input.gray.h) }
        val fmaWant = pages.sumOf { pg -> grayFmaSteps(pg.input.gray.w, pg.input.gray.h) }
        val o = diffSet(
            GRAY_KINDS, pages,
            { p: NightReadInput, rep: Int, cb: ((NightGrayWorkload.StepResult) -> Unit)? -> NightGrayWorkload.differential(p.gray, p.seg, p.charMask, rep, cb) },
            { r -> DiffRow(r.step.id, r.step.family.toString(), r.step.weight, r.kotlinNs, r.nativeNs, r.identical, r.nativePeakBytes, r.step.fmaSensitive) },
            { rs -> NightGrayWorkload.summarize(rs).map { DiffSum(it.family?.name, it.kotlinWeightedNs, it.nativeWeightedNs, it.identical) } },
        )
        val fmaRows = o.rows.filter { it.fma }
        val fmaIds = fmaRows.map { it.id }.distinct()
        // 暖機時不同的也算（[DiffOutcome.warmDifferent] 是「頁/步驟」）
        val fmaDifferent = (fmaRows.filter { !it.identical }.map { it.id } + o.warmDifferent.map { it.substringAfter('/') }.filter { it in fmaIds }).distinct()
        report.line(
            "GATE", "key" to "gray_identical", "pass" to ProbeRecords.verdict(o.different.size + o.warmDifferent.size, o.rows.size, want),
            "steps" to o.rows.size, "want" to want, "fmaSteps" to fmaRows.size, "fmaWant" to fmaWant,
            "fmaIds" to fmaIds.joinToString(","), "fmaDifferent" to fmaDifferent.joinToString(","),
            "different" to o.different.joinToString(","), "warmDifferent" to o.warmDifferent.joinToString(","),
            "workload" to NightGrayWorkload.VERSION,
        )
        val total = o.sums.last()
        report.line(
            "GATE", "key" to "gray_ratio", "ratio" to "%.3f".format(total.ratio), "perFamily" to perFamily(o.sums),
            "note" to "只做參考、不是關卡；權重＝一頁產品兩檔的實際呼叫次數（6 頁中位數，NightGrayWorkload 檔頭；20 條線、2000 點一步的已除掉），一頁中位數 0 次的權重 0；C++ 時間不含 JNI 複製",
        )
    }

    /**
     * 補充比對一種運算的累計。時間只是參考：整張影像的原語，C++ 取原語本身的耗時（不含 JNI 複製）；回傳不是 Timed 的（ellipse、rect、
     * polygonArea、npSum、npArgsort、eigh2、gaussW）含 JNI。
     */
    private class Extra {
        var cases = 0
        var diffs = 0
        var fmaCases = 0
        var fmaDiffs = 0
        var kotlinNs = 0L
        var nativeNs = 0L
        val firstDiffs = ArrayList<String>()
    }

    /** 只接一般的例外：兩邊都丟＝相同（Kotlin 會丟例外的輸入，C++ 也要丟）。OOM 之類的 Error 照樣往外丟，這一步記 FAIL。 */
    private fun attempt(f: () -> Any): Result<Any> = try {
        Result.success(f())
    } catch (e: Exception) {
        Result.failure(e)
    }

    /** 步驟表沒有的第 3a 階原語，只比位元（案例見 [ProbeBinExtras]）。紀錄與 nightprobe-3 相同：BIN_EXTRA 的 fma 欄照運算標。 */
    private fun stepBinExtra() {
        if (!NightNative.available()) {
            report.line("GATE", "key" to "bin_extra_identical", "pass" to false, "why" to "C++ 不可用（版本不符或 .so 沒載入）")
            return
        }
        extraSet(
            "BIN_EXTRA", "bin_extra_identical", "第 3a 階", ProbeBinExtras.globalCases(), { pg -> ProbeBinExtras.pageCases(pg.name, pg.input) },
            ProbeBinExtras::same, { c -> c.op in ProbeBinExtras.FMA_SENSITIVE }, perCaseFma = false,
        )
    }

    /** 步驟表沒涵蓋的第 3b 階參數與尺寸，只比位元（案例見 [ProbeGrayExtras]）。會受 FMA 影響的逐組標（電腦上核對過的整頁可分離濾波）。 */
    private fun stepGrayExtra() {
        if (!NightNative.available()) {
            report.line("GATE", "key" to "gray_extra_identical", "pass" to false, "why" to "C++ 不可用（版本不符或 .so 沒載入）")
            return
        }
        extraSet(
            "GRAY_EXTRA", "gray_extra_identical", "第 3b 階", ProbeGrayExtras.globalCases(), { pg -> ProbeGrayExtras.pageCases(pg.name, pg.input) },
            ProbeGrayExtras::same, { c -> c.fma }, perCaseFma = true,
        )
    }

    /**
     * 補充比對：跟頁無關的 [global] 先比，再逐頁建案例、比、丟掉（整頁的輸入不同時留在記憶體）。每種運算一列 [kind]、最後一列關卡 [gateKey]。
     * [perCaseFma] false（第 3a 階，紀錄格式不變）＝每列的 fma 欄是「這種運算會受 FMA 影響」；true（第 3b 階）＝每列記 fmaCases／fmaDiffs、
     * 關卡多 fmaDiffs。關卡的 fmaCases 都是 [isFma] 為真的組數。
     */
    private fun extraSet(
        kind: String,
        gateKey: String,
        label: String,
        global: List<ProbeBinExtras.Case>,
        pageCases: (PrimPage) -> List<ProbeBinExtras.Case>,
        same: (Any, Any) -> Boolean,
        isFma: (ProbeBinExtras.Case) -> Boolean,
        perCaseFma: Boolean,
    ) {
        val ex = LinkedHashMap<String, Extra>()
        fun cmp(c: ProbeBinExtras.Case) {
            val e = ex.getOrPut(c.op) { Extra() }
            val t0 = System.nanoTime()
            val a = attempt(c.kotlin)
            val t1 = System.nanoTime()
            val raw = attempt(c.native)
            val t2 = System.nanoTime()
            val timed = raw.getOrNull() as? NightNativePrims.Timed<*>
            val b = if (timed != null) Result.success(timed.value ?: Unit) else raw
            val fma = isFma(c)
            e.cases++
            if (fma) e.fmaCases++
            e.kotlinNs += t1 - t0
            e.nativeNs += timed?.nanos ?: (t2 - t1)
            val ok = when {
                a.isFailure && b.isFailure -> true
                a.isSuccess && b.isSuccess -> same(a.getOrThrow(), b.getOrThrow())
                else -> false
            }
            if (!ok) {
                e.diffs++
                if (fma) e.fmaDiffs++
                if (e.firstDiffs.size < 3) e.firstDiffs += c.tag
                report.line(
                    "WARN", "what" to "${label}補充比對不同", "op" to c.op, "case" to c.tag,
                    "kotlin" to (a.exceptionOrNull()?.javaClass?.simpleName ?: "ok"), "cpp" to (b.exceptionOrNull()?.javaClass?.simpleName ?: "ok"),
                )
            }
        }
        global.forEach(::cmp)
        for (pg in primPages()) {
            pageCases(pg).forEach(::cmp)
            report.publish(force = false)
        }
        var cases = 0
        var diffs = 0
        for ((op, e) in ex) {
            cases += e.cases
            diffs += e.diffs
            val head = arrayOf<Pair<String, Any?>>("op" to op, "cases" to e.cases, "diffs" to e.diffs, "kotlinUs" to e.kotlinNs / 1000, "cppUs" to e.nativeNs / 1000)
            if (perCaseFma) {
                report.line(kind, *head, "fmaCases" to e.fmaCases, "fmaDiffs" to e.fmaDiffs, "firstDiffs" to e.firstDiffs.joinToString(","))
            } else {
                report.line(kind, *head, "fma" to (e.fmaCases > 0), "firstDiffs" to e.firstDiffs.joinToString(","))
            }
        }
        val fmaCases = ex.values.sumOf { it.fmaCases }
        val tail = arrayOf<Pair<String, Any?>>("different" to ex.filterValues { it.diffs > 0 }.keys.joinToString(","))
        if (perCaseFma) {
            report.line(
                "GATE", "key" to gateKey, "pass" to (diffs == 0 && cases > 0), "cases" to cases, "diffs" to diffs,
                "fmaCases" to fmaCases, "fmaDiffs" to ex.values.sumOf { it.fmaDiffs }, *tail,
            )
        } else {
            report.line("GATE", "key" to gateKey, "pass" to (diffs == 0 && cases > 0), "cases" to cases, "diffs" to diffs, "fmaCases" to fmaCases, *tail)
        }
    }

    // ── (d) ─────────────────────────────────────────────────────────────────

    private fun key(page: String, m: ModelProbe.M) = "$page/${m.name}"

    /** 接著跑時，把上一輪記下的預期 sha 讀回來（紀錄檔本身就是狀態）。 */
    private fun readExpected() {
        forEachRecord("EXPECT") { kv -> expected[kv.getValue("key")] = kv.getValue("sha") }
    }

    private fun lines(): List<String> = if (report.local.exists()) report.local.readLines() else emptyList()

    private fun forEachRecord(kind: String, f: (Map<String, String>) -> Unit) = ProbeRecords.records(lines(), kind).forEach(f)

    /** 回傳 false＝模型沒準備好，這一輪不算跑完（換資料夾後可以接著跑）。 */
    private fun stepModels(): Boolean {
        if (!nativeLoaded) {
            report.line("SKIP", "step" to "d", "why" to "原生庫沒載入")
            return true
        }
        val probe = ModelProbe(ctx, tree, report, env)
        // 路徑留在記憶體，接著跑也要再登記一次（檔案已經複製過就很快）
        val ready = prep("d.models", "(d) 準備模型（第一次要從資料夾複製，約 1–2 分鐘）", limitMin = 30) { probe.prepareModels() }
        if (ready == null) {
            report.line("SKIP", "step" to "d", "why" to "模型沒準備好")
            return "d.models" in state.crashed
        }
        val pages = MODEL_PAGES.mapNotNull { asset ->
            prep("d.page.${asset.substringAfterLast('/')}", "(d) 準備頁面輸入：${asset.substringAfterLast('/')}") { probe.preparePage(asset) }
        }
        if (pages.isEmpty()) {
            report.line("SKIP", "step" to "d", "why" to "沒有一張頁準備好")
            return true
        }
        report.publish()
        cooldown()
        matrix(probe, pages)
        cooldown()
        pairs(probe, pages)
        cooldown()
        throughput(probe, pages[0])
        return true
    }

    /** 6／3 緒交錯量（交錯才不會有一邊全落在溫控之後），1 緒另外跑一輪；比 sha，並算 3 緒比 6 緒慢幾倍。 */
    private fun matrix(probe: ModelProbe, pages: List<ModelProbe.Page>) {
        step("d.matrix.6v3", "(d) 模型：6 緒、3 緒輸出比對與速度（約 2 分鐘）") {
            val ratios = HashMap<ModelProbe.M, Pair<Long, Long>>()
            var mismatches = 0
            var compared = 0
            probe.NetSet(6).use { six ->
                probe.NetSet(3).use { three ->
                    for (pg in pages) for (m in ModelProbe.M.values()) {
                        val outs = probe.alloc(m, pg)
                        // 第一次前向含 lazy 初始化，不計
                        val coldMs6 = probe.forward(m, six[m], pg, outs) / 1_000_000
                        val coldMs3 = probe.forward(m, three[m], pg, outs) / 1_000_000
                        val t6 = LongArray(3)
                        val t3 = LongArray(3)
                        var sha6 = ""
                        var bad = 0
                        for (r in 0 until 3) {
                            val order = if (r % 2 == 0) listOf(6, 3) else listOf(3, 6)
                            for (th in order) {
                                val dt = probe.forward(m, (if (th == 6) six else three)[m], pg, outs)
                                val sha = ModelProbe.shaOf(outs)
                                if (th == 6) t6[r] = dt else t3[r] = dt
                                if (sha6.isEmpty()) sha6 = sha else if (sha != sha6) bad++
                            }
                        }
                        t6.sort()
                        t3.sort()
                        expected[key(pg.name, m)] = sha6
                        report.line("EXPECT", "key" to key(pg.name, m), "sha" to sha6)
                        report.line(
                            "MATRIX", "page" to pg.name, "model" to m.name, "ms6" to t6[1] / 1_000_000, "ms3" to t3[1] / 1_000_000,
                            "ratio" to "%.3f".format(t3[1].toDouble() / t6[1]), "cold6" to coldMs6, "cold3" to coldMs3,
                            "sha16" to sha6.take(16), "mismatches" to bad,
                        )
                        mismatches += bad
                        compared++
                        val (a, b) = ratios[m] ?: (0L to 0L)
                        ratios[m] = (a + t6[1]) to (b + t3[1])
                    }
                }
            }
            for ((m, v) in ratios) report.line("RATIO3V6", "model" to m.name, "ratio" to "%.3f".format(v.second.toDouble() / v.first))
            // 依一頁實際用到的次數加權（翻譯找字＋夜讀找字＋去字＋yolo＋cseg）
            val six = ModelProbe.ROUND.sumOf { ratios.getValue(it).first }
            val three = ModelProbe.ROUND.sumOf { ratios.getValue(it).second }
            report.line("GATE", "key" to "models_ratio_3v6", "ratio" to "%.3f".format(three.toDouble() / six), "assumed" to 1.35)
            report.line(
                "GATE", "key" to "models_sha_6v3", "pass" to ProbeRecords.verdict(mismatches, compared, MODEL_CELLS),
                "compared" to compared, "want" to MODEL_CELLS, "mismatches" to mismatches,
            )
        }
        step("d.matrix.1", "(d) 模型：1 緒輸出比對（約 1 分鐘）") {
            var mismatches = 0
            var compared = 0
            probe.NetSet(1).use { one ->
                for (pg in pages) for (m in ModelProbe.M.values()) {
                    val outs = probe.alloc(m, pg)
                    probe.forward(m, one[m], pg, outs)
                    val dt = probe.forward(m, one[m], pg, outs)
                    val sha = ModelProbe.shaOf(outs)
                    val want = expected[key(pg.name, m)] // 6／3 緒那步沒留下基準：只記 sha，不能當相同
                    if (want != null) {
                        compared++
                        if (want != sha) mismatches++
                    }
                    report.line(
                        "MATRIX1", "page" to pg.name, "model" to m.name, "ms1" to dt / 1_000_000, "sha16" to sha.take(16),
                        "same" to (if (want == null) "none" else (want == sha).toString()),
                    )
                }
            }
            report.line(
                "GATE", "key" to "models_sha_1", "pass" to ProbeRecords.verdict(mismatches, compared, MODEL_CELLS),
                "compared" to compared, "want" to MODEL_CELLS, "mismatches" to mismatches,
            )
        }
    }

    /** 兩兩同時跑，比 sha。每個組合一步：閃退了下次跳過那一個組合。 */
    private fun pairs(probe: ModelProbe, pages: List<ModelProbe.Page>) {
        for (th in PAIR_THREADS) {
            val ids = pages.flatMap { pg -> ModelProbe.PAIRS.map { (a, b) -> Triple(pg, a, b) } }
            fun id(pg: ModelProbe.Page, a: ModelProbe.M, b: ModelProbe.M) = "d.pair.$th.${pg.name}.$a+$b"
            if (ids.all { (pg, a, b) -> id(pg, a, b) in state.done || id(pg, a, b) in state.crashed }) continue
            val loaded = prep("d.pair.$th.load", "(d) 載入模型，準備兩兩同時跑（$th 緒）") {
                probe.NetSet(th).also { s ->
                    try {
                        pages.firstOrNull()?.let { probe.warm(s, it) }
                    } catch (e: Throwable) {
                        s.close()
                        throw e
                    }
                }
            }
            if (loaded == null) {
                report.line("SKIP", "step" to "d.pair.$th", "why" to "d.pair.$th.load 沒成功，這個緒數的組合都沒跑")
                continue
            }
            loaded.use { set ->
                for ((pg, a, b) in ids) {
                    step(id(pg, a, b), "(d) 同時跑 $a＋$b（$th 緒）", limitMin = 3) {
                        val bad = probe.pair(a, b, set, pg, { m -> expected[key(pg.name, m)] }) { kv -> report.line("PAIR", *kv) }
                        if (bad != 0) report.line("WARN", "what" to "同時跑輸出不同", "pair" to "$a+$b", "threads" to th, "page" to pg.name, "count" to bad)
                    }
                }
            }
        }
        val all = lines()
        val g = ProbeRecords.pairs(all)
        report.line(
            "GATE", "key" to "models_sha_pairs", "pass" to ProbeRecords.verdict(g.mismatches, g.compared, PAIR_CELLS),
            "pairs" to g.pairs, "compared" to g.compared, "want" to PAIR_CELLS, "mismatches" to g.mismatches,
            "crashed" to state.crashed.filter { it.startsWith("d.pair.") }.joinToString(","),
            "failed" to ProbeRecords.failedSteps(all).filter { it.startsWith("d.pair.") }.joinToString(","),
        )
    }

    /** A＝6 緒一次一個、B＝兩個 3 緒同時、C＝兩個 6 緒同時，各 ×10，A、B 量兩次（前後對照溫控漂移）。 */
    private fun throughput(probe: ModelProbe, pg: ModelProbe.Page) {
        data class Mode(val id: String, val label: String, val th: Int, val workers: Int)
        val modes = listOf(
            Mode("A1", "6 緒一次一個", 6, 1), Mode("B1", "兩個 3 緒同時", 3, 2), Mode("C", "兩個 6 緒同時", 6, 2),
            Mode("A2", "6 緒一次一個（第二次）", 6, 1), Mode("B2", "兩個 3 緒同時（第二次）", 3, 2),
        )
        if (modes.all { "d.tp.${it.id}" in state.done || "d.tp.${it.id}" in state.crashed }) {
            tpGate()
            return
        }
        val nets = prep("d.tp.load", "(d) 載入模型，準備量吞吐") {
            val six = probe.NetSet(6)
            val three = try {
                probe.NetSet(3)
            } catch (e: Throwable) {
                six.close()
                throw e
            }
            try {
                probe.warm(six, pg)
                probe.warm(three, pg)
            } catch (e: Throwable) {
                three.close()
                six.close()
                throw e
            }
            six to three
        }
        if (nets == null) {
            report.line("SKIP", "step" to "d.tp", "why" to "d.tp.load 沒成功，吞吐都沒跑")
            tpGate()
            return
        }
        status("(d) 吞吐（約 3 分鐘）…")
        nets.first.use { six ->
            nets.second.use { three ->
                for (m in modes) {
                    val set = if (m.th == 6) six else three
                    step("d.tp.${m.id}", "(d) 吞吐 ${m.label} ×$TP_ROUNDS", limitMin = 8) {
                        val r = probe.throughput(TP_ROUNDS, m.workers, List(m.workers) { set }, pg) { k -> expected[key(pg.name, k)] }
                        val wallMs = r.wallNs / 1_000_000
                        report.line(
                            "TP", "mode" to m.id, "label" to m.label, "threads" to m.th, "workers" to m.workers, "tasks" to r.tasks, "rounds" to TP_ROUNDS,
                            "wallMs" to wallMs, "cpuMs" to r.cpuMs, "cores" to "%.2f".format(r.cpuMs.toDouble() / wallMs),
                            "msPerRound" to wallMs / TP_ROUNDS, "medianMs" to r.medianMs.entries.joinToString(",") { "${it.key}=${it.value}" },
                            "mismatches" to r.mismatches, "compared" to r.compared, "noExpect" to r.noExpect,
                        )
                        if (r.mismatches != 0) report.line("WARN", "what" to "吞吐實驗輸出不同", "mode" to m.id, "count" to r.mismatches)
                    }
                }
            }
        }
        tpGate()
    }

    private fun tpGate() {
        val g = ProbeRecords.throughput(lines())
        report.line(
            "GATE", "key" to "models_sha_tp", "pass" to ProbeRecords.verdict(g.mismatches, g.comparedModes, TP_MODES),
            "compared" to g.comparedModes, "want" to TP_MODES, "mismatches" to g.mismatches, "noExpect" to g.noExpect,
        )
        val wall = g.wallMs
        val a = listOfNotNull(wall["A1"], wall["A2"])
        val b = listOfNotNull(wall["B1"], wall["B2"])
        if (a.isEmpty() || b.isEmpty()) {
            report.line("GATE", "key" to "models_throughput", "pass" to false, "why" to "有模式沒量到（閃退或失敗）")
            return
        }
        // 吞吐＝工作數 ÷ 時間，工作數相同，所以倍數＝時間的反比
        val gain = a.average() / b.average() - 1.0
        report.line(
            "GATE", "key" to "models_throughput", "pass" to (gain >= 0.25), "gainPct" to "%.1f".format(gain * 100), "need" to "25",
            "A" to a.joinToString(","), "B" to b.joinToString(","), "C" to wall["C"], "modes" to wall.size, "mismatches" to g.mismatches,
            "note" to "兩個 3 緒同時比 6 緒一次一個快多少（只做紀錄，不擋往下）",
        )
    }

    // ── 摘要 ─────────────────────────────────────────────────────────────────

    private fun summary(modelsReady: Boolean) {
        val all = lines()
        val gates = LinkedHashMap<String, Map<String, String>>()
        for (kv in ProbeRecords.records(all, "GATE")) gates[kv.getValue("key")] = kv // 同一個關卡寫過兩次（接著跑）以最後一次為準
        val warns = ProbeRecords.records(all, "WARN").size
        val fg = ProbeRecords.foreground(all)
        val failed = ProbeRecords.failedSteps(all)
        fun g(k: String) = gates[k]
        fun pf(k: String): String = when (g(k)?.get("pass")) {
            "true" -> "通過"
            "false" -> "沒過"
            "incomplete" -> "沒比完"
            else -> "沒量到"
        }
        fun cnt(k: String) = "${g(k)?.get("compared") ?: "0"}／${g(k)?.get("want") ?: "-"} 組，不同 ${g(k)?.get("mismatches") ?: "-"}"
        val out = ArrayList<String>()
        out += "SUMMARY\t==== 關卡摘要（只列數字，不下決定）===="
        out += "SUMMARY\t總時間 ${secs()} 秒（這一輪）；可除錯=${env.debuggable()}；警告 $warns 筆；離開最上層 ${fg.left} 次；閃退過的步驟：${state.crashed.joinToString(",").ifEmpty { "無" }}；失敗的步驟：${failed.joinToString(",").ifEmpty { "無" }}"
        if (!modelsReady) out += "SUMMARY\t★ (d) 的模型沒準備好，這一輪還沒跑完（換資料夾後按「接著跑」）"
        if (fg.left > 0 || fg.steps.isNotEmpty()) {
            val hit = listOfNotNull("第 3 項".takeIf { "c" in fg.steps }, "第 5c 項".takeIf { "c3a" in fg.steps }, "第 6c 項".takeIf { "c3b" in fg.steps })
            out += "SUMMARY\t★ 中途離開過畫面：這台手機切走後只給 5 顆核心，這幾步的時間偏慢：${fg.steps.joinToString(",")}${if (hit.isEmpty()) "" else "（含 (c)，下面${hit.joinToString("、")}的倍數要打折看）"}"
        }
        out += "SUMMARY\t1. 手機上 Kotlin 摘要等於電腦：${pf("kotlin_digest")}（${g("kotlin_digest")?.get("digest")?.take(16) ?: "-"}）"
        out += "SUMMARY\t2a. 數值核心逐位元相同：${pf("numeric")}（${g("numeric")?.get("cases") ?: "-"} 組，不同 ${g("numeric")?.get("diffs") ?: "-"}）${g("numeric")?.get("why") ?: ""}"
        out += "SUMMARY\t2b. 5 族原語逐位元相同：${pf("families_identical")}（${g("families_identical")?.get("steps") ?: "-"} 步；不同：${g("families_identical")?.get("different")?.ifEmpty { "無" } ?: "-"}；暖機時不同：${g("families_identical")?.get("warmDifferent")?.ifEmpty { "無" } ?: "-"}）"
        out += "SUMMARY\t3. 5 族合計 C++ 比 ART 快幾倍（門檻 1.5）：${g("combined_ratio")?.get("ratio") ?: "-"}（${pf("combined_ratio")}）；各族 ${g("combined_ratio")?.get("perFamily") ?: "-"}"
        out += "SUMMARY\t4a. 模型輸出（只做紀錄）：6／3 緒 ${pf("models_sha_6v3")}（${cnt("models_sha_6v3")}）；1 緒 ${pf("models_sha_1")}（${cnt("models_sha_1")}）；兩兩同時 ${pf("models_sha_pairs")}（${cnt("models_sha_pairs")}）；吞吐 ${pf("models_sha_tp")}（${cnt("models_sha_tp")}）"
        out += "SUMMARY\t4b. 模型速度（只做紀錄）：3 緒比 6 緒慢 ${g("models_ratio_3v6")?.get("ratio") ?: "-"} 倍（模擬假設 1.35）；兩個 3 緒同時比 6 緒一次一個快 ${g("models_throughput")?.get("gainPct") ?: "-"}%（門檻 25%）；A=${g("models_throughput")?.get("A") ?: "-"} B=${g("models_throughput")?.get("B") ?: "-"} C=${g("models_throughput")?.get("C") ?: "-"} ms"
        out += "SUMMARY\t5a. 第 3a 階原語逐位元相同（這一階的關卡）：${pf("bin_identical")}（${g("bin_identical")?.get("steps") ?: "-"}／${g("bin_identical")?.get("want") ?: "-"} 步；不同：${g("bin_identical")?.get("different")?.ifEmpty { "無" } ?: "-"}；暖機時不同：${g("bin_identical")?.get("warmDifferent")?.ifEmpty { "無" } ?: "-"}）${g("bin_identical")?.get("why") ?: ""}"
        out += "SUMMARY\t5b. 第 3a 階補充：步驟表沒有的原語逐位元相同（也算這一階的關卡）：${pf("bin_extra_identical")}（${g("bin_extra_identical")?.get("cases") ?: "-"} 組，其中會受 FMA 影響的 ${g("bin_extra_identical")?.get("fmaCases") ?: "-"} 組；不同 ${g("bin_extra_identical")?.get("diffs") ?: "-"}${g("bin_extra_identical")?.get("different")?.takeIf { it.isNotEmpty() }?.let { "：$it" } ?: ""}）${g("bin_extra_identical")?.get("why") ?: ""}"
        out += "SUMMARY\t5c. 第 3a 階合計 C++ 比 ART 快幾倍（只做參考）：${g("bin_ratio")?.get("ratio") ?: "-"}；各族 ${g("bin_ratio")?.get("perFamily") ?: "-"}"
        val gi = g("gray_identical")
        out += "SUMMARY\t6a. 第 3b 階原語逐位元相同（這一階的關卡）：${pf("gray_identical")}（${gi?.get("steps") ?: "-"}／${gi?.get("want") ?: "-"} 步，其中會受 FMA 影響的 ${gi?.get("fmaSteps") ?: "-"}／${gi?.get("fmaWant") ?: "-"} 步${gi?.get("fmaIds")?.takeIf { it.isNotEmpty() }?.let { "（$it）" } ?: ""}、不同 ${gi?.get("fmaDifferent")?.ifEmpty { "無" } ?: "-"}；不同：${gi?.get("different")?.ifEmpty { "無" } ?: "-"}；暖機時不同：${gi?.get("warmDifferent")?.ifEmpty { "無" } ?: "-"}）${gi?.get("why") ?: ""}"
        val ge = g("gray_extra_identical")
        out += "SUMMARY\t6b. 第 3b 階補充：步驟表沒涵蓋的參數與尺寸逐位元相同（也算這一階的關卡）：${pf("gray_extra_identical")}（${ge?.get("cases") ?: "-"} 組，其中電腦上核對過會受 FMA 影響的 ${ge?.get("fmaCases") ?: "-"} 組、不同 ${ge?.get("fmaDiffs") ?: "-"}；不同 ${ge?.get("diffs") ?: "-"}${ge?.get("different")?.takeIf { it.isNotEmpty() }?.let { "：$it" } ?: ""}）${ge?.get("why") ?: ""}"
        out += "SUMMARY\t6c. 第 3b 階合計 C++ 比 ART 快幾倍（只做參考）：${g("gray_ratio")?.get("ratio") ?: "-"}；各族 ${g("gray_ratio")?.get("perFamily") ?: "-"}"
        out += "SUMMARY\t6d. 手機上的 C++ 有沒有把 Float 乘加合成 FMA（6a 的 FMA 步驟與 6b 的 FMA 組全部比到且相同才算「沒有」）：${fmaEvidence(gi, ge)}"
        out.forEach { report.raw(it) }
        report.line("END", "finished" to modelsReady)
    }

    /**
     * 6d：會受 FMA 影響的步驟（6a）與案例（6b）全部比到且相同＝「沒有」；有一處不同＝「有不同」（先看是哪一步、哪一組）；
     * 有一項沒量到或沒比完＝「不完整」（不能當證據）。
     */
    private fun fmaEvidence(gi: Map<String, String>?, ge: Map<String, String>?): String {
        val steps = gi?.get("fmaSteps")?.toIntOrNull()
        val stepsWant = gi?.get("fmaWant")?.toIntOrNull()
        val stepsDiff = gi?.get("fmaDifferent").orEmpty()
        val cases = ge?.get("fmaCases")?.toIntOrNull()
        val casesDiff = ge?.get("fmaDiffs")?.toIntOrNull()
        val n = "6a ${steps ?: "-"}／${stepsWant ?: "-"} 步、6b ${cases ?: "-"} 組"
        return when {
            stepsDiff.isNotEmpty() || (casesDiff ?: 0) > 0 -> "有不同（$n；不同的步驟：${stepsDiff.ifEmpty { "無" }}；不同的組數：${casesDiff ?: "-"}）"
            steps == null || stepsWant == null || steps == 0 || steps < stepsWant || cases == null || cases == 0 || casesDiff == null -> "不完整（$n）"
            else -> "沒有（$n 全部相同）"
        }
    }

    companion object {
        /**
         * 紀錄格式。nightprobe-3＝nightprobe-2 再加第 3a 階：步驟 c3a（BIN_WARM／BIN／BIN_PAGE／BIN_TOTAL、關卡 bin_identical／bin_ratio）、
         * 步驟 c3a.x（BIN_EXTRA 帶 fma 欄、關卡 bin_extra_identical 帶 fmaCases）、摘要第 5 項。
         * nightprobe-4＝再加第 3b 階：步驟 c3b（GRAY_WARM／GRAY（多 fma 欄）／GRAY_PAGE／GRAY_TOTAL、關卡 gray_identical 帶 fmaSteps／fmaWant／
         * fmaIds／fmaDifferent、gray_ratio）、步驟 c3b.x（GRAY_EXTRA 帶 fmaCases／fmaDiffs、關卡 gray_extra_identical 帶 fmaCases／fmaDiffs）、
         * 摘要第 6 項（6d 是 FMA 的證據）；META 多 grayWorkload。原有的列不變。
         */
        const val FORMAT = "nightprobe-4"

        /** 每個純量函式的案例數（電腦上 200 萬、8 緒）。手機控制在 1–2 分鐘。 */
        const val NUMERIC_CASES = 1_000_000L

        /** 陣列案例數的比例（電腦上 1 倍）。 */
        const val NUMERIC_ARRAY_SCALE = 0.25

        /** (c) 的 4 張公開 fixture 頁（要在 [GuardDigest.PAGES] 裡：assets 只打包那些）。 */
        val PRIM_PAGES = listOf("ch34_011", "ch34_015", "demo05", "demo06")

        /** (c) 正式量時每頁每步跑幾次（取中位數）。 */
        const val PRIM_REPEAT = 5

        private val PRIM_KINDS = DiffKinds("WARM", "PRIM", "PRIM_PAGE", "PRIM_TOTAL", "")
        private val BIN_KINDS = DiffKinds("BIN_WARM", "BIN", "BIN_PAGE", "BIN_TOTAL", "（第 3a 階）")
        private val GRAY_KINDS = DiffKinds("GRAY_WARM", "GRAY", "GRAY_PAGE", "GRAY_TOTAL", "（第 3b 階）", fma = true)

        /** 第 3a 階步驟表在 [w]×[h] 的頁上有幾步原語（準備步驟不算）：比滿這麼多步才算「全部相同」。 */
        fun binSteps(w: Int, h: Int): Int = NightBinWorkload.program(w, h).count { it.family != null }

        /** 第 3b 階步驟表在 [w]×[h] 的頁上有幾步原語（準備步驟不算）：比滿這麼多步才算「全部相同」。 */
        fun graySteps(w: Int, h: Int): Int = NightGrayWorkload.program(w, h).count { it.family != null }

        /** 其中合成 FMA 會改結果的步驟數（[NightGrayWorkload.Step.fmaSensitive]；JVM 測試在 4 張頁上逐步核對）。 */
        fun grayFmaSteps(w: Int, h: Int): Int = NightGrayWorkload.program(w, h).count { it.family != null && it.fmaSensitive }

        /** (d) 的 3 張公開頁（sandbox 內建的原圖，解碼後用產品的前處理）。 */
        val MODEL_PAGES = listOf("test/ch34_011.jpg", "test/demo05.png", "test/demo01.jpg")

        /** 吞吐實驗每種模式 10 輪（一輪＝[ModelProbe.ROUND]）。 */
        const val TP_ROUNDS = 10

        /** 吞吐的模式數（A1、B1、C、A2、B2）。 */
        const val TP_MODES = 5

        /** 兩兩同時跑的緒數。 */
        val PAIR_THREADS = listOf(3, 6)

        /** 緒數矩陣應有的頁×模型數。少了就是沒比完，不能算通過。 */
        val MODEL_CELLS = MODEL_PAGES.size * ModelProbe.M.values().size

        /** 兩兩同時跑應有的組合數（3 頁 × 10 種 × 2 種緒數）。 */
        val PAIR_CELLS = MODEL_PAGES.size * ModelProbe.PAIRS.size * PAIR_THREADS.size
    }
}
