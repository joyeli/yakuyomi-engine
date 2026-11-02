package li.joye.yakuyomi.sandbox.probe

import android.content.Context
import android.os.Process
import androidx.documentfile.provider.DocumentFile
import li.joye.yakuyomi.engine.NcnnBackend
import li.joye.yakuyomi.nightread.NightNative
import li.joye.yakuyomi.nightread.NightNumericProbe
import li.joye.yakuyomi.nightread.NightPrimWorkload
import li.joye.yakuyomi.nightread.NightRead
import java.io.File

/**
 * 夜讀 C++ 探針本體，跑在一條普通的工作執行緒上（預設堆疊、預設優先權，跟之後產品呼叫 C++ 的條件一樣）：
 *  (a) 8 頁守門摘要（Kotlin）；
 *  (b) 數值核心：C++ 對 ART 的 StrictMath、捨入、BigDecimal、常數表；
 *  (c) 5 族原語：同行程差分＋計時（先暖機）；
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
            report.line("BEGIN", "format" to "nightprobe-2", "thread" to Thread.currentThread().name, "priority" to Process.getThreadPriority(Process.myTid()))
            env.metaLines(report, buildTag)
            if (env.debuggable()) report.line("WARN", "what" to "這是可除錯的 APK：ART 比正式版慢很多，C++ 的倍數會被高估，數字不能當關卡依據")
            report.publish()
        } else {
            report.line("RESUME", "done" to state.done.size, "crashed" to state.crashed.joinToString(","))
        }
        // 載入唯一的 .so：夜讀的 native 方法要它載入後才找得到。JNI_OnLoad 死掉也要留得下紀錄，而且 (a) 不需要它，照跑。
        nativeLoaded = prep("init", "載入原生庫") {
            val so = NcnnBackend.available
            report.line(
                "META",
                "soLoaded" to so, "nightNative" to NightNative.available(),
                "numericBridge" to NightNumericProbe.available(),
                "rulesVersion" to NightRead.RULES_VERSION, "workload" to NightPrimWorkload.VERSION,
            )
            so
        } == true
        cooldown()
        step("a", "(a) 8 頁守門摘要（約 2 分鐘）") { stepGuard() }
        step("b", "(b) 數值核心對照（約 1–2 分鐘）") { stepNumeric() }
        cooldown()
        step("c", "(c) 5 族影像原語：比位元、比時間（約 2 分鐘）") { stepPrims() }
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

    private class PrimPage(val name: String, val input: li.joye.yakuyomi.nightread.NightReadInput)

    private fun stepPrims() {
        if (!NightNative.available()) {
            report.line("GATE", "key" to "families_identical", "pass" to false, "why" to "C++ 不可用（版本不符或 .so 沒載入）")
            return
        }
        val load = fixtures()
        val pages = PRIM_PAGES.map { PrimPage(it, GuardDigest.input(load, it)) }
        // 暖機：ART 的 JIT 要幾輪才穩定。每輪都記，事後看得出 Kotlin 的時間有沒有收斂；連兩輪變化 < 3% 就停（最少 3 輪、最多 8 輪）。
        // 暖機也比位元：只偶爾才不同的那種差異（重用記憶體、讀到未初始化資料）只在這裡出現一次也算沒過，所以不能只看最後那輪。
        val warmDifferent = LinkedHashSet<String>()
        var prev = Long.MAX_VALUE
        var stable = 0
        for (pass in 1..8) {
            val rs = pages.flatMap { pg ->
                NightPrimWorkload.differential(pg.input.gray, pg.input.seg, pg.input.charMask, 1).onEach { r ->
                    if (!r.identical) {
                        warmDifferent += "${pg.name}/${r.step.id}"
                        report.line("WARN", "what" to "暖機時 C++ 與 Kotlin 位元不同", "page" to pg.name, "step" to r.step.id, "pass" to pass)
                    }
                }
            }
            val kt = rs.sumOf { it.kotlinNs } / 1_000_000
            val cc = rs.sumOf { it.nativeNs } / 1_000_000
            report.line("WARM", "pass" to pass, "kotlinMs" to kt, "cppMs" to cc, "identical" to rs.all { it.identical })
            stable = if (prev != Long.MAX_VALUE && Math.abs(kt - prev) < prev * 0.03) stable + 1 else 0
            prev = kt
            if (pass >= 3 && stable >= 1) break
        }
        report.publish(force = false)
        val all = ArrayList<NightPrimWorkload.StepResult>()
        for (pg in pages) {
            val rs = NightPrimWorkload.differential(pg.input.gray, pg.input.seg, pg.input.charMask, 5) { r ->
                report.line(
                    "PRIM", "page" to pg.name, "step" to r.step.id, "family" to r.step.family, "weight" to r.step.weight,
                    "kotlinUs" to r.kotlinNs / 1000, "cppUs" to r.nativeNs / 1000, "identical" to r.identical, "peakKB" to r.nativePeakBytes / 1024,
                )
            }
            all += rs
            for (s in NightPrimWorkload.summarize(rs)) {
                report.line(
                    "PRIM_PAGE", "page" to pg.name, "family" to (s.family ?: "ALL"), "kotlinMs" to "%.1f".format(s.kotlinWeightedNs / 1e6),
                    "cppMs" to "%.1f".format(s.nativeWeightedNs / 1e6), "ratio" to "%.3f".format(s.ratio), "identical" to s.identical,
                )
            }
            report.publish(force = false)
        }
        val sums = NightPrimWorkload.summarize(all)
        for (s in sums) {
            report.line(
                "PRIM_TOTAL", "family" to (s.family ?: "ALL"), "kotlinMs" to "%.1f".format(s.kotlinWeightedNs / 1e6),
                "cppMs" to "%.1f".format(s.nativeWeightedNs / 1e6), "ratio" to "%.3f".format(s.ratio), "identical" to s.identical,
            )
        }
        val bad = all.filter { !it.identical }.map { it.step.id }.distinct()
        report.line(
            "GATE", "key" to "families_identical", "pass" to (bad.isEmpty() && warmDifferent.isEmpty()), "steps" to all.size,
            "different" to bad.joinToString(","), "warmDifferent" to warmDifferent.joinToString(","),
        )
        val total = sums.last()
        report.line(
            "GATE", "key" to "combined_ratio", "pass" to (total.ratio >= 1.5), "ratio" to "%.3f".format(total.ratio), "need" to 1.5,
            "perFamily" to sums.dropLast(1).joinToString(",") { "${it.family}=${"%.3f".format(it.ratio)}" },
            "note" to "權重＝一頁產品兩檔的實際呼叫次數（6 頁中位數，NightPrimWorkload 檔頭）；遮罩是頁面導出的代理；C++ 時間不含 JNI 複製",
        )
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
            out += "SUMMARY\t★ 中途離開過畫面：這台手機切走後只給 5 顆核心，這幾步的時間偏慢：${fg.steps.joinToString(",")}${if ("c" in fg.steps) "（含 (c)，下面第 3 項的倍數要打折看）" else ""}"
        }
        out += "SUMMARY\t1. 手機上 Kotlin 摘要等於電腦：${pf("kotlin_digest")}（${g("kotlin_digest")?.get("digest")?.take(16) ?: "-"}）"
        out += "SUMMARY\t2a. 數值核心逐位元相同：${pf("numeric")}（${g("numeric")?.get("cases") ?: "-"} 組，不同 ${g("numeric")?.get("diffs") ?: "-"}）${g("numeric")?.get("why") ?: ""}"
        out += "SUMMARY\t2b. 5 族原語逐位元相同：${pf("families_identical")}（${g("families_identical")?.get("steps") ?: "-"} 步；不同：${g("families_identical")?.get("different")?.ifEmpty { "無" } ?: "-"}；暖機時不同：${g("families_identical")?.get("warmDifferent")?.ifEmpty { "無" } ?: "-"}）"
        out += "SUMMARY\t3. 5 族合計 C++ 比 ART 快幾倍（門檻 1.5）：${g("combined_ratio")?.get("ratio") ?: "-"}（${pf("combined_ratio")}）；各族 ${g("combined_ratio")?.get("perFamily") ?: "-"}"
        out += "SUMMARY\t4a. 模型輸出（只做紀錄）：6／3 緒 ${pf("models_sha_6v3")}（${cnt("models_sha_6v3")}）；1 緒 ${pf("models_sha_1")}（${cnt("models_sha_1")}）；兩兩同時 ${pf("models_sha_pairs")}（${cnt("models_sha_pairs")}）；吞吐 ${pf("models_sha_tp")}（${cnt("models_sha_tp")}）"
        out += "SUMMARY\t4b. 模型速度（只做紀錄）：3 緒比 6 緒慢 ${g("models_ratio_3v6")?.get("ratio") ?: "-"} 倍（模擬假設 1.35）；兩個 3 緒同時比 6 緒一次一個快 ${g("models_throughput")?.get("gainPct") ?: "-"}%（門檻 25%）；A=${g("models_throughput")?.get("A") ?: "-"} B=${g("models_throughput")?.get("B") ?: "-"} C=${g("models_throughput")?.get("C") ?: "-"} ms"
        out.forEach { report.raw(it) }
        report.line("END", "finished" to modelsReady)
    }

    companion object {
        /** 每個純量函式的案例數（電腦上 200 萬、8 緒）。手機控制在 1–2 分鐘。 */
        const val NUMERIC_CASES = 1_000_000L

        /** 陣列案例數的比例（電腦上 1 倍）。 */
        const val NUMERIC_ARRAY_SCALE = 0.25

        /** (c) 的 4 張公開 fixture 頁。 */
        val PRIM_PAGES = listOf("ch34_011", "ch34_015", "demo05", "demo06")

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
