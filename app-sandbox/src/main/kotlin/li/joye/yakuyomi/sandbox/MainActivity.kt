package li.joye.yakuyomi.sandbox

import android.content.Intent
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.RectF
import android.graphics.Typeface
import android.net.Uri
import android.os.Bundle
import android.util.Log
import android.view.View
import android.widget.ImageView
import android.widget.LinearLayout
import android.widget.TextView
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.documentfile.provider.DocumentFile
import androidx.lifecycle.lifecycleScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.async
import kotlinx.coroutines.awaitAll
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import li.joye.yakuyomi.engine.CharSegmenter
import li.joye.yakuyomi.engine.Detector
import li.joye.yakuyomi.engine.EngineConfig
import li.joye.yakuyomi.engine.Grouping
import li.joye.yakuyomi.engine.InpainterConfig
import li.joye.yakuyomi.engine.Inpainter
import li.joye.yakuyomi.engine.ModelSet
import li.joye.yakuyomi.engine.NightReadRenderer
import li.joye.yakuyomi.engine.NightReadStats
import li.joye.yakuyomi.engine.OcrConfig
import li.joye.yakuyomi.engine.PageResult
import li.joye.yakuyomi.engine.RenderConfig
import li.joye.yakuyomi.engine.TextOrientation
import li.joye.yakuyomi.engine.LlmTranslator
import li.joye.yakuyomi.engine.Ocr
import li.joye.yakuyomi.engine.OcrAbResult
import li.joye.yakuyomi.engine.OcrCandidate
import li.joye.yakuyomi.engine.Renderer
import li.joye.yakuyomi.engine.TranslatorConfig
import li.joye.yakuyomi.engine.Yakuyomi
import li.joye.yakuyomi.sandbox.databinding.ActivityMainBinding

/**
 * 批量測試 sandbox（BYOM）：對 4 張內建 demo 跑完整流程，逐頁 + 總計時，去字模式由開關決定（lama/boxfill）。
 * log + 成品圖寫回所選資料夾，方便撈檔比較兩種去字的速度與品質。
 */
class MainActivity : AppCompatActivity() {

    private lateinit var binding: ActivityMainBinding
    private val prefs by lazy { getSharedPreferences("yakuyomi", MODE_PRIVATE) }

    private val logBuf = StringBuilder()
    private var runTree: DocumentFile? = null
    private var runStamp = ""
    private var runImgIdx = 0
    private var runSaveImg = false

    private val folderPicker = registerForActivityResult(ActivityResultContracts.OpenDocumentTree()) { uri ->
        if (uri != null) {
            contentResolver.takePersistableUriPermission(
                uri,
                Intent.FLAG_GRANT_READ_URI_PERMISSION or Intent.FLAG_GRANT_WRITE_URI_PERMISSION,
            )
            prefs.edit().putString(PREF_TREE, uri.toString()).apply()
            binding.logText.text = "已記住資料夾：${DocumentFile.fromTreeUri(this, uri)?.name}"
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)
        binding.pickFolderButton.setOnClickListener { folderPicker.launch(null) }
        // LLM key：只能在 app 內輸入（存 prefs）；APK 不內建、留空存檔＝清掉（之後只跑偵測/OCR/去字、不翻譯）
        binding.apiKeyInput.setText(prefs.getString(PREF_API_KEY, "") ?: "")
        binding.apiKeySaveButton.setOnClickListener {
            val k = binding.apiKeyInput.text.toString().trim()
            prefs.edit().apply { if (k.isEmpty()) remove(PREF_API_KEY) else putString(PREF_API_KEY, k) }.apply()
            Toast.makeText(this, "LLM key → ${keyLabel()}", Toast.LENGTH_SHORT).show()
        }
        binding.detectButton.setOnClickListener { runPipeline() }
        binding.inpaintCompareButton.setOnClickListener { runInpaintCompare() }
        binding.repoDemoButton.setOnClickListener { runRepoDemo() }
        binding.crossPageButton.setOnClickListener { runCrossPageBench() }
        binding.ocrAbButton.setOnClickListener { runOcrAb() }
        binding.detectOcrCheckButton.setOnClickListener { runDetectOcrCheck() }
        binding.nightReadAbButton.setOnClickListener { runNightReadAb() }
        binding.inpaintSpinner.adapter = android.widget.ArrayAdapter(
            this, android.R.layout.simple_spinner_dropdown_item, INPAINT_MODES,
        )
        binding.inpaintSpinner.setSelection(1) // 預設＝Auto-整頁（速質平均）
        buildThumbnails()
        updateButtons()
        val t = currentTree()
        binding.logText.text =
            if (t == null) "① 先按「選擇模型資料夾」選含 dbnet／ocr_48px_ctc／aot 的 .ncnn.param+.bin 的資料夾\n② 點縮圖選圖 → 診斷（多選）／效能比較（單選）"
            else "資料夾：${t.name}（點縮圖選圖 → 診斷／效能比較）"
        // 開機 Toast 標 build 版本：手動安裝後一眼確認裝對版本（沒看到＝還是舊 APK / 同步未完成）
        Toast.makeText(this, "Yakuyomi sandbox $BUILD_TAG", Toast.LENGTH_LONG).show()
        dumpLastExit(t)
    }

    /**
     * 上次行程死亡若是原生 crash／被信號殺／ANR／低記憶體，把系統保留的 ApplicationExitInfo（reason、desc、tombstone
     * backtrace）落成模型夾裡的 `<stamp>_exit.txt`——無 adb 時唯一能拿到 SIGSEGV/abort 堆疊的路（照 fork 的
     * NativeCrashReporter）。同一次死亡只寫一次（prefs 記時間戳）。
     */
    /** 翻譯用的 LLM key：只認 app 內輸入的（prefs）。APK 不再內建任何 key（防外流）；空＝引擎不翻譯（只跑偵測/OCR/去字）。 */
    private fun apiKey(): String = prefs.getString(PREF_API_KEY, null)?.trim().orEmpty()

    /** key 狀態＋尾四碼（log/Toast 對帳用，不印全文）。 */
    private fun keyLabel(): String {
        val k = apiKey()
        return if (k.isBlank()) "無（未輸入，不翻譯）" else "app 內存的 ****${k.takeLast(4)}"
    }

    private fun dumpLastExit(tree: DocumentFile?) {
        if (tree == null || android.os.Build.VERSION.SDK_INT < android.os.Build.VERSION_CODES.R) return
        runCatching {
            val am = getSystemService(android.app.ActivityManager::class.java) ?: return
            val exit = am.getHistoricalProcessExitReasons(packageName, 0, 5).firstOrNull {
                it.reason != android.app.ApplicationExitInfo.REASON_USER_REQUESTED &&
                    it.reason != android.app.ApplicationExitInfo.REASON_EXIT_SELF &&
                    it.reason != android.app.ApplicationExitInfo.REASON_OTHER
            } ?: return
            if (exit.timestamp <= prefs.getLong(PREF_LAST_EXIT, 0L)) return
            // ★ 原生 crash 的 trace 是 tombstone **protobuf 二進位**（非文字）：必須存 raw bytes，當文字 readText 會把非法
            //   UTF-8 換成 U+FFFD、rel_pc 等 varint 全爛（第一版就這樣毀了一份）。桌面用 NDK llvm-symbolizer + 未 strip .so 解。
            val raw = runCatching { exit.traceInputStream?.use { it.readBytes() } }.getOrNull()
            val text = buildString {
                appendLine("上次行程死亡（ApplicationExitInfo）：reason=${exit.reason} desc=${exit.description} status=${exit.status}")
                appendLine("time=${java.text.SimpleDateFormat("MMdd_HHmmss", java.util.Locale.US).format(java.util.Date(exit.timestamp))} pss=${exit.pss / 1024}MB rss=${exit.rss / 1024}MB build=$BUILD_TAG")
                appendLine(if (raw == null || raw.isEmpty()) "(系統未附 tombstone trace)" else "tombstone protobuf ${raw.size} bytes → 同名 .pb")
            }
            val base = "${stamp()}_exit"
            tree.createFile("text/plain", "$base.txt")?.uri?.let { uri ->
                contentResolver.openOutputStream(uri)?.use { it.write(text.toByteArray(Charsets.UTF_8)) }
            }
            if (raw != null && raw.isNotEmpty()) {
                tree.createFile("application/octet-stream", "$base.pb")?.uri?.let { uri ->
                    contentResolver.openOutputStream(uri)?.use { it.write(raw) }
                }
            }
            prefs.edit().putLong(PREF_LAST_EXIT, exit.timestamp).apply()
            Toast.makeText(this, "上次 crash 已寫 $base.txt/.pb", Toast.LENGTH_LONG).show()
        }.onFailure { Log.w(TAG, "dumpLastExit 失敗", it) }
    }

    // ===== 縮圖多選 =====
    private val selected = linkedSetOf<Int>() // 選取的 TEST_IMAGES 索引（保序）
    private val thumbViews = mutableListOf<View>()

    private fun buildThumbnails() {
        TEST_IMAGES.forEachIndexed { i, path ->
            val col = LinearLayout(this).apply {
                orientation = LinearLayout.VERTICAL
                setPadding(6, 6, 6, 6)
                layoutParams = LinearLayout.LayoutParams(
                    LinearLayout.LayoutParams.WRAP_CONTENT, LinearLayout.LayoutParams.WRAP_CONTENT,
                ).apply { marginEnd = 10 }
                setOnClickListener { toggleSelect(i) }
            }
            val iv = ImageView(this).apply {
                setImageBitmap(loadAssetThumbnail(path, 200))
                layoutParams = LinearLayout.LayoutParams(200, 280)
                scaleType = ImageView.ScaleType.FIT_CENTER
            }
            val num = TextView(this).apply {
                text = "${i + 1}"
                gravity = android.view.Gravity.CENTER
                layoutParams = LinearLayout.LayoutParams(200, LinearLayout.LayoutParams.WRAP_CONTENT)
            }
            col.addView(iv); col.addView(num)
            thumbViews.add(col)
            binding.thumbStrip.addView(col)
        }
        refreshThumbs()
    }

    private fun toggleSelect(i: Int) {
        if (i in selected) selected.remove(i) else selected.add(i)
        refreshThumbs(); updateButtons()
    }

    private fun refreshThumbs() = thumbViews.forEachIndexed { i, v ->
        v.setBackgroundColor(if (i in selected) Color.parseColor("#3F51B5") else Color.TRANSPARENT)
    }

    /** 診斷：≥1 張；效能比較：單張。執行中由 run 函式先 disable 兩鈕、finally 再 updateButtons 還原。 */
    private fun updateButtons() {
        binding.detectButton.isEnabled = selected.isNotEmpty()
        binding.inpaintCompareButton.isEnabled = selected.size == 1
        binding.repoDemoButton.isEnabled = selected.size == 1
        binding.crossPageButton.isEnabled = selected.size == 1
        binding.ocrAbButton.isEnabled = selected.isNotEmpty()
        binding.detectOcrCheckButton.isEnabled = true // 內建圖硬寫、不吃縮圖選取 ⇒ 恆可按
    }

    private fun loadAssetThumbnail(path: String, target: Int): Bitmap {
        val o1 = BitmapFactory.Options().apply { inJustDecodeBounds = true }
        assets.open(path).use { BitmapFactory.decodeStream(it, null, o1) }
        var s = 1
        while (o1.outWidth / s > target * 2) s *= 2
        val o2 = BitmapFactory.Options().apply { inSampleSize = s }
        return assets.open(path).use { BitmapFactory.decodeStream(it, null, o2)!! }
    }

    private fun runPipeline() {
        binding.detectButton.isEnabled = false
        binding.inpaintCompareButton.isEnabled = false
        val saveLog = binding.genLogSwitch.isChecked
        runSaveImg = binding.genImgSwitch.isChecked
        // 去字方法：整頁 AOT 固定 768（真機 A/B 定案）。boxfill 不用 tile。
        val (method, modeLabel) = when (binding.inpaintSpinner.selectedItemPosition) {
            0 -> "boxfill" to "快速去字"
            else -> "aot" to "AI 去字"
        }
        val tileSize = 768
        val sel = selected.toList() // 快照（鈕已 disable，執行中不變）
        val imgs = sel.map { TEST_IMAGES[it] }
        lifecycleScope.launch(Dispatchers.Default) {
            clearOutputs()
            logBuf.clear(); runImgIdx = 0; runTree = currentTree(); runStamp = stamp()
            // 方向鎖 AUTO、效能用引擎最優預設（OCR 並發/8、intraThreads 6、RenderConfig 預設 AUTO）→ 只設去字方法。
            val cfg = EngineConfig(inpainter = InpainterConfig(method = method, tileSize = tileSize))
            // 記憶體峰值取樣：背景每 150ms 抽「總 PSS（含 native）」與 native heap，記 max。
            // 量的是 runtime 真峰值（3 顆 NCNN Net + 推論工作記憶體 + bitmap），不是模型檔大小。
            // 要乾淨的「單頁峰值」就只選 1 張圖（多選會累積結果 bitmap 在 UI、把峰值墊高）。
            val memBasePss = android.os.Debug.getPss() // KB，載入模型前基線
            val memPeakPss = java.util.concurrent.atomic.AtomicLong(memBasePss)
            val memPeakNative = java.util.concurrent.atomic.AtomicLong(0)
            val memSampler = launch(Dispatchers.Default) {
                while (true) { // cancel 時下方 delay 丟 CancellationException 自動跳出
                    memPeakPss.updateAndGet { maxOf(it, android.os.Debug.getPss()) }
                    val natKb = android.os.Debug.getNativeHeapAllocatedSize() / 1024
                    memPeakNative.updateAndGet { maxOf(it, natKb) }
                    kotlinx.coroutines.delay(150)
                }
            }
            try {
                val tree = runTree
                if (tree == null) { log("✗ 請先按「選擇模型資料夾」"); return@launch }
                if (imgs.isEmpty()) { log("✗ 請先點縮圖選至少一張測試圖"); return@launch }
                // 定案：偵測 + OCR + 去字全走 NCNN（.param/.bin）。OCR 的 mixed param 由引擎 Ocr.pickParam 自己挑（見 ensureOcrNcnn）。
                val ocrNcnn = ensureOcrNcnn(tree)
                val detNcnn = ensureNcnnPair(tree, "dbnet")
                val aotNcnn = ensureNcnnPair(tree, "aot")
                if (ocrNcnn == null) { log("✗ 缺 NCNN OCR 模型（ocr_48px_ctc.ncnn.param/.bin）"); return@launch }
                if (detNcnn == null) { log("✗ 缺 NCNN 偵測模型（detector*.ncnn.param）"); return@launch }
                if (aotNcnn == null) { log("✗ 缺 NCNN 去字模型（*aot*.ncnn.param）"); return@launch }
                log("▶ 診斷 ${imgs.size} 張｜去字=$modeLabel")
                log("… 載入模型（首次複製到 filesDir 較久）")
                val alphabet = assets.open(ALPHABET).bufferedReader().use { it.readLines() }
                val tf = runCatching { Typeface.createFromAsset(assets, FONT) }.getOrNull()
                log("後端：偵測=NCNN｜去字=NCNN AOT｜OCR=NCNN（CPU 有 asimdhp 就載 mixed 精度）")
                val models = ModelSet(ocr = ocrNcnn, detectorNcnn = detNcnn, aotInpainterNcnn = aotNcnn)
                log("✓ 模型就緒，開跑")
                Yakuyomi.create(models, alphabet, apiKey(), cfg, tf).use { engine ->
                    var total = 0L
                    imgs.forEachIndexed { i, asset ->
                        val tag = "圖${sel[i] + 1}"
                        val page = loadAssetBitmap(asset)
                        when (val r = engine.translatePage(page)) { // §11：略過/失敗都保留原圖、不覆蓋
                            is PageResult.Translated -> {
                                val s = r.stats; total += s.wallMs
                                log("[$tag] 偵測${s.detectMs} OCR${s.ocrMs} 譯${s.translateMs} 去字${s.inpaintMs} 排版${s.renderMs}｜頁實際${s.wallMs}ms(階段和${s.totalMs}、去字‖翻譯重疊省${s.totalMs - s.wallMs})｜${s.lines}行${s.regions}區留${s.kept}")
                                addImage("$tag 成品（$modeLabel）", r.page)
                            }
                            is PageResult.Skipped -> {
                                val s = r.stats; total += s.totalMs
                                log("[$tag] 略過：${r.reason}｜偵測${s.detectMs} OCR${s.ocrMs}｜${s.lines}行${s.regions}區（保留原圖、不覆蓋）")
                                addImage("$tag 原圖（略過：${r.reason}）", page)
                            }
                            is PageResult.Failed -> {
                                log("[$tag] ✗ 失敗：${r.reason}（保留原圖、不覆蓋、可重試）")
                                addImage("$tag 原圖（失敗：${r.reason}）", page)
                            }
                        }
                    }
                    if (imgs.isNotEmpty()) log("★ ${imgs.size} 張總計 $total ms（去字=$modeLabel）平均 ${total / imgs.size} ms/張")
                }
            } catch (t: Throwable) {
                Log.e(TAG, "診斷失敗", t)
                log("✗✗ 例外：${t.javaClass.simpleName}: ${t.message}")
            } finally {
                memSampler.cancel()
                val peakMb = memPeakPss.get() / 1024.0
                val baseMb = memBasePss / 1024.0
                val natMb = memPeakNative.get() / 1024.0
                log(
                    "📊 記憶體峰值（去字=$modeLabel，%d張）：總PSS %.0fMB（載入前基線 %.0fMB、淨增 %.0fMB）· native heap 峰值 %.0fMB"
                        .format(imgs.size, peakMb, baseMb, peakMb - baseMb, natMb),
                )
                if (saveLog) {
                    val ok = writeLog()
                    withContext(Dispatchers.Main) {
                        binding.logText.append(
                            if (ok) "📁 已寫入：${runStamp}_log.txt（+$runImgIdx 圖）\n"
                            else "✗ 寫入失敗：請重按「選擇模型資料夾」重新授權（含寫入）\n",
                        )
                    }
                }
                withContext(Dispatchers.Main) { updateButtons() }
            }
        }
    }

    /** 把一串同尺寸的格子排成 cols 欄網格（最後一列不足補深灰空格）。用於去字全比較。 */
    private fun gridOf(cells: List<Bitmap>, cols: Int): Bitmap {
        val cw = cells.first().width
        val ch = cells.first().height
        val blank = Bitmap.createBitmap(cw, ch, Bitmap.Config.ARGB_8888).also { Canvas(it).drawColor(Color.rgb(24, 24, 24)) }
        val rows = ArrayList<Bitmap>()
        var i = 0
        while (i < cells.size) {
            val row = (0 until cols).map { cells.getOrNull(i + it) ?: blank }
            rows.add(mergeHorizontal(row))
            i += cols
        }
        return mergeVertical(rows)
    }

    /**
     * OCR 精度 A/B（NCNN fp32 真值 / NCNN fp16sa / NCNN mixed）：對選取測試圖跑 [Yakuyomi.ocrAbTest]，log 效能（各候選的
     * 載入＋recognize 耗時）+ 品質（逐行 OCR 讀取對照、只列候選間不同的行）。三個候選共用同一份 .bin，差別只在 param 與
     * fp16 開關：fp32＝全域 fp32（最穩，放第一欄當真值）；fp16sa＝全域 fp16 storage+arith（最快，但 transformer 零星讀錯）；
     * mixed＝backbone fp16 + transformer/char_pred fp32（parity/export_ocr_ncnn.py：per-layer featmask 31=7 + 手插 Cast 層），
     * 產品實際載的就是它。ORT 已整個從引擎拔掉、舊的 ORT fp32/int8 候選不再列，真機數據留在這裡當基準：
     * vivo V2429（SD 8 Gen 3）9 頁 242 行、真值＝ORT fp32：NCNN mixed 241/242（唯一不同那行其實是真值錯、mixed 對）、
     * NCNN fp32 242/242、ORT int8 223/242、NCNN 全 fp16 219/242；mixed 的 OCR 時間比 ORT int8 快 ~23%（1256 vs 1669 ms/頁）、
     * 載入 ~300–400ms；並發 8 行、num_threads=1 不進全域鎖。int8 常把小假名讀錯（なぃ／か6／だろぅ／やは自），mixed 讀對。
     * 選 demo06（第 013 頁）行數多、最能看差異。
     */
    private fun runOcrAb() {
        binding.ocrAbButton.isEnabled = false
        val sel = selected.toList()
        val imgs = sel.map { TEST_IMAGES[it] }
        lifecycleScope.launch(Dispatchers.Default) {
            clearOutputs()
            logBuf.clear(); runImgIdx = 0; runTree = currentTree(); runStamp = stamp()   // 結果落 <stamp>_log.txt + 成果圖（無 adb 靠 OneDrive 同步回來看）
            try {
                val tree = runTree
                if (tree == null) { log("✗ 請先按「選擇模型資料夾」"); return@launch }
                if (imgs.isEmpty()) { log("✗ 請先點縮圖選至少一張測試圖"); return@launch }
                // NCNN fp32（同一份 param、fp16 全關）＝標準答案放第一欄；其餘各欄印「與第一欄相同行數」
                val ocrNcnn = ensureNcnnPair(tree, "ocr_48px_ctc.ncnn")   // key 含 ".ncnn" 才不會誤抓 _mixed 那份
                // 混合精度 param（backbone fp16、transformer+head fp32，per-layer featmask）共用同一份 .bin：
                // 引擎 Ocr 由 param 名去掉 _mixed 推 .bin（兩份 param 共用 ocr_48px_ctc.ncnn.bin），mixed 只要複製 param 本身
                val ocrMixed = findFile(tree, "ocr_48px_ctc_mixed", ".param")?.let { ensureLocal(it) }?.takeIf { ocrNcnn != null }
                val detNcnn = ensureNcnnPair(tree, "dbnet")
                if (detNcnn == null) { log("✗ 缺 NCNN 偵測模型（detector*.ncnn.param）"); return@launch }
                val candidates = buildList {
                    if (ocrNcnn != null) add(OcrCandidate("NCNN fp32", ocrNcnn, OcrConfig(ncnnFp16Storage = false, ncnnFp16Arith = false, ncnnMixed = false)))   // 真值（第一欄）
                    if (ocrNcnn != null) add(OcrCandidate("NCNN fp16sa", ocrNcnn, OcrConfig(ncnnMixed = false)))   // 全 fp16：最快，但 transformer 零星讀錯（219/242）
                    if (ocrMixed != null) add(OcrCandidate("NCNN mixed", ocrMixed))  // backbone fp16 + transformer/head fp32（featmask + Cast）＝產品配方
                    // fp16 storage-only（arith 關）已測：慢 10×（每層來回 cast），不再列
                }
                if (candidates.isEmpty()) { log("✗ 缺 OCR 模型（ocr_48px_ctc.ncnn.param/.bin；mixed 另需 ocr_48px_ctc_mixed.ncnn.param）"); return@launch }
                val alphabet = assets.open(ALPHABET).bufferedReader().use { it.readLines() }
                log("▶ OCR 精度比較（${candidates.joinToString(" / ") { it.label }}）｜${imgs.size} 張｜CPU fp16=${Yakuyomi.ncnnCpuSupportsFp16()}")
                imgs.forEachIndexed { i, asset ->
                    val tag = "圖${sel[i] + 1}"
                    val page = loadAssetBitmap(asset)
                    val r = Yakuyomi.ocrAbTest(detNcnn, alphabet, page, candidates)
                    val base = r.recognizeMs.first()
                    val summary = r.labels.indices.joinToString("｜") { k ->
                        val delta = if (k > 0 && base > 0) "（${"%+.0f".format((r.recognizeMs[k] - base) / base * 100)}%）" else ""
                        val same = if (k > 0) " 同第一欄 ${r.rows.count { it[k] == it[0] }}/${r.rows.size}" else ""
                        "${r.labels[k]}[${r.backends[k]}] 載入 ${"%.0f".format(r.loadMs[k])}ms OCR ${"%.0f".format(r.recognizeMs[k])}ms$delta$same"
                    }
                    log("[$tag] 偵測 ${"%.0f".format(r.detectMs)}ms｜${r.rows.size} 行｜$summary")
                    var diff = 0
                    r.rows.forEachIndexed { j, row ->
                        if (row.any { it != row[0] }) {
                            diff++
                            row.forEachIndexed { k, text -> log("  L$j ${r.labels[k]}：${text.ifBlank { "∅" }}") }
                        }
                    }
                    log("[$tag] 候選間不同 $diff / ${r.rows.size} 行（相同的略）")
                    saveImage(tree, composeOcrAbSheet(asset.substringAfterLast('/'), page, r))
                    log("📁 成果圖 ${runStamp}_img${"%02d".format(runImgIdx)}.png")
                    writeLog()
                }
                log("★ OCR 精度比較完成")
                if (writeLog()) log("📁 已寫入：${runStamp}_log.txt")
            } catch (t: Throwable) {
                Log.e(TAG, "OCR A/B 失敗", t)
                log("✗✗ 例外：${t.javaClass.simpleName}: ${t.message}")
                log(Log.getStackTraceString(t)); writeLog()
            } finally {
                runOnUiThread { binding.ocrAbButton.isEnabled = selected.isNotEmpty() }
            }
        }
    }

    /**
     * DBNet 三 size 對照（demo06）：同一頁跑 DBNet @960 / @960+銳化 / @1024，各畫偵測框 + 計時，橫向併排肉眼比。
     * 驗桌面結論在真機重現：@960＝甜蜜點（涵蓋追上 @1024、OCR 讀對更多）、@1024 過度分割、銳化 marginal（真機 A/B）。
     * 需模型資料夾含 dbnet*.ncnn.param/.bin（DBNet 走 DetectorConfig.useDbnet 分支）。
     */
    /**
     * 偵測 + OCR 檢驗（固定圖測試）：用**產品當前設定**（[DetectorConfig] / [OcrConfig] 的預設值）跑內建測試圖，
     * 印每頁框數 / OCR 讀出塊數 / 讀出的文字 / 秒數 + 總計讀出率。**不吃縮圖選取**（內建圖硬寫）。
     * 用途＝改動引擎後的回歸檢驗：一眼看出當前 pipeline 的偵測涵蓋、OCR 讀對率、速度是否退步。
     */
    private fun runDetectOcrCheck() {
        binding.detectOcrCheckButton.isEnabled = false
        lifecycleScope.launch(Dispatchers.Default) {
            clearOutputs()
            logBuf.clear(); runImgIdx = 0; runTree = currentTree(); runStamp = stamp()
            runSaveImg = binding.genImgSwitch.isChecked
            // ★ crash 診斷：引擎 trace 即時寫進 OneDrive log 檔（native crash 前 flush → 看死在哪步、哪個 ncnn 呼叫沒回）。
            li.joye.yakuyomi.engine.EngineTrace.sink = { msg ->
                logBuf.append("[trace] ").append(msg).append('\n')
                writeLog()
            }
            try {
                val tree = runTree ?: run { log("✗ 請先按「選擇模型資料夾」"); return@launch }
                val dbnetPath = ensureNcnnPair(tree, "dbnet")
                    ?: run { log("✗ 缺 DBNet 模型（需 dbnet*.ncnn.param/.bin 放模型資料夾）"); writeLog(); return@launch }
                val ocrLocal = ensureOcrNcnn(tree)
                    ?: run { log("✗ 缺 NCNN OCR 模型（ocr_48px_ctc.ncnn.param/.bin）"); writeLog(); return@launch }
                val alphabet = assets.open(ALPHABET).bufferedReader().use { it.readLines() }
                // 內建測試圖＝章 34.1 的代表頁（006 有瘦框「その通りじゃ」/ 010 有手寫「商人」/ 011 稀疏 /
                // 013(demo06) 密集 / 014 中等 / 015 長對話）。改動引擎後跑這個當回歸檢驗。
                val imgs = listOf(
                    "test/ch34_006.jpg", "test/ch34_010.jpg", "test/ch34_011.jpg",
                    "test/demo06.jpg", "test/ch34_014.jpg", "test/ch34_015.jpg",
                )
                val pages = imgs.map { it.substringAfterLast('/').substringBeforeLast('.') to loadAssetBitmap(it) }
                val detCfg = li.joye.yakuyomi.engine.DetectorConfig()
                val ocrCfg = OcrConfig()
                log("▶ 偵測+OCR 檢驗（${pages.size} 內建圖）｜產品設定：DBNet @${detCfg.dbnetInputSize}、" +
                    "OCR stripPad=${ocrCfg.stripPad} bicubic=${ocrCfg.useBicubic} minProb=${ocrCfg.minProb}"); writeLog()
                val tOL0 = System.currentTimeMillis()
                val ocr = Ocr(ocrLocal, alphabet, ocrCfg)
                ocr.warmUp() // 暖機不計時（產品引擎常駐；第一頁才不會把冷啟算進 OCR）
                log("OCR 載入+暖機 ${"%.1f".format((System.currentTimeMillis() - tOL0) / 1000.0)}s［${ocr.backend}］（不計入下列每頁）"); writeLog()
                try {
                    Detector(dbnetPath, detCfg).use { det ->
                        det.detect(pages[0].second).textMask.recycle() // warm（丟，不計時）
                        var totBox = 0
                        var totRead = 0
                        var totMs = 0L
                        for ((name, page) in pages) {
                            val t0 = System.currentTimeMillis()
                            val d = det.detect(page)
                            val detMs = System.currentTimeMillis() - t0
                            val to0 = System.currentTimeMillis()
                            ocr.recognize(page, d.lines)
                            val ocrMs = System.currentTimeMillis() - to0
                            val texts = d.lines.mapNotNull { it.text.takeIf { t -> t.isNotBlank() } }
                            totBox += d.lines.size; totRead += texts.size; totMs += detMs + ocrMs
                            log("$name: ${d.lines.size}框 讀${texts.size}塊｜det ${"%.2f".format(detMs / 1000.0)}s " +
                                "ocr ${"%.2f".format(ocrMs / 1000.0)}s"); writeLog()
                            log("  ${texts.joinToString(" / ")}"); writeLog()
                            if (runSaveImg) {
                                addImage(
                                    "$name（${d.lines.size}框/${texts.size}讀）",
                                    page.copy(Bitmap.Config.ARGB_8888, true).also {
                                        drawLines(it, d.lines); labelBmp(it, name, "${d.lines.size}框${texts.size}讀", detMs)
                                    },
                                )
                            }
                            d.textMask.recycle()
                        }
                        val rate = if (totBox > 0) totRead * 100.0 / totBox else 0.0
                        log("★ 總計：$totBox 框、讀出 $totRead 塊（${"%.1f".format(rate)}%）｜" +
                            "偵測+OCR 共 ${"%.2f".format(totMs / 1000.0)}s"); writeLog()
                    }
                } finally {
                    ocr.close()
                    pages.forEach { it.second.recycle() }
                }
            } catch (t: Throwable) {
                Log.e(TAG, "DBNet size 對照失敗", t)
                log("✗✗ 例外：${t.javaClass.simpleName}: ${t.message}")
                log(Log.getStackTraceString(t)); writeLog()
            } finally {
                li.joye.yakuyomi.engine.EngineTrace.sink = null
                withContext(Dispatchers.Main) {
                    updateButtons(); binding.detectOcrCheckButton.isEnabled = true
                }
            }
        }
    }

    private fun runInpaintCompare() {
        binding.detectButton.isEnabled = false
        binding.inpaintCompareButton.isEnabled = false
        val imgPath = selected.firstOrNull()?.let { TEST_IMAGES[it] } // 單選（鈕已限定 size==1）
        lifecycleScope.launch(Dispatchers.Default) {
            clearOutputs()
            logBuf.clear(); runImgIdx = 0; runTree = currentTree(); runStamp = stamp()
            val saveLog = binding.genLogSwitch.isChecked // 效能比較也寫 log 檔（EP/各階段秒才進可讀檔）
            try {
                val tree = runTree
                if (tree == null) { log("✗ 請先按「選擇模型資料夾」"); return@launch }
                if (imgPath == null) { log("✗ 效能比較需選「單一」張測試圖"); return@launch }
                val ocrPath = ensureOcrNcnn(tree)
                val detPath = resolveDetectorPath(tree) // 純 NCNN
                if (detPath == null || ocrPath == null) {
                    log("✗ 模型不齊（需 dbnet + ocr_48px_ctc 的 .ncnn.param/.bin）"); return@launch
                }
                val orient = TextOrientation.AUTO // 鎖定
                log("LLM key：${keyLabel()}")
                log("▶ 去字全比較（原圖/偵測/遮罩/最佳成果 + 全去字法 boxfill·auto·AOT + 時間表）— $imgPath（需連網翻譯）")
                val page = loadAssetBitmap(imgPath)
                val alphabet = assets.open(ALPHABET).bufferedReader().use { it.readLines() }
                val tf = runCatching { Typeface.createFromAsset(assets, FONT) }.getOrNull()

                // 共用前段（3 模式共用、只跑一次）：偵測→OCR→分群→翻譯→過濾；逐階段計時供總表。
                // ★ 暖機後才計時：產品是常駐引擎，每頁成本＝暖機後的 forward；模型載入＋冷啟另外印（之前把 83MB OCR 載入
                //   和第一次 forward 都算進「辨識」，圖上 1.4 s 對不上 A/B 暖機後的 1.26 s）。
                val tDL0 = System.currentTimeMillis()
                val detector = Detector(detPath) // NCNN 偵測（.param→NCNN 後端）
                detector.warmUp()
                val tDetLoad = System.currentTimeMillis() - tDL0
                val tD0 = System.currentTimeMillis()
                val detection = detector.detect(page)
                val tDetect = System.currentTimeMillis() - tD0
                detector.close()
                val tOL0 = System.currentTimeMillis()
                val ocr = Ocr(ocrPath, alphabet, OcrConfig()) // 並發鎖最優預設(concurrent/8)
                ocr.warmUp()
                val tOcrLoad = System.currentTimeMillis() - tOL0
                val tO0 = System.currentTimeMillis()
                ocr.recognize(page, detection.lines)
                ocr.close()
                val regions = Grouping.group(detection.lines)
                val tOcr = System.currentTimeMillis() - tO0
                log("模型載入+暖機（不計入表）：偵測 ${"%.1f".format(tDetLoad / 1000.0)}s、OCR ${"%.1f".format(tOcrLoad / 1000.0)}s［${ocr.backend}］")
                val tT0 = System.currentTimeMillis()
                val translator = LlmTranslator(apiKey(), TranslatorConfig())
                val cht = translator.translate(regions.map { it.sourceText })
                regions.forEachIndexed { j, r -> r.translatedText = cht.getOrElse(j) { r.sourceText } }
                // 等效 engine TextFilter（internal 跨不過 module）：空白/純數字/譯==原 就丟（filterText 此處 null、略過 regex）
                val kept = regions.filter { r ->
                    val t = r.translatedText.trim()
                    t.isNotEmpty() && !t.all { it.isDigit() } && !r.sourceText.trim().equals(t, ignoreCase = true)
                }
                val tTranslate = System.currentTimeMillis() - tT0
                if (kept.isEmpty()) {
                    log("✗ 全數過濾（無有效譯文）：${translator.lastError}｜回應=${translator.lastRaw}")
                    return@launch
                }
                log("共用前段：偵測${"%.1f".format(tDetect / 1000.0)}s OCR${"%.1f".format(tOcr / 1000.0)}s 翻譯${"%.1f".format(tTranslate / 1000.0)}s｜${detection.lines.size}行 ${regions.size}區 留${kept.size}")

                // 兩門別比較（乾淨泡泡都平塗、差別在忙碌區）：BoxFill（忙碌也平塗）／AOT-整頁（平塗乾淨+AOT整頁忙碌·預設）。
                // 逐格已移除。去字基底＝NCNN AOT `.param`（boxfill 不跑它、auto_aot 整頁跑它；maskInp buildMask 也不跑模型）。
                val inpaintBase = resolveInpaintPath(tree) ?: run { log("✗ 缺去字模型（需 *aot*.ncnn.param）"); return@launch }
                // 去字比較：BoxFill vs AOT-整頁（固定 768·定案）。第三元素＝tileSize。
                val modes = listOf(
                    Triple("快速去字", "boxfill", 768),
                    Triple("AI 去字", "aot", 768),
                )
                val renderCfg = RenderConfig(orientation = orient)
                val pw = page.width; val ph = page.height
                fun pageCopy() = page.copy(Bitmap.Config.ARGB_8888, true)

                // 跑 3 去字 → ① 乾淨去字圖（最佳那張要拿來貼字）② 去字+路由框圖（第 2 排用）+ 各秒。
                // 框＝markRegions 讀 region.onArt/dbgStd，而 onArt/dbgStd 在「本輪 inpaint 期間」被設定、又於下一輪開頭被重置
                // → 必須在每輪 inpaint 之後、進下一輪之前，就地把「乾淨副本 + markRegions」做掉，否則只剩最後一輪的路由結果。
                val cleanByMode = ArrayList<Bitmap>()                  // 各模式乾淨去字圖（順序＝modes；最佳那張貼字用，不可帶框）
                val markedByMode = ArrayList<Bitmap>()                 // 各模式去字+路由框圖（第 2 排用）
                val timings = ArrayList<Triple<String, Long, Long>>() // name, 去字ms, 排版ms（排版只算最佳那個、其餘 0）
                // 各模式去字階段「總 PSS」峰值（KB），順序＝modes。涵蓋 Inpainter 建構（載 lama）→ inpaint→close。
                // 注意：此處 det/ocr 已 close → 只反映「單去字階段」峰值；全 app 三模型同時在的真峰值看 runPipeline。
                val peakByMode = ArrayList<Long>()
                for ((name, method, tile) in modes) {
                    regions.forEach { it.onArt = false; it.dbgStd = -1f } // 重置，避免上一輪 stale 顏色/std
                    val peakKb = java.util.concurrent.atomic.AtomicLong(android.os.Debug.getPss())
                    val sampler = launch(Dispatchers.Default) {
                        while (true) { // cancel 時 delay 丟 CancellationException 自動跳出
                            peakKb.updateAndGet { maxOf(it, android.os.Debug.getPss()) }
                            kotlinx.coroutines.delay(100)
                        }
                    }
                    val inp = Inpainter(inpaintBase, InpainterConfig(method = method, tileSize = tile))
                    val t0 = System.currentTimeMillis()
                    val cleaned = inp.inpaint(page, kept, detection.textMask)
                    val tInpaint = System.currentTimeMillis() - t0
                    inp.close()
                    sampler.cancel()
                    peakByMode.add(peakKb.get())
                    log("[$name] 去字${"%.1f".format(tInpaint / 1000.0)}s｜記憶體峰值 ${"%.0f".format(peakKb.get() / 1024.0)}MB")
                    cleanByMode.add(cleaned) // 乾淨（不畫框）：最佳那張貼字用
                    // 本輪路由（onArt/dbgStd）尚未被重置 → 立即在乾淨副本上畫辨識框（markRegions：綠=平塗／紅=onArt-重建 + std）。
                    val marked = cleaned.copy(Bitmap.Config.ARGB_8888, true)
                        .also { markRegions(it, kept); labelBmp(it, name, if (method == "aot") "768px" else "", tInpaint) }
                    markedByMode.add(marked)
                    timings.add(Triple(name, tInpaint, 0L)) // 排版時間下面只對最佳那個量、回填
                }

                // 「最佳成果」＝對 AI 去字 乾淨圖貼譯文。
                val bestIdx = modes.indexOfFirst { it.first == "AI 去字" }.takeIf { it >= 0 }
                    ?: modes.lastIndex
                val bestClean = cleanByMode[bestIdx]
                val tR0 = System.currentTimeMillis()
                val bestRendered = Renderer.render(bestClean, kept, renderCfg, tf)
                val tRender = System.currentTimeMillis() - tR0
                timings[bestIdx] = timings[bestIdx].copy(third = tRender) // 回填最佳排版時間到總表
                val tInpaintBest = timings[bestIdx].second
                log("[最佳·${modes[bestIdx].first}] 排版${"%.1f".format(tRender / 1000.0)}s 整張${"%.1f".format((tDetect + tOcr + tTranslate + tInpaintBest + tRender) / 1000.0)}s")

                // 去字遮罩（Auto-逐格 config）：白＝要擦像素 → 半透明紅疊在原圖上
                val maskInp = Inpainter(inpaintBase, InpainterConfig(method = "aot"))
                val removalMask = maskInp.buildMask(page, kept, detection.textMask)
                maskInp.close()

                // —— 網格（2 選項專用排版：6 圖格 2×3 無留白 + 資訊列橫向鋪滿在底）——
                // 序 row1：原圖・偵測行框・去字遮罩／row2：快速去字・AI去字（各含辨識框）・最佳成果／底：時間表+硬體
                val hw = hwInfoLines()
                hw.forEach { log(it) }
                val imgCells = listOf(
                    pageCopy().also { labelBmp(it, "原圖", "Raw", -1L) },
                    pageCopy().also { drawLines(it, detection.lines); labelBmp(it, "偵測行框", "Detected lines", -1L) },
                    overlayMask(page, removalMask, Color.RED).also { labelBmp(it, "去字遮罩", "Removal mask", -1L) },
                ) + markedByMode + listOf(
                    bestRendered.also { labelBmp(it, "最佳·${modes[bestIdx].first}", "Result", -1L) },
                )
                val imgGrid = gridOf(imgCells, 3) // 6 格 → 2×3、無空白格
                // 資訊列：時間表 + 硬體並排、合寬＝3 格（鋪滿 grid 寬、消除底部留白）
                val gridW = pw * 3
                val infoH = (ph * 0.62).toInt()
                val infoW1 = gridW / 2
                val infoStrip = mergeHorizontal(
                    listOf(
                        buildTimingTable(infoW1, infoH, tDetect, tOcr, tTranslate, timings, peakByMode),
                        buildHwBitmap(hw, gridW - infoW1, infoH),
                    ),
                )
                val big = mergeVertical(listOf(imgGrid, infoStrip))
                val cmpName = "${runStamp}_inpaintcmp.png"
                runCatching {
                    tree.findFile(cmpName)?.delete()
                    tree.createFile("image/png", cmpName)?.uri?.let { uri ->
                        contentResolver.openOutputStream(uri)?.use { big.compress(Bitmap.CompressFormat.PNG, 100, it) }
                    }
                }
                log("去字全比較完成（全去字法 boxfill/auto/AOT + 遮罩 + 最佳成果 + 時間表）→ 已存圖")
                addImage("去字全比較（boxfill/auto/AOT + 遮罩 + 最佳成果 + 時間表）", big)
            } catch (t: Throwable) {
                Log.e(TAG, "去背比較失敗", t)
                log("✗✗ 例外：${t.javaClass.simpleName}: ${t.message}")
            } finally {
                if (saveLog) {
                    val ok = writeLog()
                    withContext(Dispatchers.Main) {
                        binding.logText.append(
                            if (ok) "📁 已寫入：${runStamp}_log.txt\n"
                            else "✗ log 寫入失敗：請重按「選擇模型資料夾」重新授權（含寫入）\n",
                        )
                    }
                }
                withContext(Dispatchers.Main) { updateButtons() }
            }
        }
    }

    /**
     * 產生 repo demo（行銷展示圖）：對單張硬 case 頁，挑「背景最忙（字壓在畫面上）」的區，裁出乾淨 4 格特寫——
     * 原圖 / BoxFill（疊字翻譯那樣）/ Yakuyomi 去字 / Yakuyomi 翻譯成品。無 debug 框/表格，直接放 README。
     * Yakuyomi 那兩格用 Auto（整頁/逐格 由 spinner 選；BoxFill 選項退 Auto-整頁），BoxFill 永遠當對比。
     */
    private fun runRepoDemo() {
        binding.detectButton.isEnabled = false
        binding.inpaintCompareButton.isEnabled = false
        binding.repoDemoButton.isEnabled = false
        val imgPath = selected.firstOrNull()?.let { TEST_IMAGES[it] }
        // 展示圖固定用 AI 去字（AOT·NCNN·定案主去字）→ 不吃下拉、免每次記得選。BoxFill 永遠當對比。
        val method = "aot"; val modeLabel = "AI 去字"
        lifecycleScope.launch(Dispatchers.Default) {
            clearOutputs()
            logBuf.clear(); runImgIdx = 0; runTree = currentTree(); runStamp = stamp()
            try {
                val tree = runTree ?: run { log("✗ 請先按「選擇模型資料夾」"); return@launch }
                if (imgPath == null) { log("✗ repo demo 需選「單一」張測試圖"); return@launch }
                val ocrPath = ensureOcrNcnn(tree)
                val detPath = resolveDetectorPath(tree) // 純 NCNN
                val inpaintBase = resolveInpaintPath(tree) // NCNN AOT 優先
                if (detPath == null || ocrPath == null) {
                    log("✗ 模型不齊（需 dbnet + ocr_48px_ctc 的 .ncnn.param/.bin）"); return@launch
                }
                if (inpaintBase == null) {
                    log("✗ 缺去字模型（需 *aot*.ncnn.param）"); return@launch
                }
                log("▶ 產生 repo demo（硬區 4 格：原圖 / BoxFill / Yakuyomi去字 / Yakuyomi翻譯）— $imgPath（去字=$modeLabel，需連網翻譯）")
                val page = loadAssetBitmap(imgPath)
                val alphabet = assets.open(ALPHABET).bufferedReader().use { it.readLines() }
                val tf = runCatching { Typeface.createFromAsset(assets, FONT) }.getOrNull()

                // 前段：偵測 → OCR → 分群 → 翻譯 → 過濾（與「效能比較」同一套）
                val detector = Detector(detPath) // NCNN 偵測（.param→NCNN 後端）
                val detection = detector.detect(page); detector.close()
                val ocr = Ocr(ocrPath, alphabet, OcrConfig())
                ocr.recognize(page, detection.lines); ocr.close()
                val regions = Grouping.group(detection.lines)
                val translator = LlmTranslator(apiKey(), TranslatorConfig())
                val cht = translator.translate(regions.map { it.sourceText })
                regions.forEachIndexed { j, r -> r.translatedText = cht.getOrElse(j) { r.sourceText } }
                val kept = regions.filter { r ->
                    val t = r.translatedText.trim()
                    t.isNotEmpty() && !t.all { it.isDigit() } && !r.sourceText.trim().equals(t, ignoreCase = true)
                }
                if (kept.isEmpty()) { log("✗ 全數過濾（無有效譯文）：${translator.lastError}"); return@launch }

                // BoxFill 去字（對比用＝疊字翻譯的天花板）
                kept.forEach { it.onArt = false; it.dbgStd = -1f }
                val bf = Inpainter(inpaintBase, InpainterConfig(method = "boxfill"))
                val cleanBox = bf.inpaint(page, kept, detection.textMask); bf.close()
                // Yakuyomi 去字（Auto；dbgStd 在此被設定，用來挑硬區）
                kept.forEach { it.onArt = false; it.dbgStd = -1f }
                val au = Inpainter(inpaintBase, InpainterConfig(method = method))
                val cleanAuto = au.inpaint(page, kept, detection.textMask); au.close()
                // Yakuyomi 翻譯嵌字（在乾淨去字「副本」上貼字 → cleanAuto 本身保持乾淨給第 3 格）
                val translated = Renderer.render(
                    cleanAuto.copy(Bitmap.Config.ARGB_8888, true), kept, RenderConfig(orientation = TextOrientation.AUTO), tf,
                )

                // 挑硬區：背景最忙（dbgStd 最高）；無 std 則挑面積最大區
                val busiest = kept.filter { it.dbgStd >= 0f }.maxByOrNull { it.dbgStd }
                    ?: kept.maxByOrNull { (it.x1 - it.x0) * (it.y1 - it.y0) } ?: kept.first()
                val pad = ((busiest.y1 - busiest.y0) * 0.4f).coerceAtLeast(40f).toInt()
                val cx0 = (busiest.x0.toInt() - pad).coerceAtLeast(0)
                val cy0 = (busiest.y0.toInt() - pad).coerceAtLeast(0)
                val cx1 = (busiest.x1.toInt() + pad).coerceAtMost(page.width)
                val cy1 = (busiest.y1.toInt() + pad).coerceAtMost(page.height)
                fun crop(b: Bitmap) =
                    Bitmap.createBitmap(b, cx0, cy0, cx1 - cx0, cy1 - cy0).copy(Bitmap.Config.ARGB_8888, true)

                val cells = listOf(
                    crop(page).also { labelBmp(it, "1. Original", "", -1L) },
                    crop(cleanBox).also { labelBmp(it, "2. Box-fill", "", -1L) },
                    crop(cleanAuto).also { labelBmp(it, "3. Yakuyomi: erased", "", -1L) },
                    crop(translated).also { labelBmp(it, "4. Yakuyomi: translated", "", -1L) },
                )
                val demo = mergeHorizontal(cells)
                saveNamed(tree, "${runStamp}_repodemo.png", demo)
                saveNamed(tree, "${runStamp}_repodemo_fullpage.png", translated)
                log("✓ repo demo 完成（硬區 std=${"%.1f".format(busiest.dbgStd)}）→ 存 _repodemo.png（4 格）+ _repodemo_fullpage.png（全頁譯）")
                addImage("repo demo（硬區 4 格：原圖 / BoxFill / Yakuyomi 去字 / Yakuyomi 翻譯）", demo)
            } catch (t: Throwable) {
                Log.e(TAG, "repo demo 失敗", t)
                log("✗✗ 例外：${t.javaClass.simpleName}: ${t.message}")
            } finally {
                withContext(Dispatchers.Main) { updateButtons() }
            }
        }
    }

    /**
     * 夜讀上機測試：對選取的每一張圖跑幾套人物遮罩配方，把結果並排、分段耗時印在同一張大圖上。
     *
     * 跑的就是**產品用的同一條膠水**——引擎 [NightReadRenderer]（縮圖上限、偵測→人物分割→夜讀重繪、
     * 灰階／彩度／文字區換算全在引擎裡），這裡只剩「開哪套分割器、量時間、拼大圖」；測到的數字＝產品會看到的數字。
     * 配方由 [NightReadRenderer.charSegmenter] 開：
     *   · 「yolo NCNN fp16」＝單顆 yoloseg（(yolo, null)），看 mask 時間與畫面
     *   · 「yolo+cseg NCNN」＝定案的產品配方 yolo ∪ cseg（(yolo, cseg)），看總成本
     * 分割器開好先 [CharSegmenter.warmUp]（NCNN 冷啟在單緒做完），暖機計入「載入」時間、不混進逐頁數字。
     * 之前幾輪當基準的「yolo int8 ORT」（nightread-ort 的 CharMaskOrt）已退役；換 runtime 沒有精度代價，
     * 上一輪量過 cseg 單顆 ORT vs NCNN：mask 1.72 → 0.83 s、IoU 0.999。
     * 每張圖仍跑兩次取第二次：暖機只覆蓋分割器，偵測器與重繪第一次仍可能吃到冷啟，不是推論。
     * 超過 [NightReadRenderer.MAX_PIXELS] 的頁 helper 會先等比縮小再跑（輸出＝縮後尺寸），log 會標 scaled。
     */
    private fun runNightReadAb() {
        binding.nightReadAbButton.isEnabled = false
        val picked = selected.sorted().mapNotNull { TEST_IMAGES.getOrNull(it) }
        lifecycleScope.launch(Dispatchers.Default) {
            clearOutputs()
            try {
                val tree = currentTree() ?: run { log("✗ 請先按「選擇模型資料夾」"); return@launch }
                if (picked.isEmpty()) { log("✗ 請先在上方選至少一張圖"); return@launch }

                val detNcnn = resolveDetectorPath(tree)
                // 分割器只要 .param 路徑（helper 由同名推 .bin），但 .bin 要跟著進 filesDir → 走與偵測器同一條 ensureNcnnPair；
                // key 與引擎 ModelSet 的判法一致（檔名含 manga_seg／cartoonseg）
                val yoloParam = ensureNcnnPair(tree, "manga_seg")
                val csegParam = ensureNcnnPair(tree, "cartoonseg")

                log("模型：dbnet.param=${detNcnn != null} yolo.ncnn=${yoloParam != null} cseg.ncnn=${csegParam != null}")
                if (detNcnn == null) {
                    log("✗ 模型不齊：需要 dbnet 的 .param/.bin")
                    return@launch
                }
                // 配方＝(標籤, 偵測器路徑, 開分割器的工廠)；每套只在自己那輪開、用完關。分割器一律由引擎
                // NightReadRenderer.charSegmenter 開（單顆＝(yolo, null)、聯集＝(yolo, cseg)），聯集邏輯在引擎、這裡不再自己 OR。
                // yolo 有才排配方，所以工廠回 null 只可能是契約被改，用 checkNotNull 直接炸出來而非靜默略過。
                val recipes = buildList {
                    if (yoloParam != null) add(Recipe("yolo NCNN fp16", detNcnn) {
                        checkNotNull(NightReadRenderer.charSegmenter(yoloParam, null)) { "charSegmenter(yolo, null) 回 null" }
                    })
                    if (yoloParam != null && csegParam != null) add(Recipe("yolo+cseg NCNN", detNcnn) {
                        checkNotNull(NightReadRenderer.charSegmenter(yoloParam, csegParam)) { "charSegmenter(yolo, cseg) 回 null" }
                    })
                }
                if (recipes.isEmpty()) { log("✗ 沒有任何人物遮罩模型（需 manga_seg_s／cartoonseg 的 .ncnn.param/.bin）"); return@launch }
                val pages = picked.map { it.substringAfterLast('/') to loadAssetBitmap(it) }
                val results = LinkedHashMap<String, MutableList<Triple<String, Bitmap, LongArray>>>()

                for ((label, det, open) in recipes) {
                    log("▶ $label 載入模型…")
                    val tLoad = System.currentTimeMillis()
                    Detector(det).use { detector ->
                        open().use { masker ->
                            // 暖機算進載入：NCNN 第一次 extract 才真的建 pipeline，產品也是載完就 warmUp、逐頁數字不含它
                            masker.warmUp()
                            // 整合時 cseg 是逐章載入，載入成本要知道（Net 建立含權重讀取＋暖機）
                            log("  載入 ${System.currentTimeMillis() - tLoad}ms")
                            for ((name, bmp) in pages) {
                                repeat(2) { pass ->
                                    val (out, st) = nightReadOnce(bmp, detector, masker)
                                    if (pass == 1) {
                                        val t = longArrayOf(st.detectMs, st.maskMs, st.renderMs)
                                        results.getOrPut(label) { mutableListOf() }.add(Triple(name, out, t))
                                        val scaled = st.scaledTo?.let { (sw, sh) -> "  scaled ${sw}×$sh" } ?: ""
                                        log("  $name  detect ${t[0]}ms  mask ${t[1]}ms  render ${t[2]}ms  " +
                                            "total ${t.sum()}ms$scaled")
                                    } else {
                                        out.recycle()
                                    }
                                }
                            }
                        }
                    }
                }

                val sheet = composeNightReadSheet(pages, recipes.map { it.label }, results)
                val name = "nightread_${stamp()}.png"
                saveNamed(tree, name, sheet)
                addImage("夜讀", sheet)
                log("→ 已存 $name 到所選資料夾")
            } catch (e: Throwable) {
                log("✗ ${e.message ?: e.javaClass.simpleName}")
            } finally {
                withContext(Dispatchers.Main) { binding.nightReadAbButton.isEnabled = true }
            }
        }
    }

    /** 夜讀測試配方：偵測器（NCNN .param）＋ 開人物分割器的工廠（每套各自開關；分割器由引擎 [NightReadRenderer.charSegmenter] 給）。 */
    private data class Recipe(val label: String, val detector: String, val open: () -> CharSegmenter)

    /**
     * 跑一次夜讀＝呼叫產品同一條 [NightReadRenderer.render]（縮圖／偵測／分割／重繪／輸出 Bitmap 全在引擎），
     * 三段耗時與有無縮圖從 [NightReadStats] 回；sandbox 不再自己做灰階、彩度、文字區、遮罩換算——那些若跟產品不同步，
     * 這裡量到的就不是產品的數字。
     */
    private fun nightReadOnce(
        page: Bitmap,
        detector: Detector,
        masker: CharSegmenter,
    ): Pair<Bitmap, NightReadStats> {
        val stats = NightReadStats()
        val out = NightReadRenderer.render(page, detector, masker, stats = stats)
        return out to stats
    }

    /** 合成 A/B 大圖：頂部裝置與耗時，其下每列一張圖（原圖｜各配方）。 */
    private fun composeNightReadSheet(
        pages: List<Pair<String, Bitmap>>,
        labels: List<String>,
        results: Map<String, List<Triple<String, Bitmap, LongArray>>>,
    ): Bitmap {
        val colW = 520
        val gap = 10
        val headerH = 70 + 26 * (labels.size * pages.size + labels.size + 1)
        val cols = 1 + labels.size
        val rowH = pages.map { (_, b) -> (colW.toFloat() / b.width * b.height).toInt() + 24 }
        val sheet = Bitmap.createBitmap(
            cols * colW + (cols + 1) * gap,
            headerH + rowH.sum() + (rowH.size + 1) * gap,
            Bitmap.Config.ARGB_8888,
        )
        val c = Canvas(sheet)
        c.drawColor(Color.rgb(18, 18, 18))
        fun paint(size: Float, colour: Int) = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            textSize = size; color = colour; typeface = Typeface.MONOSPACE
        }
        var y = 34f
        c.drawText("Night reading", gap.toFloat(), y, paint(24f, Color.WHITE))
        y += 24
        c.drawText(
            "${android.os.Build.MANUFACTURER} ${android.os.Build.MODEL} · Android ${android.os.Build.VERSION.RELEASE}",
            gap.toFloat(), y, paint(16f, Color.rgb(140, 140, 140)),
        )
        y += 26
        for (label in labels) {
            c.drawText(label, gap.toFloat(), y, paint(18f, Color.rgb(210, 210, 210)))
            y += 22
            for ((name, _, t) in results[label].orEmpty()) {
                c.drawText(
                    "    ${name.padEnd(18)} detect ${t[0]}ms  mask ${t[1]}ms  render ${t[2]}ms  total ${t.sum()}ms",
                    gap.toFloat(), y, paint(16f, Color.rgb(190, 210, 235)),
                )
                y += 20
            }
        }
        var top = headerH.toFloat()
        pages.forEachIndexed { i, (name, src) ->
            val imgs = listOf(src) + labels.map { l -> results[l].orEmpty().first { it.first == name }.second }
            val names = listOf("original") + labels
            names.forEachIndexed { col, l ->
                val x = gap + col * (colW + gap)
                c.drawText("$l · $name", x + 2f, top + 16, paint(15f, Color.rgb(140, 140, 140)))
                c.drawBitmap(
                    imgs[col], null,
                    android.graphics.Rect(x, (top + 24).toInt(), x + colW, (top + rowH[i]).toInt()),
                    Paint(Paint.FILTER_BITMAP_FLAG),
                )
            }
            top += rowH[i] + gap
        }
        return sheet
    }

    /**
     * OCR 後端 A/B 成果圖（給使用者看圖判「誰讀對」）：左＝頁面疊偵測框＋行號（候選間讀出不同的行畫紅）；
     * 右＝逐行列第一欄（真值）的讀法，不同的行貼原圖裁片＋各候選讀法（與真值同者綠、不同者紅）。
     */
    private fun composeOcrAbSheet(pageName: String, page: Bitmap, r: OcrAbResult): Bitmap {
        val leftW = 900
        val rightW = 1100
        val gap = 12
        val scale = leftW.toFloat() / page.width
        val leftH = (page.height * scale).toInt()
        fun paint(size: Float, colour: Int, mono: Boolean = true, bold: Boolean = false) = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            textSize = size; color = colour; typeface = if (mono) Typeface.MONOSPACE else Typeface.DEFAULT; isFakeBoldText = bold
        }
        val truth = 0
        val diffRows = r.rows.indices.filter { i -> r.rows[i].any { it != r.rows[i][0] } }.toSet()
        class Row(val i: Int, val crop: Bitmap?, val h: Int)
        val rows = r.rows.indices.map { i ->
            if (i !in diffRows) return@map Row(i, null, 30)
            val q = r.quads[i]
            val x0 = q.minOf { it.x }.toInt().coerceIn(0, page.width - 2)
            val y0 = q.minOf { it.y }.toInt().coerceIn(0, page.height - 2)
            val x1 = q.maxOf { it.x }.toInt().coerceIn(x0 + 1, page.width)
            val y1 = q.maxOf { it.y }.toInt().coerceIn(y0 + 1, page.height)
            val crop = Bitmap.createBitmap(page, x0, y0, x1 - x0, y1 - y0)
            val maxW = 560
            val maxH = if (crop.height > crop.width) 420 else 200
            val sc = minOf(maxW.toFloat() / crop.width, maxH.toFloat() / crop.height, 2.5f)
            val scaled = Bitmap.createScaledBitmap(crop, (crop.width * sc).toInt().coerceAtLeast(1), (crop.height * sc).toInt().coerceAtLeast(1), true)
            if (scaled !== crop) crop.recycle() // 同尺寸時 createScaledBitmap 回同一物件（memory：recycle 坑）
            Row(i, scaled, 30 + maxOf(scaled.height, 26 * r.labels.size) + 16)
        }
        val headerH = 40 + 22 * (r.labels.size + 1) + 20
        val rightH = rows.sumOf { it.h } + 20
        val sheet = Bitmap.createBitmap(leftW + rightW + 3 * gap, headerH + maxOf(leftH, rightH) + gap, Bitmap.Config.ARGB_8888)
        val c = Canvas(sheet)
        c.drawColor(Color.rgb(18, 18, 18))
        var y = 30f
        c.drawText(
            "OCR 後端比較 · $pageName · ${r.rows.size} 行 · 偵測 ${"%.0f".format(r.detectMs)}ms · 紅框＝候選讀法不同",
            gap.toFloat(), y, paint(22f, Color.WHITE, mono = false, bold = true),
        )
        y += 26
        r.labels.forEachIndexed { k, l ->
            val same = r.rows.count { it[k] == it[truth] }
            c.drawText(
                "${l.padEnd(12)}[${r.backends[k]}]  載入 ${"%.0f".format(r.loadMs[k])}ms  OCR ${"%.0f".format(r.recognizeMs[k])}ms  同真值 $same/${r.rows.size}",
                gap.toFloat(), y, paint(17f, if (k == truth) Color.WHITE else Color.rgb(190, 210, 235)),
            )
            y += 22
        }
        val top = headerH.toFloat()
        c.drawBitmap(page, null, android.graphics.RectF(gap.toFloat(), top, (gap + leftW).toFloat(), top + leftH), Paint(Paint.FILTER_BITMAP_FLAG))
        val boxOk = Paint().apply { color = Color.rgb(80, 200, 120); style = Paint.Style.STROKE; strokeWidth = 2f }
        val boxDiff = Paint().apply { color = Color.rgb(255, 80, 80); style = Paint.Style.STROKE; strokeWidth = 4f }
        val lblBg = Paint().apply { color = Color.argb(200, 0, 0, 0) }
        r.quads.forEachIndexed { i, q ->
            val path = android.graphics.Path()
            q.forEachIndexed { j, p ->
                val px = gap + p.x * scale
                val py = top + p.y * scale
                if (j == 0) path.moveTo(px, py) else path.lineTo(px, py)
            }
            path.close()
            c.drawPath(path, if (i in diffRows) boxDiff else boxOk)
            val lx = gap + q[0].x * scale
            val ly = top + q[0].y * scale
            c.drawRect(lx, ly - 18, lx + 34, ly + 2, lblBg)
            c.drawText("$i", lx + 2, ly - 2, paint(16f, if (i in diffRows) Color.rgb(255, 120, 120) else Color.rgb(160, 255, 180), bold = true))
        }
        val rx = (2 * gap + leftW).toFloat()
        var ry = top
        for (row in rows) {
            val i = row.i
            val head = "L${"%02d".format(i)}"
            if (row.crop == null) {
                c.drawText(head, rx, ry + 22, paint(17f, Color.rgb(160, 160, 160)))
                c.drawText(r.rows[i][truth].ifBlank { "∅" }, rx + 60, ry + 22, paint(20f, Color.rgb(220, 220, 220), mono = false))
            } else {
                c.drawRect(rx - 4, ry + 2, rx + rightW - 4, ry + row.h - 4, Paint().apply { color = Color.rgb(40, 28, 28) })
                c.drawText(head, rx, ry + 22, paint(17f, Color.rgb(255, 120, 120), bold = true))
                c.drawBitmap(row.crop, rx + 60, ry + 28, null)
                var ty = ry + 28 + 22
                val tx = rx + 60 + row.crop.width + 20
                r.labels.forEachIndexed { k, l ->
                    val t = r.rows[i][k]
                    val col = when {
                        k == truth -> Color.WHITE
                        t == r.rows[i][truth] -> Color.rgb(160, 255, 180)
                        else -> Color.rgb(255, 120, 120)
                    }
                    c.drawText(l.padEnd(12), tx, ty, paint(16f, Color.rgb(170, 170, 170)))
                    c.drawText(t.ifBlank { "∅" }, tx + 150, ty, paint(22f, col, mono = false))
                    ty += 26
                }
                row.crop.recycle()
            }
            ry += row.h
        }
        return sheet
    }

    /** 用指定檔名存 PNG 到資料夾（覆蓋同名）。 */
    private fun saveNamed(tree: DocumentFile, name: String, bmp: Bitmap) {
        runCatching {
            tree.findFile(name)?.delete()
            tree.createFile("image/png", name)?.uri?.let { uri ->
                contentResolver.openOutputStream(uri)?.use { bmp.compress(Bitmap.CompressFormat.PNG, 100, it) }
            }
        }
    }

    /**
     * 在 bmp 上對每個 region 畫彩框：onArt=true→RED(lama 重建)，false→GREEN(boxfill/平塗)。
     * auto 模式另把引擎實測背景 std/亮度標在框左上（dbgStd≥0 才有）＝調 autoStdThreshold 用，直接看引擎真值不靠桌面 parity。
     */
    private fun markRegions(bmp: Bitmap, regions: List<li.joye.yakuyomi.engine.TextRegion>) {
        val canvas = Canvas(bmp)
        val paint = Paint().apply {
            style = Paint.Style.STROKE
            strokeWidth = 5f
            isAntiAlias = true
        }
        val txt = Paint().apply {
            color = Color.WHITE; textSize = 26f; isAntiAlias = true; isFakeBoldText = true
        }
        val txtBg = Paint().apply { color = Color.argb(210, 0, 0, 0); style = Paint.Style.FILL }
        for (r in regions) {
            paint.color = if (r.onArt) Color.RED else Color.GREEN
            canvas.drawRect(r.x0, r.y0, r.x1, r.y1, paint)
            if (r.dbgStd >= 0f) {
                val label = "s%.1f w%.0f".format(r.dbgStd, r.dbgWhite)
                val tw = txt.measureText(label)
                val ty = r.y0 + 26f
                canvas.drawRect(r.x0, r.y0, r.x0 + tw + 8f, r.y0 + 32f, txtBg)
                canvas.drawText(label, r.x0 + 4f, ty, txt)
            }
        }
    }

    /**
     * 在 bmp 上把每條偵測行框（TextLine.quad＝旋轉四邊形 4 點）描成細青線，直接改傳入的 bmp。
     * 對照 markRegions 的「合併後氣泡框」：這裡是合併前的逐行框，看偵測/分群前的原始粒度。
     */
    private fun drawLines(bmp: Bitmap, lines: List<li.joye.yakuyomi.engine.TextLine>) {
        val canvas = Canvas(bmp)
        val paint = Paint().apply {
            style = Paint.Style.STROKE; strokeWidth = 3f; isAntiAlias = true; color = Color.CYAN
        }
        for (line in lines) {
            val q = line.quad
            if (q.size < 4) continue
            val path = android.graphics.Path().apply {
                moveTo(q[0].x, q[0].y)
                for (i in 1..3) lineTo(q[i].x, q[i].y)
                close()
            }
            canvas.drawPath(path, paint)
        }
    }

    /**
     * 回傳 raw 的副本，並在 mask 為白（>127）的每個像素以 color 半透明（~50%）疊色。視覺化去字遮罩覆蓋範圍。
     * mask 與 raw 同尺寸（去字遮罩本就以原圖尺寸生成）。
     */
    private fun overlayMask(raw: Bitmap, mask: Bitmap, color: Int): Bitmap {
        val w = raw.width
        val h = raw.height
        val out = raw.copy(Bitmap.Config.ARGB_8888, true)
        val src = IntArray(w * h); out.getPixels(src, 0, w, 0, 0, w, h)
        val mpx = IntArray(w * h); mask.getPixels(mpx, 0, w, 0, 0, w, h)
        val cr = (color shr 16) and 0xFF
        val cg = (color shr 8) and 0xFF
        val cb = color and 0xFF
        for (i in src.indices) {
            if ((mpx[i] and 0xFF) > 127) { // 遮罩像素＝白 → 與原像素 50/50 混色
                val p = src[i]
                val r = (((p shr 16) and 0xFF) + cr) / 2
                val g = (((p shr 8) and 0xFF) + cg) / 2
                val b = ((p and 0xFF) + cb) / 2
                src[i] = Color.rgb(r, g, b)
            }
        }
        out.setPixels(src, 0, w, 0, 0, w, h)
        return out
    }

    /** 把硬體/設定橫幅獨立畫成 w×h 的白底格（取代疊在圖上的 drawHwInfo）：mono 黑字、整塊垂直置中、字級自適應塞滿。 */
    private fun buildHwBitmap(lines: List<String>, w: Int, h: Int): Bitmap {
        val bmp = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(bmp)
        canvas.drawColor(Color.WHITE)
        val txt = Paint().apply {
            color = Color.BLACK; isAntiAlias = true; typeface = Typeface.MONOSPACE; isFakeBoldText = true
        }
        // 字級：以 100px 試量「最寬行」實際寬度（含中日全形字，比 length 準）→ 寬塞 84% / 高塞 92%(每行1.5倍距) 取小者，夾合理範圍。
        val probe = 100f
        txt.textSize = probe
        val widest = lines.maxOf { txt.measureText(it) }.coerceAtLeast(1f)
        val byW = probe * (w * 0.84f) / widest
        val byH = h * 0.92f / (lines.size * 1.5f)
        val ts = minOf(byW, byH).coerceIn(16f, 48f)
        txt.textSize = ts
        val lineH = ts * 1.5f
        val blockH = lineH * lines.size
        val pad = w * 0.06f
        var y = (h - blockH) / 2f + ts * 0.85f // 整塊垂直置中
        for (line in lines) {
            canvas.drawText(line, pad, y, txt)
            y += lineH
        }
        return bmp
    }

    /** 單行標籤（en 留空）：舊呼叫端沿用，委派雙行版。 */
    private fun labelBmp(bmp: Bitmap, name: String, ms: Long): Bitmap = labelBmp(bmp, name, "", ms)

    /**
     * 在 bmp 右上角疊印雙行標籤（黑底圓角矩形）：上＝中文 zh（ms≥0 接「 %.1fs」），下＝英文 en（小灰字）。
     * en 為空＝只畫一行。直接修改傳入的 bmp 並回傳。
     */
    private fun labelBmp(bmp: Bitmap, zh: String, en: String, ms: Long): Bitmap {
        val canvas = Canvas(bmp)
        val zhText = if (ms >= 0) "$zh %.1fs".format(ms / 1000.0) else zh
        val textSize = (bmp.width / 26f).coerceAtLeast(14f)
        val enSize = textSize * 0.72f
        val zhPaint = Paint().apply {
            color = Color.WHITE; this.textSize = textSize; isAntiAlias = true; typeface = Typeface.MONOSPACE
        }
        val enPaint = Paint().apply {
            color = Color.LTGRAY; this.textSize = enSize; isAntiAlias = true; typeface = Typeface.MONOSPACE
        }
        val hasEn = en.isNotEmpty()
        val pad = textSize * 0.4f
        val lineGap = if (hasEn) textSize * 0.2f else 0f
        val textW = maxOf(zhPaint.measureText(zhText), if (hasEn) enPaint.measureText(en) else 0f)
        val boxW = textW + pad * 2
        val boxH = pad * 2 + textSize + (if (hasEn) lineGap + enSize else 0f)
        val right = bmp.width.toFloat() - pad
        val top = pad
        val bgPaint = Paint().apply { color = Color.BLACK; alpha = 200; isAntiAlias = true }
        canvas.drawRoundRect(RectF(right - boxW, top, right, top + boxH), 8f, 8f, bgPaint)
        val left = right - boxW + pad
        canvas.drawText(zhText, left, top + pad + textSize * 0.85f, zhPaint)
        if (hasEn) canvas.drawText(en, left, top + pad + textSize + lineGap + enSize * 0.85f, enPaint)
        return bmp
    }

    /** 硬體/設定資訊 5 行（版本・裝置/SoC/核數/RAM・Android/ABI・效能參數・LLM）＝去背時間數據的對照背景；由 buildHwBitmap 畫成第 3 排右格。 */
    private fun hwInfoLines(): List<String> {
        val cores = Runtime.getRuntime().availableProcessors()
        val mi = android.app.ActivityManager.MemoryInfo()
        getSystemService(android.app.ActivityManager::class.java).getMemoryInfo(mi)
        val ramGB = mi.totalMem / (1024.0 * 1024 * 1024)
        val soc = if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.S) {
            "${android.os.Build.SOC_MANUFACTURER} ${android.os.Build.SOC_MODEL}"
        } else {
            android.os.Build.HARDWARE
        }
        val abi = android.os.Build.SUPPORTED_ABIS.firstOrNull() ?: "?"
        val tc = TranslatorConfig()
        return listOf(
            BUILD_TAG,
            "${android.os.Build.MANUFACTURER} ${android.os.Build.MODEL} · $soc · %d核 · %.1fGB".format(cores, ramGB),
            "Android ${android.os.Build.VERSION.RELEASE} (API ${android.os.Build.VERSION.SDK_INT}) · $abi",
            "效能：OCR並發 x%d · 偵測/去字/OCR 全 NCNN CPU（OCR mixed：backbone fp16＋transformer fp32）".format(OcrConfig().concurrency),
            "去字與翻譯並發重疊 → 整張＝牆鐘(非各段相加) · 各段皆暖機後計時（引擎常駐時的每頁成本）",
            "LLM：${tc.provider} · ${tc.model}",
        )
    }

    /** 把多張 Bitmap 橫向拼接（頂部對齊）。 */
    private fun mergeHorizontal(list: List<Bitmap>): Bitmap {
        val h = list.maxOf { it.height }
        val w = list.sumOf { it.width }
        val out = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(out)
        canvas.drawColor(Color.WHITE)
        var x = 0
        for (bmp in list) {
            canvas.drawBitmap(bmp, x.toFloat(), 0f, null)
            x += bmp.width
        }
        return out
    }

    /** 把多張 Bitmap 縱向堆疊（左邊對齊）。 */
    private fun mergeVertical(list: List<Bitmap>): Bitmap {
        val w = list.maxOf { it.width }
        val h = list.sumOf { it.height }
        val out = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(out)
        canvas.drawColor(Color.WHITE)
        var y = 0
        for (bmp in list) {
            canvas.drawBitmap(bmp, 0f, y.toFloat(), null)
            y += bmp.height
        }
        return out
    }

    /** 逐階段時間總表（縱＝偵測/OCR/翻譯/去字/排版/整張，橫＝各模式），畫成一張 w×h 格子表填左下空格。 */
    private fun buildTimingTable(
        w: Int, h: Int,
        tDetect: Long, tOcr: Long, tTranslate: Long,
        timings: List<Triple<String, Long, Long>>, // name, 去字ms, 排版ms
        peakKbByMode: List<Long>, // 各模式去字階段「總 PSS」峰值（KB），順序＝timings
    ): Bitmap {
        val bmp = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(bmp)
        canvas.drawColor(Color.WHITE)
        val stages = listOf(
            "偵測" to "Detect", "辨識" to "OCR", "翻譯" to "Translate",
            "去字" to "Inpaint", "整張" to "Wall total",
            "記憶體" to "Mem MB", // 去字階段總 PSS 峰值（非時間，故 fmt 不同）
        )
        val cols = 1 + timings.size
        val rowsN = 1 + stages.size
        val colW = w.toFloat() / cols
        val rowH = h.toFloat() / rowsN
        val ts = (rowH * 0.3f).coerceIn(20f, 40f)
        val txt = Paint().apply {
            color = Color.BLACK; textSize = ts; isAntiAlias = true; typeface = Typeface.MONOSPACE
        }
        val txtB = Paint().apply {
            color = Color.BLACK; textSize = ts; isAntiAlias = true
            typeface = Typeface.MONOSPACE; isFakeBoldText = true
        }
        val grid = Paint().apply { color = Color.LTGRAY; strokeWidth = 2f }
        for (c in 0..cols) canvas.drawLine(c * colW, 0f, c * colW, rowsN * rowH, grid)
        for (r in 0..rowsN) canvas.drawLine(0f, r * rowH, cols * colW, r * rowH, grid)
        fun put(col: Int, row: Int, s: String, bold: Boolean) {
            val p = if (bold) txtB else txt
            val x = col * colW + (colW - p.measureText(s)) / 2
            val y = row * rowH + rowH / 2 + ts * 0.35f
            canvas.drawText(s, x, y, p)
        }
        // 中英雙行（中文粗體在上、英文小灰字在下）＝項目名稱中英顯示，雙行才不擠出格
        val small = Paint(txt).apply { textSize = ts * 0.72f; color = Color.DKGRAY }
        fun putTwoLine(col: Int, row: Int, top: String, bottom: String) {
            val cx = col * colW + colW / 2
            val cyTop = row * rowH + rowH / 2 - ts * 0.1f
            canvas.drawText(top, cx - txtB.measureText(top) / 2, cyTop, txtB)
            canvas.drawText(bottom, cx - small.measureText(bottom) / 2, cyTop + ts * 0.95f, small)
        }
        fun fmt(ms: Long) = "%.1f".format(ms / 1000.0)
        putTwoLine(0, 0, "階段", "Stage")
        timings.forEachIndexed { i, t -> put(i + 1, 0, t.first, true) }
        stages.forEachIndexed { si, stage ->
            putTwoLine(0, si + 1, stage.first, stage.second)
            timings.forEachIndexed { i, t ->
                val v = when (si) {
                    0 -> fmt(tDetect)
                    1 -> fmt(tOcr)
                    2 -> fmt(tTranslate)
                    3 -> fmt(t.second)
                    // 整張＝牆鐘：去字(t.second)與翻譯併發重疊 ⇒ 取 max 而非相加（對齊 PageStats.wallMs、§8）。
                    // 排版(t.third)幾乎恆 0，已從顯示列拿掉，但仍計入牆鐘（不影響、~0）。
                    4 -> fmt(tDetect + tOcr + maxOf(tTranslate, t.second) + t.third)
                    else -> "%.0f".format((peakKbByMode.getOrNull(i) ?: 0L) / 1024.0) // 記憶體峰值 MB
                }
                put(i + 1, si + 1, v, si == 4) // 整張(牆鐘)那列粗體
            }
        }
        return bmp
    }

    private suspend fun log(msg: String) {
        logBuf.append(msg).append('\n')
        withContext(Dispatchers.Main) {
            binding.logText.append("$msg\n")
            binding.scroll.post { binding.scroll.fullScroll(View.FOCUS_DOWN) }
        }
    }

    private suspend fun addImage(label: String, bmp: Bitmap) {
        if (runSaveImg) runTree?.let { saveImage(it, bmp) }
        withContext(Dispatchers.Main) {
            val lbl = TextView(this@MainActivity).apply {
                text = label
                setTypeface(typeface, Typeface.BOLD)
                setPadding(0, 24, 0, 4)
            }
            val iv = ImageView(this@MainActivity).apply {
                layoutParams = LinearLayout.LayoutParams(
                    LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.WRAP_CONTENT,
                )
                adjustViewBounds = true
                setImageBitmap(scaledForView(bmp))
            }
            binding.outputContainer.addView(lbl)
            binding.outputContainer.addView(iv)
        }
    }

    private fun findFile(tree: DocumentFile, contains: String, suffix: String): DocumentFile? =
        tree.listFiles().firstOrNull { f ->
            val n = f.name?.lowercase() ?: return@firstOrNull false
            n.contains(contains) && n.endsWith(suffix)
        }


    private suspend fun clearOutputs() = withContext(Dispatchers.Main) {
        binding.logText.text = ""
        // 輸出區第 0 個＝logText，其餘＝上次結果圖/標籤 → 清掉只留 logText（控件區在另一容器，不受影響）
        while (binding.outputContainer.childCount > 1) binding.outputContainer.removeViewAt(1)
    }

    private fun stamp(): String =
        java.text.SimpleDateFormat("MMdd_HHmmss", java.util.Locale.US).format(java.util.Date())

    private fun saveImage(tree: DocumentFile, bmp: Bitmap) {
        runImgIdx++
        val name = "${runStamp}_img${"%02d".format(runImgIdx)}.png"
        runCatching {
            tree.findFile(name)?.delete()
            tree.createFile("image/png", name)?.uri?.let { uri ->
                contentResolver.openOutputStream(uri)?.use { bmp.compress(Bitmap.CompressFormat.PNG, 100, it) }
            }
        }
    }

    private fun writeLog(): Boolean = runCatching {
        val tree = runTree ?: return false
        val name = "${runStamp}_log.txt"
        tree.findFile(name)?.delete()
        val uri = tree.createFile("text/plain", name)?.uri ?: return false
        contentResolver.openOutputStream(uri)?.use { it.write(logBuf.toString().toByteArray(Charsets.UTF_8)) }
        true
    }.getOrDefault(false)

    private fun scaledForView(src: Bitmap, maxW: Int = 1000): Bitmap {
        if (src.width <= maxW) return src
        val h = (src.height.toLong() * maxW / src.width).toInt().coerceAtLeast(1)
        return Bitmap.createScaledBitmap(src, maxW, h, true)
    }

    /** SAF 模型串流複製到 filesDir（64KB 緩衝、不佔 heap），回傳路徑；已存在且同大小則跳過。 */
    private fun ensureLocal(doc: DocumentFile): String {
        val name = doc.name ?: "model.bin"
        val out = java.io.File(filesDir, name)
        if (out.exists() && out.length() == doc.length()) return out.absolutePath
        contentResolver.openInputStream(doc.uri)!!.use { input ->
            out.outputStream().use { o -> input.copyTo(o, 1 shl 16) }
        }
        return out.absolutePath
    }

    /**
     * 跨頁吞吐 benchmark：同一張圖、同一個 warm 引擎，併發度 D=1→5 各跑 D 頁（全併發 [async]+[awaitAll]）。
     * 一次驗兩件事：
     *  ① **併發安全**——同引擎被 D 條同時呼叫 [translatePage]（共用 detector/ocr/translator/inpainter session）會不會
     *     崩/亂/失敗。崩(app 死)或成功數<D ＝共用實例併發不安全 → 真做跨頁要每頁分 session 或加階段鎖。
     *  ② **吞吐**——每頁＝總時間/D 有沒有隨 D 下降（＝第 N 頁翻譯的網路等待被第 N+1 頁 CPU 填滿）。平的＝CPU 已飽和或 session 序列化。
     * 需連網（真 translate 才有網路等待可重疊）。
     */
    private fun runCrossPageBench() {
        binding.detectButton.isEnabled = false
        binding.inpaintCompareButton.isEnabled = false
        binding.repoDemoButton.isEnabled = false
        binding.crossPageButton.isEnabled = false
        val imgPath = selected.firstOrNull()?.let { TEST_IMAGES[it] }
        // 去字方法讀下拉選單（0=boxfill 快速去字 / 其餘=aot AI 去字）——測 boxfill vs aot 的跨頁差異要靠這個。
        val (method, methodLabel) = when (binding.inpaintSpinner.selectedItemPosition) {
            0 -> "boxfill" to "快速去字(boxfill)"
            else -> "aot" to "AI 去字(aot)"
        }
        lifecycleScope.launch(Dispatchers.Default) {
            clearOutputs()
            logBuf.clear(); runImgIdx = 0; runTree = currentTree(); runStamp = stamp()
            val saveLog = binding.genLogSwitch.isChecked
            try {
                val tree = runTree ?: run { log("✗ 請先按「選擇模型資料夾」"); return@launch }
                if (imgPath == null) { log("✗ 跨頁測試需選單張測試圖"); return@launch }
                val ocrNcnn = ensureOcrNcnn(tree) ?: run { log("✗ 缺 NCNN OCR 模型（ocr_48px_ctc.ncnn.param/.bin）"); return@launch }
                val detNcnn = ensureNcnnPair(tree, "dbnet")
                val aotNcnn = ensureNcnnPair(tree, "aot")
                if (detNcnn == null || aotNcnn == null) { log("✗ 缺 NCNN 偵測/去字 .param"); return@launch }
                val alphabet = assets.open(ALPHABET).bufferedReader().use { it.readLines() }
                val tf = runCatching { Typeface.createFromAsset(assets, FONT) }.getOrNull()
                val cfg = EngineConfig(inpainter = InpainterConfig(method = method, tileSize = 768))
                val models = ModelSet(ocr = ocrNcnn, detectorNcnn = detNcnn, aotInpainterNcnn = aotNcnn)
                val page = loadAssetBitmap(imgPath)
                log("LLM key：${keyLabel()}")
                log("▶ 跨頁吞吐測試（同一圖·併發 D=1→5·需連網翻譯·去字=$methodLabel）— $imgPath")
                Yakuyomi.create(models, alphabet, apiKey(), cfg, tf).use { engine ->
                    engine.translatePage(page) // 熱身（載模型/暖快取），不計時
                    log("  熱身完成，開始 D 掃描（每頁＝總時間/D；D 併發同時翻同一圖）")
                    for (d in 1..5) {
                        val t0 = System.currentTimeMillis()
                        val results = (1..d).map { async { engine.translatePage(page) } }.awaitAll()
                        val total = System.currentTimeMillis() - t0
                        val ok = results.count { it is PageResult.Translated }
                        val wall = results.filterIsInstance<PageResult.Translated>().joinToString(",") { "${it.stats.wallMs}" }
                        log("  併發 $d：總 ${"%.1f".format(total / 1000.0)}s｜每頁 ${"%.2f".format(total / 1000.0 / d)}s｜成功 $ok/$d｜各頁wall=[$wall]ms")
                    }
                    log("— 看點：① 每頁隨 D 下降＝跨頁重疊有效；平的＝CPU 飽和/session 序列化。② 崩或成功數<D＝共用引擎併發不安全→真做要每頁分 session 或加鎖。")
                }
            } catch (t: Throwable) {
                Log.e(TAG, "跨頁測試失敗", t)
                log("✗✗ 例外：${t.javaClass.simpleName}: ${t.message}（併發不安全或翻譯失敗）")
            } finally {
                if (saveLog) {
                    val ok = writeLog()
                    withContext(Dispatchers.Main) {
                        binding.logText.append(if (ok) "📁 已寫入：${runStamp}_log.txt\n" else "✗ log 寫入失敗\n")
                    }
                }
                withContext(Dispatchers.Main) { updateButtons() }
            }
        }
    }

    private fun ensureNcnnPair(tree: DocumentFile, key: String): String? {
        val param = findFile(tree, key, ".param") ?: return null
        val bin = findFile(tree, key, ".bin") ?: return null
        ensureLocal(bin) // .bin 必須與 .param 同在 filesDir（Detector/Inpainter 由 .param 推 .bin）
        return ensureLocal(param)
    }

    /**
     * OCR 模型路徑（產品配方）：主檔 `ocr_48px_ctc.ncnn.param`+`.bin`（key 故意含 ".ncnn" 才不會誤抓 `_mixed` 那份），並把同夾的
     * `ocr_48px_ctc_mixed.ncnn.param`（backbone fp16、transformer/char_pred fp32；共用同一份 .bin）一起放進 filesDir——
     * 引擎 [Ocr] 由主檔路徑推 mixed 路徑、CPU 有 asimdhp 才載，跟產品一樣；夾裡沒 mixed 就退回全域精度 param（全 fp16，
     * 小假名零星讀錯）。只給主檔而不帶 mixed 會讓「診斷」量到的不是產品精度，所以集中在這裡處理。回主檔 .param 的本機路徑；缺主檔回 null。
     */
    private fun ensureOcrNcnn(tree: DocumentFile): String? {
        val base = ensureNcnnPair(tree, "ocr_48px_ctc.ncnn") ?: return null
        findFile(tree, "ocr_48px_ctc_mixed", ".param")?.let { ensureLocal(it) }
        return base
    }

    /** 偵測模型路徑：NCNN `.param`（純 NCNN）；缺回 null。給直接建 Detector 的 dev 工具用。 */
    private fun resolveDetectorPath(tree: DocumentFile): String? = ensureNcnnPair(tree, "dbnet")

    /** 去字模型路徑：NCNN AOT `.param`（純 NCNN）；缺回 null。給直接建 Inpainter 的 dev 工具用。 */
    private fun resolveInpaintPath(tree: DocumentFile): String? = ensureNcnnPair(tree, "aot")

    private fun currentTree(): DocumentFile? {
        val s = prefs.getString(PREF_TREE, null) ?: return null
        return runCatching { DocumentFile.fromTreeUri(this, Uri.parse(s)) }.getOrNull()
    }

    private fun loadAssetBitmap(path: String): Bitmap =
        assets.open(path).use { BitmapFactory.decodeStream(it) }

    companion object {
        private const val TAG = "MainActivity"
        private const val BUILD_TAG = "v2.6-warm" // 改一次就 bump，手動安裝確認版本用（橫幅/Toast 只標這個）
        private const val PREF_LAST_EXIT = "last_exit_ts_v2" // v2＝raw .pb 版；換 key 讓上一版毀掉的那次 crash 重吐一次
        // NCNN 推論由引擎 NcnnBackend（libyakuyomi_ncnn）負責；sandbox 不再自帶 benchmark 用的 libncnn_jni。

        private const val PREF_TREE = "modelTree"
        private const val PREF_API_KEY = "llmApiKey" // app 內輸入的 LLM key（覆蓋 build 內建）
        // 去字兩門別：0 快速去字（BoxFill·就近取色平塗）/ 1 AI 去字（AOT-GAN 重建背景·整頁 768·預設）
        private val INPAINT_MODES = listOf(
            "快速去字（極速·低質）",
            "AI 去字（高品質·預設）",
        )
        // 測試圖（縮圖選單順序＝編號 1..N）
        private val TEST_IMAGES = listOf(
            "test/demo01.jpg", // 1
            "test/demo02.jpg", // 2
            "test/demo03.png", // 3
            "test/demo04.png", // 4
            "test/demo05.png", // 5
            // 6 = OCR 回歸測試素材（Chapter 34.1 p013）：句尾否定/反問被 bilinear 縮放糊掉→漏讀→意思相反
            //     （はないのかね→「居然有」、貴族でもなく→「就算是貴族」）＋多行密集小字。
            //     用途：改 OCR 前處理（如 bicubic warp）前後跑診斷對照、驗證「意思相反」是否救回。
            "test/demo06.jpg", // 6
            // 7–9 = 夜讀用的代表頁：有框黑白 / 多泡 / 無框彩頁
            "test/ch34_011.jpg", // 7
            "test/ch34_014.jpg", // 8
            // 10 = **已經貼好譯文的成品頁**：夜讀在產品上處理的一律是翻譯完成後的頁，
            //      譯文是純色黑字，墨度與邊緣都跟手寫原文不同，量測要帶著它。
            "test/translated.webp", // 9
        )
        private const val ALPHABET = "models/alphabet-all-v5.txt"
        private const val FONT = "fonts/NotoSansMonoCJK.ttc"
    }
}
