package li.joye.yakuyomi.sandbox.probe

import android.content.Intent
import android.graphics.Color
import android.net.Uri
import android.os.Bundle
import android.view.Gravity
import android.view.WindowManager
import android.widget.Button
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.TextView
import android.widget.Toast
import androidx.activity.OnBackPressedCallback
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.FileProvider
import androidx.core.view.ViewCompat
import androidx.core.view.WindowInsetsCompat
import androidx.documentfile.provider.DocumentFile
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

/**
 * 「夜讀 C++ 探針」畫面。全程不關螢幕（FLAG_KEEP_SCREEN_ON）：這台手機在 app 不在最上層時只給 5 顆核心，切走的時間不能當數字，所以切走會被記成警告。
 * 紀錄一邊跑一邊寫進所選資料夾，也可以用「分享紀錄」傳出去。
 */
class NightProbeActivity : AppCompatActivity() {
    private val prefs by lazy { getSharedPreferences("yakuyomi", MODE_PRIVATE) }
    private lateinit var state: ProbeState
    private lateinit var folderText: TextView
    private lateinit var stepText: TextView
    private lateinit var logText: TextView
    private lateinit var scroll: ScrollView
    private lateinit var startButton: Button
    private lateinit var resumeButton: Button
    private lateinit var shareButton: Button

    private val folderPicker = registerForActivityResult(ActivityResultContracts.OpenDocumentTree()) { uri ->
        if (uri != null) {
            contentResolver.takePersistableUriPermission(
                uri,
                Intent.FLAG_GRANT_READ_URI_PERMISSION or Intent.FLAG_GRANT_WRITE_URI_PERMISSION,
            )
            prefs.edit().putString(PREF_TREE, uri.toString()).apply()
            refresh()
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        state = ProbeState(this)
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        val pad = (16 * resources.displayMetrics.density).toInt()
        val root = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(pad, pad, pad, pad)
        }
        fun btn(text: String, onClick: () -> Unit) = Button(this).apply {
            this.text = text
            setOnClickListener { onClick() }
            root.addView(this)
        }
        root.addView(
            TextView(this).apply {
                text = "夜讀 C++ 探針"
                textSize = 22f
            },
        )
        root.addView(
            TextView(this).apply {
                text = "跑之前：手機放涼、接上電源。跑的時候留在這個畫面、不要切走（約 15 分鐘）。"
                setPadding(0, pad / 2, 0, pad / 2)
            },
        )
        folderText = TextView(this).also { root.addView(it) }
        btn("選擇資料夾（放模型與紀錄）") { folderPicker.launch(null) }
        startButton = btn("開始") { start(resume = false) }
        resumeButton = btn("接著跑（跳過做完的與閃退的步驟）") { start(resume = true) }
        shareButton = btn("分享紀錄") { share() }
        stepText = TextView(this).apply {
            textSize = 16f
            setTextColor(Color.parseColor("#3F51B5"))
            setPadding(0, pad / 2, 0, pad / 2)
            gravity = Gravity.START
            root.addView(this)
        }
        logText = TextView(this).apply {
            typeface = android.graphics.Typeface.MONOSPACE
            textSize = 11f
        }
        scroll = ScrollView(this).apply {
            addView(logText)
            root.addView(this, LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT, 0, 1f))
        }
        setContentView(root)
        // targetSdk 36：內容畫到系統列底下，要自己讓開
        ViewCompat.setOnApplyWindowInsetsListener(root) { v, ins ->
            val b = ins.getInsets(WindowInsetsCompat.Type.systemBars())
            v.setPadding(pad + b.left, pad + b.top, pad + b.right, pad + b.bottom)
            ins
        }
        onBackPressedDispatcher.addCallback(
            this,
            object : OnBackPressedCallback(true) {
                override fun handleOnBackPressed() {
                    if (ProbeSession.running) {
                        Toast.makeText(this@NightProbeActivity, "探針還在跑，請留在這個畫面", Toast.LENGTH_LONG).show()
                    } else {
                        finish()
                    }
                }
            },
        )
        val recovered = ProbeRecovery.check(this, prefs.getString(PREF_TREE, null))
        stepText.text = recovered ?: if (!ProbeSession.running && state.resumable) {
            "上次沒跑完（${state.fileName}），可以按「接著跑」。"
        } else if (state.finished) {
            "上次跑完了：${state.fileName}。"
        } else {
            ""
        }
        refresh()
    }

    override fun onStart() {
        super.onStart()
        ProbeSession.activityVisible = true
    }

    override fun onStop() {
        super.onStop()
        ProbeSession.activityVisible = false
        if (ProbeSession.running) {
            // 不在主執行緒寫：紀錄的鎖可能正被「複製到所選資料夾」拿著。摘要從紀錄算次數，所以閃退後接著跑也不會歸零。
            ProbeSession.report?.let { r -> Thread { r.line("WARN", "what" to ProbeRecords.LEFT_FOREGROUND) }.start() }
        }
    }

    private fun tree(): DocumentFile? = prefs.getString(PREF_TREE, null)?.let { s ->
        runCatching { DocumentFile.fromTreeUri(this, Uri.parse(s)) }.getOrNull()
    }

    private fun refresh() {
        val t = tree()
        folderText.text = if (t == null) "還沒選資料夾" else "資料夾：${t.name}"
        val running = ProbeSession.running
        startButton.isEnabled = !running && t != null
        resumeButton.isEnabled = !running && t != null && state.resumable
        shareButton.isEnabled = state.fileName != null
    }

    /** 先看資料夾裡有沒有 4 顆模型（在別的執行緒：資料夾可能是雲端），缺的話直接說，別等 (a)(b)(c) 跑完才失敗。 */
    private fun start(resume: Boolean) {
        val t = tree() ?: return
        startButton.isEnabled = false
        resumeButton.isEnabled = false
        stepText.text = "檢查資料夾裡的模型…"
        Thread {
            val missing = runCatching { ModelProbe.missing(t) }.getOrElse { listOf("讀不了資料夾：${it.message}") }
            runOnUiThread {
                if (missing.isEmpty()) {
                    launch(t, resume)
                } else {
                    stepText.text = "所選資料夾（或它底下的 models）缺這些模型，請重選資料夾：\n${missing.joinToString("\n")}"
                    refresh()
                }
            }
        }.start()
    }

    private fun launch(t: DocumentFile, resume: Boolean) {
        val now = System.currentTimeMillis()
        if (!resume) state.reset(SimpleDateFormat("MMdd_HHmmss", Locale.US).format(Date(now)) + "_nightprobe.txt", now)
        val report = ProbeReport(applicationContext, t, state.fileName!!)
        ProbeSession.report = report
        ProbeSession.tree = t
        ProbeSession.running = true
        logText.text = ""
        // 逐步的細項不上畫面：探針量時間時別讓主執行緒一直排版
        report.onLine = { line ->
            if (UI_HIDDEN.none { line.startsWith(it) }) runOnUiThread { appendLog(line) }
        }
        refresh()
        val app = applicationContext
        // 普通工作執行緒：預設堆疊、預設優先權（別在主執行緒、也別開大堆疊，條件要跟之後產品呼叫 C++ 時一樣）
        Thread({
            try {
                NightProbeRunner(app, t, report, state, BUILD_TAG) { s -> runOnUiThread { stepText.text = s } }.run()
            } catch (e: Throwable) {
                report.line("FAIL", "step" to "runner", "error" to "${e.javaClass.name}: ${e.message}")
                report.publish()
                runOnUiThread { stepText.text = "探針本身出錯：${e.message}（已寫進紀錄）" }
            } finally {
                report.onLine = null
                report.close()
                ProbeSession.running = false
                ProbeSession.report = null
                runOnUiThread { refresh() }
            }
        }, "night-probe").start()
    }

    private fun appendLog(line: String) {
        if (logText.length() > 20_000) logText.text = logText.text.subSequence(logText.length() - 10_000, logText.length())
        logText.append(line.replace('\t', ' ') + "\n")
        scroll.post { scroll.fullScroll(ScrollView.FOCUS_DOWN) }
    }

    private fun share() {
        val name = state.fileName ?: return
        val dir = ProbeReport.dirOf(this)
        val files = dir.listFiles().orEmpty().filter { it.isFile && ProbeRecords.belongsToReport(it.name, name) }.sortedBy { it.name }
        if (files.isEmpty()) return
        val uris = ArrayList(files.map { FileProvider.getUriForFile(this, "$packageName.nightprobe", it) })
        val send = if (uris.size == 1) {
            Intent(Intent.ACTION_SEND).setType("text/plain").putExtra(Intent.EXTRA_STREAM, uris[0])
        } else {
            Intent(Intent.ACTION_SEND_MULTIPLE).setType("*/*").putParcelableArrayListExtra(Intent.EXTRA_STREAM, uris)
        }
        send.addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
        startActivity(Intent.createChooser(send, "分享夜讀探針紀錄"))
    }

    companion object {
        private const val PREF_TREE = "modelTree" // 跟 MainActivity 同一個資料夾
        const val BUILD_TAG = "v3.0-nightprobe"
        private val UI_HIDDEN = listOf("ENV\t", "PRIM\t", "BIN\t", "PAIR\t", "NUM\t", "LOAD\t", "EXPECT\t", "STEP_OK\t", "STEP\t")
    }
}
