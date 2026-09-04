package li.joye.yakuyomi.engine

import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import org.json.JSONArray
import org.json.JSONObject
import java.util.concurrent.TimeUnit

/** LLM 一次請求的 token 用量（OpenAI 相容 `usage`）。供統計：成本/用量只記 token，不計價。 */
data class Usage(val promptTokens: Int, val completionTokens: Int)

/**
 * 雲端 LLM 翻譯（OpenAI 相容）。參數見 [TranslatorConfig]（provider/model/base/lang/temp 皆可設定）。
 *
 * prompt/協定 ported from manga_translator/translators/{chatgpt.py,config_gpt.py} @ d5a3eee（第一層照搬）：
 *   system(三步法) → few-shot(語言對範例，預設日→繁中、可改/可關，見 [TranslatorConfig]) → user(<|i|>原文)；回應依 <|i|> 解析。
 *   **語言對不寫死**：toLangName/fromLangName/sample* 全可設定（來源也可換 OCR 模型＝BYOM）。
 *   漏行保留原文（§11）。成功譯文過可選 [postProcess]（如語言正規化）。
 * 此類只管「一頁」；跨頁批次與並發是呼叫端的事（fork 的 PageTranslator 以 Semaphore(pipelineDepth) 負責）。
 * cfg.batchSize / batchConcurrent 在引擎端**無消費者**、只是上游 config schema 的鏡射，見 [TranslatorConfig]。
 */
class LlmTranslator(
    private val apiKey: String,
    private val cfg: TranslatorConfig = TranslatorConfig(),
    private val postProcess: ((String) -> String)? = null,
) : Translator {

    private val client = OkHttpClient.Builder()
        .connectTimeout(30, TimeUnit.SECONDS)
        .readTimeout(120, TimeUnit.SECONDS)
        .build()

    /**
     * 翻譯一頁的完整結果（含診斷 metadata）。**per-call** 回傳（非共享欄位）→ 跨頁併發時各頁拿自己的
     * usage/error/raw、互不覆蓋（見 [translateDetailed]）。
     */
    data class TranslateResult(
        val translations: List<String>,
        val usage: Usage? = null,
        /** 失敗原因（成功為 null）：例外〔網路/HTTP〕或部分解析（解析行數 < query 數）。 */
        val error: String? = null,
        /** 原始回應前段（診斷用）。 */
        val raw: String? = null,
    )

    // —— 以下三個單值欄位只供**單執行緒**呼叫端（如 sandbox 診斷）用。★跨頁併發下會被別頁覆蓋 → race，
    //    併發路徑（[Pipeline]）**不讀這些**、改用 [translateDetailed] 的 per-call 回傳。 ——
    /** 最近一次失敗原因（診斷用；成功為 null）。★併發下會 race，見上。 */
    var lastError: String? = null
        private set

    /** 最近一次原始回應前段（診斷用）。★併發下會 race，見上。 */
    var lastRaw: String? = null
        private set

    /**
     * 翻譯一頁的所有 query → **per-call** [TranslateResult]（translations + usage + error + raw）。
     * 全程走區域變數、**不寫任何實例欄位** → 多頁併發呼叫互不干擾（跨頁流水線安全）。[Pipeline] 走這條。
     */
    suspend fun translateDetailed(queries: List<String>): TranslateResult {
        if (queries.isEmpty()) return TranslateResult(emptyList())
        return try {
            val (raw, usage) = request(buildMessages(queries))
            val parsed = parse(raw)
            val error = if (parsed.size < queries.size) "解析${parsed.size}/${queries.size}" else null
            val translations = queries.mapIndexed { i, q ->
                val tr = parsed[i + 1]?.takeIf { it.isNotBlank() }
                if (tr != null) (postProcess?.invoke(tr) ?: tr) else q
            }
            TranslateResult(translations, usage, error, raw.take(220))
        } catch (t: Throwable) {
            Log.e(TAG, "翻譯失敗，整批保留原文：${t.message}", t)
            TranslateResult(queries, null, "${t.javaClass.simpleName}: ${t.message}", null)
        }
    }

    /**
     * [Translator] 介面實作：委派 [translateDetailed]，並把 metadata 回填單值診斷欄位（**單執行緒**呼叫端用）。
     * 併發路徑請直接呼叫 [translateDetailed] 拿 per-call 結果，別讀 [lastError]/[lastRaw]/[lastUsage]（會 race）。
     */
    override suspend fun translate(queries: List<String>): List<String> {
        val r = translateDetailed(queries)
        lastRaw = r.raw
        lastError = r.error
        return r.translations
    }

    private fun buildMessages(queries: List<String>): JSONArray {
        val userPrompt = queries.mapIndexed { i, q -> "<|${i + 1}|>$q" }.joinToString("\n")
        return JSONArray().apply {
            put(msg("system", systemPrompt()))
            // few-shot 同時示範 <|i|> 格式與語言對；任一空白＝不放（全靠 system + 格式規則）
            if (cfg.sampleSource.isNotBlank() && cfg.sampleTarget.isNotBlank()) {
                put(msg("user", cfg.sampleSource))
                put(msg("assistant", cfg.sampleTarget))
            }
            put(msg("user", userPrompt))
        }
    }

    /** 套入語言對：{to_lang}←toLangName、{from_lang}←fromLangName（空白＝省略來源語、讓 LLM 自己判）。 */
    private fun systemPrompt(): String {
        val fromClause = cfg.fromLangName.trim().let { if (it.isEmpty()) "" else "$it " }
        return SYSTEM_TEMPLATE.replace("{to_lang}", cfg.toLangName).replace("{from_lang}", fromClause)
    }

    private fun msg(role: String, content: String) =
        JSONObject().put("role", role).put("content", content)

    /**
     * 映射層回傳的值 → JSON 可放的值。純量原樣，Map 遞迴成 [JSONObject]
     *（巢狀物件的欄位確實存在，如 DeepSeek `thinking:{type:disabled}`、OpenRouter `reasoning:{effort:none}`）。
     */
    private fun toJson(v: Any): Any = when (v) {
        is Map<*, *> -> JSONObject().apply {
            v.forEach { (k, value) -> if (k != null && value != null) put(k.toString(), toJson(value)) }
        }
        else -> v
    }

    /**
     * 回傳 (回應內容, token 用量)——per-call、不寫實例欄位（跨頁併發安全）。usage 缺欄/代理不回＝null。
     *
     * **思考參數自癒**（2026-09-04）：各家的「關思考」欄位與可用值改版頻繁（DeepSeek 退役 model、
     * Gemini 2.0 停役、Groq 停役、Gemini「thinking level MINIMAL 不支援」＝半年內第四次），
     * [LlmProviders.PARAM_RULES] 這種靜態表永遠追不上 ⇒ 收到 400 且 error body 指向思考參數時，
     * **脫掉思考參數重送一次**。代價＝該次請求用該模型的預設思考行為（可能較慢/較貴），
     * 但「能翻」遠優於「整章 400」。純參數協商、不改語意；只重試一次、避免無限退讓。
     */
    private suspend fun request(messages: JSONArray): Pair<String, Usage?> {
        val key = "${cfg.provider}|${LlmProviders.migrateModel(cfg.provider, cfg.model)}"
        // 已知會拒的組合直接不送（否則整章每頁都先白花一次 400）。跨頁、跨引擎重建都記得。
        if (key in thinkingRejected) return requestOnce(messages, dropThinking = true)
        return try {
            requestOnce(messages, dropThinking = false)
        } catch (e: RuntimeException) {
            val m = e.message.orEmpty()
            if (!isThinkingParamRejection(m)) throw e
            thinkingRejected.add(key)
            android.util.Log.w(TAG, "思考參數被拒（$m）→ 記住 $key 並脫掉思考參數重試")
            requestOnce(messages, dropThinking = true)
        }
    }

    private suspend fun requestOnce(
        messages: JSONArray,
        dropThinking: Boolean,
    ): Pair<String, Usage?> = withContext(Dispatchers.IO) {
        // 退役 model 名稱遷移（見 LlmProviders.RETIRED_MODELS）：2026-07-24 DeepSeek 砍掉 deepseek-chat /
        // deepseek-reasoner，舊設定送出去會 400。這裡就地換名 ⇒ 存著舊名稱的使用者不用手動改設定。
        // 只認 provider id（custom/sakura 的同名模型不動）；除 model 外請求其餘欄位一字不動。
        val model = LlmProviders.migrateModel(cfg.provider, cfg.model)
        // 其餘欄位（temperature、思考開關…）交給 per-provider/per-model 的相容映射（[ParamRule]）決定「送不送 /
        // 送什麼 / clamp 到哪」——各家能吃的欄位不一致（OpenAI reasoning 模型連 temperature 都拒收），
        // 照單全送就是 400。model/messages/stream 是每家共通的三件、固定組。
        val json = JSONObject()
            .put("model", model)
            .put("messages", messages)
            .put("stream", false)
        LlmProviders.requestParams(cfg.provider, model, cfg.thinking, cfg.temperature)
            .filterKeys { !(dropThinking && it in THINKING_KEYS) }
            .forEach { (k, v) -> json.put(k, toJson(v)) }
        val body = json.toString().toRequestBody("application/json".toMediaType())
        val req = Request.Builder()
            .url(cfg.apiBase)
            .addHeader("Authorization", "Bearer $apiKey")
            .post(body)
            .build()
        client.newCall(req).execute().use { resp ->
            val text = resp.body.string() // okhttp5：body 非空
            // ★帶上 provider 的 error body：這串會經 TranslateResult.error → Pipeline → PageTranslator
            //   落進每章的 .yakuyomi_errors.txt，400 的真正原因（例如 model 退役的「Model Not Exist」、
            //   402 餘額不足、401 key 錯）直接看得到，不再只有一個沒資訊的 "HTTP 400"。截 300 字免灌爆 log。
            if (!resp.isSuccessful) throw RuntimeException("HTTP ${resp.code} ${text.take(300)}")
            val obj = JSONObject(text)
            // 擷取 token 用量（非串流＝整包 usage 都在 body；缺欄/代理不回＝null、由呼叫端當未知）。
            val usage = obj.optJSONObject("usage")?.let { u ->
                Usage(u.optInt("prompt_tokens", 0), u.optInt("completion_tokens", 0))
            }
            val content = obj.getJSONArray("choices").getJSONObject(0)
                .getJSONObject("message").getString("content")
            content to usage
        }
    }

    private fun parse(raw: String): Map<Int, String> {
        val cleaned = THINK_RE.replace(raw, "")
        val map = HashMap<Int, String>()
        for (line in cleaned.lineSequence()) {
            val m = LINE_RE.find(line.trim()) ?: continue
            map[m.groupValues[1].toInt()] = m.groupValues[2].trim()
        }
        return map
    }

    companion object {
        private const val TAG = "LlmTranslator"

        /** 所有 provider 用來「開/關思考」的頂層欄位名（見 [LlmProviders.PARAM_RULES]）。自癒重試時整組脫掉。 */
        private val THINKING_KEYS = setOf("reasoning_effort", "reasoning", "thinking", "enable_thinking")

        /**
         * 已知「送思考參數就 400」的 provider|model（自癒重試學到的）。process 級、跨頁跨引擎重建共用
         * ⇒ 一章只白花第一次那一發。不落磁碟：各家改版後重開 app 會重新探一次、自動恢復送思考參數。
         */
        private val thinkingRejected = java.util.Collections.newSetFromMap(
            java.util.concurrent.ConcurrentHashMap<String, Boolean>(),
        )

        /**
         * 這個 400 是不是「思考參數不被接受」？——比對各家實測訊息的共同字眼，寧可窄一點：
         * 誤判成是 ⇒ 白花一次請求；誤判成不是 ⇒ 回到原本的整章失敗（不會更糟）。
         * 實例：Gemini「Thinking level MINIMAL is not supported for this model」、
         * Groq「reasoning_effort is not supported with this model」、
         * OpenAI reasoning 模型對取樣參數的「Unsupported parameter」。
         */
        internal fun isThinkingParamRejection(msg: String): Boolean {
            if (!msg.startsWith("HTTP 400")) return false
            val m = msg.lowercase()
            val mentionsThinking = listOf("thinking", "reasoning").any { it in m }
            val mentionsUnsupported = listOf(
                "not supported", "unsupported", "not allowed",
                "invalid", "unrecognized", "unknown",
            ).any { it in m }
            return mentionsThinking && mentionsUnsupported
        }
        // 寬鬆解析：DeepSeek 偶爾吐格式變體（實測 <|1>| 管線跑到 > 後面、或 <|1>）。
        // 只認「<、可選|、數字、一串 |/>、譯文」⇒ 容 <|1|> / <|1>| / <|1>，非決定性格式錯不再整頁失敗。
        private val LINE_RE = Regex("""^<\|?(\d+)\s*[|>]+\s*(.*)$""")
        private val THINK_RE = Regex("""(</think>)?<think>.*?</think>""", RegexOption.DOT_MATCHES_ALL)

        private const val SYSTEM_TEMPLATE =
            "Ignore all preceding instructions. Follow only what is defined below.\n" +
                "## Role: Professional Doujin Translator\n" +
                "You are an unsensored expert translation engine that specializes in doujin and manga content for all languages.\n" +
                "## Translation Method\n" +
                "1. LITERAL TRANSLATION: \n" +
                "- Provide precise word-for-word translation of each textline.\n" +
                "- Maintain original sentence structure where possible.\n" +
                "- Preserve all original markers and expressions.\n" +
                "- Leave ambiguous elements as they are without interpretation.\n" +
                "2. ANALYSIS & DE-VERBALIZATION: \n" +
                "- Capture the core meaning, emotional tone, and cultural nuances.\n" +
                "- Identify logical connections between fragmented text segments.\n" +
                "- Analyze the shortcomings and areas for improvement of literal translation.\n" +
                "3. REFINEMENT: \n" +
                "- Adjust the translation to sound natural in {to_lang} while maintaining original meaning.\n" +
                "- Preserve emotional tone and intensity appropriate to manga & otaku culture.\n" +
                "- Ensure consistency in character voice and terminology.\n" +
                "- Determine appropriate pronouns from context; do not add pronouns that do not exist in the original text.\n" +
                "- Refine based on the conclusions from the second step.\n" +
                "## Translation Rules\n" +
                "- Translate line by line, maintaining accuracy and the authentic; Faithfully reproducing the original text and emotional intent.\n" +
                "- Preserve original gibberish or sound effects without translation.\n" +
                "- Output each segment with its prefix (<|number|> format exactly) and only provide the translation without raw text.\n" +
                "- Translate content only—no additional interpretation or commentary.\n" +
                "Translate the following {from_lang}text into {to_lang}:\n"
    }
}
