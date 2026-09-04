package li.joye.yakuyomi.engine

/**
 * LLM provider 預設表（多 provider + 自動撈模型清單）。
 *
 * 涵蓋 m-i-t 的 LLM translator（openai / deepseek / gemini / groq / custom_openai / sakura / qwen2 @ d5a3eee）
 * ＋ OpenRouter 便利預設。**全部走 OpenAI 相容「聊天」端點**（含 Gemini 的 OpenAI-compat 端點）
 * ⇒ [LlmTranslator] 一個 client 通吃、零新請求 builder。差別只在「列模型」端點（見 [ModelSource]）。
 *
 * 借鏡 nextai-translator 的 listModels：能 `GET /v1/models` 就自動撈、跟著官方更新（模型迭代快、不寫死清單）。
 */
enum class ModelSource {
    /** OpenAI 相容：`GET {modelsUrl}` 帶 Bearer → `data[].id`。涵蓋 deepseek/openai/groq/qwen/openrouter/sakura/custom。 */
    OPENAI,

    /** Google Gemini native：`GET {modelsUrl}?key=` → `models[]`（濾 generateContent）→ name 去掉 `models/` 前綴。 */
    GEMINI,

    /** 無清單端點：使用者自行輸入 model id。 */
    NONE,
}

/**
 * 一個 provider 預設。
 *
 * @param baseEditable 自架 / 自訂（sakura / custom）：由使用者填 base，[chatUrl]/[modelsUrl] 留空、由 base 推導
 *                     （見 [LlmProviders.chatUrlOf]/[LlmProviders.modelsUrlOf]）。
 */
data class LlmProvider(
    val id: String,
    val displayName: String,
    val chatUrl: String,
    val modelsUrl: String,
    val modelSource: ModelSource,
    val defaultModel: String,
    val baseEditable: Boolean = false,
)

/**
 * 一條 **request 參數相容規則**（資料表驅動：加 provider / 加特例 model 只改表、不動邏輯，見 [LlmProviders.PARAM_RULES]）。
 *
 * 為什麼需要：各家雖然都是「OpenAI 相容」，能吃的欄位其實不一致，**同一家不同世代也不同**，照單全送就是 400——
 *   · OpenAI 的 reasoning 模型（o 系列 / gpt-5 系列）**整組拒收** `temperature`/`top_p`/`max_tokens`…
 *   · 「思考開關」每家欄位形狀都不同：`thinking` / `reasoning_effort` / `enable_thinking` / `reasoning`
 *   · 同一個欄位的合法值還逐代不同（`reasoning_effort` 的 none / minimal）
 * 規則本身純資料 ⇒ [LlmProviders.requestParams] 是純函式、可單測（`LlmParamsTest`）。
 *
 * @param modelPattern  model id 比對（比對前小寫化、用 `containsMatchIn`，故要錨定請自己寫 `^`）；
 *                      null＝不比對＝該 provider 的 fallback 規則（放清單最後）。
 * @param temperature   送不送 `temperature`（OpenAI reasoning 模型拒收 → false）。
 * @param temperatureRange 合法範圍，超出就 clamp（OpenAI 相容主流是 0–2）。
 * @param maxTokensField 「最大輸出 token」欄位名（OpenAI reasoning 模型只認 `max_completion_tokens`）。
 *                      ★引擎目前不送這欄，先備著（見 [LlmProviders.maxTokensFieldOf]）。
 * @param thinkingOff   [TranslatorConfig.thinking]=false（預設）時要附加的欄位；空＝該家沒這概念或關不掉 → 不送。
 * @param thinkingOn    [TranslatorConfig.thinking]=true 時要附加的欄位；空＝用該家預設（多數家預設就會思考）。
 */
data class ParamRule(
    val modelPattern: Regex? = null,
    val temperature: Boolean = true,
    val temperatureRange: ClosedFloatingPointRange<Double> = 0.0..2.0,
    val maxTokensField: String = "max_tokens",
    val thinkingOff: Map<String, Any> = emptyMap(),
    val thinkingOn: Map<String, Any> = emptyMap(),
)

object LlmProviders {

    /** m-i-t 全部 LLM provider ＋ OpenRouter 便利預設。順序＝設定頁下拉順序。 */
    val ALL: List<LlmProvider> = listOf(
        LlmProvider(
            "deepseek", "DeepSeek",
            "https://api.deepseek.com/chat/completions",
            "https://api.deepseek.com/v1/models",
            // deepseek-chat 於 2026-07-24 15:59 UTC 退役（相容 shim 一併移除）→ 改用其對應的
            // deepseek-v4-flash（原 deepseek-chat＝此模型的非思考模式）。舊名稱的遷移見 [RETIRED_MODELS]。
            ModelSource.OPENAI, "deepseek-v4-flash",
        ),
        LlmProvider(
            "openai", "OpenAI",
            "https://api.openai.com/v1/chat/completions",
            "https://api.openai.com/v1/models",
            ModelSource.OPENAI, "gpt-4o-mini",
        ),
        LlmProvider(
            "gemini", "Google Gemini",
            // 走 Gemini 的 OpenAI 相容聊天端點 ⇒ 既有 LlmTranslator 直接通；列模型走 native（compat 路徑無 /models）。
            "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
            "https://generativelanguage.googleapis.com/v1beta/models",
            // 舊預設 gemini-2.0-flash 於 2026-06-01 **停役**（送出去 404）→ 換成官方點名的替代、且是 models 頁
            // 列為 stable 的現役泛用 flash。舊 id 的自動遷移見 [RETIRED_MODELS]。
            // https://ai.google.dev/gemini-api/docs/deprecations
            ModelSource.GEMINI, "gemini-3.6-flash",
        ),
        LlmProvider(
            "groq", "Groq",
            "https://api.groq.com/openai/v1/chat/completions",
            "https://api.groq.com/openai/v1/models",
            // 舊預設 llama-3.3-70b-versatile 於 2026-08-16 **停役**→ 換成 Groq 自己點名的替代、且在 models 頁
            // 列為 **production** 的 openai/gpt-oss-120b（另一個建議 qwen/qwen3.6-27b 只是 preview
            // 「evaluation only」，不當預設）。停役日已過 ⇒ 舊 id 的自動遷移見 [RETIRED_MODELS]（2026-08-25 收錄）。
            // https://console.groq.com/docs/deprecations ／ https://console.groq.com/docs/models
            ModelSource.OPENAI, "openai/gpt-oss-120b",
        ),
        LlmProvider(
            "qwen", "通義千問 Qwen",
            // **國際版端點**（dashscope-intl＝新加坡；2026-08-25 驗活、無 key 回 401 invalid_api_key）。
            // 原本用的 dashscope.aliyuncs.com 是**中國大陸**端點——本 app 受眾多在國際版 console
            // （alibabacloud.com）開 key，打大陸端點必 401。大陸 key 使用者請改走「自訂」provider 填
            // https://dashscope.aliyuncs.com/compatible-mode/v1。官方新推的 {WorkspaceId}.*.maas 專屬
            // 網域因含個人 workspace id 無法當 preset；官方明言舊網域 remains fully functional。
            "https://dashscope-intl.aliyuncs.com/compatible-mode/v1/chat/completions",
            "https://dashscope-intl.aliyuncs.com/compatible-mode/v1/models",
            ModelSource.OPENAI, "qwen-plus",
        ),
        LlmProvider(
            "openrouter", "OpenRouter",
            "https://openrouter.ai/api/v1/chat/completions",
            "https://openrouter.ai/api/v1/models",
            // OpenRouter 的 deepseek/deepseek-chat 仍存在（＝DeepSeek V3、OpenRouter 自己的命名空間、沒被退役），
            // 但既然對應的新世代 deepseek/deepseek-v4-flash 已在 OpenRouter 上架（2026-07 對 /api/v1/models 查證過），
            // 預設就跟 DeepSeek 官方那筆對齊。
            ModelSource.OPENAI, "deepseek/deepseek-v4-flash",
        ),
        // m-i-t sakura：自架 JA→ZH 專精 LLM（SAKURA_API_BASE，OpenAI 相容）。
        LlmProvider(
            "sakura", "Sakura（自架）",
            "", "", ModelSource.OPENAI, "sakura-14b-qwen2.5-v1.0",
            baseEditable = true,
        ),
        // m-i-t custom_openai：OpenRouter / LM Studio / SiliconFlow / 任何 OpenAI 相容端點的傘。
        LlmProvider(
            "custom", "自訂（OpenAI 相容）",
            "", "", ModelSource.OPENAI, "",
            baseEditable = true,
        ),
    )

    val DEFAULT: LlmProvider = ALL.first() // deepseek

    fun byId(id: String?): LlmProvider = ALL.firstOrNull { it.id == id } ?: DEFAULT

    /**
     * 已退役的 model id → 現行替代（**per-provider**，key＝[LlmProvider.id]）。
     *
     * 2026-07-24 15:59 UTC：DeepSeek 退役 `deepseek-chat` / `deepseek-reasoner` 兩個舊名稱、相容 shim
     * 一併移除 → 舊名稱送出去一律 HTTP 400（`Model Not Exist`）。key / base URL / 請求格式都沒變，**只有名稱要換**：
     * 兩者原本分別對應 `deepseek-v4-flash` 的非思考 / 思考模式，故都遷到 `deepseek-v4-flash`。
     *
     * 為什麼不是只改 [LlmProvider.defaultModel] 就好：使用者的 model 設定若**存著**舊名稱（曾手動輸入 /
     * 從舊的「抓取模型」清單挑過），預設值救不到他 → 在送出請求前就地換名（見 [migrateModel] 的呼叫端
     * `LlmTranslator.request`），使用者不必手動改設定。
     *
     * **只認 provider id**：custom / sakura / 自架端點上的同名模型不動（那是別人的命名空間，可能真的存在）。
     *
     * **只收「已停役＝送出去會報錯」的名稱，不收「deprecated 但還能用」的**——後者硬換掉等於偷偷改使用者
     * 選的模型（例：Groq 的 `llama-3.3-70b-versatile` 在 deprecated 期間官方寫明「Model remains fully
     * functional during this period」⇒ 當時只改預設、不遷移；2026-08-16 真的停役後才進表，見下）。
     */
    private val RETIRED_MODELS: Map<String, Map<String, String>> = mapOf(
        "deepseek" to mapOf(
            "deepseek-chat" to "deepseek-v4-flash",      // 舊＝非思考模式
            "deepseek-reasoner" to "deepseek-v4-flash",  // 舊＝思考模式（v4-flash 預設就是思考模式）
        ),
        // Gemini 2.0 系四個 id 於 **2026-06-01 停役**（已不可存取＝送出去報錯，非只是 deprecated）。
        // 替代照官方 deprecations 頁的建議、再對 models 頁挑「列為 stable」的現役 id：
        // 泛用 flash → gemini-3.6-flash、lite → gemini-3.5-flash-lite（維持原本的價位/延遲級距，不硬升級費率）。
        // https://ai.google.dev/gemini-api/docs/deprecations ／ https://ai.google.dev/gemini-api/docs/models
        "gemini" to mapOf(
            "gemini-2.0-flash" to "gemini-3.6-flash",
            "gemini-2.0-flash-001" to "gemini-3.6-flash",
            "gemini-2.0-flash-lite" to "gemini-3.5-flash-lite",
            "gemini-2.0-flash-lite-001" to "gemini-3.5-flash-lite",
        ),
        // ── 2026-08-25 五家全面稽核（DeepSeek/OpenAI/Gemini/Qwen/OpenRouter 官方 deprecation 頁逐一核過）──
        // 收表判準補充：**除了「已停役」，還要「使用者可能存著」**——本 app 2026-06-09 首發，凡停役日早於
        // 首發的 id（Gemini 全部 preview 系、Qwen 2026-01-30 前各批、OpenAI o1-mini/o1-preview 等）不可能
        // 出現在我們的「撈模型」清單裡 ⇒ 不收（手輸古 id 屬極端例外；mixtral 例外是因 m-i-t 文件教人輸它）。
        // 【到期看板：日期到了收下一波】
        //   2026-08-31  OpenRouter moonshotai/kimi-k2.5（官方無指定替代、到期後從 /models 消失即 400）
        //   2026-10-10  Qwen qwen3 大批（qwen3-32b/-coder-plus/-max-preview…→ 官方指 qwen3.6-flash/3.7-plus/3.7-max）
        //   2026-10-23  OpenAI 大批（gpt-4/gpt-4-turbo/gpt-3.5-turbo/o1/o3-mini/o4-mini/gpt-4.1-nano → gpt-5.6-sol/terra/luna）
        //   2026-12-11  OpenAI 快照批（gpt-5/-mini/-nano/-pro 2025-08-07 快照、o3/o3-pro → gpt-5.6 系）
        //   2027-05-07  Gemini gemini-3.1-flash-lite → gemini-3.5-flash-lite（我們的 lite 遷移目標已直指 3.5，無鏈風險）
        // 預設 model 全數現役：deepseek-v4-flash（2026-07-31 GA）/ gpt-4o-mini（不在任何停役名單）/
        // gemini-3.6-flash（stable）/ openai/gpt-oss-120b（production）/ qwen-plus（穩定別名）/
        // openrouter deepseek/deepseek-v4-flash（expiration_date:null）。
        // Groq 三波停役（2026-07-17 / 2026-08-16）全數到期後收錄（2026-08-25）。替代照官方 deprecations 頁
        // 建議、並維持原本的大小/價位級距（8b-instant → 20b、其餘 → 120b）。migrateModel 在 PARAM_RULES
        // 比對之前跑 ⇒ 遷移目標自動命中既有的 gpt-oss 規則（reasoning_effort=low），不用另加參數列。
        // mixtral-8x7b-32768 停役更早（m-i-t 也是 2026-08 才修它的預設，PR #1166）：使用者若照 m-i-t 文件
        // 手輸過就會存著它 → 一併遷移；級距對齊 m-i-t 的選擇（→ gpt-oss-20b）。
        // https://console.groq.com/docs/deprecations
        "groq" to mapOf(
            "llama-3.3-70b-versatile" to "openai/gpt-oss-120b",
            "llama-3.1-8b-instant" to "openai/gpt-oss-20b",
            "qwen/qwen3-32b" to "openai/gpt-oss-120b",
            "meta-llama/llama-4-scout-17b-16e-instruct" to "openai/gpt-oss-120b",
            "mixtral-8x7b-32768" to "openai/gpt-oss-20b",
        ),
    )

    /** 送出請求前的 model 名稱遷移：命中 [RETIRED_MODELS] 換成替代名，否則原樣回傳。 */
    fun migrateModel(providerId: String?, model: String): String =
        RETIRED_MODELS[providerId]?.get(model.trim()) ?: model

    // ───────────────────────── request 參數相容映射（per provider / per model） ─────────────────────────

    /** OpenAI reasoning 模型的「最大輸出 token」欄位名（Chat Completions 只認這個、送 max_tokens 會 400）。 */
    private const val MAX_COMPLETION = "max_completion_tokens"

    /**
     * provider → 規則清單。**由上而下第一個命中 [ParamRule.modelPattern] 者勝**、pattern=null 的 fallback 放最後；
     * **不在表內的 provider（custom / sakura / 未知）→ [DEFAULT_RULE]＝只送 temperature、不加任何欄位**
     * （自架端點的相容性未知，亂送未知欄位就是 400）。
     *
     * 加一家 provider / 一個特例 model ＝**只加一列資料**，[requestParams] 的邏輯不用動。
     */
    private val PARAM_RULES: Map<String, List<ParamRule>> = mapOf(
        // DeepSeek：思考開關＝**頂層物件** {"thinking":{"type":"disabled"}}（OpenAI SDK 的 extra_body ＝ body 頂層欄位）。
        // v4-flash / v4-pro 兩顆都支援雙模式、**預設思考開**（比舊 deepseek-chat 慢且貴）→ 我們預設關＝復刻舊行為。
        // thinking=true 不送欄位（該家預設就是思考）。思考模式下 temperature/top_p/presence/frequency
        // 「不支援但不報錯、只是無效」⇒ 照送無妨（關思考時才真的生效）。
        // https://api-docs.deepseek.com/guides/thinking_mode/
        "deepseek" to listOf(
            ParamRule(thinkingOff = mapOf("thinking" to mapOf("type" to "disabled"))),
        ),
        // OpenAI：★最大地雷＝**reasoning 模型整組拒收取樣參數**。官方（Azure 同一份 API 規格）明列
        // 「currently unsupported with reasoning models: temperature, top_p, presence_penalty, frequency_penalty,
        //   logprobs, top_logprobs, logit_bias, max_tokens」→ 送了 400，且長度上限要改叫 max_completion_tokens。
        // reasoning_effort 的可用值又**逐代不同**：none 只有 gpt-5.1 以後有／minimal 只有初代 gpt-5 系有
        //（gpt-5.1+ 拿掉、gpt-5-codex 也不支援）／o 系列只有 low|medium|high（o1-mini 根本沒這參數）／
        // gpt-5-pro 只吃 high（＝關不掉）。gpt-5*-chat 是**非** reasoning 的聊天模型（吃 temperature、
        // 送 reasoning_effort 反而 400）⇒ 放第一條先攔下來。
        // https://learn.microsoft.com/en-us/azure/foundry/openai/how-to/reasoning
        "openai" to listOf(
            ParamRule(Regex("^gpt-5.*chat")), // gpt-5-chat-latest / gpt-5.1-chat：非 reasoning，走一般規則
            ParamRule(Regex("^o1-mini"), temperature = false, maxTokensField = MAX_COMPLETION), // 無 reasoning_effort
            // o 系列（o1/o3/o4-mini/o3-pro/codex-mini）：關不掉，只能降到最低檔 low
            ParamRule(
                Regex("^(o\\d|codex-mini)"), temperature = false, maxTokensField = MAX_COMPLETION,
                thinkingOff = mapOf("reasoning_effort" to "low"),
            ),
            ParamRule(Regex("^gpt-5-pro"), temperature = false, maxTokensField = MAX_COMPLETION), // 只支援 high＝關不掉
            ParamRule( // gpt-5-codex：不支援 minimal → 降到 low
                Regex("^gpt-5-codex"), temperature = false, maxTokensField = MAX_COMPLETION,
                thinkingOff = mapOf("reasoning_effort" to "low"),
            ),
            ParamRule( // 初代 gpt-5 / -mini / -nano：最低檔＝minimal（沒有 none）
                Regex("^gpt-5(-mini|-nano)?$"), temperature = false, maxTokensField = MAX_COMPLETION,
                thinkingOff = mapOf("reasoning_effort" to "minimal"),
            ),
            ParamRule( // gpt-5.1 以後（5.1/5.2/5.4/5.5/5.6…）：none＝完全不思考
                Regex("^gpt-5\\."), temperature = false, maxTokensField = MAX_COMPLETION,
                thinkingOff = mapOf("reasoning_effort" to "none"),
            ),
            // 其餘沒列到的 gpt-5 變體：確定是 reasoning 模型（不送 temperature），但 effort 值沒把握 → 不送
            ParamRule(Regex("^gpt-5"), temperature = false, maxTokensField = MAX_COMPLETION),
            ParamRule(), // gpt-4o / gpt-4.1 / 其他＝非 reasoning，一般規則
        ),
        // Gemini（走 OpenAI 相容端點）：思考走 reasoning_effort，compat 層映射到 thinkingBudget(2.5 系)／
        // thinking_level(3.x)。**送不支援的值＝400 INVALID_ARGUMENT，不會被忽略也不會自動降級**
        // （官方相容層那張映射表只涵蓋 3.1-pro/3.1-flash-lite/3-flash/2.5，對表上沒列的新世代是直接透傳）。
        // 2026-09-04 全面查證後的可用值（官方 thinking 頁逐模型表）：
        //   3.8/3.7-flash        low|medium|high   ← minimal 明確報錯（使用者實測 3.8-flash 400）
        //   3.6/3.5-flash(-lite) minimal|low|medium|high
        //   3.1-flash-lite       low|high          ← minimal、medium 皆拒
        //   3-pro-preview        low|high          ← medium 拒
        //   3.1-pro-preview      low|medium|high
        //   2.5-flash(-lite)     none 可真的關思考
        //   2.5-pro              不能關（none/budget 0 → 400「Thinking can't be disabled」）
        // ⇒ **`low` 是唯一跨全部現役 chat 模型都不 400 的值**，除 2.5 非 Pro（能關就關、更省）之外一律送 low。
        // 不送會更糟：官方「無 reasoning_effort 時用模型預設」，而 3 系預設 On(medium/high) ⇒ 更慢更貴。
        // 3 系官方明說「Reasoning cannot be turned off for Gemini 3 models」，low 已是最省。
        // 影像/語音類（-image/-tts/-live）值域另有一套（如 3.1-flash-lite-image 只吃 minimal|high）⇒ 不送、
        // 用模型預設（那類本就不是我們的翻譯目標，只是可能出現在「抓取模型」清單被選到）。
        // https://ai.google.dev/gemini-api/docs/thinking ／ https://ai.google.dev/gemini-api/docs/openai
        // temperature：官方 changelog 2026-07-21 對「最新 Gemini 模型」標 deprecated，但 compat 層明文
        // silently ignore 不支援的參數 ⇒ 照送無妨（同 DeepSeek v4 的立場）；若日後開始報錯再加 temperature=false 規則。
        // https://ai.google.dev/gemini-api/docs/openai
        "gemini" to listOf(
            ParamRule(Regex("-(image|tts|live)")),                       // 非文字模型：值域另一套 → 不送
            ParamRule(Regex("^gemini-2\\.5-pro"), thinkingOff = mapOf("reasoning_effort" to "low")),
            ParamRule(Regex("^gemini-2\\.5"), thinkingOff = mapOf("reasoning_effort" to "none")),
            ParamRule(Regex("^gemini-[3-9]"), thinkingOff = mapOf("reasoning_effort" to "low")),
            ParamRule(),
        ),
        // Groq：reasoning_effort **只有部分模型吃**——Qwen 3.x 支援 none|default（真的能關）、GPT-OSS 只有
        // low|medium|high（關不掉、只能降到 low）；送給 llama 系會 400「reasoning_effort is not supported with
        // this model」⇒ llama 什麼都不送。**現行預設 openai/gpt-oss-120b 走 gpt-oss 那條**。
        // https://console.groq.com/docs/reasoning
        "groq" to listOf(
            ParamRule(Regex("qwen"), thinkingOff = mapOf("reasoning_effort" to "none")),
            ParamRule(Regex("gpt-oss"), thinkingOff = mapOf("reasoning_effort" to "low")),
            ParamRule(),
        ),
        // Qwen（DashScope compatible-mode）：思考開關＝頂層 enable_thinking。★本引擎一律 stream=false，
        // 而 DashScope 對「會思考的模型 + 非串流」直接 400
        //「parameter.enable_thinking must be set to false for non-streaming calls」
        // ⇒ **兩種狀態都送 false**（非串流下本來就拿不到思考，寧可保證跑得動）。
        // qwen-plus/max/flash/turbo 預設關、Qwen3.5+ 預設開，統一顯式關掉最穩。
        // https://www.alibabacloud.com/help/en/model-studio/deep-thinking
        // temperature：DashScope 官方明文「Range: [0, 2). Do not set to 0.」——兩端都不合法 ⇒ clamp 到 (0,2) 內。
        "qwen" to listOf(
            ParamRule(
                temperatureRange = 0.01..1.99,
                thinkingOff = mapOf("enable_thinking" to false),
                thinkingOn = mapOf("enable_thinking" to false),
            ),
        ),
        // OpenRouter：統一的 reasoning 物件，effort="none"＝「Disables reasoning entirely」。對不支援推理的模型
        // 也安全——OpenRouter 預設「providers that don't support all the LLM parameters ... will ignore unknown
        // parameters」（要改成排除該 provider 得自己開 require_parameters）。同理，`openai/o3` 這種轉手的
        // reasoning 模型即使收到 temperature 也由 OpenRouter 吸收掉 ⇒ 不必在這裡逐家重列一次規則。
        // https://openrouter.ai/docs/use-cases/reasoning-tokens ／ https://openrouter.ai/docs/features/provider-routing
        "openrouter" to listOf(
            ParamRule(thinkingOff = mapOf("reasoning" to mapOf("effort" to "none"))),
        ),
    )

    /** 表外 provider（custom / sakura / 未知）的保守預設：只送 temperature，其餘一律不送。 */
    private val DEFAULT_RULE = ParamRule()

    /** 解出這次請求該套哪條規則（純函式；model 大小寫不敏感）。 */
    fun ruleFor(providerId: String?, model: String): ParamRule {
        val m = model.trim().lowercase()
        val rules = PARAM_RULES[providerId] ?: return DEFAULT_RULE
        return rules.firstOrNull { it.modelPattern?.containsMatchIn(m) ?: true } ?: DEFAULT_RULE
    }

    /**
     * 這次請求要**附加**到 body 的參數（**不含** model / messages / stream——那三個由 [LlmTranslator] 固定組）。
     *
     * 純函式、無 IO ⇒ 可單測（見 `LlmParamsTest`）。回傳值型別限 String / Boolean / Number / Map（巢狀物件，
     * 如 DeepSeek 的 `thinking:{type:disabled}`），由呼叫端轉成 JSON。
     *
     * @param model      **已經過 [migrateModel]** 的名稱（規則按實際送出的 model 比對）。
     * @param thinking   [TranslatorConfig.thinking]：false（預設）＝送該家「關思考」欄位；true＝用該家預設。
     * @param temperature 會 clamp 到該家合法範圍；該 model 不吃 temperature（OpenAI reasoning 模型）就整個不送。
     */
    fun requestParams(
        providerId: String?,
        model: String,
        thinking: Boolean,
        temperature: Double,
    ): Map<String, Any> {
        val rule = ruleFor(providerId, model)
        val out = LinkedHashMap<String, Any>()
        if (rule.temperature) out["temperature"] = temperature.coerceIn(rule.temperatureRange)
        out.putAll(if (thinking) rule.thinkingOn else rule.thinkingOff)
        return out
    }

    /**
     * 「最大輸出 token」的欄位名（OpenAI reasoning 模型＝`max_completion_tokens`，其餘＝`max_tokens`）。
     * ★引擎目前不設輸出上限、**沒送**這個欄位；映射層先備著，日後要限長（或 fork 要用）直接查這裡。
     */
    fun maxTokensFieldOf(providerId: String?, model: String): String = ruleFor(providerId, model).maxTokensField

    /** 最終聊天端點（→ [TranslatorConfig.apiBase]）：baseEditable 用使用者 [base]、否則用預設 [LlmProvider.chatUrl]。 */
    fun chatUrlOf(p: LlmProvider, base: String): String =
        if (p.baseEditable) deriveChatUrl(base) else p.chatUrl

    /** 最終列模型端點：baseEditable 用使用者 [base]、否則用預設 [LlmProvider.modelsUrl]。 */
    fun modelsUrlOf(p: LlmProvider, base: String): String =
        if (p.baseEditable) deriveModelsUrl(base) else p.modelsUrl

    /** 使用者 base → 聊天端點。容忍填 origin / `.../v1` / 完整端點。 */
    private fun deriveChatUrl(base: String): String {
        val b = base.trim().trimEnd('/')
        return when {
            b.isEmpty() -> ""
            b.contains("/chat/completions") || b.endsWith("/completions") -> b
            b.endsWith("/v1") -> "$b/chat/completions"
            else -> "$b/v1/chat/completions"
        }
    }

    /** 使用者 base → 列模型端點（同 origin 的 `/v1/models`）。 */
    private fun deriveModelsUrl(base: String): String {
        var b = base.trim().trimEnd('/')
        if (b.isEmpty()) return ""
        if (b.contains("/chat/completions")) b = b.substringBefore("/chat/completions").trimEnd('/')
        return when {
            b.endsWith("/models") -> b
            b.endsWith("/v1") -> "$b/models"
            else -> "$b/v1/models"
        }
    }
}
