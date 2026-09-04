package li.joye.yakuyomi.engine

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * [LlmProviders.requestParams]：per-provider / per-model 的 request 參數相容映射。
 *
 * 這層存在的理由就是「照單全送會 400」，所以測的重點是**該省的有省掉**（OpenAI reasoning 模型不能帶
 * temperature）與**該送的形狀正確**（各家思考開關欄位名/值不同）。純函式、無 IO，跑在 JVM。
 */
class LlmParamsTest {

    private fun params(provider: String, model: String, thinking: Boolean = false, temp: Double = 0.3) =
        LlmProviders.requestParams(provider, model, thinking, temp)

    // ── DeepSeek：思考預設開 → 我們預設送 thinking:{type:disabled} ──

    @Test fun deepseekDisablesThinkingByDefault() {
        val p = params("deepseek", "deepseek-v4-flash")
        assertEquals(mapOf("type" to "disabled"), p["thinking"])
        assertEquals(0.3, p["temperature"] as Double, 1e-9) // 思考關掉後 temperature 才真的生效
    }

    @Test fun deepseekThinkingOnSendsNoOverride() {
        // 該家預設就是思考 → 開啟時不送任何 thinking 欄位（用預設）
        assertFalse(params("deepseek", "deepseek-v4-flash", thinking = true).containsKey("thinking"))
    }

    // ── OpenAI：reasoning 模型整組拒收 temperature（★這批要修的地雷） ──

    @Test fun openAiReasoningModelsDropTemperature() {
        for (m in listOf("o1", "o3", "o3-mini", "o4-mini", "o1-mini", "codex-mini", "gpt-5", "gpt-5.6", "gpt-5-pro")) {
            assertFalse("$m 不該帶 temperature", params("openai", m).containsKey("temperature"))
            assertEquals("$m 的長度上限欄位名", "max_completion_tokens", LlmProviders.maxTokensFieldOf("openai", m))
        }
    }

    @Test fun openAiChatModelsKeepTemperature() {
        for (m in listOf("gpt-4o-mini", "gpt-4.1", "gpt-5-chat-latest", "gpt-5.1-chat")) {
            assertEquals("$m 應保留 temperature", 0.3, params("openai", m)["temperature"] as Double, 1e-9)
            assertFalse("$m 不該帶 reasoning_effort", params("openai", m).containsKey("reasoning_effort"))
            assertEquals("max_tokens", LlmProviders.maxTokensFieldOf("openai", m))
        }
    }

    @Test fun openAiReasoningEffortIsPerGeneration() {
        assertEquals("none", params("openai", "gpt-5.1")["reasoning_effort"])     // 5.1+ 才有 none
        assertEquals("none", params("openai", "gpt-5.6")["reasoning_effort"])
        assertEquals("minimal", params("openai", "gpt-5")["reasoning_effort"])    // 初代只有 minimal
        assertEquals("minimal", params("openai", "gpt-5-nano")["reasoning_effort"])
        assertEquals("low", params("openai", "gpt-5-codex")["reasoning_effort"])  // codex 不支援 minimal
        assertEquals("low", params("openai", "o3")["reasoning_effort"])           // o 系列只有 low|medium|high
        assertFalse(params("openai", "o1-mini").containsKey("reasoning_effort"))  // 沒這參數
        assertFalse(params("openai", "gpt-5-pro").containsKey("reasoning_effort")) // 只吃 high＝關不掉
    }

    @Test fun openAiThinkingOnSendsNoEffort() {
        assertFalse(params("openai", "gpt-5.6", thinking = true).containsKey("reasoning_effort"))
    }

    // ── 其餘各家的思考開關形狀 ──

    @Test fun geminiEffortDependsOnGeneration() {
        // 逐模型的完整覆蓋見 geminiThinkingEffortIsSafeAcrossEveryLiveModel；這裡只守世代分界。
        assertEquals("none", params("gemini", "gemini-2.5-flash")["reasoning_effort"])   // none 只有 2.5 非 Pro 吃
        assertEquals("low", params("gemini", "gemini-3.6-flash")["reasoning_effort"])    // 3 系關不掉、low 最省且安全
        assertFalse(params("gemini", "gemini-2.0-flash").containsKey("reasoning_effort")) // 非思考模型 → 不送
    }

    @Test fun groqEffortDependsOnModelFamily() {
        assertEquals("none", params("groq", "qwen/qwen3.6-27b")["reasoning_effort"])       // qwen 真的能關
        assertEquals("low", params("groq", "openai/gpt-oss-120b")["reasoning_effort"])     // 只吃 low|medium|high
        assertFalse(params("groq", "llama-3.3-70b-versatile").containsKey("reasoning_effort")) // 送了會 400
    }

    @Test fun qwenAlwaysDisablesThinkingBecauseWeAreNonStreaming() {
        // DashScope：思考模型 + 非串流會 400「enable_thinking must be set to false for non-streaming calls」
        assertEquals(false, params("qwen", "qwen-plus")["enable_thinking"])
        assertEquals(false, params("qwen", "qwen-plus", thinking = true)["enable_thinking"])
    }

    @Test fun openRouterUsesReasoningObject() {
        assertEquals(mapOf("effort" to "none"), params("openrouter", "deepseek/deepseek-v4-flash")["reasoning"])
        assertFalse(params("openrouter", "deepseek/deepseek-v4-flash", thinking = true).containsKey("reasoning"))
    }

    // ── 未知端點：最保守（只送 temperature） ──

    @Test fun selfHostedAndUnknownSendNothingExtra() {
        for (p in listOf("custom", "sakura", "totally-unknown")) {
            for (thinking in listOf(false, true)) {
                val out = LlmProviders.requestParams(p, "whatever-model", thinking, 0.3)
                assertEquals("$p 只該有 temperature", setOf("temperature"), out.keys)
            }
        }
        assertEquals("max_tokens", LlmProviders.maxTokensFieldOf("custom", "whatever-model"))
    }

    // ── 其他 ──

    @Test fun temperatureIsClampedToProviderRange() {
        assertEquals(2.0, params("deepseek", "deepseek-v4-flash", temp = 9.9)["temperature"] as Double, 1e-9)
        assertEquals(0.0, params("deepseek", "deepseek-v4-flash", temp = -1.0)["temperature"] as Double, 1e-9)
    }

    @Test fun modelMatchIsCaseInsensitiveAndTrimmed() {
        assertFalse(params("openai", "  GPT-5.6  ").containsKey("temperature"))
        assertEquals("none", params("openai", "  GPT-5.6  ")["reasoning_effort"])
    }

    // ── 預設模型 × 退役名稱遷移（「開箱即 400」的防線） ──

    @Test fun noPresetShipsARetiredModelId() {
        // 每個 provider 的 defaultModel 都必須是現役名稱——若它自己會被 migrateModel 換掉，
        // 代表預設表沒跟上退役（DeepSeek 2026-07-24、Gemini 2.0 系 2026-06-01 都踩過）。
        for (p in LlmProviders.ALL) {
            assertEquals(
                "${p.id} 的預設模型是退役名稱",
                p.defaultModel,
                LlmProviders.migrateModel(p.id, p.defaultModel),
            )
        }
    }

    @Test fun retiredGeminiIdsMigrateToLiveOnes() {
        // 2026-06-01 停役（送出去會報錯）→ 遷移；lite 維持 lite 級距，不硬升費率
        assertEquals("gemini-3.6-flash", LlmProviders.migrateModel("gemini", "gemini-2.0-flash"))
        assertEquals("gemini-3.6-flash", LlmProviders.migrateModel("gemini", "gemini-2.0-flash-001"))
        assertEquals("gemini-3.5-flash-lite", LlmProviders.migrateModel("gemini", "gemini-2.0-flash-lite"))
        // 還在服役的不能亂動
        assertEquals("gemini-2.5-flash", LlmProviders.migrateModel("gemini", "gemini-2.5-flash"))
        // 別家的同名 model 不受影響（只認 provider id）
        assertEquals("gemini-2.0-flash", LlmProviders.migrateModel("custom", "gemini-2.0-flash"))
    }

    @Test fun retiredGroqIdsMigrateToLiveOnes() {
        // 2026-08-16 停役日過後收錄（deprecated 期間刻意不遷移＝不偷換使用者選的模型，見 RETIRED_MODELS 註解）。
        // 替代維持大小/價位級距：8b-instant → 20b、70b/32b/scout → 120b；mixtral 對齊 m-i-t PR #1166 → 20b。
        assertEquals("openai/gpt-oss-120b", LlmProviders.migrateModel("groq", "llama-3.3-70b-versatile"))
        assertEquals("openai/gpt-oss-20b", LlmProviders.migrateModel("groq", "llama-3.1-8b-instant"))
        assertEquals("openai/gpt-oss-120b", LlmProviders.migrateModel("groq", "qwen/qwen3-32b"))
        assertEquals("openai/gpt-oss-20b", LlmProviders.migrateModel("groq", "mixtral-8x7b-32768"))
        // 現役的不能亂動；別家（custom/自架）的同名 model 不受影響
        assertEquals("openai/gpt-oss-120b", LlmProviders.migrateModel("groq", "openai/gpt-oss-120b"))
        assertEquals("llama-3.3-70b-versatile", LlmProviders.migrateModel("custom", "llama-3.3-70b-versatile"))
    }

    @Test fun geminiThinkingEffortIsSafeAcrossEveryLiveModel() {
        // low ＝ 唯一跨全部現役 gemini chat 模型都不 400 的值（2026-09-04 查證，見 PARAM_RULES 註記）。
        // 3.8/3.7-flash 拒 minimal（使用者實測 400）、3.1-flash-lite 與 3-pro-preview 只吃 low|high。
        for (m in listOf(
            "gemini-3.8-flash", "gemini-3.7-flash", "gemini-3.6-flash", "gemini-3.5-flash",
            "gemini-3.5-flash-lite", "gemini-3.1-flash-lite", "gemini-3.1-pro-preview",
            "gemini-3-pro-preview", "gemini-3-flash-preview",
        )) {
            assertEquals("送給 $m 的 reasoning_effort 不安全", "low", params("gemini", m)["reasoning_effort"])
        }
        // 2.5 非 Pro＝唯一能真的關思考的一群（更省，維持 none）
        assertEquals("none", params("gemini", "gemini-2.5-flash")["reasoning_effort"])
        assertEquals("none", params("gemini", "gemini-2.5-flash-lite")["reasoning_effort"])
        // 2.5-pro 不能關（送 none → 400「Thinking can't be disabled for this model」）
        assertEquals("low", params("gemini", "gemini-2.5-pro")["reasoning_effort"])
        // 影像類值域另一套（3.1-flash-lite-image 只吃 minimal|high）→ 什麼都不送
        assertFalse(params("gemini", "gemini-3.1-flash-lite-image").containsKey("reasoning_effort"))
    }

    // ── 思考參數自癒（收到 400 就脫掉思考欄位重送）──

    @Test fun thinkingRejectionIsRecognisedAcrossProviders() {
        // 各家實測過的拒收訊息都要認得（Gemini 3.8-flash 實例＝2026-09-04 使用者回報）
        assertTrue(
            LlmTranslator.isThinkingParamRejection(
                """HTTP 400 {"error":{"code":400,"message":"Thinking level MINIMAL is not supported """ +
                    """for this model. Please retry with other thinking level.","status":"INVALID_ARGUMENT"}}""",
            ),
        )
        assertTrue(
            LlmTranslator.isThinkingParamRejection(
                """HTTP 400 {"error":{"message":"reasoning_effort is not supported with this model"}}""",
            ),
        )
        assertTrue(
            LlmTranslator.isThinkingParamRejection(
                """HTTP 400 {"error":{"message":"Unsupported parameter: 'reasoning_effort'"}}""",
            ),
        )
    }

    @Test fun unrelatedFailuresAreNotTreatedAsThinkingRejection() {
        // 脫思考參數救不了的錯不能誤判（否則白花一次請求、還蓋掉真正的錯誤訊息）
        assertFalse(LlmTranslator.isThinkingParamRejection("""HTTP 400 {"error":{"message":"Model Not Exist"}}"""))
        assertFalse(LlmTranslator.isThinkingParamRejection("""HTTP 401 {"error":{"message":"invalid api key"}}"""))
        assertFalse(LlmTranslator.isThinkingParamRejection("""HTTP 402 insufficient balance"""))
        // 非 400 的思考字眼（例如 429 訊息裡提到 reasoning）不重試
        assertFalse(LlmTranslator.isThinkingParamRejection("""HTTP 429 reasoning tokens rate limit"""))
    }

    @Test fun qwenTemperatureClampsInsideOpenInterval() {
        // DashScope 官方：「Range: [0, 2). Do not set to 0.」——0 與 2 都不合法，送了 400。
        assertEquals(0.01, params("qwen", "qwen-plus", temp = 0.0)["temperature"])
        assertEquals(1.99, params("qwen", "qwen-plus", temp = 2.0)["temperature"])
        assertEquals(0.3, params("qwen", "qwen-plus", temp = 0.3)["temperature"])
        // 其他家不受影響（OpenAI 相容主流 0–2 含兩端）
        assertEquals(0.0, params("deepseek", "deepseek-v4-flash", temp = 0.0)["temperature"])
    }

    @Test fun groqMigrationTargetsHitTheGptOssParamRule() {
        // migrateModel 在 PARAM_RULES 之前跑 ⇒ 遷移後的 id 要命中 gpt-oss 規則（thinkingOff=low），
        // 否則遷移救回了 model、卻送錯參數又 400。
        assertEquals("low", params("groq", LlmProviders.migrateModel("groq", "llama-3.3-70b-versatile"))["reasoning_effort"])
        assertEquals("low", params("groq", LlmProviders.migrateModel("groq", "llama-3.1-8b-instant"))["reasoning_effort"])
    }

    @Test fun defaultModelsHitTheIntendedRule() {
        // 換預設模型後，規則是否還命中對的那條（換 id 卻沒對到規則＝白改）
        fun defaultParams(id: String) = LlmProviders.byId(id).let { params(it.id, it.defaultModel) }
        assertEquals(mapOf("type" to "disabled"), defaultParams("deepseek")["thinking"])
        assertEquals("low", defaultParams("gemini")["reasoning_effort"])      // 3 系關不掉、low 是跨模型安全底
        assertEquals("low", defaultParams("groq")["reasoning_effort"])        // gpt-oss 只吃 low|medium|high
        assertEquals(mapOf("effort" to "none"), defaultParams("openrouter")["reasoning"])
        assertFalse(defaultParams("openai").containsKey("reasoning_effort"))  // gpt-4o-mini＝非推理模型
    }

    @Test fun ruleLookupFallsBackWhenProviderMissing() {
        // provider 為 null（設定未填）＝走保守預設，不因為 model 長得像 o3 就套 OpenAI 規則
        val out = LlmProviders.requestParams(null, "o3", false, 0.3)
        assertTrue(out.containsKey("temperature"))
        assertEquals(setOf("temperature"), out.keys)
    }
}
