package li.joye.yakuyomi.engine

/**
 * 引擎參數（CLAUDE.md §5 Config：第一層 schema + 預設值）。
 *
 * 參考使用者的 m-i-t config_deepseek.json，但**預設值對齊本專案實際使用的模型**：
 *   偵測 = DBNet（m-i-t default 偵測器，ResNet34+DB head）
 *   OCR  = 48px CTC（非 48px 自回歸）
 *   去字 = AOT-GAN（m-i-t inpainting.ckpt，NCNN·整頁 768；LaMa 已退役）
 * 因此部分數值與該 config 的 default/48px 不同，差異處以註解標出。
 *
 * 設定粒度：§11 v1 全域。標〔設定〕者＝預期在設定頁開放調整（頻繁調，如直/橫排、provider/key/語言）；
 * 未標者＝預設，留可控空間（仍在此結構內），需要時可快速加進設定 UI。
 */
data class EngineConfig(
    val detector: DetectorConfig = DetectorConfig(),
    val ocr: OcrConfig = OcrConfig(),
    val translator: TranslatorConfig = TranslatorConfig(),
    val inpainter: InpainterConfig = InpainterConfig(),
    val render: RenderConfig = RenderConfig(),
)

data class DetectorConfig(
    val minSide: Float = 3f,
    // seg 文字筆畫遮罩二值門檻（去字用）。★ 0.3 會濾掉漢字旁注音「假名」的弱訊號 → 去字留一排假名殘留。
    // 降到 0.12＝偵測器其實看得到假名、只是 prob 弱（桌面 parity/auto_diag.py dev_furi3 實證）。只影響去字遮罩、不動偵測框/OCR。
    val segThreshold: Float = 0.12f,
    // ── DBNet（m-i-t default 偵測器，本專案唯一偵測器）：ResNet34+DB head，讀對率贏退役的 ctd 1.6–2.5×（真機定案）──
    //   out0=db（2ch，ch0=raw logits，Kotlin 補 sigmoid）、out1=mask（1ch，半/全解析度平台不定、已 sigmoid）。DB 後處理見 Detector.linesFromProbMap。
    val dbnetInputSize: Int = 1024,       // DBNet 甜蜜點（真機 3頁×size×OCR 定案：@1024 字對率最高 + warm ~0.9s；@960 字糙、@1280+ 慢又字誤、@768 漏。resize_aspect → input canvas 768×1024、矩形繞開正方形 832-992 crash 帶）
    val detectUnsharp: Boolean = false,   // 可選：偵測輸入銳利化（marginal + OOD；真機 demo06 A/B 定預設關）
    val dbBinThreshold: Float = 0.5f,     // DB binarize：sigmoid(db ch0) > 此（m-i-t text_threshold=0.5）
    val dbBoxThreshold: Float = 0.7f,     // DB score 過濾：component-mean prob < 此丟（m-i-t box_threshold=0.7）
    val dbUnclipRatio: Float = 2.3f,      // DB unclip 膨脹（m-i-t unclip_ratio=2.3）
)

data class OcrConfig(
    val textHeight: Int = 48,         // 48px CTC
    val minTextLength: Int = 0,       // config.ocr.min_text_length
    val ignoreBubble: Int = 0,        // 〔設定〕config.ocr.ignore_bubble：1–50 開啟，跳過彩色/非氣泡 SFX 類文字（預設 0＝關）
    val minProb: Float = 0.5f,        // config.ocr.prob：OCR 平均信心 < 此值就丟（剃除低信心誤讀；m-i-t 預設 0.5）
    // OCR 裁切前把偵測 quad 四邊各外擴 N px（RotRect.expand；只動 OCR 裁切、**不動偵測框** ⇒ 去字遮罩走 seg 筆畫不受影響）。
    // 病根：偵測框太瘦把字切掉 → 48px CTC 空讀（model_48px_ctc 對 0 字元框在 prob 門檻前就丟）→ 該區被 Pipeline 的
    // textRegions filter 濾掉 → 留原文不翻＝使用者看到的「漏氣泡」。桌面 16 頁實測：pad=4 讀出 345→398(+15%)、框數不變、
    // 弄壞 9 vs 救回 350；006「その通りじゃ」框僅 23px 寬「通」被切 → pad=0 空讀、pad=12 讀對 p=0.993。
    // ★ 預設 4＝真機 A/B 定案（sandbox 6 頁 161 框、按偵測框 index 精確配對）：救回 2（含 006「その通りじゃ」＝
    //   使用者回報的漏氣泡）、**弄壞 0**、實質修復 ~14（'と一も百白です'→'とても面白いですね'、'あいませか'→
    //   'ありませんか'、'お父雄'→'お父様'、'そんな学識も'→'そんな常識も'…），代價＝微小雜訊（'！'→'ー'、少個假名，
    //   LLM 容錯；同 [useBicubic] 的權衡）。**且 OCR 快 ~20%**（框變寬→warp 後 strip 比例→CTC 序列變短）。
    //   pad=8/12 開始退步（8：弄壞 2；12：弄壞 1）⇒ 4 是甜蜜點。桌面 m-i-t warp 模擬曾給 +15% 讀出，真機只 +2
    //   （引擎自刻 bicubic warp 的 baseline 已達 98%），故真機定值不可照抄桌面。
    val stripPad: Int = 4,
    // 逐行並發 OCR：小圖塊（48px 高、窄）吃不滿 intra-op 4 緒 → 改「每行單緒、N 行並發」把核填滿。
    // concurrent=true → NCNN Net num_threads 設 1（單行單緒、逐條 forward 不進 ncnn 全域鎖）、靠 Semaphore(concurrency) 並發；
    // false → 單行用滿 NUM_THREADS、序列。純 CPU ⇒ 收益需實測（sandbox 去背比較 OCR 列 A/B）。與「批次 padding」不同：零 padding 浪費。
    val concurrent: Boolean = true,   // 預設開：真機 8.9s→4.8s(快46%)、零品質風險(每行邏輯不變)、8 核全填滿
    val concurrency: Int = 8,         // 同時在飛的行數上限（＝核數，全核並發；conc=核數為甜蜜點，再高無核可用）
    // 裁切縮放內插法：true=手刻 perspective bicubic（救小假名漏讀→句尾否定不再翻反）、false=Canvas bilinear（舊）。
    // parity 517字/100行 vs bilinear 486/96；真機 A/B 驗過：+6% OCR 時間、把「才能がある/言わないから/言いなさい」
    // 等句尾/關鍵詞讀回（消滅最陰險的「意思相反」），偶爾多讀個雜訊字（LLM 可容錯）。淨正面 → 預設開。
    val useBicubic: Boolean = true,
    // OCR strip 銳化（unsharp mask，見 Ocr.unsharp）：抵銷 warp 把 ~30px 寬直行上採樣到 48px 的模糊、
    // 救回被糊掉漏讀的小假名（v0.16.9 加）。★預設開＝實測救回小假名（對本就讀對 p≈1.0 的乾淨行無副作用）；
    // 關＝退回無銳化（strip 較糊、小假名可能漏讀）。原本硬編碼永遠開，2026-07-16 抽成設定（進階玩家可關）。
    val ocrUnsharp: Boolean = true,
    // OCR 半精度，兩段分開開：storage＝權重/中間值存 fp16（省一半記憶體頻寬）；arith＝用 fp16 指令算。
    // 真機 A/B（2026-09-21）：兩者全開快 27–32% 但 transformer 零星讀錯字、全關同 fp32 但慢 32–45%；storage-only（fp16s）
    // 是第三條路。產品解法＝下面的 ncnnMixed（backbone 吃 fp16 的速度、transformer 留 fp32 的準度）；這兩個仍是 mixed 的前提。
    val ncnnFp16Storage: Boolean = true,
    val ncnnFp16Arith: Boolean = true,
    // 混合精度（backbone fp16、transformer+char_pred fp32）：模型旁有 `<name>_mixed.ncnn.param` 就優先用它（同一份 .bin）。
    // 只在 ncnnFp16Storage 開且 CPU 有 asimdhp 時生效（否則 mixed param 裡的 Cast 會把 fp32 讀成垃圾），Ocr 會自動退回原 param。
    // 真機 9 頁 242 行：mixed 241（唯一不同那行是真值錯）、全 fp16 219 → 預設開；A/B 想固定跑「全 fp16」時關掉。
    val ncnnMixed: Boolean = true,
)

// 預設 few-shot（日→繁中）：示範 <|i|> 逐行格式。改語言對時連同 toLangName/fromLangName 一起換成對應譯文。
private const val DEFAULT_SAMPLE_SOURCE =
    "<|1|>恥ずかしい… 目立ちたくない… 私が消えたい…\n<|2|>きみ… 大丈夫⁉\n<|3|>なんだこいつ 空気読めて ないのか…？"
private const val DEFAULT_SAMPLE_TARGET =
    "<|1|>好尷尬…我不想引人注目…我想消失…\n<|2|>你…沒事吧⁉\n<|3|>這傢伙是看不懂氣氛嗎…？"

/**
 * 翻譯設定。**語言對可任意**（不寫死日→繁中，只是預設）：
 *  - [toLangName] = 目標語言（LLM 直接照這個翻）。
 *  - [fromLangName] = 來源語言標註（進 prompt 措辭；實際來源其實由 OCR 模型決定＝BYOM 換模型就換來源）。
 *  - [sampleSource]/[sampleTarget] = few-shot 範例（同時示範格式與語言對）。
 * 換語言對：設 toLangName + fromLangName + 對應的 few-shot（三者要一致，否則 few-shot 會把輸出帶偏）。
 */
data class TranslatorConfig(
    val provider: String = "deepseek",                                  // 〔設定〕config.translator
    val targetLang: String = "CHT",                                     // 〔設定〕config.target_lang
    // 預設同 LlmProviders 的 deepseek 那筆。舊名 deepseek-chat 於 2026-07-24 15:59 UTC 退役（送出去會 400）
    // → 改為其對應的 deepseek-v4-flash；存著舊名的既有設定由 LlmProviders.migrateModel 就地換名。
    val model: String = "deepseek-v4-flash",                            // 〔設定〕
    val apiBase: String = "https://api.deepseek.com/chat/completions",  // 〔設定〕custom_openai 用
    // ⚠️ toLangName / fromLangName 預設被 fork :domain 的 TranslationPreferences.DEFAULT_TARGET_LANG /
    //   DEFAULT_SOURCE_LANG 鏡像（:domain 不能 import 引擎）。改這兩個請同步改那邊，否則 few-shot 判斷會 drift。
    val toLangName: String = "Traditional Chinese (Taiwan, 台灣慣用的繁體中文用語)",  // 〔設定〕目標語言
    val fromLangName: String = "Japanese",          // 〔設定〕來源語言標註（空白＝讓 LLM 自己判）
    val sampleSource: String = DEFAULT_SAMPLE_SOURCE, // 〔設定〕few-shot 原文（空白＝不放範例）
    val sampleTarget: String = DEFAULT_SAMPLE_TARGET, // 〔設定〕few-shot 譯文（要跟 toLangName 同語言）
    // 〔設定〕**取樣溫度**。★不是每家/每個模型都吃：OpenAI 的 reasoning 模型（o 系列、gpt-5 系列）**拒收**
    //   temperature（送了 400）、DeepSeek 思考模式下則是「收下但無效」⇒ 由 LlmProviders.requestParams 決定
    //   這次請求送不送、以及 clamp 到該家的合法範圍（見 [ParamRule]）。
    val temperature: Double = 0.3,
    // 〔設定〕**思考模式（reasoning）**，預設 **關**。
    //   為什麼預設關：各家新世代模型（DeepSeek v4 系、Gemini 3 系…）**預設就會思考**——對「逐行照翻」這種
    //   結構化任務多花數秒與數倍 token 卻沒明顯品質提升 ⇒ 關掉＝復刻舊 deepseek-chat 的非思考行為（快又便宜）。
    //   開＝允許模型思考：**回應更慢、token 更多、費用更高**。
    //   欄位形狀 per-provider（thinking / reasoning_effort / enable_thinking / reasoning），映射表見 [ParamRule]；
    //   ★不是每家都能關（OpenAI o 系列只能降到最低檔、Gemini 3.x 只能 minimal、Groq 的 GPT-OSS 關不掉、
    //   自架的 custom/sakura 一律不送欄位）。
    val thinking: Boolean = false,
    // 跨頁批次翻譯（對映 m-i-t --batch-size / --batch-concurrent；§2 翻譯批次策略、§10 並發旋鈕）。
    // ★ 現況：引擎端**無消費者**——原本讀這兩欄的 BatchTranslator 已移除（跨頁併發改由 fork 的 PageTranslator
    //   以 Semaphore(pipelineDepth) 負責、不吃這裡）。保留＝§4 第一層「config schema 照搬上游」（上游調參 ⇒
    //   這裡零 code 改動）；日後引擎要自帶批次器可直接接回。**非產品設定頁項目**（故不標〔設定〕）。
    val batchSize: Int = 8,              // m-i-t --batch-size：concurrent 模式＝同時並發頁數上限；merged 模式＝每 prompt 併幾頁
    val batchConcurrent: Boolean = true, // m-i-t --batch-concurrent：true＝逐頁分開請求、批內並發（防 truncation/幻覺）；false＝併大 prompt
    val filterText: String? = null,   // 〔設定〕config.filter_text：regex 命中譯文則濾掉該區（例 ".*badtext.*"）
)

data class InpainterConfig(
    // 〔設定〕去字方法（真機 A/B 定案＝兩門別，皆純 NCNN；LaMa/逐格/auto 逐區路由/GPU 皆已退役移除）：
    //   boxfill＝「快速去字」：取字區就近的背景色平塗（瞬間、平/單色泡泡最乾淨；壓畫面/多彩會塗錯色塊＝粗糙）。不跑去字模型。
    //   aot（預設）＝「AI 去字」：全區都跑 AOT-GAN 重建背景，整頁一次縮到 [tileSize]。走 NCNN AOT；去字被翻譯的網路等待蓋住(§8)，故 tile 大小幾乎不加牆鐘。
    val method: String = "aot",
    val tileSize: Int = 768,          // 整頁 AOT 去字解析度。AOT 全卷積·任意尺寸；768＝畫質/記憶體/藏在翻譯下的甜蜜點（真機 A/B 定案：512 忙碌區糊、1024 記憶體 2× 且貼翻譯天花板）。
    // 去字遮罩膨脹（半徑 = maskDilate/2）。★關鍵：漫畫在臉/頭髮上的字會描一圈白邊；遮罩太薄只蓋黑筆畫、
    // 白邊留在外面 → 去字後殘白塊。加厚到半徑~12 吞掉白邊後，AOT 周圍 context 全變底圖 → 重建乾淨（桌面 inpaint_dev DIL=12 實證 ≈ MIT）。
    val maskDilate: Float = 24f,      // 半徑 12px：吞掉文字白邊（之前 7=半徑4 會殘白塊）
    val bboxPad: Int = 16,            // 去字 allow 用區域 bbox 矩形外擴 px：涵蓋貼 bbox 邊界的假名（行框太緊會漏）
)

data class RenderConfig(
    val orientation: TextOrientation = TextOrientation.AUTO, // 〔設定〕對應 config.render.direction=auto（CJK→直排）
    val fontBorder: Boolean = true,                              // 〔設定〕config.render.disable_font_border=false
    val artStrokeRatio: Float = 0.16f,                           // 壓畫面區(aot 重建·onArt)的白邊寬＝字級×此（比一般 0.10 粗；busy 背景上黑字粗白邊更好讀）
    val fontSizeMax: Int = 60,
    val fontSizeMin: Int = 9,
    // 排版幾何（純文字框法，對齊 parity/typeset_parity.py；不常動，留可控空間）
    val expandW: Float = 1.3f,   // 文字框放大倍率（寬）給呼吸空間
    val expandH: Float = 1.5f,   // 文字框放大倍率（直欄高 / 橫排列高）
    val colTrim: Int = 3,        // 直排每欄少放幾字（縮短欄長、減少凸出；欄變多→字級自動縮）
    val rowTrim: Int = 3,        // 橫排每行少放幾字（colTrim 的橫排對映：行變短、列變多→字級自動縮）
    val fontScale: Float = 0.85f, // 算好字級後整體縮放（<1＝更小、更 fit 格子、留邊距）
    // 文字顏色：auto＝取去字後背景亮度判黑/白字（最穩、白底黑字/黑底白字）；mono＝一律黑字白邊
    val colorMode: String = "auto",
    val bgDark: Int = 110,       // auto：去字後背景平均亮度 < 此值＝暗底 → 白字
    // 縱中橫（tate-chu-yoko）：直排時把連續短 ASCII 串（2–4 字的數字/字母/!?）併成一格水平並排（年齡「20」、年份「2020」、「!?」不再上下堆疊歪頭讀）。
    // §4 第三層知情偏離：m-i-t/parity 逐字畫、無此邏輯；只影響直排內 ASCII 短串，CJK 不變。預設開、可關回逐字。
    val tateChuYoko: Boolean = true,
)
