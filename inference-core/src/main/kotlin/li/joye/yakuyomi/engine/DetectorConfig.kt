package li.joye.yakuyomi.engine

/**
 * 偵測參數（DBNet）。`EngineConfig.detector`（:engine）與夜讀（:nightread-android 的 NightReadRenderer 經 [Detector]）共用，
 * 所以放在推論核心；其餘翻譯參數（OCR／翻譯／去字／排版）在 :engine 的 Config.kt。
 */
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
