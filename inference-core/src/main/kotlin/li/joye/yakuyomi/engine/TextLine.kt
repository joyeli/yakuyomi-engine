package li.joye.yakuyomi.engine

/**
 * 一條文字行：偵測給出旋轉四邊形（原圖座標）+ 分數；OCR 再填方向與辨識文字。
 * 取代舊的軸對齊 Box（OCR 需要旋轉框餵入）。
 */
class TextLine(val quad: List<Pt>, val score: Float) {
    var direction: String = "h" // 'h' 橫書 / 'v' 直書（由 OCR 的 sortPnts 判定）
    var text: String = ""        // OCR 日文
    var translatedText: String = "" // 翻譯後繁中
}
