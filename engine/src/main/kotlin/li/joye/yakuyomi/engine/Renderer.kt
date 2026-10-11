package li.joye.yakuyomi.engine

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Typeface
import kotlin.math.abs
import kotlin.math.ceil
import kotlin.math.min
import kotlin.math.roundToInt

/** 排版方向。AUTO＝依內容（CJK 直排、純英數橫排），對應 m-i-t config 的 direction:auto。 */
enum class TextOrientation { VERTICAL, HORIZONTAL, AUTO }

/**
 * 排版（純文字框法，可靠）：定位 + 大小都用文字框，框適度放大（[RenderConfig.expandW]/[RenderConfig.expandH]）給呼吸空間。
 * 對齊 parity/typeset_parity.py（§4 第二層：同輸入近輸出）。
 *   不靠氣泡 flood-fill——相鄰氣泡會連通成一塊、整個算錯，已棄用。
 *   直排：CJK 上→下、欄右→左、向上對齊、每欄少 [RenderConfig.colTrim] 字（縮短欄長、減少凸出）、標點旋轉、短 ASCII 串縱中橫（tate-chu-yoko，見 [RenderConfig.tateChuYoko]）。
 *   橫排：左→右、上→下、向上對齊。
 *   文字色：[RenderConfig.colorMode]=auto 取去字後背景亮度判黑/白字（白底黑字、黑底白字）；OCR color head 色相太雜不採用。
 * ★ 後續：字型可選。
 */
object Renderer {

    fun render(
        page: Bitmap,
        regions: List<TextRegion>,
        cfg: RenderConfig = RenderConfig(),
        tf: Typeface? = null,
    ): Bitmap {
        val out = page.copy(Bitmap.Config.ARGB_8888, true)
        val canvas = Canvas(out)
        val face = tf ?: Typeface.DEFAULT
        val fill = Paint().apply { color = Color.BLACK; isAntiAlias = true; typeface = face }
        val stroke = Paint().apply {
            color = Color.WHITE; style = Paint.Style.STROKE; strokeWidth = 4f
            isAntiAlias = true; typeface = face
        }
        for (region in regions) {
            val text = region.translatedText.ifBlank { region.sourceText }
            if (text.isBlank()) continue
            if (region.x1 - region.x0 < 8f || region.y1 - region.y0 < 8f) continue
            val (fillColor, outlineColor) = textColors(page, region, cfg)
            fill.color = fillColor
            stroke.color = outlineColor
            val vertical = when (cfg.orientation) {
                TextOrientation.AUTO -> region.direction == "v" // 跟著偵測到的原文方向（對齊 m-i-t），不再無腦直排
                TextOrientation.VERTICAL -> true
                TextOrientation.HORIZONTAL -> false
            }
            // 斜框：繞區域中心旋轉畫布、用去傾斜框排版。
            // 旋轉方向經 PCA 量測對齊 m-i-t 樣本：region 正角＝文字右端下斜。
            // Android Canvas 正角＝順時針（y 朝下）＝右端下斜 → 直接 +angle（parity 用 PIL、正角逆時針故取 -angle）。
            val rotate = abs(region.angle) >= 1f
            val x0: Float; val y0: Float; val x1: Float; val y1: Float
            if (rotate) {
                x0 = region.cx - region.boxW / 2f; y0 = region.cy - region.boxH / 2f
                x1 = region.cx + region.boxW / 2f; y1 = region.cy + region.boxH / 2f
                canvas.save(); canvas.rotate(region.angle, region.cx, region.cy)
            } else {
                x0 = region.x0; y0 = region.y0; x1 = region.x1; y1 = region.y1
            }
            val boxes = region.layoutBoxes
            if (vertical) drawVertical(canvas, x0, y0, x1, y1, text, fill, stroke, cfg, region.onArt)
            else if (boxes.isNotEmpty() && !rotate) drawHorizontalFit(canvas, boxes, region.lineHeight, text, fill, stroke, cfg, region.onArt)
            else drawHorizontal(canvas, x0, y0, x1, y1, text, fill, stroke, cfg, region.onArt)
            if (rotate) canvas.restore()
        }
        return out
    }

    /** 文字色 (fill, outline)：auto＝取去字後背景亮度（暗底白字、亮底黑字，對齊 parity auto_colors）；mono＝黑字白邊。 */
    private fun textColors(page: Bitmap, r: TextRegion, cfg: RenderConfig): Pair<Int, Int> {
        if (cfg.colorMode == "mono") return Color.BLACK to Color.WHITE
        // 壓在畫面上(lama 重建的 busy 背景)：一律黑字+粗白邊。白邊把字框出來，任何雜亂背景都讀得到（對齊 m-i-t 做法）。
        if (r.onArt) return Color.BLACK to Color.WHITE
        val lum = bgLuminance(page, r.x0, r.y0, r.x1, r.y1)
        return if (lum < cfg.bgDark) Color.WHITE to Color.BLACK else Color.BLACK to Color.WHITE
    }

    /** 去字後背景在文字框內的平均亮度（Rec.601）。 */
    private fun bgLuminance(page: Bitmap, rx0: Float, ry0: Float, rx1: Float, ry1: Float): Float {
        val w = page.width
        val h = page.height
        val x0 = rx0.toInt().coerceIn(0, w - 1)
        val y0 = ry0.toInt().coerceIn(0, h - 1)
        val x1 = rx1.toInt().coerceIn(x0 + 1, w)
        val y1 = ry1.toInt().coerceIn(y0 + 1, h)
        val bw = x1 - x0
        val bh = y1 - y0
        val px = IntArray(bw * bh)
        page.getPixels(px, 0, bw, x0, y0, bw, bh)
        var sum = 0.0
        for (p in px) {
            sum += 0.299 * ((p shr 16) and 0xFF) + 0.587 * ((p shr 8) and 0xFF) + 0.114 * (p and 0xFF)
        }
        return (sum / px.size).toFloat()
    }

    private fun isCjk(text: String): Boolean = text.any {
        val o = it.code
        o in 0x3040..0x30FF || o in 0x4E00..0x9FFF || o in 0x3400..0x4DBF || o in 0xFF00..0xFFEF
    }

    /** 直排：欄右→左、格上→下、向上對齊；大小填滿放大後的文字框、每欄少 colTrim 格。每格＝1 字或 1 個縱中橫短串。 */
    private fun drawVertical(canvas: Canvas, x0: Float, y0: Float, x1: Float, y1: Float, text: String, fill: Paint, stroke: Paint, cfg: RenderConfig, onArt: Boolean = false) {
        val chars = text.filter { it != '\n' }
        if (chars.isEmpty()) return
        val cells = toVerticalCells(chars, cfg.tateChuYoko)  // 切格：一般字一格、短 ASCII 串併成一個縱中橫格
        val bw = (x1 - x0) * cfg.expandW         // 寬：放大後的文字框寬
        val colRoom = (y1 - y0) * cfg.expandH    // 直欄可用高（從文字框頂往下）
        var size = cfg.fontSizeMin
        var s = min(colRoom.toInt(), cfg.fontSizeMax)
        while (s >= cfg.fontSizeMin) {
            val lh = s * 1.05f; val cw = s * 1.1f
            val cpc = maxOf(1, (colRoom / lh).toInt() - cfg.colTrim)
            if (ceil(cells.size / cpc.toFloat()).toInt() * cw <= bw) { size = s; break }
            s--
        }
        size = maxOf(cfg.fontSizeMin, (size * cfg.fontScale).roundToInt())  // 整體縮小、更 fit
        fill.textSize = size.toFloat(); stroke.textSize = size.toFloat()
        stroke.strokeWidth = maxOf(2f, size * (if (onArt) cfg.artStrokeRatio else STROKE_RATIO))  // 描邊隨字級；壓畫面區用更粗白邊
        val lh = size * 1.05f; val cw = size * 1.1f
        val cpc = maxOf(1, (colRoom / lh).toInt() - cfg.colTrim)
        val columns = splitColumnsV(cells, cpc)       // 禁則：欄不以行頭禁則字開頭
        val cols = columns.size
        val tcx = (x0 + x1) / 2f                  // 定位：水平置中於文字框中心
        val rightCx = tcx + cols * cw / 2f - cw / 2f
        val blockH = columns.maxOf { it.size } * lh // 垂直置中：以最長欄格數為塊高，置中於框
        val startCy = (y0 + y1) / 2f - blockH / 2f
        for (col in 0 until cols) {
            val cx = rightCx - col * cw
            var cy = startCy
            for (cell in columns[col]) {
                if (cell.length == 1) {
                    drawCharVertical(canvas, cell[0], cx, cy + lh / 2f, fill, stroke, cfg.fontBorder)
                } else {
                    drawTateChuYoko(canvas, cell, cx, cy + lh / 2f, cw, fill, stroke, cfg.fontBorder)
                }
                cy += lh
            }
        }
    }

    /**
     * 直排切格：一般字一格；連續短 ASCII 串（2–[MAX_TCY] 字的數字/字母/!?）併成一個縱中橫格（tate-chu-yoko）。
     * 單字 ASCII（如獨立「5」）維持單格；過長串（> MAX_TCY，如英文長詞）退回逐字（避免水平壓太扁）。
     */
    private fun toVerticalCells(chars: String, enabled: Boolean): List<String> {
        if (!enabled) return chars.map { it.toString() }
        val cells = ArrayList<String>()
        var i = 0
        val n = chars.length
        while (i < n) {
            if (isTcyChar(chars[i])) {
                var j = i + 1
                while (j < n && isTcyChar(chars[j])) j++
                if (j - i in 2..MAX_TCY) {
                    cells.add(chars.substring(i, j))                       // 一個縱中橫格
                } else {
                    for (k in i until j) cells.add(chars[k].toString())   // 單字或過長：逐字（維持原行為）
                }
                i = j
            } else {
                cells.add(chars[i].toString()); i++
            }
        }
        return cells
    }

    private fun isTcyChar(c: Char): Boolean =
        c in '0'..'9' || c in 'A'..'Z' || c in 'a'..'z' || c == '!' || c == '?'

    /** 直排切欄＋行頭禁則：禁則字（單字標點）不置於欄頭、併回前一欄（最多 +2，避免暴衝）。 */
    private fun splitColumnsV(cells: List<String>, cpc: Int): List<List<String>> {
        val cols = ArrayList<List<String>>()
        var i = 0
        val n = cells.size
        while (i < n) {
            var end = minOf(i + cpc, n)
            var ext = 0
            while (end < n && cells[end].length == 1 && cells[end][0] in NO_START && ext < 2) { end++; ext++ }
            cols.add(cells.subList(i, end).toList())
            i = end
        }
        return cols
    }

    /** 縱中橫：把短 ASCII 串在一個直排格內水平並排、置中於欄心；超出格寬只橫向壓縮（高度不變、與鄰字視覺一致）。 */
    private fun drawTateChuYoko(canvas: Canvas, group: String, cx: Float, cyc: Float, cellW: Float, fill: Paint, stroke: Paint, border: Boolean) {
        val w = fill.measureText(group)
        val fm = fill.fontMetrics
        val baseline = cyc - (fm.ascent + fm.descent) / 2f
        val target = cellW * 0.92f
        val scaleX = if (w > target) target / w else 1f
        canvas.save()
        if (scaleX != 1f) canvas.scale(scaleX, 1f, cx, baseline)  // 只橫向縮、繞欄心
        val tx = cx - w / 2f
        if (border) canvas.drawText(group, tx, baseline, stroke)
        canvas.drawText(group, tx, baseline, fill)
        canvas.restore()
    }

    /** 橫排：列上→下、字左→右、向上對齊；大小填滿放大後的文字框。 */
    private fun drawHorizontal(canvas: Canvas, x0: Float, y0: Float, x1: Float, y1: Float, text: String, fill: Paint, stroke: Paint, cfg: RenderConfig, onArt: Boolean = false) {
        val bw = (x1 - x0) * cfg.expandW
        val rowRoom = (y1 - y0) * cfg.expandH
        var size = cfg.fontSizeMin
        var lines = listOf(text)
        var s = min(rowRoom.toInt(), cfg.fontSizeMax)
        while (s >= cfg.fontSizeMin) {
            fill.textSize = s.toFloat()
            val ls = wrapCjk(text, fill, (bw - cfg.rowTrim * s).coerceAtLeast(s.toFloat())) // 每行少 rowTrim 字（橫向字數）
            val maxW = ls.maxOfOrNull { fill.measureText(it) } ?: 0f
            if (ls.size * s * 1.18f <= rowRoom && maxW <= bw) { size = s; lines = ls; break }
            s--
        }
        size = maxOf(cfg.fontSizeMin, (size * cfg.fontScale).roundToInt())  // 整體縮小、更 fit
        fill.textSize = size.toFloat(); stroke.textSize = size.toFloat()
        stroke.strokeWidth = maxOf(2f, size * (if (onArt) cfg.artStrokeRatio else STROKE_RATIO))  // 描邊隨字級；壓畫面區用更粗白邊
        lines = wrapCjk(text, fill, (bw - cfg.rowTrim * size).coerceAtLeast(size.toFloat()))  // 縮小後重排（含 rowTrim）
        val lh = size * 1.18f
        val tcx = (x0 + x1) / 2f
        var baseline = (y0 + y1) / 2f - lines.size * lh / 2f + size * ASCENT  // 垂直置中於框
        val pageW = canvas.width.toFloat()
        for (ln in lines) {
            val lw = fill.measureText(ln)
            // 置中後若超出頁緣（框貼頁邊、放大後的框比頁寬多）→ 推回頁內，避免句首／句尾被裁掉。
            val tx = if (lw + 2 * EDGE_MARGIN <= pageW) {
                (tcx - lw / 2f).coerceIn(EDGE_MARGIN, pageW - lw - EDGE_MARGIN)
            } else {
                tcx - lw / 2f
            }
            if (cfg.fontBorder) canvas.drawText(ln, tx, baseline, stroke)
            canvas.drawText(ln, tx, baseline, fill)
            baseline += lh
        }
    }

    /**
     * 橫排、排進已知可用框（氣泡內部，見 TextRegion.layoutBox）：字級上限＝原文字高×[FIT_SIZE_RATIO]（與畫面相稱），
     * 由大往小找第一個「換行後塞得進框」的字級；行距用字型實際高度（阿拉伯文等上下伸部較長的字也不互撞）。
     */
    private fun drawHorizontalFit(canvas: Canvas, boxes: List<FloatArray>, srcLineH: Float, text: String, fill: Paint, stroke: Paint, cfg: RenderConfig, onArt: Boolean) {
        // 阿拉伯字母（阿拉伯文、波斯文、烏爾都文…）同 px 字面明顯比韓／日／拉丁大寫小，上限放寬才看起來與原文同大
        val ratio = if (text.any { it.code in 0x0600..0x06FF || it.code in 0x0750..0x077F || it.code in 0xFB50..0xFEFF }) FIT_SIZE_RATIO_ARABIC else FIT_SIZE_RATIO
        val cap = if (srcLineH > 0f) (srcLineH * ratio).roundToInt() else cfg.fontSizeMax
        val maxS = min(cfg.fontSizeMax, cap).coerceAtLeast(cfg.fontSizeMin)
        var size = cfg.fontSizeMin
        var lines: List<String>? = null
        var box = boxes.last()
        search@ for (s in maxS downTo cfg.fontSizeMin) {
            fill.textSize = s.toFloat()
            for (b in boxes) {
                val bw = (b[2] - b[0]) * (1f - 2 * FIT_INSET)
                val bh = (b[3] - b[1]) * (1f - 2 * FIT_INSET)
                if (bw < s) continue
                val ls = wrapCjk(text, fill, bw)
                val maxW = ls.maxOfOrNull { fill.measureText(it) } ?: 0f
                if (blockInkHeight(ls, fill) <= bh && maxW <= bw) { size = s; lines = ls; box = b; break@search }
            }
        }
        val bw = (box[2] - box[0]) * (1f - 2 * FIT_INSET)
        fill.textSize = size.toFloat(); stroke.textSize = size.toFloat()
        val ls = lines ?: wrapCjk(text, fill.apply { textSize = size.toFloat() }, bw.coerceAtLeast(size.toFloat()))
        stroke.strokeWidth = maxOf(2f, size * (if (onArt) cfg.artStrokeRatio else STROKE_RATIO))
        val lh = size * LINE_SPACING
        val (top, bottom) = inkExtent(ls, fill)
        val tcx = (box[0] + box[2]) / 2f
        // 以實際墨跡置中：整塊高＝(行數−1)×行距＋單行墨跡高
        var baseline = (box[1] + box[3]) / 2f - ((ls.size - 1) * lh + (bottom - top)) / 2f - top
        val pageW = canvas.width.toFloat()
        for (ln in ls) {
            val lw = fill.measureText(ln)
            val tx = if (lw + 2 * EDGE_MARGIN <= pageW) (tcx - lw / 2f).coerceIn(EDGE_MARGIN, pageW - lw - EDGE_MARGIN) else tcx - lw / 2f
            if (cfg.fontBorder) canvas.drawText(ln, tx, baseline, stroke)
            canvas.drawText(ln, tx, baseline, fill)
            baseline += lh
        }
    }

    /** 各行墨跡相對基線的最高（負）與最低點。 */
    private fun inkExtent(lines: List<String>, p: Paint): Pair<Float, Float> {
        val r = android.graphics.Rect()
        var top = 0f; var bottom = 0f
        for (ln in lines) {
            if (ln.isEmpty()) continue
            p.getTextBounds(ln, 0, ln.length, r)
            top = min(top, r.top.toFloat()); bottom = maxOf(bottom, r.bottom.toFloat())
        }
        if (bottom - top <= 0f) { top = -p.textSize * ASCENT; bottom = p.textSize * (1f - ASCENT) }
        return top to bottom
    }

    private fun blockInkHeight(lines: List<String>, p: Paint): Float {
        val (top, bottom) = inkExtent(lines, p)
        return (lines.size - 1) * p.textSize * LINE_SPACING + (bottom - top)
    }

    private fun drawCharVertical(canvas: Canvas, ch: Char, cx: Float, cyc: Float, fill: Paint, stroke: Paint, border: Boolean) {
        val s = ch.toString()
        val w = fill.measureText(s)
        val fm = fill.fontMetrics
        val baseline = cyc - (fm.ascent + fm.descent) / 2f
        val rotate = ch in ROTATE_CHARS
        if (rotate) { canvas.save(); canvas.rotate(90f, cx, cyc) }
        if (border) canvas.drawText(s, cx - w / 2f, baseline, stroke)
        canvas.drawText(s, cx - w / 2f, baseline, fill)
        if (rotate) canvas.restore()
    }

    private fun wrapCjk(text: String, paint: Paint, maxW: Float): List<String> {
        // 非中日文（阿拉伯文、拉丁字母、韓文…）按詞換行：逐字切會把詞拆兩半（阿拉伯文連寫字形也會斷開）。
        if (!isCjk(text)) return wrapWords(text, paint, maxW)
        val lines = ArrayList<String>()
        val cur = StringBuilder()
        for (ch in text) {
            if (ch == '\n') { lines.add(cur.toString()); cur.clear(); continue }
            if (cur.isNotEmpty() && paint.measureText(cur.toString() + ch) > maxW && ch !in NO_START) {
                lines.add(cur.toString()); cur.clear()  // 行頭禁則：禁則字不另起行
            }
            cur.append(ch)
        }
        if (cur.isNotEmpty()) lines.add(cur.toString())
        return lines
    }

    /** 按空白分詞換行；詞本身比行寬長也不拆（交給呼叫端縮字級——拆詞會讓阿拉伯文等連寫字形斷開）。 */
    private fun wrapWords(text: String, paint: Paint, maxW: Float): List<String> {
        val lines = ArrayList<String>()
        for (para in text.split('\n')) {
            var cur = ""
            for (word in para.split(' ').filter { it.isNotEmpty() }) {
                val cand = if (cur.isEmpty()) word else "$cur $word"
                if (cur.isEmpty() || paint.measureText(cand) <= maxW) { cur = cand; continue }
                lines.add(cur)
                cur = word
            }
            lines.add(cur)
        }
        return lines.filter { it.isNotEmpty() }.ifEmpty { listOf("") }
    }

    private const val ASCENT = 0.82f
    private const val EDGE_MARGIN = 4f  // 橫排行與頁緣的最小距離（px）
    private const val FIT_INSET = 0.04f       // 可用框四邊各內縮此比例（離氣泡外框留白；框本身已在外框內）
    private const val FIT_SIZE_RATIO = 1.2f   // 譯文字級上限＝原文字高×此值（阿拉伯文等字面比韓／日文小，同 px 看起來較小）
    private const val FIT_SIZE_RATIO_ARABIC = 1.8f   // 阿拉伯字母主體高約 0.45em（韓／日／拉丁大寫約 0.75em）
    private const val LINE_SPACING = 1.3f     // 排進框時的行距＝字級×此值
    private const val STROKE_RATIO = 0.10f  // 描邊寬＝字級×此比例（隨字級縮放）
    private const val MAX_TCY = 4  // 縱中橫一格最多併幾個 ASCII（涵蓋 2 位數年齡、4 位數年份；更長退回逐字避免壓太扁）
    private const val ROTATE_CHARS = "ー－—―‐~〜～…‥（）()「」『』【】〔〕［］｛｝〈〉《》＜＞<>｜|：;"
    // 行頭禁則：不可置於欄/行開頭（收尾標點、小假名）→ 併回前一欄/行（kinsoku）
    private const val NO_START = "、。，．：；！？”’）〕】｝」』》〉…‥ーゝゞヽヾ々ぁぃぅぇぉっゃゅょゎァィゥェォッャュョヮ"
}
