package li.joye.yakuyomi.engine

import kotlin.math.PI
import kotlin.math.abs
import kotlin.math.acos
import kotlin.math.cos
import kotlin.math.hypot
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sin
import kotlin.math.sqrt

/**
 * 文字行 → 氣泡區分群（翻譯前合併，避免逐行碎裂）。
 *
 * 移植自 manga_translator/textline_merge/__init__.py + utils/generic.py @ d5a3eee
 * （透過已驗證的 parity/mit_grouping.py 作規格本；§4 第一/二層：同輸入近輸出）。
 *
 * 兩階段（缺一就會「靠近的多顆氣泡黏成一大塊」＝色塊 + 文字疊住）：
 *   1. **連邊**：quadrilateralCanMergeRegion（寬鬆：同方向、字級相近、間距 < 1×字、邊對齊 < 3×字）。
 *   2. **MST 分裂**：splitTextRegion——對每個連通塊建最小生成樹，若最大邊（間距）相對其餘是離群值就在那裡切開、遞迴。
 *      這步把「被第 1 步過度連起來的相鄰氣泡」拆回去，是不過度合併的關鍵。
 *
 * 註：合併階段 m-i-t 的 assigned_direction 為 None（OCR 才設）→ distance 一律走 v 模式；此處照搬。
 *    區域文字色不在此算（Renderer 取去字後背景亮度判黑/白字）。
 */
class TextRegion(
    val lines: List<TextLine>,
    val direction: String,
    /** 區域傾斜角（度，對齊 m-i-t TextBlock.angle）；排版沿此角度旋轉，<3° 視為 0。 */
    val angle: Float = 0f,
    /** 區域中心（各行角點均值），旋轉排版的樞紐。 */
    val cx: Float = 0f,
    val cy: Float = 0f,
    /** 去傾斜後（文字自然座標）的框尺寸，旋轉排版時當文字框用。 */
    val boxW: Float = 0f,
    val boxH: Float = 0f,
) {
    var translatedText: String = ""

    /** 此區是否「壓在畫面上」(去字走 lama 重建，非乾淨白泡)。由 :engine 的 `Inpainter` 設、`Renderer` 據此給黑字粗白邊（busy 背景好讀）。 */
    var onArt: Boolean = false

    /** auto 路由除錯：此區背景量到的 std / 亮度均值（由 :engine 的 `Inpainter` auto 分支設；-1＝未量）。sandbox 去背比較標在框上、調門檻用，產品不讀。 */
    var dbgStd: Float = -1f
    var dbgWhite: Float = -1f

    /** 合併原文（lines 已依閱讀序排好）。日文無空白，直接相接。 */
    val sourceText: String get() = lines.joinToString("") { it.text }

    val x0: Float = lines.minOf { ln -> ln.quad.minOf { it.x } }
    val y0: Float = lines.minOf { ln -> ln.quad.minOf { it.y } }
    val x1: Float = lines.maxOf { ln -> ln.quad.maxOf { it.x } }
    val y1: Float = lines.maxOf { ln -> ln.quad.maxOf { it.y } }
}

object Grouping {

    fun group(lines: List<TextLine>): List<TextRegion> {
        val n = lines.size
        if (n == 0) return emptyList()
        val q = lines.map { GQuad(it) }

        // step 1：寬鬆連邊 → union-find 連通塊
        val parent = IntArray(n) { it }
        fun find(x: Int): Int {
            var r = x
            while (parent[r] != r) r = parent[r]
            var c = x
            while (parent[c] != c) { val nxt = parent[c]; parent[c] = r; c = nxt }
            return r
        }
        for (i in 0 until n) {
            for (j in i + 1 until n) {
                if (canMerge(q[i], q[j])) parent[find(i)] = find(j)
            }
        }
        val comps = LinkedHashMap<Int, MutableList<Int>>()
        for (i in 0 until n) comps.getOrPut(find(i)) { mutableListOf() }.add(i)

        // step 2：每個連通塊跑 MST 分裂
        val regionIdx = ArrayList<List<Int>>()
        for (comp in comps.values) regionIdx.addAll(splitTextRegion(q, comp))

        // step 3：每區內排閱讀序、定方向、算傾斜角 + 去傾斜框
        return regionIdx.map { members ->
            val dir = majorityDir(q, members)
            val ordered = if (dir == "h") {
                members.sortedBy { q[it].centroid.y }            // 橫書：上→下
            } else {
                members.sortedByDescending { q[it].centroid.x }  // 直書：右→左
            }
            val angle = regionAngle(q, members)
            val pts = members.flatMap { q[it].pts }
            val cx = pts.map { it.x }.average().toFloat()
            val cy = pts.map { it.y }.average().toFloat()
            val (bw, bh) = orientedBox(pts, cx, cy, angle)
            TextRegion(ordered.map { lines[it] }, dir, angle, cx, cy, bw, bh)
        }
    }

    /** 區域角度＝各行角度均值-90（度，對齊 textline_merge dispatch）；<3° 視為 0。 */
    private fun regionAngle(q: List<GQuad>, members: List<Int>): Float {
        val meanRad = members.map { q[it].angle.toDouble() }.average()
        val deg = (Math.toDegrees(meanRad) - 90.0).toFloat()
        return if (abs(deg) < 3f) 0f else deg
    }

    /** 把區域所有角點去傾斜（R(-angle)）後的軸對齊範圍＝文字自然座標下的框尺寸。 */
    private fun orientedBox(pts: List<Pt>, cx: Float, cy: Float, angleDeg: Float): Pair<Float, Float> {
        val rad = Math.toRadians(angleDeg.toDouble())
        val cos = cos(rad).toFloat()
        val sin = sin(rad).toFloat()
        var minX = Float.MAX_VALUE; var maxX = -Float.MAX_VALUE
        var minY = Float.MAX_VALUE; var maxY = -Float.MAX_VALUE
        for (p in pts) {
            val dx = p.x - cx; val dy = p.y - cy
            val rx = cos * dx + sin * dy
            val ry = -sin * dx + cos * dy
            if (rx < minX) minX = rx; if (rx > maxX) maxX = rx
            if (ry < minY) minY = ry; if (ry > maxY) maxY = ry
        }
        return (maxX - minX) to (maxY - minY)
    }

    // —— 連邊判定：quadrilateral_can_merge_region（merge_bboxes_text_region 用的參數）——
    private fun canMerge(
        a: GQuad, b: GQuad,
        ratio: Float = 1.9f,
        discardConnectionGap: Float = 2f,
        charGapTolerance: Float = 1f,       // m-i-t merge_bboxes 傳入值
        charGapTolerance2: Float = 3f,
        fontSizeRatioTol: Float = 2f,
        aspectRatioTol: Float = 1.3f,
    ): Boolean {
        val charSize = min(a.fontSize, b.fontSize)
        if (charSize <= 0f) return false
        val x1 = a.aabb[0]; val y1 = a.aabb[1]; val w1 = a.aabb[2]; val h1 = a.aabb[3]
        val x2 = b.aabb[0]; val y2 = b.aabb[1]; val w2 = b.aabb[2]; val h2 = b.aabb[3]
        val d = Geometry.polyDistance(a.pts, b.pts)
        if (d > discardConnectionGap * charSize) return false
        if (max(a.fontSize, b.fontSize) / charSize > fontSizeRatioTol) return false
        if (a.aspectRatio > aspectRatioTol && b.aspectRatio < 1f / aspectRatioTol) return false
        if (b.aspectRatio > aspectRatioTol && a.aspectRatio < 1f / aspectRatioTol) return false
        if (a.isApproxAxisAligned && b.isApproxAxisAligned) {
            if (d < charSize * charGapTolerance) {
                if (abs(x1 + w1 / 2f - (x2 + w2 / 2f)) < charGapTolerance2) return true
                if (w1 > h1 * ratio && h2 > w2 * ratio) return false
                if (w2 > h2 * ratio && h1 > w1 * ratio) return false
                if (w1 > h1 * ratio || w2 > h2 * ratio) {  // 橫
                    return abs(x1 - x2) < charSize * charGapTolerance2 ||
                        abs(x1 + w1 - (x2 + w2)) < charSize * charGapTolerance2
                } else if (h1 > w1 * ratio || h2 > w2 * ratio) {  // 直
                    return abs(y1 - y2) < charSize * charGapTolerance2 ||
                        abs(y1 + h1 - (y2 + h2)) < charSize * charGapTolerance2
                }
                return false
            }
            return false
        }
        // 非軸對齊（旋轉框）分支
        if (abs(a.angle - b.angle) < 15f * PI.toFloat() / 180f) {
            val fs = min(a.fontSize, b.fontSize)
            if (Geometry.polyDistance(a.pts, b.pts) > fs * charGapTolerance2) return false
            if (abs(a.fontSize - b.fontSize) / fs > 0.25f) return false
            return true
        }
        return false
    }

    // —— MST 分裂：split_text_region ——
    private fun splitTextRegion(
        q: List<GQuad>, idx: List<Int>, gamma: Float = 0.5f, sigma: Float = 2f,
    ): List<List<Int>> {
        if (idx.size == 1) return listOf(idx)
        if (idx.size == 2) {
            val fs = max(q[idx[0]].fontSize, q[idx[1]].fontSize)
            val close = q[idx[0]].distance(q[idx[1]]) < (1f + gamma) * fs
            val sameAngle = abs(q[idx[0]].angle - q[idx[1]].angle) < 0.2f * PI.toFloat()
            return if (close && sameAngle) listOf(idx) else listOf(listOf(idx[0]), listOf(idx[1]))
        }
        // 完全圖 → Kruskal MST
        val edges = ArrayList<Edge>()
        for (a in idx.indices) for (b in a + 1 until idx.size) {
            edges.add(Edge(idx[a], idx[b], q[idx[a]].distance(q[idx[b]])))
        }
        edges.sortBy { it.w }
        val par = HashMap<Int, Int>().apply { idx.forEach { put(it, it) } }
        fun find(x: Int): Int { var r = x; while (par[r] != r) r = par[r]!!; return r }
        val mst = ArrayList<Edge>()
        for (e in edges) {
            val ra = find(e.u); val rb = find(e.v)
            if (ra != rb) { par[ra] = rb; mst.add(e); if (mst.size == idx.size - 1) break }
        }
        mst.sortByDescending { it.w }
        val distances = mst.map { it.w }
        val fontsize = idx.map { q[it].fontSize }.average().toFloat()
        val mean = distances.average().toFloat()
        val std = sqrt(distances.map { (it - mean) * (it - mean) }.average()).toFloat()
        val stdThreshold = max(0.3f * fontsize + 5f, 5f)
        val b1 = q[mst[0].u]; val b2 = q[mst[0].v]
        val maxPolyDistance = Geometry.polyDistance(b1.pts, b2.pts)
        val maxCentroidAlignment = min(abs(b1.centroid.x - b2.centroid.x), abs(b1.centroid.y - b2.centroid.y))
        val keep = (distances[0] <= mean + std * sigma || distances[0] <= fontsize * (1f + gamma)) &&
            (std < stdThreshold || (maxPolyDistance == 0f && maxCentroidAlignment < 5f))
        if (keep) return listOf(idx)
        // 切掉最大邊（mst[0]）→ 其餘 MST 邊的連通塊 → 遞迴
        val par2 = HashMap<Int, Int>().apply { idx.forEach { put(it, it) } }
        fun find2(x: Int): Int { var r = x; while (par2[r] != r) r = par2[r]!!; return r }
        for (k in 1 until mst.size) { par2[find2(mst[k].u)] = find2(mst[k].v) }
        val sub = LinkedHashMap<Int, MutableList<Int>>()
        for (i in idx) sub.getOrPut(find2(i)) { mutableListOf() }.add(i)
        val ans = ArrayList<List<Int>>()
        for (c in sub.values) ans.addAll(splitTextRegion(q, c, gamma, sigma))
        return ans
    }

    private fun majorityDir(q: List<GQuad>, members: List<Int>): String {
        val counts = members.groupingBy { q[it].direction }.eachCount()
        val top2 = counts.entries.sortedByDescending { it.value }
        if (top2.size == 1) return top2[0].key
        if (top2[0].value == top2[1].value) {  // 平手 → 取長寬比最極端那行的方向
            var maxAr = -100f
            var dir = top2[0].key
            for (i in members) {
                val ar = q[i].aspectRatio
                if (ar > maxAr) { maxAr = ar; dir = q[i].direction }
                if (1f / ar > maxAr) { maxAr = 1f / ar; dir = q[i].direction }
            }
            return dir
        }
        return top2[0].key
    }

    private data class Edge(val u: Int, val v: Int, val w: Float)
}

/** 一條文字行的幾何（對齊 generic.py:Quadrilateral 的 grouping 子集）。pts 排成 [左上,右上,右下,左下]。 */
private class GQuad(val line: TextLine) {
    val pts: List<Pt>
    val direction: String

    init {
        val (ordered, isV) = sortPnts(line.quad)
        pts = ordered
        direction = if (isV) "v" else "h"
    }

    /** 四邊中點 p1,p2,p3,p4（截斷成整數座標，對齊 m-i-t structure 的 .astype(int)）。 */
    private val structure: List<Pt> by lazy {
        fun mid(a: Pt, b: Pt) = Pt(((a.x + b.x) / 2f).toInt().toFloat(), ((a.y + b.y) / 2f).toInt().toFloat())
        listOf(mid(pts[0], pts[1]), mid(pts[2], pts[3]), mid(pts[1], pts[2]), mid(pts[3], pts[0]))
    }

    /** [x, y, w, h]（軸對齊外接框）。 */
    val aabb: FloatArray by lazy {
        val mnx = pts.minOf { it.x }; val mny = pts.minOf { it.y }
        floatArrayOf(mnx, mny, pts.maxOf { it.x } - mnx, pts.maxOf { it.y } - mny)
    }

    val fontSize: Float by lazy {
        val v1 = sub(structure[1], structure[0])
        val v2 = sub(structure[3], structure[2])
        min(norm(v2), norm(v1))
    }

    val aspectRatio: Float by lazy {
        val v1 = sub(structure[1], structure[0])
        val v2 = sub(structure[3], structure[2])
        val n1 = norm(v1)
        if (n1 <= 0f) 1f else norm(v2) / n1
    }

    val angle: Float by lazy {
        val v1 = sub(structure[1], structure[0])
        val n1 = norm(v1)
        val cos = if (n1 <= 0f) 1f else (v1.x / n1)
        ((acos(cos.coerceIn(-1f, 1f)) + PI.toFloat()) % PI.toFloat())
    }

    val centroid: Pt by lazy {
        Pt(pts.map { it.x }.average().toFloat(), pts.map { it.y }.average().toFloat())
    }

    val isApproxAxisAligned: Boolean by lazy {
        val v1 = sub(structure[1], structure[0]); val n1 = norm(v1)
        val v2 = sub(structure[3], structure[2]); val n2 = norm(v2)
        if (n1 <= 0f || n2 <= 0f) return@lazy true
        val u1 = Pt(v1.x / n1, v1.y / n1)
        val u2 = Pt(v2.x / n2, v2.y / n2)
        abs(u1.y) < 0.05f || abs(u1.x) < 0.05f || abs(u2.y) < 0.05f || abs(u2.x) < 0.05f
    }

    /** distance_impl（assigned_direction=None → v 模式）：依頂/底邊較近者取對應角點距離。 */
    fun distance(other: GQuad, rho: Float = 0.5f): Float {
        val fs = max(fontSize, other.fontSize)
        val a1 = Geometry.polyArea(Geometry.convexHull(listOf(pts[0], pts[1], other.pts[0], other.pts[1]))) / fs
        val a2 = Geometry.polyArea(Geometry.convexHull(listOf(pts[2], pts[3], other.pts[2], other.pts[3]))) / fs
        var top = true
        if (a1 < fs * rho) top = true
        if (a2 < fs * rho && a2 < a1) top = false
        return if (top) ptDist(pts[0], other.pts[0]) else ptDist(pts[2], other.pts[2])
    }

    private fun sub(a: Pt, b: Pt) = Pt(a.x - b.x, a.y - b.y)
    private fun norm(p: Pt) = hypot(p.x.toDouble(), p.y.toDouble()).toFloat()
    private fun ptDist(a: Pt, b: Pt) = hypot((a.x - b.x).toDouble(), (a.y - b.y).toDouble()).toFloat()
}

/** sort_pnts：用長邊向量定直/橫書，並把 4 點排成 [左上,右上,右下,左下]。對齊 generic.py:sort_pnts。 */
private fun sortPnts(quad: List<Pt>): Pair<List<Pt>, Boolean> {
    val n = quad.size
    val norms = FloatArray(n * n)
    for (i in 0 until n) for (j in 0 until n) {
        val vx = quad[i].x - quad[j].x
        val vy = quad[i].y - quad[j].y
        norms[i * n + j] = hypot(vx.toDouble(), vy.toDouble()).toFloat()
    }
    val order = (0 until n * n).sortedBy { norms[it] }
    val b0 = order[8]; val b1 = order[10]
    var l0x = quad[b0 / n].x - quad[b0 % n].x
    var l0y = quad[b0 / n].y - quad[b0 % n].y
    val l1x = quad[b1 / n].x - quad[b1 % n].x
    val l1y = quad[b1 / n].y - quad[b1 % n].y
    if (l0x * l1x + l0y * l1y < 0f) { l0x = -l0x; l0y = -l0y }
    val sx = abs((l0x + l1x) / 2f)
    val sy = abs((l0y + l1y) / 2f)
    val isV = sx <= sy
    return if (isV) {
        val byY = quad.sortedBy { it.y }
        val first2 = listOf(byY[0], byY[1]).sortedBy { it.x }
        val last2 = listOf(byY[2], byY[3]).sortedBy { it.x }
        Pair(listOf(first2[0], first2[1], last2[1], last2[0]), true)
    } else {
        val byX = quad.sortedBy { it.x }
        val ls = listOf(byX[0], byX[1]).sortedBy { it.y }
        val rs = listOf(byX[2], byX[3]).sortedBy { it.y }
        Pair(listOf(ls[0], rs[0], rs[1], ls[1]), false)
    }
}
