package li.joye.yakuyomi.engine

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

/** 手刻幾何（取代 cv2.minAreaRect / shapely）——分組正確性的地基，純函式好測。 */
class GeometryTest {

    private fun sq(x0: Float, y0: Float, x1: Float, y1: Float) =
        listOf(Pt(x0, y0), Pt(x1, y0), Pt(x1, y1), Pt(x0, y1))

    @Test fun polyArea_square_and_triangle() {
        assertEquals(100f, Geometry.polyArea(sq(0f, 0f, 10f, 10f)), 0.01f)
        assertEquals(50f, Geometry.polyArea(listOf(Pt(0f, 0f), Pt(10f, 0f), Pt(0f, 10f))), 0.01f)
    }

    @Test fun segPointDistance_perpendicular_and_pastEndpoint() {
        // 線段 (0,0)-(10,0)：點 (5,3) → 垂距 3；點 (-4,0) → 落在起點外 → 到 (0,0) = 4
        assertEquals(3f, Geometry.segPointDistance(Pt(0f, 0f), Pt(10f, 0f), Pt(5f, 3f)), 0.01f)
        assertEquals(4f, Geometry.segPointDistance(Pt(0f, 0f), Pt(10f, 0f), Pt(-4f, 0f)), 0.01f)
    }

    @Test fun pointInPoly_insideOutside() {
        val s = sq(0f, 0f, 10f, 10f)
        assertTrue(Geometry.pointInPoly(s, Pt(5f, 5f)))
        assertFalse(Geometry.pointInPoly(s, Pt(15f, 5f)))
    }

    @Test fun polyDistance_gap_overlap_touch() {
        val a = sq(0f, 0f, 10f, 10f)
        assertEquals(3f, Geometry.polyDistance(a, sq(13f, 0f, 23f, 10f)), 0.01f) // 間隔 3
        assertEquals(0f, Geometry.polyDistance(a, sq(5f, 0f, 15f, 10f)), 0.01f)  // 重疊
        assertEquals(0f, Geometry.polyDistance(a, sq(10f, 0f, 20f, 10f)), 0.01f) // 共邊相切
    }

    @Test fun convexHull_dropsInteriorPoint() {
        val hull = Geometry.convexHull(sq(0f, 0f, 10f, 10f) + Pt(5f, 5f))
        assertEquals(4, hull.size)
        assertFalse(hull.contains(Pt(5f, 5f)))
    }

    @Test fun minAreaRect_axisAlignedRect() {
        val r = Geometry.minAreaRect(sq(0f, 0f, 20f, 10f))!!
        val dims = listOf(r.w, r.h).sorted()
        assertEquals(10f, dims[0], 0.5f)
        assertEquals(20f, dims[1], 0.5f)
    }
}
