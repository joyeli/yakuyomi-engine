package li.joye.yakuyomi.engine

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Test

/** ModelSet.resolve：把「哪個檔是哪顆模型」的命名比對收進引擎（三顆全 NCNN；OCR 兩份 param 共用一份 bin、優先原版）。 */
class ModelSetTest {

    @Test fun resolvesNcnn() {
        val m = ModelSet.resolve(
            listOf(
                "dbnet_detect.ncnn.param" to "/m/det.param",
                "dbnet_detect.ncnn.bin" to "/m/det.bin",
                "ocr_48px_ctc.ncnn.param" to "/m/ocr.param",
                "ocr_48px_ctc.ncnn.bin" to "/m/ocr.bin",
                "mit_aot_fixed512.ncnn.param" to "/m/aot.param",
                "mit_aot_fixed512.ncnn.bin" to "/m/aot.bin",
            ),
        )!!
        assertEquals("/m/det.param", m.detectorNcnn)
        assertEquals("/m/ocr.param", m.ocr)
        assertEquals("/m/aot.param", m.aotInpainterNcnn)
    }

    /** 原版與 mixed 都在（models-v4 兩份 param 共用 bin）→ 回原版，不管清單順序；能不能用 mixed 由 Ocr.pickParam 決定。 */
    @Test fun prefersBaseOverMixed() {
        val m = ModelSet.resolve(
            listOf(
                "ocr_48px_ctc_mixed.ncnn.param" to "/m/ocr_mixed.param", // mixed 排前面也不該被選到
                "ocr_48px_ctc.ncnn.param" to "/m/ocr.param",
                "ocr_48px_ctc.ncnn.bin" to "/m/ocr.bin",
                "dbnet_detect.ncnn.param" to "/m/det.param",
                "mit_aot_fixed512.ncnn.param" to "/m/aot.param",
            ),
        )!!
        assertEquals("/m/ocr.param", m.ocr)
    }

    /** 只有 mixed 版也接受（不擋在 resolve；缺原版時 Ocr 建構會明講）。 */
    @Test fun acceptsMixedOnly() {
        val m = ModelSet.resolve(
            listOf(
                "ocr_48px_ctc_mixed.ncnn.param" to "/m/ocr_mixed.param",
                "dbnet_detect.ncnn.param" to "/m/det.param",
                "mit_aot_fixed512.ncnn.param" to "/m/aot.param",
            ),
        )!!
        assertEquals("/m/ocr_mixed.param", m.ocr)
    }

    @Test fun caseInsensitive() {
        val m = ModelSet.resolve(listOf("DBNET.PARAM" to "d", "OCR.Param" to "o", "AOT.param" to "a"))!!
        assertEquals("d", m.detectorNcnn)
        assertEquals("o", m.ocr)
        assertEquals("a", m.aotInpainterNcnn)
    }

    @Test fun nullWhenAnyMissing() {
        assertNull(ModelSet.resolve(listOf("ocr.param" to "o", "aot.param" to "a")))   // 缺偵測
        assertNull(ModelSet.resolve(listOf("dbnet.param" to "d", "aot.param" to "a"))) // 缺 ocr
        assertNull(ModelSet.resolve(listOf("dbnet.param" to "d", "ocr.param" to "o"))) // 缺去字
        assertNull(ModelSet.resolve(emptyList()))
    }

    /** ORT 已拔除：`.onnx` 不再算 OCR 模型（只有 .bin 也不算，要 .param）。 */
    @Test fun onnxNotAccepted() {
        assertNull(ModelSet.resolve(listOf("dbnet.param" to "d", "ocr_int8.onnx" to "o", "aot.param" to "a")))
        assertNull(ModelSet.resolve(listOf("dbnet.param" to "d", "ocr_48px_ctc.ncnn.bin" to "o", "aot.param" to "a")))
    }

    /** 夜讀兩顆選配（models-v5 真實檔名）：都在 → 兩欄都填、.bin 不算；翻譯三顆不受夜讀檔干擾。 */
    @Test fun resolvesCharSeg() {
        val m = ModelSet.resolve(
            listOf(
                "dbnet_detect.ncnn.param" to "/m/det.param",
                "ocr_48px_ctc.ncnn.param" to "/m/ocr.param",
                "mit_aot_fixed512.ncnn.param" to "/m/aot.param",
                "manga_seg_s.ncnn.param" to "/m/yolo.param",
                "manga_seg_s.ncnn.bin" to "/m/yolo.bin",
                "cartoonseg.ncnn.param" to "/m/cseg.param",
                "cartoonseg.ncnn.bin" to "/m/cseg.bin",
            ),
        )!!
        assertEquals("/m/yolo.param", m.charSegYoloNcnn)
        assertEquals("/m/cseg.param", m.charSegCsegNcnn)
        assertEquals("/m/det.param", m.detectorNcnn)
        assertEquals("/m/ocr.param", m.ocr)
        assertEquals("/m/aot.param", m.aotInpainterNcnn)
    }

    /** 缺夜讀模型不影響翻譯就緒：resolve 仍成功、兩欄 null；只有一顆也接受（NightReadRenderer 會用單顆）。 */
    @Test fun charSegOptional() {
        val base = listOf("dbnet.param" to "d", "ocr.param" to "o", "aot.param" to "a")
        val none = ModelSet.resolve(base)!!
        assertNull(none.charSegYoloNcnn)
        assertNull(none.charSegCsegNcnn)
        val yoloOnly = ModelSet.resolve(base + ("manga_seg_s.ncnn.param" to "y"))!!
        assertEquals("y", yoloOnly.charSegYoloNcnn)
        assertNull(yoloOnly.charSegCsegNcnn)
        val csegOnly = ModelSet.resolve(base + ("cartoonseg.ncnn.param" to "c"))!!
        assertNull(csegOnly.charSegYoloNcnn)
        assertEquals("c", csegOnly.charSegCsegNcnn)
    }

    /**
     * 關鍵字不互撞：夜讀檔名（manga_seg／cartoonseg）不含 dbnet/ocr/aot——只有夜讀檔時翻譯三顆仍缺 → null、
     * 不會被誤認成任何翻譯角色；翻譯檔名（含 mixed OCR）也不含夜讀關鍵字。
     */
    @Test fun charSegKeysDoNotCollide() {
        assertNull(ModelSet.resolve(listOf("manga_seg_s.ncnn.param" to "y", "cartoonseg.ncnn.param" to "c")))
        val m = ModelSet.resolve(
            listOf(
                "dbnet_detect.ncnn.param" to "d",
                "ocr_48px_ctc_mixed.ncnn.param" to "o",
                "mit_aot_fixed512.ncnn.param" to "a",
            ),
        )!!
        assertNull(m.charSegYoloNcnn)
        assertNull(m.charSegCsegNcnn)
    }
}
