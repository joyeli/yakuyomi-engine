package li.joye.yakuyomi.engine

import android.graphics.Bitmap
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import java.lang.reflect.Modifier

/**
 * 借出偵測器的 API（[TranslationEngine.detectorModelPath]／[TranslationEngine.detectWith]）。建 [Pipeline] 要原生，JVM 上
 * 只守結構：介面預設＝不借（路徑 null，呼叫端據此不共用），[Yakuyomi.create] 回的唯一實作 [Pipeline] 兩個都覆寫——少覆寫
 * 一個，fork 會默默永遠不共用（拿到 null）或一呼叫就拋。委派本身（`detector.detect(page, cfg)`）與 cfg 的用法見
 * inference-core 的 DetectorCfgOverloadTest；真機由 fork 的一次性比對（verify=same）驗。
 */
class DetectorSharingApiTest {

    @Test
    fun interfaceDefaultDoesNotLend() {
        val engine = object : TranslationEngine {
            override suspend fun translatePage(page: Bitmap): PageResult = PageResult.Failed("unused")
            override fun warmUp() = Unit
            override fun close() = Unit
        }
        assertNull(engine.detectorModelPath)
    }

    @Test
    fun pipelineOverridesBothMembers() {
        val detectWith = Pipeline::class.java.getDeclaredMethod(
            "detectWith",
            Bitmap::class.java,
            DetectorConfig::class.java,
        )
        assertEquals(Detection::class.java, detectWith.returnType)
        assertTrue(Modifier.isPublic(detectWith.modifiers))
        val path = Pipeline::class.java.getDeclaredMethod("getDetectorModelPath")
        assertEquals(String::class.java, path.returnType)
    }

    @Test
    fun detectorExposesItsModelPath() {
        // 共用的判準＝兩邊 `.param` 路徑相同：Detector 要把建構時的路徑原樣公開
        val getter = Detector::class.java.getDeclaredMethod("getModelPath")
        assertEquals(String::class.java, getter.returnType)
        assertTrue(Modifier.isPublic(getter.modifiers))
    }
}
