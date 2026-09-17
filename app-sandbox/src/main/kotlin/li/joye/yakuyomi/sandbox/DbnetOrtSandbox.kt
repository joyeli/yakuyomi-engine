package li.joye.yakuyomi.sandbox

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.graphics.Bitmap
import li.joye.yakuyomi.engine.Detection
import li.joye.yakuyomi.engine.Detector
import java.nio.FloatBuffer

/**
 * DBNet 的 ONNX Runtime 版偵測器，給 int8 量化模型用（上機驗證）。
 *
 * 產品路徑跑的是 NCNN fp16（145.9 MB）。靜態量化（QDQ、per-channel）把模型壓到 73.4 MB 且桌面
 * 快 1.78 倍，但那是 ONNX 格式，要另一條推論路徑才跑得起來。要不要把偵測搬回 ORT，得先有真機數字。
 *
 * **前後處理一律借用 [Detector] 的 companion**，換的只有中間那一次前向。否則前處理差一點，後面
 * 每個門檻都會歪，A/B 就變成在比兩套實作而不是比量化。
 *
 * 這個類別住在 sandbox 而不是 `:nightread-ort`，因為它同時要看到 `:engine`（前後處理）與 ORT，
 * 而夜讀核心刻意不依賴引擎。
 */
class DbnetOrtSandbox(modelPath: String, intraOpThreads: Int = 4) : AutoCloseable {

    private val env: OrtEnvironment = OrtEnvironment.getEnvironment()
    private val session: OrtSession = env.createSession(
        modelPath,
        OrtSession.SessionOptions().apply { setIntraOpNumThreads(intraOpThreads) },
    )

    fun detect(page: Bitmap): Detection {
        val input = Detector.preprocess(page)
        val name = session.inputNames.first()
        OnnxTensor.createTensor(
            env,
            FloatBuffer.wrap(input.chw),
            longArrayOf(1, 3, input.h.toLong(), input.w.toLong()),
        ).use { tensor ->
            session.run(mapOf(name to tensor)).use { res ->
                @Suppress("UNCHECKED_CAST")
                val dbOut = (res[0].value as Array<Array<Array<FloatArray>>>)[0]    // [2, h, w]
                @Suppress("UNCHECKED_CAST")
                val maskOut = (res[1].value as Array<Array<Array<FloatArray>>>)[0]  // [1, mh, mw]

                val area = input.w * input.h
                val db = FloatArray(2 * area)
                for (ch in 0 until 2) {
                    for (y in 0 until input.h) {
                        System.arraycopy(dbOut[ch][y], 0, db, ch * area + y * input.w, input.w)
                    }
                }
                val mh = maskOut[0].size
                val mw = maskOut[0][0].size
                val mask = FloatArray(mw * mh)
                for (y in 0 until mh) System.arraycopy(maskOut[0][y], 0, mask, y * mw, mw)

                return Detector.postprocess(db, mask, mw, mh, input, page.width, page.height)
            }
        }
    }

    override fun close() = session.close()
}
