package li.joye.yakuyomi.engine

/**
 * NCNN 推論後端（P1 去字 + P2 偵測）。OCR 留在 ORT（NCNN 把 transformer 相對位置編碼寫死在轉換寬度、native 寬會壞）。
 *
 * handle-based：`createNet` 載入 pnnx 轉出的 .param/.bin 回一個 native handle（ncnn::Net*），pipeline 每頁復用、
 * 收工 `releaseNet`。模型換到手機 CPU 的 NEON/Winograd → 偵測實測比 ORT-XNNPACK 快 ~3.7×（見 memory litert-gpu-blocked）。
 */
internal object NcnnBackend {
    /** 原生庫是否載得起來（缺 .so / 非 arm64 → false，呼叫端可退回 ORT）。 */
    val available: Boolean = try {
        System.loadLibrary("yakuyomi_ncnn")
        true
    } catch (t: Throwable) {
        false
    }

    /** 載入 .param/.bin，回 native handle（純 CPU；NEON/Winograd）；0=失敗。★ 不改此 external 名（JNI 符號 = Java_..._createNet，改名會 UnsatisfiedLinkError）。 */
    external fun createNet(paramPath: String, binPath: String): Long

    external fun releaseNet(handle: Long)

    /**
     * ★ 全域鎖：**序列化所有 ncnn 原生推論**（detect + 去字）。
     *
     * ncnn 內部用 **OpenMP**（libomp 靜態連進 libyakuyomi_ncnn.so）做卷積平行化。多個 app 執行緒**同時**進入
     * ncnn forward（跨頁併發把 detect/去字 派到多個 Dispatchers.Default 緒）→ 各自開 OpenMP parallel region →
     * OpenMP 全域 runtime 不容許多個並發 master → **`__kmp_abort_process` 直接 abort 行程（SIGABRT）**。
     * 真機 tombstone 實證（thread=DefaultDispatch, #01 __kmp_abort_process），2026-07-14。
     *
     * detect 與 去字共用同一把鎖（同一個 ncnn OpenMP runtime，任兩個並發的 parallel region 都會撞）。
     * OCR 走 ORT（另一個 .so、自有執行緒模型）不受此鎖，翻譯走網路 → 併發保留。detect/去字 皆 CPU-bound，
     * 序列化幾乎不損吞吐（本就塞在翻譯的網路等待窗內、CPU 也無法真的同時跑兩份）。
     */
    private val ncnnLock = Any()

    private external fun detectDbnetNative(handle: Long, chw: FloatArray, inW: Int, inH: Int, db: FloatArray, mask: FloatArray): Int

    private external fun inpaintAotNative(handle: Long, img: FloatArray, mask: FloatArray, s: Int, out: FloatArray): Int

    private external fun extractNative(handle: Long, chw: FloatArray, inW: Int, inH: Int, inC: Int, outNames: Array<String>, outs: Array<FloatArray>): Int

    /** DBNet 偵測（矩形 resize_aspect 輸入，繞開正方形 832-992 crash 帶）：chw=[3,inH,inW] → db 填 [2*inW*inH]（raw logits 2ch 全解析）、mask 填 [(inW/2)*(inH/2)]（已 sigmoid 半解析）。回 mask.h（>0=OK）。序列化（見 [ncnnLock]）。 */
    fun detectDbnet(handle: Long, chw: FloatArray, inW: Int, inH: Int, db: FloatArray, mask: FloatArray): Int {
        EngineTrace.log("ncnn.detectDbnet.enter ${inW}x$inH")
        return synchronized(ncnnLock) {
            EngineTrace.log("ncnn.detectDbnet.call ${inW}x$inH")
            val rc = detectDbnetNative(handle, chw, inW, inH, db, mask)
            EngineTrace.log("ncnn.detectDbnet.exit rc=$rc")
            rc
        }
    }

    /** 去字 AOT：img=NCHW[3,s,s]（[-1,1] holes-zeroed）+ mask=[s*s] → out 填 [3*s*s]（[-1,1]）。回 0=OK。序列化（見 [ncnnLock]）。 */
    fun inpaintAot(handle: Long, img: FloatArray, mask: FloatArray, s: Int, out: FloatArray): Int {
        EngineTrace.log("ncnn.inpaint.enter s=$s")
        return synchronized(ncnnLock) {
            EngineTrace.log("ncnn.inpaint.call s=$s")
            val rc = inpaintAotNative(handle, img, mask, s, out)
            EngineTrace.log("ncnn.inpaint.exit rc=$rc")
            rc
        }
    }

    /**
     * 通用抽取（後處理在 Kotlin 的模型，如人物分割）：chw=[inC,inH,inW] 進 in0，依 [outNames] 抽出各 blob、
     * 逐 channel 複製進 [outs]（每個陣列大小要等於該 blob 的 w×h×c）。回 0=OK、-2 大小不合、-3 抽取失敗。序列化（見 [ncnnLock]）。
     */
    fun extract(handle: Long, chw: FloatArray, inW: Int, inH: Int, inC: Int, outNames: Array<String>, outs: Array<FloatArray>): Int {
        EngineTrace.log("ncnn.extract.enter ${inW}x$inH x$inC → ${outNames.size} blobs")
        return synchronized(ncnnLock) {
            val rc = extractNative(handle, chw, inW, inH, inC, outNames, outs)
            EngineTrace.log("ncnn.extract.exit rc=$rc")
            rc
        }
    }
}
