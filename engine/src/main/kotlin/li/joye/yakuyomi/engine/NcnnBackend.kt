package li.joye.yakuyomi.engine

/**
 * NCNN 推論後端（去字 + 偵測 + 人物分割 + OCR）。
 *
 * handle-based：`createNet` 載入 pnnx 轉出的 .param/.bin 回一個 native handle（ncnn::Net*），pipeline 每頁復用、
 * 收工 `releaseNet`。模型換到手機 CPU 的 NEON/Winograd → 偵測實測比 ORT-XNNPACK 快 ~3.7×（見 memory litert-gpu-blocked）。
 * OCR 原本卡在「transformer 位置編碼被 trace 烤死在轉換寬度」的寬度牆，2026-09-21 改把 PE 當第二個輸入餵進去
 * （parity/export_ocr_ncnn.py）後也搬過來；[ocrCtc] 的並發規則見其註解。
 */
internal object NcnnBackend {
    /** 原生庫是否載得起來（缺 .so / 非 arm64 → false；三顆模型全 NCNN、沒有備援，呼叫端只能報錯）。 */
    val available: Boolean = try {
        System.loadLibrary("yakuyomi_ncnn")
        true
    } catch (t: Throwable) {
        false
    }

    /** 載入 .param/.bin，回 native handle（純 CPU；NEON/Winograd）；0=失敗。★ 不改此 external 名（JNI 符號 = Java_..._createNet，改名會 UnsatisfiedLinkError）。 */
    external fun createNet(paramPath: String, binPath: String): Long

    /**
     * [createNet] 加選項：numThreads（>0 才設；OCR 並發模式設 1）、fp16Storage／fp16Arith 分開控制半精度
     * （storage=權重與中間值存 fp16、arith=用 fp16 指令算；transformer 若半精度算術讀錯字，可只留 storage）。
     */
    external fun createNetEx(paramPath: String, binPath: String, numThreads: Int, fp16Storage: Boolean, fp16Arith: Boolean): Long

    external fun releaseNet(handle: Long)

    private external fun cpuSupportsFp16Native(): Boolean

    /** 這顆 CPU 有 fp16 storage/arithmetic（arm82 asimdhp）；OCR 混合精度 param 只在此為 true 時才能用（見 [Ocr]）。 */
    val cpuSupportsFp16: Boolean by lazy { available && runCatching { cpuSupportsFp16Native() }.getOrDefault(false) }

    /**
     * ★ 全域鎖：**序列化所有 ncnn 原生推論**（detect + 去字）。
     *
     * ncnn 內部用 **OpenMP**（libomp 靜態連進 libyakuyomi_ncnn.so）做卷積平行化。多個 app 執行緒**同時**進入
     * ncnn forward（跨頁併發把 detect/去字 派到多個 Dispatchers.Default 緒）→ 各自開 OpenMP parallel region →
     * OpenMP 全域 runtime 不容許多個並發 master → **`__kmp_abort_process` 直接 abort 行程（SIGABRT）**。
     * 真機 tombstone 實證（thread=DefaultDispatch, #01 __kmp_abort_process），2026-07-14。
     *
     * detect 與 去字共用同一把鎖（同一個 ncnn OpenMP runtime，任兩個並發的 parallel region 都會撞）。
     * OCR 走 [ocrCtc]（1 緒 Net、SimpleOMP inline，見該註解）不進此鎖，翻譯走網路 → 併發保留。detect/去字 皆 CPU-bound，
     * 序列化幾乎不損吞吐（本就塞在翻譯的網路等待窗內、CPU 也無法真的同時跑兩份）。
     */
    private val ncnnLock = Any()

    private external fun detectDbnetNative(handle: Long, chw: FloatArray, inW: Int, inH: Int, db: FloatArray, mask: FloatArray): Int

    private external fun inpaintAotNative(handle: Long, img: FloatArray, mask: FloatArray, s: Int, out: FloatArray): Int

    private external fun extractNative(handle: Long, chw: FloatArray, inW: Int, inH: Int, inC: Int, outNames: Array<String>, outs: Array<FloatArray>): Int

    private external fun ocrCtcNative(handle: Long, chw: FloatArray, w: Int, h: Int, pe: FloatArray, t: Int, idx: IntArray, logp: FloatArray): Int

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

    /**
     * 48px CTC OCR 單條：chw=[3,h,w]（h=48）+ pe=正弦位置表（至少 t×320，只讀前 t 列）→ JNI 內算完 argmax 與
     * top-1 log_softmax，填 idx[t]、logp[t]。回 t（>0=OK）；負值見 ncnn_jni.cpp。
     *
     * [serialize]=false 時**不進 [ncnnLock]**：OCR 的 Net 以 num_threads=1 建（[Ocr] 並發模式），SimpleOMP 對
     * num_threads==1 的 parallel region 走 inline（simpleomp.cpp `__kmpc_fork_call`：不碰共用 task queue、全域初始化
     * 用 pthread_once）→ 多條 strip 可同時 forward，也不會與持鎖中的偵測/去字（走 task queue）互撞。這是 OCR 保住
     * 「8 行並發快 46%」的前提（ncnnLock 當初是為 libomp 的並發 master abort 加的，SimpleOMP 後偵測/去字仍保守持鎖）。
     * 非並發模式（num_threads>1）照舊序列化。
     */
    fun ocrCtc(handle: Long, chw: FloatArray, w: Int, h: Int, pe: FloatArray, t: Int, idx: IntArray, logp: FloatArray, serialize: Boolean): Int {
        return if (serialize) {
            synchronized(ncnnLock) { ocrCtcNative(handle, chw, w, h, pe, t, idx, logp) }
        } else {
            ocrCtcNative(handle, chw, w, h, pe, t, idx, logp)
        }
    }
}
