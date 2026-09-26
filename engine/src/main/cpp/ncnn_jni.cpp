// Yakuyomi 引擎的 NCNN 推論橋（去字 + 偵測 + 人物分割 + OCR）。handle-based：模型 createNet 載一次、
// pipeline 每頁復用同一 ncnn::Net（不像 sandbox benchmark 每次重載）。
#include <jni.h>
#include <android/log.h>
#include <cmath>
#include <cstring>
#include "net.h"
#include "cpu.h"

#define LOGW(...) __android_log_print(ANDROID_LOG_WARN, "yakuyomi_ncnn", __VA_ARGS__)

// 載 .param/.bin 進 net；失敗 delete 並回 0。
static jlong loadNet(JNIEnv* env, ncnn::Net* net, jstring paramPath, jstring binPath) {
    const char* pp = env->GetStringUTFChars(paramPath, nullptr);
    const char* bp = env->GetStringUTFChars(binPath, nullptr);
    int r1 = net->load_param(pp);
    int r2 = net->load_model(bp);
    env->ReleaseStringUTFChars(paramPath, pp);
    env->ReleaseStringUTFChars(binPath, bp);
    if (r1 != 0 || r2 != 0) {
        LOGW("ncnn load fail param=%d model=%d", r1, r2);
        delete net;
        return 0;
    }
    return (jlong) net;
}

// 載入 pnnx 轉出的 .param/.bin，回 native handle（ncnn::Net*，純 CPU、ncnn 預設選項）；0=失敗。
// （GPU/Vulkan 已移除：NCNN Vulkan 實測算不對 AOT-GAN，見 memory ncnn-vulkan-fp16。）
extern "C" JNIEXPORT jlong JNICALL
Java_li_joye_yakuyomi_engine_NcnnBackend_createNet(
        JNIEnv* env, jobject, jstring paramPath, jstring binPath) {
    return loadNet(env, new ncnn::Net(), paramPath, binPath);
}

// createNet 加選項：numThreads（>0 才設；OCR 逐條並發時設 1——SimpleOMP 對 num_threads==1 的 parallel region 走 inline、
// 不碰共用 task queue，多條 strip 才能同時 forward 而不必進 Kotlin 端的 ncnnLock）；fp16Storage／fp16Arith 分開關：
// storage 關＝權重載入時轉 fp32、中間值 fp32；arith 關＝不用 fp16 指令算（ncnn 的 fp16s 路徑：存半精度、算單精度）。
extern "C" JNIEXPORT jlong JNICALL
Java_li_joye_yakuyomi_engine_NcnnBackend_createNetEx(
        JNIEnv* env, jobject, jstring paramPath, jstring binPath, jint numThreads, jboolean fp16Storage, jboolean fp16Arith) {
    ncnn::Net* net = new ncnn::Net();
    if (numThreads > 0) net->opt.num_threads = numThreads;
    if (!fp16Storage) {
        net->opt.use_fp16_packed = false;
        net->opt.use_fp16_storage = false;
        net->opt.use_bf16_storage = false;
    }
    if (!fp16Arith) net->opt.use_fp16_arithmetic = false;
    return loadNet(env, net, paramPath, binPath);
}

extern "C" JNIEXPORT void JNICALL
Java_li_joye_yakuyomi_engine_NcnnBackend_releaseNet(JNIEnv*, jobject, jlong handle) {
    if (handle) delete (ncnn::Net*) handle;
}

// 這顆 CPU 有沒有 fp16 storage/arithmetic（arm82 asimdhp）。ncnn 的 fp16 路徑（convert_layout、各 arm 層的
// support_fp16_storage）全看這個；OCR 的混合精度 param 裡的 Cast 層宣告「進來的是 fp16」，沒 asimdhp 的機器 blob 是 fp32、
// Cast 會靜默吐垃圾 → Kotlin 端據此退回原 param。
extern "C" JNIEXPORT jboolean JNICALL
Java_li_joye_yakuyomi_engine_NcnnBackend_cpuSupportsFp16Native(JNIEnv*, jobject) {
#if __aarch64__
    return ncnn::cpu_support_arm_asimdhp() ? JNI_TRUE : JNI_FALSE;
#else
    return JNI_FALSE;
#endif
}

// DBNet 偵測（m-i-t default 偵測器）：in0[3,s,s] → out0=db[2,s,s]（raw logits，全解析度）+ out1=mask[1,s/2,s/2]（已 sigmoid，半解析度）。
// ★ out1 半/全解析度平台不定（x86 半、arm64 全）→ 動態讀 mat 的 .w/.h/.c 決定複製量，絕不寫死 s*s（否則越界）。
// 填 dbOut[2*s*s]（逐 channel db.w*db.h）+ maskOut[(s/2)*(s/2)]。回傳 mask.h（>0=OK，Kotlin 據此驗半解析度假設）。
extern "C" JNIEXPORT jint JNICALL
Java_li_joye_yakuyomi_engine_NcnnBackend_detectDbnetNative(
        JNIEnv* env, jobject, jlong handle,
        jfloatArray chw, jint inW, jint inH, jfloatArray dbOut, jfloatArray maskOut) {
    if (!handle) return -1;
    ncnn::Net* net = (ncnn::Net*) handle;

    size_t area = (size_t) inW * inH;
    jfloat* in = env->GetFloatArrayElements(chw, nullptr);
    ncnn::Mat inMat(inW, inH, 3);   // 矩形（resize_aspect pad256）：繞開正方形 832-992 crash 帶
    for (int c = 0; c < 3; c++) {
        memcpy(inMat.channel(c), in + (size_t) c * area, sizeof(float) * area);
    }
    env->ReleaseFloatArrayElements(chw, in, JNI_ABORT);

    ncnn::Mat db, mask;
    ncnn::Extractor ex = net->create_extractor();
    ex.input("in0", inMat);
    ex.extract("out0", db);
    ex.extract("out1", mask);
    LOGW("dbnet in=%dx%d out0=%dx%dx%d out1=%dx%dx%d", inW, inH, db.w, db.h, db.c, mask.w, mask.h, mask.c);

    // ★ 防越界：db 緩衝＝2*inW*inH、mask 緩衝＝inW*inH（全解析上限，因 mask 半/全解析平台不定）。超出回負碼（變 exception 不 crash）。
    if ((size_t) db.c * db.w * db.h > 2 * area || (size_t) mask.w * mask.h > area) {
        return -(db.w * 1000 + mask.w);
    }
    // 空輸出（forward 出問題或 blob 名不符）也回報。
    if (db.w == 0 || mask.w == 0) return -1;

    // db：全解析度、2 channel（ch0=logits、ch1 threshold-map 不用；仍複製兩通道與緩衝對齊）。逐 channel 用 mat 實際 w*h。
    jfloat* od = env->GetFloatArrayElements(dbOut, nullptr);
    for (int c = 0; c < db.c && c < 2; c++) {
        memcpy(od + (size_t) c * db.w * db.h, db.channel(c), sizeof(float) * db.w * db.h);
    }
    env->ReleaseFloatArrayElements(dbOut, od, 0);

    // mask：半解析度、1 channel。用 mat 實際尺寸複製（絕不假設 s*s，否則越界）。
    jfloat* om = env->GetFloatArrayElements(maskOut, nullptr);
    memcpy(om, mask.channel(0), sizeof(float) * mask.w * mask.h);
    env->ReleaseFloatArrayElements(maskOut, om, 0);

    return mask.w * 10000 + mask.h; // 回實際 mask 尺寸（半/全解析平台不定，Kotlin 解 mw=rc/10000 mh=rc%10000）
}

// 去字 AOT：in0=img[3,s,s]（[-1,1] holes-zeroed）+ in1=mask[1,s,s] → out0[3,s,s]（[-1,1]）。填 outArr[3*s*s]。
extern "C" JNIEXPORT jint JNICALL
Java_li_joye_yakuyomi_engine_NcnnBackend_inpaintAotNative(
        JNIEnv* env, jobject, jlong handle,
        jfloatArray img, jfloatArray mask, jint s, jfloatArray outArr) {
    if (!handle) return -1;
    ncnn::Net* net = (ncnn::Net*) handle;

    jfloat* pi = env->GetFloatArrayElements(img, nullptr);
    jfloat* pm = env->GetFloatArrayElements(mask, nullptr);
    ncnn::Mat imgMat(s, s, 3);
    for (int c = 0; c < 3; c++) {
        memcpy(imgMat.channel(c), pi + (size_t) c * s * s, sizeof(float) * s * s);
    }
    ncnn::Mat maskMat(s, s, 1);
    memcpy(maskMat.channel(0), pm, sizeof(float) * s * s);
    env->ReleaseFloatArrayElements(img, pi, JNI_ABORT);
    env->ReleaseFloatArrayElements(mask, pm, JNI_ABORT);

    ncnn::Mat out;
    ncnn::Extractor ex = net->create_extractor();
    ex.input("in0", imgMat);
    ex.input("in1", maskMat);
    ex.extract("out0", out);

    jfloat* o = env->GetFloatArrayElements(outArr, nullptr);
    for (int c = 0; c < 3; c++) {
        memcpy(o + (size_t) c * s * s, out.channel(c), sizeof(float) * s * s);
    }
    env->ReleaseFloatArrayElements(outArr, o, 0);
    return 0;
}

// 通用抽取（人物分割等「後處理在 Kotlin」的模型）：in0=[c,h,w] → 依 outNames 逐一 extract，
// 各 blob 逐 channel 複製進 outs[i]（Java 陣列大小必須 == w*h*c，否則回 -2；extract 失敗回 -3）。
// ★ ncnn::Mat 每個 channel 有對齊 padding（cstep），不能整塊 memcpy，要 m.channel(c) 逐 channel 搬。
extern "C" JNIEXPORT jint JNICALL
Java_li_joye_yakuyomi_engine_NcnnBackend_extractNative(
        JNIEnv* env, jobject, jlong handle,
        jfloatArray chw, jint inW, jint inH, jint inC,
        jobjectArray outNames, jobjectArray outs) {
    if (!handle) return -1;
    ncnn::Net* net = (ncnn::Net*) handle;

    size_t area = (size_t) inW * inH;
    jfloat* in = env->GetFloatArrayElements(chw, nullptr);
    ncnn::Mat inMat(inW, inH, inC);
    for (int c = 0; c < inC; c++) {
        memcpy(inMat.channel(c), in + (size_t) c * area, sizeof(float) * area);
    }
    env->ReleaseFloatArrayElements(chw, in, JNI_ABORT);

    ncnn::Extractor ex = net->create_extractor();
    ex.input("in0", inMat);
    int n = env->GetArrayLength(outNames);
    for (int i = 0; i < n; i++) {
        jstring jname = (jstring) env->GetObjectArrayElement(outNames, i);
        const char* name = env->GetStringUTFChars(jname, nullptr);
        ncnn::Mat m;
        int rc = ex.extract(name, m);
        if (rc != 0) {
            LOGW("extract %s fail rc=%d", name, rc);
            env->ReleaseStringUTFChars(jname, name);
            return -3;
        }
        jfloatArray arr = (jfloatArray) env->GetObjectArrayElement(outs, i);
        size_t need = (size_t) m.w * m.h * m.c;
        if ((size_t) env->GetArrayLength(arr) != need) {
            LOGW("extract %s size mismatch: blob %dx%dx%d=%zu java=%d", name, m.w, m.h, m.c, need, env->GetArrayLength(arr));
            env->ReleaseStringUTFChars(jname, name);
            return -2;
        }
        env->ReleaseStringUTFChars(jname, name);
        jfloat* out = env->GetFloatArrayElements(arr, nullptr);
        size_t plane = (size_t) m.w * m.h;
        for (int c = 0; c < m.c; c++) {
            memcpy(out + (size_t) c * plane, m.channel(c), sizeof(float) * plane);
        }
        env->ReleaseFloatArrayElements(arr, out, 0);
        env->DeleteLocalRef(arr);
        env->DeleteLocalRef(jname);
    }
    return 0;
}

// 48px CTC OCR（parity/export_ocr_ncnn.py 的 blob 契約）：in0=[3,48,W]（(x−127.5)/127.5、右側白邊由 Kotlin 補）+
// in1=PE[T,320]（T=floor(W/4)−1；正弦表由 Kotlin 算好整張傳進來、這裡只讀前 T 列）→ out0=char_logits[T,D]。
// D=19264 一列列搬回 JVM 太重，這裡逐時步算 argmax 與 top-1 log_softmax（= v_max − log Σ exp(v − v_max)，同 Kotlin
// ORT 路徑的 ctcDecodeArr），只填 idx[T]、logp[T]；收合重複／查字表在 Kotlin。回 T（>0）；-1 無 handle、
// -2 idx/logp 緩衝小於 T、-3 抽取失敗、-4 輸出形狀不是 [T,D]（T 與 Kotlin 算的不合＝t_of_w 假設破了）。
extern "C" JNIEXPORT jint JNICALL
Java_li_joye_yakuyomi_engine_NcnnBackend_ocrCtcNative(
        JNIEnv* env, jobject, jlong handle,
        jfloatArray chw, jint w, jint h, jfloatArray pe, jint t,
        jintArray idxOut, jfloatArray logpOut) {
    if (!handle) return -1;
    ncnn::Net* net = (ncnn::Net*) handle;
    if (env->GetArrayLength(idxOut) < t || env->GetArrayLength(logpOut) < t) return -2;

    size_t area = (size_t) w * h;
    jfloat* in = env->GetFloatArrayElements(chw, nullptr);
    ncnn::Mat inMat(w, h, 3);
    for (int c = 0; c < 3; c++) {
        memcpy(inMat.channel(c), in + (size_t) c * area, sizeof(float) * area);
    }
    env->ReleaseFloatArrayElements(chw, in, JNI_ABORT);

    const int D_MODEL = 320;
    jfloat* pp = env->GetFloatArrayElements(pe, nullptr);
    ncnn::Mat peMat(D_MODEL, t);                       // 2-D：w=320、h=T（c=1 無 cstep 問題、可整塊搬）
    memcpy(peMat.data, pp, sizeof(float) * D_MODEL * t);
    env->ReleaseFloatArrayElements(pe, pp, JNI_ABORT);

    ncnn::Mat out;
    ncnn::Extractor ex = net->create_extractor();
    ex.input("in0", inMat);
    ex.input("in1", peMat);
    if (ex.extract("out0", out) != 0) return -3;
    // out0 應為 2-D [T,D]（pnnx 去掉 batch）；容忍 c=1 的 3-D。elemsize 必須是 4（fp32）——extract 預設會 cast 回 fp32，
    // 但若日後 mask 範圍變動讓 char_pred 吐 fp16、這裡當 float 讀就是垃圾，先擋。
    if (!((out.dims == 2) || (out.dims == 3 && out.c == 1)) || out.h != t || out.w <= 0 || out.elemsize != 4) {
        LOGW("ocr out0 shape mismatch: dims=%d w=%d h=%d c=%d elemsize=%zu expect h=%d fp32", out.dims, out.w, out.h, out.c, out.elemsize, t);
        return -4;
    }
    const int D = out.w;
    const float* base = (const float*) out.data;       // dims 2／3(c=1) 皆為 w*h 連續
    jint* oi = env->GetIntArrayElements(idxOut, nullptr);
    jfloat* ol = env->GetFloatArrayElements(logpOut, nullptr);
    for (int ti = 0; ti < t; ti++) {
        const float* row = base + (size_t) ti * D;
        int best = 0;
        float bestV = row[0];
        for (int c = 1; c < D; c++) {
            if (row[c] > bestV) { bestV = row[c]; best = c; }
        }
        double sum = 0.0;
        for (int c = 0; c < D; c++) sum += exp((double) (row[c] - bestV));
        oi[ti] = best;
        ol[ti] = (jfloat) (-log(sum));
    }
    env->ReleaseIntArrayElements(idxOut, oi, 0);
    env->ReleaseFloatArrayElements(logpOut, ol, 0);
    return t;
}
