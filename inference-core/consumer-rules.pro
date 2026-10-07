# NCNN JNI（NcnnBackend 的 native 方法）：保留原生方法名，否則 R8 縮小後 JNI 對不上符號、runtime crash。
# ONNX Runtime 的 `-keep class ai.onnxruntime.**` 已隨 ORT 拔除（2026-09-26）一併移除。
-keepclasseswithmembernames class * {
    native <methods>;
}
