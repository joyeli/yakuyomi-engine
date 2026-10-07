# `:inference-core` — 共用的裝置端推論核心

[English](README.md) ｜ 中文

翻譯和夜讀都要用的那一塊：NCNN 原生層、文字偵測、文字行分群、模型下載。上面有兩個模組，彼此不互相依賴：

```
li.joye.yakuyomi:nightread   （純 Kotlin 的夜讀函式庫，巢狀 includeBuild）
        ▲ api
:nightread-android ──api──▶ :inference-core ◀──api── :engine
   （夜讀）                   （libyakuyomi_ncnn.so）     （翻譯）
```

一般不必直接依賴這個模組。要翻譯就依賴 [`:engine`](../engine/README_zh.md)，要夜讀就依賴 [`:nightread-android`](../nightread-android/README_zh.md)；兩者都用 `api` 把這裡帶進來。

Group：`li.joye.yakuyomi:inference-core`。Min SDK 26。只出 arm64-v8a。

## 內容

| 檔案 | 內容 |
|---|---|
| `NcnnBackend.kt` + `src/main/cpp/ncnn_jni.cpp` | NCNN 的 JNI：載入與釋放 Net、推論全域鎖、夜讀用的低優先權閘、各模型的前向。編出 `libyakuyomi_ncnn.so`。 |
| `src/main/cpp/ncnn/` | 預編的 NCNN（標頭、`libncnn.a`、CMake 設定）：arm64、SimpleOMP、不含 Vulkan。 |
| `Detector.kt` | DBNet 文字偵測：文字行四邊形加逐像素筆畫遮罩（`Detection`）。 |
| `DetectorConfig.kt` | 偵測參數。`:engine` 的 `EngineConfig.detector` 用它，夜讀也用它。 |
| `TextLine.kt`、`Grouping.kt` | 偵測到的文字行，以及 m-i-t 兩階段分群成區塊（`TextRegion`）。 |
| `Geometry.kt`、`ImageOps.kt` | `Pt`、`RotRect` 與幾何、前處理 helper。 |
| `ModelDownloader.kt` | 依 manifest 下載模型、驗 sha256。 |
| `EngineTrace.kt` | 原生呼叫前後的診斷 trace 掛鉤。 |

這裡所有 Kotlin 程式碼的套件都維持 `li.joye.yakuyomi.engine`。JNI 符號名由它決定（`Java_li_joye_yakuyomi_engine_NcnnBackend_*`），reader app 也照這個套件 import，所以拆模組沒有改任何名字。

## 只能有一個原生庫

`libyakuyomi_ncnn.so` 只在這裡編。NCNN 是靜態庫，如果另一個模組也編自己的 `.so`，兩邊各帶一份 NCNN，也就各有一個 SimpleOMP worker 池、各有一套 `pthread_once`。`NcnnBackend` 在一般優先權的專用執行緒上建池、用一把鎖序列化多緒前向，這兩條規則到時只管得到其中一份。

所以所有原生入口都留在這個模組，包括只有一邊用到的：

| JNI 入口（`NcnnBackend_…`） | Kotlin 包裝 | 誰用 |
|---|---|---|
| `createNetNative`、`createNetExNative` | `createNet`、`createNetEx` | 每顆模型 |
| `initThreadPoolNative` | private `ensureThreadPool` | `NcnnBackend` 自己 |
| `releaseNet` | `releaseNet` | 每顆模型 |
| `detectDbnetNative` | `detectDbnet` | `Detector`（翻譯與夜讀） |
| `cpuSupportsFp16Native` | `cpuSupportsFp16` | OCR 選 param（`:engine`） |
| `ocrCtcNative` | `ocrCtc` | `Ocr`（`:engine`） |
| `inpaintAotNative` | `inpaintAot` | `Inpainter`（`:engine`） |
| `extractNative` | `extract` | 人物分割（`:nightread-android`） |

因此單獨用夜讀時，會多帶 OCR 和去字那兩個用不到的小函式。NCNN 的層實作本來就整包連進來。

`consumer-rules.pro` 讓 R8 保留原生方法名，跟著這個模組帶到每個使用它的 app。

## 建置

- NDK `28.2.13676358`、CMake `3.22.1`（`externalNativeBuild`）。reader app 釘同一版 NDK，才能 strip 這個庫。
- 只編 `arm64-v8a`；預編的 NCNN 只有 arm64。
- 要換預編的 NCNN，請用 SimpleOMP（`NCNN_SIMPLEOMP=ON`）、不含 Vulkan 重編，並確認 `ncnn.cmake` 沒有連結裸的 `pthread`（NDK 沒有 `libpthread`；`Threads::Threads` 沒問題）。

## `@InternalEngineApi`

`NcnnBackend` 和 `Geometry` 拆模組前是 `internal`。Kotlin 的 `internal` 跨不了 Gradle 模組，而 `:engine` 和 `:nightread-android` 都要用，所以改成公開，但標上 `@InternalEngineApi`——錯誤等級的 opt-in 註解。引擎的三個模組整模組 opt-in。外部程式不要 opt-in：它們不是公開 API，隨時可能改。`ImageOps` 仍是 `internal`。

## 下載模型

`ModelDownloader` 讀 manifest（預設是本 repo `main` 上的 [`models.json`](../models.json)），確保檔案都在你指定的資料夾、逐檔驗 sha256。已存在且大小與雜湊都對的會跳過。

```kotlin
val remote = ModelDownloader.fetchManifest()
val dir = File(context.filesDir, "models")
ModelDownloader.ensure(remote.filter { it.role == "detector" }, dir) { progress -> /* ModelProgress */ }
```

role 是 manifest 裡的字串（目前有 `detector`、`ocr`、`inpainter`、`charseg`），用它篩出要的檔。`ensure` 回傳 role → 檔案的 map，但同一個 role 有多個檔（`.param` 和 `.bin`）時 map 只留最後一個，所以請用檔名到資料夾裡找。app 要自己宣告 `INTERNET` 權限，這個模組不宣告任何權限。

模型要用本機路徑載入。別用 `readBytes()` 把權重讀進 JVM heap：heap 上限約 512 MB，跟裝置 RAM 無關。

## `EngineTrace`

`EngineTrace.sink` 預設是 `null`，零開銷。設了之後，每個原生呼叫前後會收到一行（`xxx.enter`／`xxx.call`／`xxx.exit`）。行程如果死在原生碼裡，最後一行就是那個呼叫。reader app 把這些行寫進它的診斷紀錄。
