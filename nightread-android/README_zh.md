# `:nightread-android` — 在 Android 上跑夜讀

[English](README.md) ｜ 中文

夜讀把漫畫頁重建成頁面本身變暗：對話框變深底亮字、空白背景填黑、人物受保護。演算法在純 Kotlin 的 [`yakuyomi-nightread`](https://github.com/joyeli/yakuyomi-nightread) 函式庫。這個模組是餵它的 Android 那一半：把 `Bitmap` 轉成函式庫的輸入，用 NCNN 跑文字偵測與人物分割，再把結果轉回 `Bitmap`。

它**不依賴**翻譯引擎。依賴只有 [`:inference-core`](../inference-core/README_zh.md)（NCNN、DBNet 偵測、分群、模型下載）和夜讀函式庫，兩個都是 `api`：

```
li.joye.yakuyomi:nightread ◀──api── :nightread-android ──api──▶ :inference-core
```

Group：`li.joye.yakuyomi:nightread-android`。Min SDK 26。只出 arm64-v8a。

## 內容

| 檔案 | 內容 |
|---|---|
| `NightReadRenderer.kt` | `NightReadRenderer`（入口）、`NightReadStats`（計時）、`UnionCharSegmenter`。 |
| `CharSeg.kt` | `CharSegmenter` 與兩顆 NCNN 分割器：`YoloSegSegmenter`（YOLO11-seg `manga_seg_s`）和 `CsegSegmenter`（CartoonSegmentation RTMDet-Ins），加上各自的後處理（`YoloSegPost`、`CsegPost`）。 |

Kotlin 套件是 `li.joye.yakuyomi.engine`，不是 `li.joye.yakuyomi.nightread`。這是歷史原因：程式碼原本在引擎模組裡，reader app 也照這個套件 import。夜讀函式庫自己的型別（`NightReadParams`、`NightTier`…）在 `li.joye.yakuyomi.nightread`。

## 接進 app

引擎 repo 以原始碼透過 Gradle composite build 引入。

1. 把引擎 repo 加成 git submodule，要加 `--recursive`：它裡面還有夜讀函式庫的 submodule。

   ```
   git submodule add https://github.com/joyeli/yakuyomi-engine.git yakuyomi-engine
   git submodule update --init --recursive
   ```

2. `settings.gradle.kts`：

   ```kotlin
   includeBuild("yakuyomi-engine")
   ```

3. app 的 `build.gradle.kts`：

   ```kotlin
   dependencies {
       implementation("li.joye.yakuyomi:nightread-android:0.5.0") // 版號只是佔位，includeBuild 會換成原始碼
   }
   ```

   app 只打 `arm64-v8a`（`ndk { abiFilters += "arm64-v8a" }`）；其他 ABI 沒有推論後端。

條件：裝好 NDK `28.2.13676358` 與 CMake `3.22.1`（`:inference-core` 從原始碼編 `libyakuyomi_ncnn.so`），app 的 `minSdk` 至少 26。引擎這個 build 和它裡面巢狀的夜讀函式庫 build 各自找 Android SDK，讀不到 app 的 `local.properties`：請設 `ANDROID_HOME`，或在 `yakuyomi-engine/` 和 `yakuyomi-engine/yakuyomi-nightread/` 各放一份寫了 `sdk.dir=...` 的 `local.properties`。

## 模型

引擎 [`models.json`](../models.json) 裡的兩個 role：

| Role | 檔案 | 大小 |
|---|---|---|
| `detector` | `dbnet_detect.ncnn.param` + `.bin` | 約 153 MB |
| `charseg` | `manga_seg_s.ncnn.param` + `.bin`、`cartoonseg.ncnn.param` + `.bin` | 約 147 MB |

OCR 與去字模型只有翻譯用，夜讀不會載。只下載這兩個 role：

```kotlin
val dir = File(context.filesDir, "models")
val wanted = ModelDownloader.fetchManifest().filter { it.role == "detector" || it.role == "charseg" }
ModelDownloader.ensure(wanted, dir) { progress -> /* ModelProgress */ }
```

兩個呼叫都是 `suspend`，要在協程裡跑。所以 app 要自己宣告 kotlinx-coroutines（`:inference-core` 以 `implementation` 依賴它，不在你的編譯 classpath 上）。不管用不用，`:inference-core` 在 runtime 都會把 OkHttp 和 kotlinx-coroutines 帶進 APK。

也可以自己把檔案放進本機資料夾。傳路徑就好，Net 直接從檔案載進 native 記憶體。別用 `readBytes()` 把權重讀進 JVM heap。

兩顆人物分割模型**不是** GPL-3.0，各有條件：YOLO11-seg 權重（模型卡 `license: other`、Ultralytics AGPL-3.0、「Copyrighted by Minshan Xie」）與 CartoonSegmentation 權重（上游未寫授權）。Yakuyomi 散布的是它們的 NCNN 轉檔，僅供研究與非商業用途，權利人要求即下架。細節見 [docs/MODELS_zh.md](../docs/MODELS_zh.md#夜讀模型)。

## 快速開始

```kotlin
import li.joye.yakuyomi.engine.Detector
import li.joye.yakuyomi.engine.NightReadRenderer
import li.joye.yakuyomi.nightread.NightTier

Detector(dir.resolve("dbnet_detect.ncnn.param").path).use { detector ->
    val seg = NightReadRenderer.charSegmenter(
        dir.resolve("manga_seg_s.ncnn.param").path,
        dir.resolve("cartoonseg.ncnn.param").path,
    ) ?: error("沒有人物分割模型")
    seg.use {
        detector.warmUp(); it.warmUp()   // 原生的首次初始化在單緒做完，之後才能並行
        val night: Bitmap = NightReadRenderer.render(page, detector, it, NightTier.L2.apply())
    }
}
```

`NightTier.L2.apply()` 就是 reader 的「標準」；`NightTier.L3.apply()` 是「更多」（更多背景塗黑）。`renderTiers(...)` 用一次偵測、一次分析產生好幾檔，逐檔交給回呼。

`charSegmenter` 吃兩個 `.param` 路徑（各自的 `.bin` 要放在旁邊）。兩個都有＝取聯集（定案配方）；只有一個＝就用那個；都沒有＝`null`，夜讀不可用。

## 入口

- `render(page, detector, charSeg, params)`：必要時縮圖、偵測、分割、重建，回一張新的 `ARGB_8888` bitmap。超過 `MAX_PIXELS`（3.5 MPx）的頁會先縮，**輸出是縮後尺寸**（看 `NightReadStats.scaledTo`）。
- `render(page, detect = …, segment = …, beforeRender = …)`：同上，但兩次推論由你的 lambda 做，可以插自己的檢查點（暫停、讓路）。`beforeRender` 回 `false` 就放棄這頁、回 `null`。
- `renderTiers(...)`：一次產好幾檔。每張輸出在回呼返回後立刻 recycle，所以要寫檔就在回呼裡寫完。和前一檔相同的檔會收到 `null`。
- `render(page, detection, charMask, params)`：已經有同尺寸的 `Detection` 和人物遮罩時用。

`extraLines`（從別處知道的文字行，頁面座標）和 `inpaintMask`（譯後頁的去字遮罩）是選配，只對翻過的頁有意義。

## 執行緒與記憶體

- 偵測器和分割器在並行跑頁之前，先在單緒暖機一次。多頁同時打進剛載好的 Net 在真機上會閃退。
- 暖機之後可以多頁並行。這個模組和函式庫都沒有共享的可變狀態；多緒的 NCNN 前向由核心的全域鎖序列化。
- 記憶體由呼叫端管。一頁在飛約需每像素 58 bytes 的 Java heap，一般頁約 150 MB，所以 512 MB 的 heap 大約只放得下兩頁。
- `parallel: Executor?`（預設 `null`）把一頁分析裡互相獨立的分支丟給你的 executor 跑，輸出逐位元相同。它會拉高單頁的 heap 尖峰（估算從 58 改成至少 70 B/px），只有這頁是唯一在飛的頁時才划算。
- `NcnnFlavor` 決定 Net 怎麼建。只做夜讀的 app 用 `DEFAULT` 就對了。`NIGHT_LOCKED`／`NIGHT_FREE` 與 `NcnnLowPriorityHook` 是給 reader 讓夜讀讓路給翻譯用的。
- 這些呼叫不會 recycle 輸入的 bitmap。

## 測試

`:nightread-android:testDebugUnitTest` 跑 10 個 JVM 測試：兩顆分割器的後處理對 Python fixture 逐像素比對（`src/test/resources/charseg/`），以及讀夜讀 submodule 測試頁的整頁測試（`../yakuyomi-nightread/nightread/src/test/resources/page`）。
