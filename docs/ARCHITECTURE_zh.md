# 架構

[English](ARCHITECTURE.md) ｜ 中文

Yakuyomi 怎麼翻一頁、專案為什麼這樣切、裝置端引擎跟桌面驗證工具怎麼搭。引擎的 API 看 [`engine/README_zh.md`](../engine/README_zh.md)。

## 兩半

| | 裝置端（`engine/`，Kotlin） | 桌面（`parity/`，Python） |
|---|---|---|
| 角色 | 產品，跑在手機上 | 驗證工具，跑在筆電上 |
| 技術 | Kotlin、NCNN、Android Canvas | Python、numpy/cv2、torch、onnxruntime、ncnn、PIL |
| 出貨 | 是（library） | 否（開發用） |
| 用途 | 翻頁 | 拿 Kotlin port 對照參考實作 |

引擎是要交付的東西。桌面 parity 之所以存在，是因為引擎把 [manga-image-translator](https://github.com/zyddnys/manga-image-translator)（m-i-t，Python/torch）用 Kotlin（NCNN）重寫，這種 port 沒辦法逐行 diff。桌面在 Python 跑同樣的階段，讓我們先確認「同輸入、近輸出」，再信任 Kotlin 版。見 [`parity/README_zh.md`](../parity/README_zh.md)。

## 單頁資料流

```
頁 bitmap
  Detector (NCNN)    文字行（旋轉四邊形）+ 逐像素筆畫遮罩
  Ocr (NCNN)         每行日文（48px CTC，fp16/fp32 混合精度，貪婪解碼）
  Grouping (Kotlin)  行 -> 氣泡區塊（連邊，再 MST 分裂；閱讀序；傾斜角）
  Translator (LLM)   目標語言，每頁一個請求，無滾動上文
  TextFilter         判定哪些區塊有可用的譯文
  Inpainter (NCNN)   抹掉原文（平塗，或 AOT-GAN）
  Renderer (Canvas)  把譯文畫回去（直/橫排，沿區塊角度旋轉）
翻好的頁 bitmap
```

每階段一句話：

- **Detector** — DBNet（ResNet34 + DB head），m-i-t 的 default detector，跑 NCNN、fp16。前處理是 resize_aspect 到 1024、再 pad 到 256 的倍數（**矩形**輸入；這也繞開 ncnn 在正方形 832–992 尺寸上的 heap 損壞）。後處理：機率圖 sigmoid、二值化 0.5、連通元件、minAreaRect，box score ≥ 0.7 留下、unclip 2.3；第二個輸出是 `seg` 筆畫遮罩（門檻 0.12 + 膨脹）。去字用這張遮罩，抹的是筆畫不是方塊。真機讀對率比先前用的 comic-text-detector 高 1.6–2.5×，後者已整條移除。
- **Ocr** — 48px CTC 跑 NCNN、混合精度：卷積 backbone 走 fp16、transformer encoder 與字元頭走 fp32（ncnn 的逐層 featmask，中間插一個明確的 Cast）。全 fp16 會把小假名讀錯、全 fp32 慢 32–45%；混合精度在真機上 242 行有 241 行與 fp32 相同，還比它取代的 int8 ONNX Runtime 模型快 ~23%。正弦位置編碼每條字條現算（`sinusoidalPe`，T = ⌊W/4⌋ − 1）、當第二個輸入餵進去而不是烤進圖裡——任何寬度的字條都能跑就是靠這個。把每行裁出來（手刻 bicubic perspective warp、直書行轉正），辨識、對字典貪婪解碼（argmax 與 log-softmax 在 JNI 算、CTC 收合在 Kotlin），丟掉低於 `minProb` 的行。裁切前先把偵測到的四邊形以 `stripPad`（4px）往外擴：框太瘦會切掉最後一個字，CTC 隨即吐出空字串，整個區塊就被丟掉、不翻。外擴只影響 OCR 的裁切——偵測框本身不動，所以以筆畫為準的去字遮罩不受影響。文字行並發跑（見下）。
- **Grouping** — 把行併成氣泡大小的區塊，分兩階段：寬鬆的連邊，再用 MST 分裂把只靠傳遞才連起來的鄰居切開。這個分裂就是讓密集氣泡不黏成一塊的關鍵。同時算每個區塊的閱讀序跟傾斜角。
- **Translator** — m-i-t 的 `chatgpt.py` prompt 與協定，走 OpenAI 相容呼叫，每頁一個請求、無跨頁上文。某區塊翻譯失敗就保留它的原文。語言對可設：目標走 `toLangName`、來源由 OCR 模型加 prompt 標籤決定。預設日翻繁中，不寫死。服務商是預設選單、全 OpenAI 相容（Gemini 走它的 compat 端點），各家的模型清單即時撈取——見 [PROVIDERS.md](PROVIDERS_zh.md)。
- **TextFilter** — m-i-t 的譯後過濾。一個區塊算「可用」的條件：譯文非空白、非純數字、不命中過濾 regex、且不等於原文。
- **Inpainter** — 兩個門別：`boxfill`（**快速去字**：用就近背景色平塗遮罩區，快但粗糙）跟 AOT-GAN（**AI 去字**，預設：NCNN 整頁一次、768px，重建背景、品質較高）。舊的逐區（逐格）路徑已移除；LaMa 已退役。
- **Renderer** — 文字框排版，不做氣泡 flood-fill：字級自適應、行頭禁則、直或橫排、文字顏色取去字後背景亮度、畫布沿區塊傾斜角旋轉。

`Pipeline.kt` 管 orchestration，以及「絕不用比原圖更糟的東西覆蓋」這條不變式。

## 並發

三個地方，為了吞吐：

- **OCR** 把一頁的文字行並發跑，每行一條緒。圖塊小、吃不滿多緒推論，所以用不同的行把核填滿比較快。OCR 的 Net 以單緒建，這也是它的 forward 可以不進「序列化偵測與去字」那把全域鎖的原因——那把鎖是因為兩個 OpenMP team 同時進 ncnn 會把行程 abort，而單緒的 Net 根本不組 team。
- **去字跟翻譯重疊。** 去字只要 OCR 過的區塊，這在翻譯回來前就確定了，所以去字在背景 coroutine 上跑、同時 LLM 請求在飛。一頁省下兩者裡較短的那個。這也是為什麼失敗的區塊保留「重貼的原文」而不是原封不動的原圖：把去字跟翻譯結果解耦，才能讓兩者重疊。
- **跨頁** 批次是 reader app 的事、不是引擎的：fork 的 `PageTranslator` 把一章的頁丟進 `translatePage`、以 `Semaphore(pipelineDepth)` 限同時在飛的頁數，下載 worker 在你讀到之前先翻好。（`TranslatorConfig.batchSize` / `batchConcurrent` 只是 m-i-t config schema 的鏡射——引擎裡沒有任何東西讀它們。）

## 與 manga-image-translator 對齊

引擎對齊的是行為，不是原始碼。三層，由緊到鬆：

1. **照抄。** 跨語言不變的資料：prompt 與協定、config schema 與多數階段門檻、模型選擇、處理順序——能沿用的直接用 m-i-t 的調校。但若干預設在裝置端重調（偵測輸入 1024 而非 m-i-t 的 2048、去字 tile 768、strip pad、遮罩膨脹）。
2. **對齊行為、自由實作。** 做「什麼」跟 m-i-t，「怎麼寫」隨 Kotlin/NCNN 慣例。偵測後處理、座標反算、遮罩生成、分群兩階段、閱讀序、並發翻譯。判準是「同輸入、近輸出」。
3. **知情偏離，在碼裡註記。** 平台逼的或刻意的取捨：NCNN 取代 torch、OCR 的混合精度與「位置編碼當輸入」、不用 CUDA、不做滾動上文、兩個去字門別。

被迫改寫的（其餘照搬）：torch 換 NCNN forward；cv2/PIL 換 Bitmap/Canvas/手刻；numpy 換 Kotlin；async httpx/CLI/YAML 換 OkHttp、coroutines、小型 config 載入；manga-ocr 的自回歸 decode；任何 CUDA/GPU 假設。

## 裝置端的現實

真機跑出來的：

- **OCR 的精度是逐層的、不是全域的。** 全 fp16 的 NCNN 會在 transformer 把小假名讀錯（9 頁 242 行只對 219）；全 fp32 全對但慢 32–45%；只開 fp16 storage 更慢 10×（每層進出都要 cast）。所以出貨的 `_mixed` param 讓 backbone 卷積留在 fp16、用 ncnn 的逐層 featmask（`31=7`）把 transformer 與字元頭強制成 fp32，中間插一個明確的 Cast——241/242、OCR 時間比舊的 int8 ONNX 模型少 ~23%。這份 param 只在有 ARMv8.2 fp16（`asimdhp`）的 CPU 上正確；其他機器 `Ocr.pickParam` 會載原版 param、吃同一份 `.bin`。（歷史：到 v3 為止 OCR 走 ONNX Runtime、且必須關 XNNPACK，因為 XNNPACK 會把這顆模型算錯；v4 起引擎已無 ONNX Runtime。）
- **模型載 native 記憶體。** NCNN 的 Net 直接從檔案路徑把 `.param`/`.bin` 讀進 native；用 `readBytes()` 讀進 JVM heap 會撞到每 app 的 heap 上限（約 512MB，跟裝置 RAM 無關）而 OOM。BYOM 先把選的檔複製到 `filesDir` 再傳路徑。
- **前處理要跟 Python 匯出完全一致** — resize、normalize、NCHW 順序。這是最大的隱形分歧來源，也是 parity 工具大半的存在理由。
- **推論執行緒。** 偵測跟 AOT-GAN 去字用對齊裝置大核數的緒數（測試機 Snapdragon 8 Gen 3 是 6），把慢的小核加進去會讓一次推論更慢而不是更快。
- **GPU/NPU 試過、對這些模型不管用——全部跑 CPU。** NCNN 的 Vulkan 把 AOT-GAN 去字模型**算錯**（fp16/fp32 都輸出垃圾、tile 越大越糟——Adreno 上這組 op 的 shader 級 bug），偵測器在 Vulkan 上也輸給 CPU；LiteRT 連把這些模型編到 GPU 都失敗；NPU（Hexagon）後端需要 int8 QDQ、被 OCR 模型的動態寬度堵住。偵測器的 int8 量化也試過：完全吐不出框、在 ARM 上也沒更快，所以維持 fp16。所以三顆模型都跑 CPU、都走 NCNN 的手機核心（NEON/Winograd；安全的地方用 fp16、OCR 的 transformer 用 fp32）——而這夠快（Snapdragon 8 Gen 3、6 張代表頁：偵測 + OCR 合計約 10.3 秒——以 v3 的 int8 OCR 量的、NCNN OCR 那份再快 ~23%——161 個偵測框讀出 160）。

## Repo 結構

```
engine/        Android library，裝置端 pipeline（產品）
  src/main/kotlin/li/joye/yakuyomi/engine/
    Yakuyomi.kt, TranslationEngine.kt, ModelSet.kt   對外入口（facade + 型別）
    Pipeline.kt, Detector.kt, Ocr.kt, Grouping.kt,
    LlmTranslator.kt, Inpainter.kt, Renderer.kt      各階段
    LlmProviders.kt, LlmModels.kt                    供應商預設 + 即時撈模型清單
    NcnnBackend.kt (+ src/main/cpp/ncnn_jni.cpp)     NCNN JNI：Net、全域鎖、OCR CTC forward
    Geometry.kt, ImageOps.kt, TextFilter.kt          內部 helper
  src/test/kotlin/…                                  JVM 單元測試
app-sandbox/   sandbox 測試 app（裝置計時、比較圖）
parity/        桌面 Python 驗證工具（不出貨）
docs/          這份，加參數參考
```

reader app（[Yakuyomi](https://github.com/joyeli/Yakuyomi) mihon fork）用 git submodule + Gradle composite build 引入 `engine`，只加整合層：下載 hook、設定、模型管理。
