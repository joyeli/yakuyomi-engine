# 模型

[English](MODELS.md) ｜ 中文

引擎不內建任何模型權重。翻譯需要三顆模型——偵測、OCR、去字（inpaint）——有兩種取得方式：手動（自備模型）或從本 repo 的 releases 自動下載。三顆全跑 NCNN、以 `.param` + `.bin` 交付；OCR 是兩份 `.param`（原版與混合精度版）共用一份 `.bin`——翻譯共六個檔。夜讀（頁面本身變暗、人物保護；預設關、reader 端整合進行中）另加兩顆選配的人物分割模型——再四個檔、同樣 NCNN——見[夜讀模型](#夜讀模型)。兩種方式都落在同一個 models 資料夾，下游解析完全一樣。引擎已經不依賴 ONNX Runtime。

## 模型

| 角色 | 後端 | 檔案 | 大小 | 授權 | 出處 |
|---|---|---|---|---|---|
| 偵測 | NCNN（fp16） | `dbnet_detect.ncnn.param` + `.bin` | ~153 MB | GPL-3.0 | DBNet（ResNet34 + DB head），出自 [manga-image-translator](https://github.com/zyddnys/manga-image-translator) 的 default detector |
| OCR | NCNN（fp16/fp32 混合） | `ocr_48px_ctc.ncnn.param` + `ocr_48px_ctc_mixed.ncnn.param` + `ocr_48px_ctc.ncnn.bin` | ~83 MB | GPL-3.0 | 48px CTC，由 [manga-image-translator](https://github.com/zyddnys/manga-image-translator) 權重在本專案轉檔 |
| 去字 | NCNN（fp16） | `mit_aot_fixed512.ncnn.param` + `.bin` | ~11 MB | GPL-3.0 | AOT-GAN，出自 [manga-image-translator](https://github.com/zyddnys/manga-image-translator) |
| 人物分割（夜讀；選配） | NCNN（fp16） | `manga_seg_s.ncnn.param` + `.bin` | ~20 MB | 模型卡 `license: other`；Ultralytics AGPL-3.0——見[夜讀模型](#夜讀模型) | YOLO11-seg，權重來自 Hugging Face [anonimkaq4/manga-page-element-segmentation](https://huggingface.co/anonimkaq4/manga-page-element-segmentation)，以 MangaSeg／Manga109-s 標註訓練 |
| 人物分割（夜讀；選配） | NCNN（fp16） | `cartoonseg.ncnn.param` + `.bin` | ~126 MB | 上游未寫授權——見[夜讀模型](#夜讀模型) | CartoonSegmentation（RTMDet-Ins），權重來自 Hugging Face [Jakaline/CartoonSegmentationOnnx](https://huggingface.co/Jakaline/CartoonSegmentationOnnx)；上游 [CartoonSegmentation/CartoonSegmentation](https://github.com/CartoonSegmentation/CartoonSegmentation) |

前三列是翻譯那組、缺一不可；後兩列只有夜讀用得到、解析時可缺（見[夜讀模型](#夜讀模型)）。

**後端。** 翻譯那三顆都跑 NCNN（ARM NEON / Winograd 核心）、都在 CPU 上——GPU/NPU 試過、對這些模型不管用（NCNN Vulkan 把 AOT-GAN 算成垃圾、LiteRT 編不出來）。v1 的 LaMa 去字已退役移除，改由 AOT-GAN（manga-image-translator 的 inpaint）取代。OCR 到 v3 為止走 ONNX Runtime（int8）；v4 把它搬到 NCNN，引擎從此整個拔掉 ONNX Runtime。夜讀那兩顆同樣是 NCNN（fp16）、跑 CPU。

**v4 OCR。** 48px CTC 模型跑在 NCNN 上、**混合精度**：卷積 backbone（約 96% 的 MACs）維持 fp16，transformer encoder 與字元頭用 ncnn 的逐層 featmask（`31=7`）強制 fp32，兩者之間插一個明確的 `Cast` 層。精度怎麼切就是重點。9 頁 242 行、以 fp32 為真值：全 fp16 的 NCNN 讀對 219 行、舊的 int8 ONNX 模型 223 行——兩者都把小假名讀錯（なぃ、か6、だろぅ）——全 fp32 的 NCNN 242 行、混合精度 241 行（唯一不同的那行其實是真值讀錯、混合精度是對的）。OCR 時間比 int8 那顆少 ~23%（例：每頁 1256 vs 1669 ms），模型載入 ~0.3–0.4 秒。正弦位置編碼不烤進圖裡、而是當第二個輸入餵進去，所以任何寬度的字條都能跑——先前把 OCR 卡在 ONNX Runtime 上的那道牆就是它。文字行 8 條並發、跑在單緒建的 Net 上，不進 NCNN 的全域推論鎖。兩份 `.param` 共用同一份 `.bin`：

- `ocr_48px_ctc_mixed.ncnn.param`——混合精度版，預設用它。**只在**CPU 有 ARMv8.2 fp16（`asimdhp`）且 fp16 storage 開著時才正確：它的 `Cast` 層宣告「進來的是 fp16」，其他情況下會把 fp32 資料當 fp16 讀、靜默吐垃圾。
- `ocr_48px_ctc.ncnn.param`——原版，每層都用 Net 的全域精度。混合版用不了時（CPU 沒 fp16、`OcrConfig.ncnnFp16Storage` / `ncnnMixed` 關、或 `_mixed` 檔不在）引擎（`Ocr.pickParam`）就載這份——沒 fp16 的手機一樣讀得對，只是跑 fp32、慢一點。

fp16 的 `.bin` 約 83 MB，退役的 int8 是 44 MB；救回小假名的是精度、不是體積。

**v3 偵測器。** comic-text-detector 已退役、整條移除；改用 manga-image-translator 的 default detector（DBNet：ResNet34 + DB head），真機讀對的文字多 **1.6–2.5×**。權重維持 fp16 storage——int8 量化實測**完全吐不出框**、在 ARM 上也沒有比較快，因此不採用；這也是為什麼光偵測器就佔了裝置端 ~247 MB 權重裡的 ~153 MB。前處理是 resize_aspect 到 1024、再 pad 到 256 的倍數；這樣得到的**矩形**輸入同時繞開 ncnn 對 832–992 正方形尺寸的 heap corruption。SD 8 Gen 3 上的實測：9 頁、242 個偵測行，偵測平均每頁 0.79 秒、v4 OCR 每頁 1.25 秒（比 v3 的 int8 OCR 少 23%）；242 行全部讀出、241 行與 fp32 參考一致。

精確 bytes 與雜湊釘在 [`models.json`](../models.json)：

```
dbnet_detect.ncnn.param            13392  sha256 9e6db2f8…ee1ff7b5
dbnet_detect.ncnn.bin          153010556  sha256 f57bdbed…fcc55c3d
ocr_48px_ctc.ncnn.param            18133  sha256 e701cfc5…cd9901bd
ocr_48px_ctc_mixed.ncnn.param      18438  sha256 32e298de…198309e3
ocr_48px_ctc.ncnn.bin           83037664  sha256 3e0a8094…a057c7a9
mit_aot_fixed512.ncnn.param        33810  sha256 f21ef860…ee7d32b5
mit_aot_fixed512.ncnn.bin       11366088  sha256 a52db45e…5e3560b6
manga_seg_s.ncnn.param             25188  sha256 b0ed82b8…8bea16cc
manga_seg_s.ncnn.bin            20426200  sha256 680ddd68…a622d71a
cartoonseg.ncnn.param              37237  sha256 e259a57a…284b6266
cartoonseg.ncnn.bin            126418212  sha256 f3bf313f…1cc2d83f
```

這些雜湊是**散布用的完整性檢查**——用來確認你手上的檔就是我們發行的那個。它們**不是**判斷「重建是否正確」的準則：一次重建可以數值上完全等價、雜湊卻不同（去字那顆就永遠如此）。要從上游 ckpt 重建這些模型，見 [BUILD_MODELS_zh.md](BUILD_MODELS_zh.md)。

## 夜讀模型

夜讀是把頁面本身變暗——分鏡格與背景分區壓暗——但不動人物，所以夜裡讀的頁是暗的、裡面的人卻不會跟著變灰。它**預設關**、reader 端整合仍在進行中，這兩顆模型也只有夜讀開著時才會用到：翻譯完全不碰它們，`ModelSet.resolve` 也不要求它們在。

配方是 **yolo ∪ cseg**，兩顆都跑 NCNN（fp16、CPU）：兩個人物分割器取聯集，因為各自都會漏掉對方抓得到的人物。兩顆檔都在就聯集、只有一顆就用那顆、都不在則夜讀不可用。文字區塊用的是翻譯同一顆 DBNet 偵測器。超過 3.5 MPx 的頁會先等比縮到那個預算之內再跑偵測／分割／重繪，輸出＝縮後尺寸——工作集約 40 B／px，7 MPx 的頁要 20 秒以上。

| | 模型 | 輸入 | 真機（vivo V2429、SD 8 Gen 3） |
|---|---|---|---|
| yolo | YOLO11-seg `manga_seg_s`——3 類（分鏡格／對話框／人物），只用人物那類 | 1024、letterbox | ~0.46 秒 |
| cseg | CartoonSegmentation RTMDet-Ins `cartoonseg`——單類別實例分割 | 640、右下角 pad | ~0.83 秒 |
| 聯集 | | | ~1.2–1.4 秒 |

整頁夜讀（偵測 + 分割 + 重繪）在那台機器上要 **6–25 秒／頁**，所以這是離線預算——背景工作、不是翻頁當下跑的東西。這一對在翻譯那 ~247 MB 之外再加 ~147 MB。

**授權與散布。** 這兩顆都不是 manga-image-translator 的、也都不是 GPL-3.0。以下只寫查證過的：

- **yolo**——[YOLO11-seg](https://github.com/ultralytics/ultralytics) 權重來自 Hugging Face [anonimkaq4/manga-page-element-segmentation](https://huggingface.co/anonimkaq4/manga-page-element-segmentation)。模型卡宣告 `license: other`；Ultralytics YOLO11 本身是 AGPL-3.0。以 MangaSeg／Manga109-s 標註訓練。模型卡要求：再散布或商用前自行確認 MangaSeg、Manga109-s 與 Ultralytics 的授權、標註「Copyrighted by Minshan Xie」、並引用 MangaSeg 論文（CVPR 2025）。
- **cseg**——[CartoonSegmentation](https://github.com/CartoonSegmentation/CartoonSegmentation)（RTMDet-Ins）權重來自 Hugging Face [Jakaline/CartoonSegmentationOnnx](https://huggingface.co/Jakaline/CartoonSegmentationOnnx)。上游 repo 沒有 LICENSE 檔、README 也未寫授權；訓練資料含 Manga109。

我們把這兩顆的 NCNN 轉檔透過 `models-v5` release 散布，**僅供研究／非商業用途**、附上述出處歸屬；權利人提出要求即下架。你不必用我們的副本：自行從出處取得權重、用 [BUILD_MODELS_zh.md](BUILD_MODELS_zh.md#人物分割--ncnn夜讀) 的腳本轉檔、走自備模型。

## 散布與授權

**翻譯模型——GPL-3.0。** 偵測、OCR、去字三顆權重都是 GPL-3.0。本專案在該授權下、附 manga-image-translator 出處歸屬地重新散布它們，純粹是為了「自動下載」的方便。三顆都是我們自己對 manga-image-translator 權重做的 NCNN 轉檔（OCR 那顆把位置編碼抽成輸入，再從原版 param 衍生出第二份混合精度 param）——沒有上游現成的可散布檔可指，所以由本 repo host。你也可以自己從出處取得原始權重、走自備模型。

**夜讀模型——非 GPL-3.0，逐顆各有條件。** `manga_seg_s`（YOLO11-seg）：模型卡 `license: other`、Ultralytics AGPL-3.0、以 MangaSeg／Manga109-s 訓練、須標註「Copyrighted by Minshan Xie」。`cartoonseg`（CartoonSegmentation）：上游未寫授權、訓練資料含 Manga109。兩者皆以研究／非商業用途、附出處歸屬散布，權利人要求即下架——完整聲明見[夜讀模型](#夜讀模型)。

從上游 ckpt 到這六個檔（翻譯）＋四個檔（夜讀）的完整轉檔路徑——腳本、釘住的版本、以及怎麼驗證產出——見 [BUILD_MODELS_zh.md](BUILD_MODELS_zh.md)。

## 怎麼取得模型

**自動下載（reader）。** reader 一鍵抓齊 manifest 列的每個檔，逐檔對 [`models.json`](../models.json) 的 sha256 驗證。每一筆都帶自己的 url：偵測器來自 `models-v3` release、OCR 三個檔來自 `models-v4`，去字沒有變動、仍由 `models-v2` 供應，夜讀四個檔（角色 `charseg`）來自 `models-v5`。夜讀那四個檔就在 manifest 裡，所以不管夜讀有沒有開、每台裝置都會下載。結果跟自備模型一樣，只是自動化。

**自備模型（手動）。** 把檔案放進你指給 app 的 models 資料夾（每個角色都要 `.param` 與對應的 `.bin`；OCR 請把兩份 `.param` 和 `.bin` 都放進去）。它們按檔名 + 副檔名解析——`.param` 含 `dbnet` → 偵測、`.param` 含 `aot` → 去字、`.param` 含 `ocr` → OCR（兩份 OCR param 哪份當入口都行；引擎查過 CPU 後自己切到 `_mixed` 或退回原版）。三個翻譯角色缺一即視為未備齊。夜讀模型的解析方式相同——`.param` 含 `manga_seg` → yolo 人物分割、含 `cartoonseg` → cseg——但它們是選配：缺了 `resolve` 照樣成功（翻譯就緒永遠不看夜讀模型），夜讀用手上有的那些（兩顆 → 聯集、一顆 → 那顆、零顆 → 夜讀不可用）。沒有備援 runtime：ONNX Runtime 的 OCR 路徑、ONNX 偵測/去字路徑與 LaMa 全都移除了，v3 留下的 `ocr_int8.onnx` 不再被使用。

## 驗證

`ModelDownloader.verify(models, dir)` 逐角色回報「本機檔的 size 與 sha256 是否符合我們發行的版本」。因為 manifest 跟檔案一起版本化，雜湊永遠正確——更新權重 = 出新版 manifest，不會有過時雜湊。這能確認你手上的檔案跟我們散布的逐位元相同。

## API

```kotlin
// 引擎 — ModelDownloader
val models = ModelDownloader.fetchManifest()              // models.json -> List<RemoteModel>
ModelDownloader.ensure(models, destDir) { progress -> }   // 下載缺的/不符的，驗 sha256
ModelDownloader.verify(models, destDir)                   // role -> 是否相符（只驗、不下載）
```

引擎只負責「抓、驗、落檔」；下載到哪、何時觸發、進度 UI 都是 reader 的事——跟 LLM 模型清單同一套引擎/reader 分法（見 [PROVIDERS.md](PROVIDERS_zh.md)）。
