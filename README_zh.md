# Yakuyomi — 漫畫翻譯引擎

偵測、OCR、去字都在裝置上跑（全 NCNN），翻譯走雲端 LLM。預設日文翻繁體中文，來源與目標語言都可改。

[English](README.md) ｜ 中文

狀態：整條 pipeline 在裝置上跑、驅動 Yakuyomi reader app——下載即翻、邊讀邊翻（即時翻譯）、換去字法免重翻的重繪都能用。reader 的[第一版公開發佈](https://github.com/joyeli/Yakuyomi/releases/latest)已出。

本 repo 是**引擎**（`yakuyomi-engine`）——翻譯函式庫，不是可安裝的 app。**要 app？** 那是 reader **Yakuyomi**，一個 [mihon](https://github.com/mihonapp/mihon) fork：[**下載簽章 APK**](https://github.com/joyeli/Yakuyomi/releases/latest) 或看它的 [repo](https://github.com/joyeli/Yakuyomi)。這個引擎 repo 以 submodule 被它引入，見[儲存庫結構](#儲存庫結構)。

## 從哪開始

| 你想做的事 | 去哪 |
|---|---|
| **看它跑起來** | 下面的[**試跑**](#試跑)——把 sandbox app 編出來裝到 arm64 手機上，看一頁走完整條 pipeline。LLM key 是選配：不給就跳過翻譯，偵測、OCR、去字照樣看得到。 |
| **把引擎整合進自己的 app** | [**`engine/README_zh.md`**](engine/README_zh.md)——API 面：`translatePage`、模型怎麼進去、設定、結果處理、執行緒。 |
| **自己重建模型** | [**`docs/BUILD_MODELS_zh.md`**](docs/BUILD_MODELS_zh.md)——從上游 checkpoint 轉，含各種坑與驗證判準。屬進階：我們轉好的可以直接下載，**用**引擎完全不需要走這條。 |

## 這是什麼

Yakuyomi 翻譯漫畫頁。五個階段裡四個在裝置上跑（偵測、OCR、去字走 NCNN，排版走 Canvas），只有翻譯連網：

```
頁 bitmap
  偵測  (NCNN)      文字行框 + 逐像素筆畫遮罩
  OCR   (NCNN)      每行一次前向  ->  原文
  分群              把對齊的行併成氣泡區塊
  翻譯  (LLM)       每頁一個請求
  去字  (NCNN)      抹掉原文（平塗，或 AOT-GAN 重建）
  排版  (Canvas)    把譯文畫回去
  翻好的 bitmap
```

引擎只對外開一個呼叫，`translatePage(page): PageResult`（翻好／略過／失敗）。覆蓋原檔、「已翻譯」標記、續傳、背景翻譯佇列、邊讀邊翻是 reader app 的事。

![效能比較](docs/img/showcase.png)

來自 sandbox app：同一頁走過整條 pipeline——偵測、去字遮罩、兩種去字模式（含各自偵測到的區塊）、貼完譯文的成品——表把每個階段的用時與記憶體峰值拆開列出；橫幅記了裝置、生效的設定、跟 LLM。（圖是先前 int8 OCR 版本產的——橫幅仍寫 ORT-int8；換成 NCNN OCR 後各階段用時略有不同，以下方「目標」的數字為準。）

![去字 vs 平塗](docs/img/removal-compare.png)

壓在畫面上的字是難題。平塗（多數疊字翻譯的做法）會在頭髮上塗一塊色塊；Yakuyomi 的 AI 去字會先重建底下的髮絲，再把譯文排上去。

## 目標

- **速度優先於極致品質——這是為手機刻意做的權衡。** 一開始的直覺是追畫質：LaMa 去字、逐區原生解析度重建、最銳的去字。**在手機上那是死路**——那些每頁要好幾秒、吃掉好幾 GB 記憶體，reader 就卡住了。在終端裝置上，目標不是那最後幾個百分點的品質，而是**速度**：你在讀，頁就得出得來。所以每個階段都停在「品質/效率的拐點」、而非品質天花板：
  - **OCR** 走 NCNN 混合精度——backbone fp16、transformer 頭 fp32。全 fp16 會把小假名讀錯（9 頁 242 行只讀對 219），它取代的那顆 int8 ONNX Runtime 模型也一樣（223/242）；混合精度讀對 241/242，還比那顆 int8 快 ~23%。
  - **偵測與去字** 同樣走 NCNN 的手機核心（NEON/Winograd）。偵測器維持 fp16——int8 量化試過，完全吐不出框，在 ARM 上也沒有比較快。
  - **去字 tile 768** ——整頁 AOT-GAN，品質夠好、**又剛好藏在翻譯等待底下**（見「併發」）的那個點。更大的 tile 或逐區重建只是銳一咪咪、卻會戳出翻譯等待；LaMa 又慢又糊。**GPU/NPU 試過、對這些模型不管用**——NCNN 的 Vulkan 把 AOT-GAN 算成垃圾、LiteRT 根本編不出來——所以全部跑在 **CPU**，而 CPU 結果就夠了。

  Snapdragon 8 Gen 3 上的實測：9 頁、242 個偵測行——偵測平均 **每頁 0.79 秒**、混合精度 OCR 平均 **每頁 1.25 秒**（比它取代的 int8 OCR 少 23%）；242 行全部讀出，其中 241 行與 fp32 參考完全一致。翻譯與去字在這之外，且兩者互相重疊（見「併發」）。記憶體峰值在前一版量到 ~1.9–2.1 GB（換 OCR 後尚未重量）——不用 GPU、遠不到 16 GB RAM。
- **併發，兩層。**
  - *頁內* ——去字只需要 OCR 過的區塊（LLM 回來前就知道），所以它在背景 coroutine 上跑、同時翻譯請求在飛；一頁只付兩者中較長的那個。（這也是為什麼失敗的區塊保留重貼的原文、而非原封的圖——把去字跟翻譯結果解耦，才能重疊。）
  - *跨頁* ——`translatePage` 可以對同一個 warm 引擎併發呼叫（共用偵測／OCR／翻譯／去字 session；真機 benchmark 過——不崩、不亂）。所以 reader 能做流水線：第 N 頁的網路翻譯，跟第 N+1 頁的裝置端 detect/OCR 重疊。搭配便宜的 box-fill 去字，流水線能撞到「網路上限」——淺深度（~4）下約 **2× 循序速度**。讀到的頁先是 box-fill 畫質、閒置時再升級成完整 AOT-GAN 去字（見下方重繪）。
- **可設定，能公開。** 服務商、模型、API base、金鑰、語言對都是設定（自備金鑰）。模型從你選的資料夾載入，不打包進 APK（自備模型）。約 20 個引擎參數可調，見 [docs/PARAMETERS_zh.md](docs/PARAMETERS_zh.md)。
- **絕不讓書庫變更糟。** 只有翻譯成功才覆蓋該頁。整頁沒文字、整頁翻譯失敗、或網路斷線，原圖原封不動。單一區塊翻譯失敗時保留它的日文，而不是清空。

![跨頁併發流水線](docs/img/crosspage_showcase.png)

兩層併發。*頁內*，去字（CPU）跟翻譯（網路）重疊——一頁只付兩者中較長的那個。*跨頁*，`translatePage` 可以對同一個 warm 引擎併發呼叫，所以頁能流水線：第 N 頁的翻譯跟第 N+1 頁的裝置端 detect/OCR 重疊。真機 benchmark——搭配 box-fill 去字、淺深度下約 **2× 循序速度**。

## 能做什麼

- **偵測** — DBNet（ResNet34 + DB head，manga-image-translator 的 default detector）跑在 NCNN 上。它取代了 comic-text-detector，後者已整條移除：DBNet 在真機上**讀對的文字多 1.6–2.5×**。頁面以 resize_aspect 縮到 1024、再 pad 到 256 的倍數——**矩形**輸入，這也繞開了 ncnn 對 832–992 正方形尺寸的 heap corruption。回傳文字行四邊形，加一張逐像素筆畫遮罩，用來把去字限制在筆畫上。
- **OCR** — 48px CTC 模型跑在 NCNN 上、**混合精度**：卷積 backbone 走 fp16、transformer 與字元頭走 fp32（ncnn 的逐層 featmask，中間插一個明確的 Cast 層）。正弦位置編碼每條字條現算、當第二個輸入餵進去而不是烤進圖裡，所以任何字條寬度都能跑。真機 242 行讀對 241 行、與 fp32 相同（int8 223、全 fp16 219——兩者都把小假名讀錯），比它取代的 int8 ONNX Runtime 模型快 ~23%。每行一次前向、貪婪解碼、文字行並發辨識。混合精度 param 需要 ARMv8.2 fp16 的 CPU；沒有的話引擎載原版 param、跑 fp32。
- **翻譯** — 雲端 LLM，沿用 manga-image-translator 的行號協定。任何 OpenAI 相容服務商都行；預設選單涵蓋 manga-image-translator 那組（OpenAI、DeepSeek、Gemini、Groq、Qwen、Sakura、自訂）外加 OpenRouter，各家的模型清單即時撈取。預設 DeepSeek。引擎每頁送一個請求；把多頁並發跑（以及限流）是呼叫端的事——reader 用 semaphore 做，見上面的「併發」。某行翻譯失敗時退回原文，不會弄壞整頁。詳見 [docs/PROVIDERS_zh.md](docs/PROVIDERS_zh.md)。
- **去字** — NCNN 上兩個模式。對話框一律平塗（乾淨、無黃暈），模式之間只差在壓在畫面上的字怎麼處理：

  | 模式 | 做法 | 速度 |
  |---|---|---|
  | 快速去字（BoxFill） | 就近取背景色平塗（壓畫面會變色塊） | 最快 |
  | AI 去字（預設） | AOT-GAN 重建字底下的畫面，整頁跑 tile 768 | 較慢、銳——而且藏在翻譯等待底下 |

- **排版** — 文字框排版，直排或橫排，字級自適應、垂直置中、描邊隨字級、行頭禁則、沿傾斜氣泡角度擺放。文字顏色依去字後的背景決定（亮底黑字、暗底白字）。
- **重繪（analyze | render 切分）** — 翻好的頁會連同它的分析素材一起回傳：文字遮罩，加上帶著原文與譯文的區塊。之後換去字法、重新排版時不必重跑偵測／OCR／LLM——換去字模式、升級品質只花去字 + 排版兩個階段，不耗 token。
- **語言** — 開箱即用日翻繁中。換目標語言、來源語言、few-shot 範例就能翻任何語言對。繁中輸出靠 prompt，沒有 OpenCC 後處理。

## 儲存庫結構

兩個 repo：

| Repo | 角色 |
|---|---|
| `yakuyomi-engine`（這個） | 引擎：`:engine`（整條 pipeline，只開 `translatePage`）、`:app-sandbox`（真機跑 pipeline 用）、`parity/` 桌面驗證工具。沒有 reader 程式碼。 |
| `Yakuyomi`（mihon fork） | reader app：mihon 加上下載 hook、翻譯設定、模型管理。用 git submodule + Gradle `includeBuild` 引入引擎。 |

引擎跟 reader 解耦，才能自己單獨測；app 是真正的 mihon fork。引擎的改動 commit 在這裡，app 端 bump submodule 指標。

## 模型

權重不 commit、也不打包進 APK。reader 可自動下載，或你手動放進指定的資料夾——出處、雜湊、授權見 [docs/MODELS_zh.md](docs/MODELS_zh.md)。

| 階段 | 模型 | 後端 | 來源 |
|---|---|---|---|
| 偵測 | DBNet，ResNet34 + DB head（`.ncnn.param`/`.bin`） | NCNN | 來自 [manga-image-translator](https://github.com/zyddnys/manga-image-translator)（它的 default detector） |
| OCR | 48px CTC，fp16/fp32 混合精度（`.ncnn.param` ×2 + `.bin`） | NCNN | 權重來自 [manga-image-translator](https://github.com/zyddnys/manga-image-translator) |
| 去字 | AOT-GAN 漫畫 inpaint（`.ncnn.param`/`.bin`） | NCNN | 來自 [manga-image-translator](https://github.com/zyddnys/manga-image-translator) |
| 字型 | Noto Sans/Serif CJK、思源 | — | CJK 算繪（OFL / Apache） |

全部都是 NCNN，`.param` + `.bin` 成對（都要）；OCR 是兩份 `.param`（原版與 `_mixed`）共用一份 `.bin`，引擎載入時自己挑。整套約 247 MB，大半是 fp16 偵測器（153 MB）與 fp16 的 OCR 權重（83 MB）。

## 試跑

引擎是 Android library（arm64、NCNN），所以要試跑就是把 sandbox app（`:app-sandbox`）編出來裝上去。**需要真的 arm64 Android 裝置**——sandbox 只打 `arm64-v8a`，x86 模擬器跑不起來。

**1. 拿模型。** 模型不在 repo 裡。把 [`models.json`](models.json) 列的六個檔抓下來——偵測器的 `.param`+`.bin` 在 `models-v3` release、OCR 的兩份 `.param`（原版與 `_mixed`）+ `.bin` 在 `models-v4`、去字的 `.param`+`.bin` 在 `models-v2`——全放進同一個手機讀得到的資料夾。來源、雜湊與授權見 [docs/MODELS_zh.md](docs/MODELS_zh.md)。

**2.（選配）給 LLM key。** 把 `api-keys.properties.example` 複製成 `api-keys.properties`、填入 `DEEPSEEK_API_KEY`。**不給也沒關係，翻譯那步會自動跳過**——偵測、OCR、去字照跑，一樣看得到 pipeline 在做什麼。

**3. 編譯安裝。**

```
./gradlew :app-sandbox:assembleDebug
adb install app-sandbox/build/outputs/apk/debug/app-sandbox-debug.apk
```

**4. 跑起來。** 開 app → 按「選擇模型資料夾」指到第 1 步那個夾 → 然後：

| 按鈕 | 做什麼 |
|---|---|
| **偵測 + OCR 檢驗** | **先按這個。** 用產品預設把內建測試圖跑一遍偵測 + OCR，印出框數／讀出塊數／秒數。不用選圖、不用 key。 |
| **診斷** | 點一張縮圖 → 跑**完整 pipeline**（偵測 → OCR → 翻譯 → 去字 → 排版），附各階段耗時。沒 key 就跳過翻譯那步。 |
| **效能比較** | 單張圖，兩種去字方法並排比。 |

按鈕分成「選圖測試」（跑你選的縮圖）與「固定圖測試」（跑內建圖、不理選取）兩區。

想自己從上游 checkpoint 把模型轉出來、而不是下載我們轉好的？見 [docs/BUILD_MODELS_zh.md](docs/BUILD_MODELS_zh.md)。

reader app（Yakuyomi）在另一個 fork repo。

## 設定

每個可調參數、值域、改了會怎樣，都寫在 [docs/PARAMETERS_zh.md](docs/PARAMETERS_zh.md)。reader 的翻譯設定開放同一組，按階段分組，進階旋鈕收在開關後面。

## 與 manga-image-translator 的關係

引擎是純 Kotlin 從頭實作，不含 manga-image-translator 的原始碼。借的是行為——翻譯 prompt 與協定、參數 schema（不少預設值針對裝置重調）、模型選擇與處理順序。細節與分層對齊原則在 [docs/ARCHITECTURE_zh.md](docs/ARCHITECTURE_zh.md)。

## 致謝

- [mihon](https://github.com/mihonapp/mihon) — app fork 的來源（Apache-2.0）
- [manga-image-translator](https://github.com/zyddnys/manga-image-translator) — prompt 與行為參考；DBNet 偵測、OCR 與 AOT-GAN 去字模型權重
- [ncnn](https://github.com/Tencent/ncnn) — 三顆模型的裝置端推論 runtime
- Noto Sans/Serif CJK、思源 — 字型

## 授權

**GPL-3.0** — 見 [LICENSE](LICENSE)。這裡的程式碼是用 Kotlin 從頭寫的，但移植了 manga-image-translator 的 prompt、參數 schema 與分組；作為該 GPL-3.0 專案的衍生，本引擎為 GPL-3.0。

各組件授權：
- [manga-image-translator](https://github.com/zyddnys/manga-image-translator) — GPL-3.0（prompt/協定、偵測/OCR/去字行為、文字行分組；DBNet 偵測模型、48px CTC OCR 模型與 AOT-GAN 去字模型）
- [ncnn](https://github.com/Tencent/ncnn) — BSD-3-Clause（推論 runtime，靜態連結）
- [mihon](https://github.com/mihonapp/mihon) — Apache-2.0（reader fork 在另一個產品 repo；Apache-2.0 與 GPL-3.0 相容，故組合後的 app 為 GPL-3.0）

模型權重皆 GPL-3.0，透過本 repo 的 release **散布**供一鍵自動下載——manifest 是 [`models.json`](models.json)，指向 [`models-v3`](https://github.com/joyeli/yakuyomi-engine/releases/tag/models-v3) 的偵測器、[`models-v4`](https://github.com/joyeli/yakuyomi-engine/releases/tag/models-v4) 的 OCR，以及 [`models-v2`](https://github.com/joyeli/yakuyomi-engine/releases/tag/models-v2) 裡未變動的去字檔（見 [docs/MODELS_zh.md](docs/MODELS_zh.md)）；也可從上述來源自備。字型未 bundle（系統 CJK fallback）。
