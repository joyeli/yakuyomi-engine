# 重建模型

[English](BUILD_MODELS.md) ｜ 中文

引擎載的三顆模型，都是我們自己對 [manga-image-translator](https://github.com/zyddnys/manga-image-translator) 上游 ckpt 的轉檔——沒有上游現成的可散布檔可指，所以由我們自己轉、自己 host。[MODELS_zh.md](MODELS_zh.md) 講這三顆是什麼、怎麼取得；這頁講**怎麼從上游 ckpt 重建它們**，以及**怎麼判斷你重建出來的是對的**。夜讀用的那兩顆選配人物分割模型也是轉檔——但來源是第三方權重、不是 manga-image-translator 的——在三條重建路徑之後[另有一節](#人物分割--ncnn夜讀)。

這裡沒有任何需要你自己拼回去的配方——每條路徑就是一支腳本。以下是那些腳本要的環境、判斷產出的準則，以及「不驗證就會安靜出錯」的那些坑。

> **你多半用不到這頁。** 我們轉好的權重可以直接下載、而且有 checksum 驗證（見 [MODELS_zh.md](MODELS_zh.md)），reader 與 sandbox 走的都是那條。會來這頁的情況是：想稽核我們的轉檔、想換一顆上游 checkpoint、或想用自己的 toolchain 重建。只想**看引擎跑起來** → [repo README](../README_zh.md#試跑)；想**整合** → [engine/README_zh.md](../engine/README_zh.md)。

## 「可重現」在這裡的意思

比對任何雜湊之前先讀這段，因為三顆裡有一顆最直覺的那個檢查是錯的、另一顆還沒確認過。

**[`models.json`](../models.json) 裡的 sha256 是「散布用的完整性檢查」，不是「重現性判準」。** 它存在的目的是讓 app 確認「下載到的檔＝我們發行的檔」。它不是「重建正確」的定義——一次重建可以數值上完全等價，雜湊卻不同。

| 模型 | 判準 | 說明 |
|---|---|---|
| 偵測（DBNet） | **逐位元相同**——sha256 對得上 `models.json` | 實測：冷啟動重跑仍精確重現 `9e6db2f8…` / `f57bdbed…` |
| OCR（NCNN） | **數值等價**——30 條 fixture 字條解碼文字與 ORT fp32 參考逐行相同、每個 timestep 的 argmax 相同、寬度掃描全過。`models.json` 的雜湊是出貨當時的值 | 與 DBNet 同一條 trace → pnnx 路徑，釘住的 toolchain 應能重現位元，但尚未在冷啟動重跑上確認過——用判準看、別看雜湊 |
| 去字（AOT-GAN） | **數值等價**——`out0` 逐位元相同 ＋ 逐層權重比對。**sha256 對不上** | 這是預期內、已查清的：pnnx 的層自動命名與排序不同。見下 |
| 人物分割（yolo／cseg，夜讀） | **數值等價**——NCNN 原始輸出對來源 ONNX 在 fp16 storage 容差內、人物聯集遮罩對 ONNX 路徑在測試頁上的 IoU。`models.json` 的雜湊是出貨當時的值 | 尚未在冷啟動重跑上確認過；yolo 走的是 ultralytics 內建的 pnnx，別指望換版本後位元不變——用判準看 |

**為什麼 AOT 的雜湊永遠對不上。** 我們重建出的 `mit_aot_fixed512.ncnn.param` 是 33,762 B，release 是 33,810 B（差 −48 B）；`.bin` 則是**跟 release 一模一樣的 size、但位元不同**。兩個差異都是 pnnx 版本造成的，而且都追到底了：

- **param**：層自動命名（我們是 `conv_24` / `relu_0` / `reflectpad2d_40`，release 是 `conv_70` / `relu_6` / `pad_0`），加上 release 多寫了 Padding 的預設值 `5=0 6=0`。**op 直方圖 diff 完全為空**，layer/blob 數也都是 402/500。
- **bin**：77% 的位元組不同，但逐層解析後，**76 層權重是同一組、每層 bit-identical**——只是新版 pnnx 把 AOTBlock 的平行 dilated 分支與 fuse conv 排成了不同順序。

真正該看的判準是行為面的，而且只要你把 release 權重給它比對，腳本就會跑：對真頁比對，`out0` 在 s=512 與 s=768 **皆逐位元相同**（`np.array_equal` 為真、max|d| = 0.0）。

**「逐位元」這件事綁在釘住的版本上。** 上面 DBNet 的結果，成立於 torch 2.1.1 + pnnx 1.0.20260526、x86 Linux。換版本大概率會退化成「數值等價但非逐位元相同」——**那是預期，不是失敗**，這種情況請改用各模型段落列的容差來判斷。

## 前置

### Python 環境

```bash
pip install -r parity/requirements.txt
```

該檔的「模型重建」區段是**刻意釘死版本**的——逐位元重現就是綁這一組：

| 套件 | 版本 | 誰要用 |
|---|---|---|
| torch | 2.1.1 | 三條路徑都要 |
| torchvision | 0.16.1 | DBNet——ResNet34 backbone |
| onnx | 1.17.0 | OCR——只有退役的 int8 路徑（`quantize_ocr_int8.py`）用 |
| onnxruntime | 1.23.0 | OCR——跑 NCNN 轉檔拿來驗證的 fp32 ONNX 參考（以及退役的 int8 路徑） |
| pnnx | 1.0.20260526 | torch → ncnn |
| ncnn | 1.0.20260526 | 驗證：載入產出、比對 forward |

實測環境：Python 3.10.12 / numpy 1.26.4 / opencv 4.11、x86 Linux（WSL2）。

**pnnx 是 pip 套件，不是要自己編的外部 binary。** `pip install pnnx` 會同時給你 Python module（`import pnnx`，AOT 腳本用）與 `~/.local/bin/pnnx` 這支 console script（DBNet 與 OCR 腳本以 subprocess 呼叫）。裝到別的地方就用 `YAKU_PNNX` 指過去。

### 上游 clone

所有匯出腳本都是從 manga-image-translator 的 clone 讀模型定義，而不是把副本 vendored 進來：

```bash
git clone https://github.com/zyddnys/manga-image-translator
export YAKU_MIT_CLONE=/path/to/manga-image-translator   # 預設 /mnt/d/Gits/manga-image-translator
```

**腳本只讀 clone——不動它的任何檔案，也不動它的 git 狀態。** 但沒辦法直接 import：上游 `manga_translator/__init__` 會把 translators → tiktoken → openai 整包拖進來，`detection/__init__` 又會 import 根本不在的 `rusty_manga_image_translator`。每支腳本都在記憶體裡繞過去——塞假 module stub 與 package shell（有 `__path__`、但**不執行** `__init__` body）。模型類別本身是純 torch 模組，這樣就解得開。

### `.upstream-ref` 的不一致

有一條不一致值得講白。[`.upstream-ref`](../.upstream-ref) 釘的是 `efdc229`（2026-07-01），但這些模型實際對著建的 clone 停在 `d5a3eee`（2026-05-24），腳本的 `ported spec` 檔頭寫的也都是 `@ d5a3eee`（`export_ocr_ncnn.py` 重用 `export_ocr_onnx.py` 的 loader，所以繼承同一個 pin）。

**這對重建沒有影響**，而且這是查過的、不是假設的。把 `d5a3eee..efdc229` 對 watched 的模型路徑逐一 diff：

- `detection/default_utils/DBNet_resnet34.py`——**完全零差異**
- `inpainting/inpainting_aot.py` 與 `ocr/model_48px_ctc.py`——**各 1 行**，而且兩者都落在 *loader* 類別的 device dispatch（`cuda`/`mps`/`xpu`），**不在**我們匯出的 `nn.Module` 裡
- `detection/default.py`——同一行 device dispatch
- 整個 `detection/`、`ocr/`、`inpainting/` 範圍內，**沒有任何 class 或 function 簽章變動**

所以我們 trace 的架構在兩個 commit 上是相同的。腳本的 `@ d5a3eee` 檔頭記錄的是「實際建出並驗過的版本」；pin 比 clone 新的那些差異，對我們而言是 no-op（§4 第三層那類——刻意不追的上游漂移）。

## 上游 ckpt

**這些你都不用自己抓。** 每支腳本會透過 [`parity/paths.py`](../parity/paths.py) 裡同一支 `fetch()` 下載它要的檔，並在使用前驗過。以下每個雜湊都是照抄上游自己 `_MODEL_MAPPING` 的宣告、不是我們自己編的；全部出自 [beta-0.3 release](https://github.com/zyddnys/manga-image-translator/releases/tag/beta-0.3)。

| 上游檔案 | 大小 | sha256（上游宣告值） | 誰去抓 |
|---|---|---|---|
| `detect-20241225.ckpt` | 308,380,176 B | `67ce1c4ed4793860f038c71189ba9630a7756f7683b1ee5afb69ca0687dc502e` | `export_dbnet_ncnn.py` |
| `inpainting.ckpt` | 22,785,303 B | `878d541c68648969bc1b042a6e997f3a58e49b6c07c5636ad55130736977149f` | `export_aot_ncnn.py` |
| `ocr-ctc.zip` | — | `fc61c52f7a811bc72c54f6be85df814c6b60f63585175db27cb94a08e0c30101` | `export_ocr_onnx.py` / `export_ocr_ncnn.py`（並解壓） |

ckpt 快取在 `parity/out/ckpt/`（已 gitignore，大檔不會進 repo）。`fetch()` 是**每次跑都驗 sha256**（不只是剛下載完才驗）；hash 不符的檔會重下一次，仍不符就拒用、不會硬吃。下載走 `.part` 再 `os.replace`，所以中斷的執行不會留一個半截檔冒充成品。已經有現成檔就用 `YAKU_DET_CKPT` / `YAKU_INPAINT_CKPT` / `YAKU_OCR_CTC_DIR` 指過去跳過下載。

**OCR 的 ckpt 多一步：它是 zip 發的。** `export_ocr_onnx.py`（`export_ocr_ncnn.py` 走同一個 loader）會下載 `ocr-ctc.zip`、驗過、再把 `ocr-ctc.ckpt` + `alphabet-all-v5.txt` 解壓到 `parity/out/ckpt/ocr-ctc/`。注意**上游宣告 hash 的只有 zip 本身**——解壓出來的那兩個檔上游沒宣告 hash，所以腳本**不會**自己算一個塞進去釘住。驗 zip 才是在證明來源；自己算的 hash 只能證明「解壓沒把檔弄壞」，那是另一件事。（供參考，本機觀察到的值：`ocr-ctc.ckpt` 169,075,247 B、`alphabet-all-v5.txt` 95,997 B / `c1295ae1…54da33`。）

所有路徑與 env 覆蓋都集中在 `parity/paths.py`。

## 重建偵測器（DBNet）

```bash
python3 parity/export_dbnet_ncnn.py              # 匯出 + 驗證（約 2-3 分鐘，pnnx 那步最久）
python3 parity/export_dbnet_ncnn.py --skip-verify
```

驗 ckpt → 從 clone 取出 `TextDetection` → `load_state_dict`（strict）→ `model.eval()` → `torch.jit.trace` @ `[1,3,1024,768]` → pnnx → ncnn。

**產出**——`parity/out/dbnet/dbnet.ncnn.param` + `.bin`。應該精確等於：

```
dbnet.ncnn.param      13,392 B  sha256 9e6db2f8c6b0662ab00eb2100b3373d3c984a235eaac0e61c0b2a484ee1ff7b5
dbnet.ncnn.bin   153,010,556 B  sha256 f57bdbede7764a534c56e88be0269602259a7fcd47e54e8b7d954fd0fcc55c3d
```

這是 `models.json` 的值。**但檔名不是**——見[上線](#上線)。

**驗證**會自動跑：用 ncnn 載入產出、檢查 blob 契約（`in0` / `out0` / `out1`），並把 forward 拿去跟 torch eager 比對。測試頁預設是 `app-sandbox/src/main/assets/test/ch34_006.jpg`——在 repo 裡，所以**從空白 clone 就跑得起來**（`YAKU_DBNET_TESTPAGE` 可覆蓋）。跟 **release 權重**比對則是額外、選配的一步：手上有那份權重才用 `YAKU_DBNET_REF` 指過去；沒有的話腳本會明說「跳過」，不會假裝驗過。而這顆其實不需要它——對 `models.json` 的 sha256 檢查是更強的陳述。

對 torch eager 的已知容差是：

- `out0` sigmoid(ch0)：maxdiff 0.0034（mean 4.5e-05、corr 0.9999997）
- `out1` mask：maxdiff 0.272——那是單點離群。mean 6.9e-06、只有 0.004% 的像素差 >0.05，二值化 @0.5 只翻 196,608 中的 ~3 個像素。

那是 ncnn fp16 storage 在 sigmoid 陡峭處的捨入，對框與遮罩無實質影響；而且因為我們的重建與 release 逐位元相同，**這個差異本來就存在於現行上線的模型裡**，不是重建引入的。

**這顆別量化。** int8 實測**完全吐不出框**，在 ARM 上也沒比較快。維持 fp16 storage——這也是為什麼光偵測器就 ~153 MB。

**trace 的 shape 不是執行限制。** 這個網路是全卷積的——產出的 param 只有 Convolution/Deconvolution/Pooling/Concat/Split/ReLU/BinaryOp，沒有 Reshape 或 Interp——所以換任何尺寸照跑。用 768×1024 矩形是為了對齊引擎實際餵的形狀，順便繞開 ncnn 對 832–992 正方形輸入的 heap corruption。

## 重建 OCR（48px CTC、NCNN 混合精度）

兩棒接力，零手動下載。第 1 棒產出的是第 2 棒拿來驗證的 fp32 ONNX；出貨的只有第 2 棒的產物。

```bash
python3 parity/export_ocr_onnx.py                 # ckpt（自動抓 + 解壓）→ fp32 ONNX：驗證用的參考
python3 parity/export_ocr_ncnn.py                 # ckpt → NCNN（原版 + _mixed param、一份 .bin）+ 驗證
python3 parity/export_ocr_ncnn.py --skip-export   # 用現成產出重做 _mixed param、重驗
python3 parity/export_ocr_ncnn.py --fixture       # …並順便寫 JVM 測試 fixture
```

**第 1 棒**把 `OCR.forward` 以 opset 17 + 動態軸 N/W 匯出 → `parity/out/ocr_48px_ctc.onnx`，164,974,063 B、sha256 `3019b406…2c35d8`。重跑結果一致；這一棒的位元結果取決於 torch 版本。它已經不再出貨：它是第 2 棒的驗證（以及 `ocr_parity.py`）在桌面用 onnxruntime 跑的 fp32 參考。

兩棒都要讀 ckpt zip——第 2 棒是從 ckpt trace、不是從 ONNX。純解碼的 parity 腳本只需要 alphabet，而 `paths.ALPHABET` 在解壓那份不在時會**自動退回引擎資產裡那份**（`engine/src/main/assets/models/alphabet-all-v5.txt`，與上游逐位元相同）——所以它們從空白 clone 不抓任何東西就跑得起來。

**第 2 棒**是 `export_ocr_ncnn.py`，依序：

1. 從 clone 取 `OCR`、載 ckpt，丟掉 `pe.pe` buffer（烤好的位置編碼表；non-strict 載入）。
2. 對十種寬度（64 … 1024）實測 backbone 的寬度下採樣、對一組候選公式逐一比對——吻合的是 `T = ⌊W/4⌋ − 1`，記進 `t_of_w.txt`；引擎的 `sinusoidalPe` 用同一條公式，沒有任何候選吻合時腳本會停下來。
3. 包一層（`build_wrapper`）：三層 encoder 的 `PositionalEncoding.forward` 改成「加上一個**輸入張量**」而不是切自己的 buffer，沒人用的 `color` 頭拿掉，唯一輸出是 `char_logits`（文字顏色由去字後的背景判、不靠 OCR）。餵模型自己的 PE 表時，包裝後的 `char_logits` 與原模型完全相同（斷言 `max|Δ| = 0`），所以包裝只改接線、不改計算。
4. `torch.jit.trace` @ W = 256、配一張現算的正弦 PE（`make_pe`，T = 63），然後 pnnx 帶 `inputshape=[1,3,48,256],[1,63,320]` **加** `inputshape2=[1,3,48,1024],[1,255,320]`。第二組 shape 就是讓 pnnx 把 attention 的 Reshape 維度保持動態、而不是烤死在 trace 寬度的關鍵——兩組差越大，它越能確定哪個維度在變。
5. `write_mixed_param()` 從原版 param 以純文字衍生出 `ocr_48px_ctc_mixed.ncnn.param`：從 backbone 輸出的 `Squeeze` 起每一層、加上 PE 輸入 `in1` 的 `Split`，尾巴都補 `31=7`（ncnn 的逐層 featmask：bit0 關 fp16 arithmetic、bit1 關 fp16 storage/packing、bit2 關 bf16——bf16 本來就全域關、順手 mask 掉免得日後有人打開），並在最後一層 backbone 卷積與 `Squeeze` 之間插一層 `Cast`（`0=2 1=1`，fp16 → fp32）。只有表頭的數量與那些尾巴不同；權重順序不變，所以兩份 param 讀同一份 `.bin`。

**為什麼 PE 要當輸入。** 上游 `PositionalEncoding.forward` 是 `x + self.pe[:, :x.size(1)]`；trace 時 `size(1)` 是常數，pnnx 就把切片烤死在 trace 寬度，換任何寬度都是垃圾。到 v3 為止 OCR 留在 ONNX Runtime 上就是撞了這道牆。把編碼當 `in1` 餵進去——純正弦、`pe[t, 2i] = sin(t / 10000^(2i/320))`、`pe[t, 2i+1] = cos(同)`，裝置上幾行算得出來、三層 encoder 共用——圖裡就只剩 `x + in1`。引擎 `ncnn_jni.cpp:ocrCtcNative` 吃的 blob 契約：`in0` = 影像 `[3,48,W]`、值域 `(x − 127.5)/127.5`；`in1` = PE `[T,320]`；`out0` = `char_logits [T,19264]` raw；greedy CTC 與 top-1 log-softmax 信心在 JNI 算，收合與查字表在 Kotlin。

**為什麼是混合精度。** 真機（SD 8 Gen 3、9 頁 242 行、以 fp32 為真值）：全 fp16 快 27–32% 但只讀對 219——transformer 把小假名讀錯（なぃ、か6、だろぅ、やは自）；全 fp32 讀對 242 但慢 32–45%；只開 fp16 storage 慢 10×（每層進出都 cast）。backbone fp16 + transformer fp32 讀對 241（唯一不同那行是真值自己讀錯），OCR 時間比 int8 ONNX 模型少 ~23%，載入 ~0.3–0.4 秒。

**產出**——`parity/out/ocr_ncnn/ocr_48px_ctc.ncnn.param` + `ocr_48px_ctc_mixed.ncnn.param` + `ocr_48px_ctc.ncnn.bin`。檔名本來就對得上 `models.json`；出貨的位元是：

```
ocr_48px_ctc.ncnn.param          18,133 B  sha256 e701cfc5df9d3c55c9fd0a36725499d01d45c18e32946271fc26e580cd9901bd
ocr_48px_ctc_mixed.ncnn.param    18,438 B  sha256 32e298deca3acb8ba95897ca8cccbd028582ae42a697c3c05b4eaefe198309e3
ocr_48px_ctc.ncnn.bin        83,037,664 B  sha256 3e0a809441f5284871d18a3d757a7ec098c4b9ee64bdfd7ac776a84aa057c7a9
```

**驗證**會自動跑、從空白 clone 就行，跑的是**原版** param、所有 fp16 選項全關——x86 CPU 也只能這樣：

- **真 strip**——一直以來那 30 個凍結 quad（`faithful_boxes.json` on `demo03.png`）：NCNN 對 ORT fp32 參考，逐行比解碼文字、逐 timestep 比 argmax；int8 ONNX 在的話多一欄一起比。實測：**文字 30/30 逐行相同、argmax 30/30 全同**（對 int8 是 29/30——就是 int8 與 fp32 本來就不同的那行）。
- **寬度牆掃描**——取最長的那條字條、右側補白到 W+1、W+7、300、333、512、777、1000、1024、1500：每個寬度都必須解出跟 ORT fp32 在該寬度相同的文字。實測：沒有任何寬度不同。這是「trace 寬度沒有漏進圖裡」的檢查；這裡失敗就是坑 7。
- **`--fixture`** 寫 `engine/src/test/resources/ocr/`——第 0 條字條的 `in0`、它的 PE、期望的 argmax 索引與 log-prob、以及兩份 param 的副本——給 JVM 測試用：`OcrCtcParityTest`（引擎的 `sinusoidalPe` 與 CTC 收合對 numpy）與 `OcrMixedParamTest`（混合 param 必須滿足的結構規則——坑 6）。

**桌面驗不到的：混合 param。** x86 沒有 fp16 storage，所以 `_mixed` 在這裡只能 parse 與結構檢查；它**算得對不對**是真機的事（上面的 241/242）。坑 6 說明為什麼它只會壞在那一種方式。

**退役的 int8 模型。** 到 v3 為止出貨的 OCR 是 `ocr_int8.onnx`：第 1 棒的 ONNX 經 `quantize_ocr_int8.py` 動態量化（`quant_pre_process(skip_symbolic_shape=True)`、再 `quantize_dynamic(weight_type=QUInt8)`——43,625,294 B、sha256 `353e68a5…29fa4c5c`、與 `models-v2` release 逐位元相同；30 條字條 29/30、也就是以前寫的「96.7% CTC parity」）。腳本與 `ocr_parity.py` 都留著，好讓第 2 棒的 int8 那欄還能重現；它們不再有任何東西出貨。坑 4 記著那個量化器的兩個不直覺前置。

## 重建去字模型（AOT-GAN）

```bash
python3 parity/export_aot_ncnn.py            # 轉檔 + 驗證（約 1-2 分鐘）
python3 parity/export_aot_ncnn.py --skip-ref # 略過與 release 權重的比對
```

ckpt（自動下載）→ 從 clone 取 `AOTGenerator` → `load_state_dict` → `model.eval()` → **`my_layer_norm` monkey-patch**（見[坑 1](#坑-1pnnx-下不了-torchstd產出一顆死模型卻不報錯)——沒有它你會得到一顆死模型）→ `torch.jit.trace` @512 → `pnnx.convert(fp16=True, optlevel=2)` → ncnn。

如果你有真的 m-i-t 安裝，注意它的模型夾裡可能**只有** `lama_large_512px.ckpt`——`inpainting.ckpt` 是另外一個下載，腳本會處理。

**產出**——`parity/out/aot/mit_aot_fixed512.ncnn.param`（33,762 B）+ `.bin`（11,366,088 B）。檔名本來就對得上 `models.json`，不用改名。

**用判準看它、不要看雜湊**——sha256 一定不同，理由見[前面](#可重現在這裡的意思)。腳本會驗 blob 契約（`in0`/`in1`/`out0`）、在 512 與 768 兩種 shape 對 torch 比對。對 torch fp32 的容差是 s=512 max|d| 0.0477（mean 5.3e-4）、s=768 max|d| 0.1017（mean 4.8e-4）——fp16 storage 的正常誤差，release 權重同樣有。

**這是三顆裡唯一「完整判準需要空白 clone 沒有的東西」的模型**：要確認 **`out0` 與 release 權重逐位元相同**、以及跑逐層權重比對，都需要那份 release 權重。用 `YAKU_REF_MODELS` 指到放著它的資料夾（從 [`models.json`](../models.json) 裡 `models-v2` 的 url 下載），或者用 `--skip-ref` 只靠上面那組 torch 比對。

**檔名裡的 `fixed512` 是 trace shape、不是限制。** AOT-GAN 是全卷積的；引擎實跑的是 **tile 768**（`InpainterConfig.tileSize`）。這名字純屬歷史包袱——改名要連 `models.json` 與 release asset 一起換，不值得。@512 trace 的一個副產物是 layer-norm 的元素數被烤成常數（`mul_10 2=16384.0` / `div_11 2=16383.0` ＝ 128×128）。跑 768 時真值該是 36864/36863，但這只讓 Bessel 係數從 1.0000271 變成 1.0000610——相對誤差 ~3e-5，而 reduction 本身仍是動態的。768 的輸出與 release 逐位元相同，就是最實際的背書。

## 人物分割 → NCNN（夜讀）

兩顆選配模型、只有夜讀用得到——裝置上的配方是兩者人物遮罩的聯集（見 [MODELS_zh.md](MODELS_zh.md#夜讀模型)）。跟上面三顆不同：它們的來源不是 manga-image-translator 的 ckpt，腳本也**不會**去抓——權重請自己拿：`manga_seg_s.pt`（加上它的 ONNX 匯出，當驗證參考）來自 Hugging Face [anonimkaq4/manga-page-element-segmentation](https://huggingface.co/anonimkaq4/manga-page-element-segmentation)、`cartoonseg.onnx` 來自 [Jakaline/CartoonSegmentationOnnx](https://huggingface.co/Jakaline/CartoonSegmentationOnnx)——再把腳本指過去。預設找的是 `../yakuyomi-nightread/research/out/models/`（放在本 repo 旁邊的夜讀研究 repo）；`YAKU_CSEG_ONNX`、`YAKU_YOLOSEG_PT`、`YAKU_YOLOSEG_ONNX` 可覆蓋。yolo 匯出需要 `ultralytics`，`parity/requirements.txt` 沒有釘它。兩支腳本都會印出產出的 sha256 與大小。

```bash
python3 parity/export_cseg_ncnn.py                # 切 ONNX 圖 → pnnx → ncnn，+ 三層驗證
python3 parity/export_cseg_ncnn.py --fixture      # …並寫 JVM 測試 fixture
python3 parity/export_cseg_ncnn.py --skip-export  # 只驗證，用既有 param/bin
python3 parity/export_yoloseg_ncnn.py             # ultralytics export format=ncnn → ncnn，+ 驗證
python3 parity/export_yoloseg_ncnn.py --fixture
python3 parity/export_yoloseg_ncnn.py --skip-export
```

**cseg**——CartoonSegmentation 的 RTMDet-Ins，來源 `cartoonseg.onnx`。mmdeploy 匯出的 ONNX 把 NonMaxSuppression、TopK 與逐實例的動態卷積都放在**圖裡**，而 ncnn 沒有這些層。所以腳本把圖切在**原始頭輸出**之後（`onnx.utils.extract_model`）——剩下 CSPNeXt backbone + PAFPN neck + head，全是 Convolution／Swish／Pooling／Interp 這類標準層——把輸入固定成 `[1,3,640,640]`，再跑 pnnx（它的預設 `fp16=1`，即 fp16 storage；不量化）。後處理搬到 Kotlin：`CsegPost` 是腳本 numpy `postprocess()` 的移植，而後者照 mmdet 3.3 的 `RTMDetInsHead`——先驗在 `(col × stride, row × stride)`、row-major 展平，l/t/r/b 解碼框並夾到 `[0, 640]`，score > 0.05 → NMS IoU 0.6 → 最多 100 個實例，再過動態遮罩頭（相對座標 + 8 個遮罩特徵 → 1×1 卷積 10→8→8→1）→ 雙線性 ×8 到 640 → sigmoid。管線取 score > 0.3 的實例、機率 > 0.5 二值化、裁掉 pad、最近鄰放回原尺寸、取聯集。blob 契約（`engine/src/main/cpp/ncnn_jni.cpp:extractNative`；名字與順序不可改）：`in0` = `[3,640,640]` **BGR**、`(x − mean) / std`，mean `(103.53, 116.28, 123.675)`、std `(57.375, 57.12, 58.395)`，等比縮到長邊 640、**右下角** pad 114（在正規化之前填；mmdet 慣例、非置中）；`out0..out2` = `rtm_cls [1,H,W]` 三層 stride 8/16/32（單類別 logit，sigmoid 是分數）、`out3..out5` = `rtm_reg [4,H,W]`（l,t,r,b——relu 後 × stride 才是像素距離，relu 在 Kotlin 做）、`out6..out8` = `rtm_kernel [169,H,W]`（動態卷積 weights `[80,64,8]` 再 biases `[8,8,1]`）、`out9` = `mask_feat [8,80,80]`。產出：`parity/out/cseg/cartoonseg.ncnn.param` + `.bin`。

驗證分三層：(1) 切圖 ONNX（ORT）vs NCNN——十個原始輸出的 max|Δ|、fp16 storage 容差（logit 值域到 ~40）；(2) 切圖 + numpy 後處理 vs **完整 ONNX**（含圖內 NMS）——實例以框 IoU 配對後，比遮罩機率差與二值 IoU；(3) 對 `app-sandbox/src/main/assets/test/` 的測試頁，三條路（完整 ONNX／ORT 切圖 + 後處理／NCNN + 後處理）的**聯集遮罩** IoU，聯集 IoU 偏低的頁腳本會標出來。`--fixture` 把 `ch34_011` 的十個原始輸出（fp16）與期望遮罩寫進 `engine/src/test/resources/charseg/`，給 `CsegPostParityTest` 逐像素比對 Kotlin 後處理。

**yolo**——YOLO11-seg（`manga_seg_s.pt`，以 MangaSeg／Manga109-s 訓練）。整個轉檔就是 ultralytics 自己的匯出器——`YOLO(pt).export(format="ncnn", imgsz=1024, half=True)`——底層是 pnnx、跟 DBNet／AOT 同一條路：一步到位、fp16 storage、不量化。腳本把產出的 `model.ncnn.param` / `.bin` 複製成 `parity/out/yoloseg/manga_seg_s.ncnn.param` + `.bin`。blob 契約（`NcnnBackend.extract`；不可改）：`in0` = `[3,1024,1024]` **RGB**、`/255`、ultralytics letterbox（等比縮到長邊 1024、**置中** pad 114）；`out0` = `[39,21504]`，每個 anchor 是 `cx,cy,w,h`（1024 座標）+ 3 類分數（0 = frame、1 = speech_bubble、2 = character）+ 32 個遮罩係數；`out1` = `[32,256,256]` 遮罩 prototypes。後處理（`YoloSegPost`，移植自腳本的 `postprocess()`，亦即研究端產出桌面守護框數字的 `run_yoloseg_onnx`）：只取 character 類且分數 > 0.25 → NMS IoU 0.45（貪婪、分數遞減）→ `sigmoid(係數 · prototypes)` 在 256×256 → 裁到 bbox（`crop_mask`：行 `[int(y1), ceil(y2))`、列同）→ 雙線性放到 1024 → 去 letterbox → 雙線性放到原尺寸 → > 0.5 → 聯集。兩段雙線性都是 `cv2.resize INTER_LINEAR`（半像素中心、邊界夾住）；Kotlin 只在框的支撐區內算，結果相同。

驗證：NCNN vs ONNX（ORT）的 `out0` 分數通道與 `out1`，再比兩條路經同一套 numpy 後處理後、在測試頁上的**聯集遮罩** IoU。`--fixture` 把 `ch34_011` 的 `out0`／`out1`（fp16）與期望遮罩寫進 `engine/src/test/resources/charseg/`，給 `YoloSegPostParityTest`。

**兩顆都用判準看、別看雜湊。** 兩者都跟 AOT 一樣走過 pnnx（yolo 走的是 ultralytics 內建那份），冷啟動重跑能不能重現雜湊都還沒確認過；`models.json` 的值就是出貨當時的位元。產出檔名本來就對得上 `models.json`，原名直用。

## 那些坑

這些就是為什麼在此之前，除了當初做的人以外沒人重建得出來。

### 坑 1：pnnx 下不了 `torch.std`——產出一顆死模型卻不報錯

`AOTBlock.my_layer_norm` 用了 `feat.std((2,3))`。pnnx 1.0.20260526 轉得出 pnnx IR，但**下不到 ncnn 層**：`layer torch.std not exists or registered` → `network graph not ready` → `find_blob_index_by_name in0/in1/out0 failed`，extract 回 −1。

**而它不會讓轉檔失敗。** pnnx 開開心心 exit，照樣寫出一個看起來完全正常的 `.param`（29,852 B）與 `.bin`。要等到有人載入它，才會發現這顆模型是死的。**重建完不驗證，你就會發出一顆死模型。**

解法是腳本在匯出時（純記憶體、不動 clone）把 `my_layer_norm` monkey-patch 成手刻的等價式：mean → sub → `d*d` → mean → ×N ÷(N−1) → sqrt → +1e-9。兩個很容易寫錯的細節：

- **`×N ÷(N−1)` 是 Bessel 修正。** `torch.std` 預設 `unbiased=True`——它不是除以 N。漏掉這個，你的輸出會有微妙的偏差。
- **用 `d*d`、不要用 `d**2`**，這樣 pnnx 才會出 BinaryOp mul、對齊 release。

這同時也是「當初那次轉檔做的是同一件事」的證據：release 的 param 裡**沒有** `std`，而是同一組展開（`mean_87` / `mul_10 2=16384.0` / `div_11 2=16383.0` / `sqrt_12`），而且它的 op 直方圖與我們的重建**逐項相同**。

### 坑 2：DBNet 的 `out0` 是 raw logits——沒有 sigmoid

`out0` ch0 是 shrink map 的 **raw logits**。上游是在模型**外面**套 sigmoid（`detection/default.py:23`，`db = db.sigmoid()`），引擎也是（`Detector.kt:59`）。如果你「好心」把 sigmoid 併進匯出的模型，它就會被套兩次，**框會全爆**。不要加。

（ch1 那個 threshold map 則**確實**是在模型內就 sigmoid 過的。這個不對稱是上游的設計，不是我們的。）

### 坑 3：`model.eval()` 是硬性要求，不是衛生習慣

`DBHead.forward` 是照 `self.training` 分支的：train 模式會多吐一個 `binary_maps`，於是 `out0` 變成 3 channel、跟引擎介面對不上。DBNet 腳本用 `assert db.shape[1] == 2` 擋這個。

AOT-GAN 也一樣，但理由不同：`AOTGenerator.forward` 的 training 分支**不含 `clip(-1,1)`**，輸出值域會安靜地改掉。

### 坑 4：OCR 量化有兩個不直覺的前置

*（退役的 int8 路徑——只有你要重建 `ocr_int8.onnx` 給比對欄用時才相關。）*

1. **必須先常數摺疊。** 在 torch 匯出的圖裡，`layer4.5/conv1` 的權重是以 `Conv <- Identity <- initializer` 進來的。ORT 的 Conv 量化器只認「input[1] 直接是 initializer」，不會穿過那顆 Identity，於是裸跑 `quantize_dynamic` 直接死在 `ValueError: Expected onnx::Conv_1267 to be an initializer`。匯出時已經 `do_constant_folding=True` 也**不會**消掉這一顆；`quant_pre_process` 才會（節點 646 → 437、非-initializer 的 Conv 權重 1 → 0）。
2. **`skip_symbolic_shape=True` 是必要的。** 符號形狀推論碰到動態 W 就算不下去：`Cannot determine if floor(floor(W/2)/2) - 1 < 0` → `Incomplete symbolic shape inference`。反正動態量化本來就不需要形狀推論——我們要的只是它的常數摺疊那一段。

### 坑 5：ncnn 的 `.param` 與 `.bin` 必須「同一次轉檔」配對出貨

ncnn 的 `.bin` 就是照 `.param` 的層順序線性排的權重流。pnnx 版本不同 → 層順序不同 → `.bin` 位元組整個變——即使每一顆張量其實都 bit-identical。**混用（新 `.param` + 舊 `.bin`）不會報錯，而是安靜吐出全 0**（去字結果整片黑）。

實測：`ours.param` + `release.bin` → 0.0、`release.param` + `ours.bin` → 0.0；而各自配對則都得到相同的 513071.40625。在 `models.json` 裡 `.param` 與 `.bin` 是**兩個獨立 asset**——**永遠要用同一次轉檔的產物一起換**，而且別忘了 app 端可能還快取著舊的那顆。AOT 腳本的 `compare_weights()` 就是為了擋這一類錯誤而存在。

### 坑 6：featmask 改的是「這層怎麼算」、不是「這層收到什麼」

混合精度 OCR param 把 transformer 各層標 `31=7` 讓它們跑 fp32。第一版只做了這件事，結果真機 SIGSEGV 死在 `conv3x3s1_winograd43_fp16sa`——一層 **backbone** 卷積、在**另一頁**的執行緒上、離 transformer 十萬八千里。

機制（對著 ncnn 的 `net.cpp` 驗證過）：`convert_layout` 只在 `opt.use_fp16_storage && !layer->support_fp16_storage` 時才把 fp16 blob cast 成 fp32，而 masked 層的 opt 已經把 `use_fp16_storage` 關掉——條件不成立、不 cast，backbone 吐出的 fp16 pack8 blob 就原封流進 fp32 層。`Squeeze` 只 reshape、看不出事；`Permute` 用 elemsize 2 建輸出、卻以 `float*` 寫 `w*h` 個元素——兩倍的量——堆積溢出砸到隔壁哪條並發執行緒的 Winograd workspace。量小（T=30 約 19 KB）、無聲、離病因很遠。

解法是在最後一層 backbone 卷積與 `Squeeze` 之間明確插一層 `Cast`（`0=2 1=1`）：`Cast_arm` 在 asimdhp 上自己 support fp16 storage，Net 不會預先動它；它吃 fp16 pack8、吐 fp32 pack8，而 `Squeeze`（base 層、無 packing）會被 Net 自動 unpack 到 pack1。`Cast` 本身不標 mask。`OcrMixedParamTest` 守這條規則：每個 masked 層的每個輸入，都必須來自 masked 層、來自 `in1`（PE，由 extractor 直接以 fp32 塞入）、或來自 `Cast`。

推論就是混合 param 的前提：它的 `Cast` 宣告「進來的是 fp16」，這只在 CPU 有 fp16 storage（ARMv8.2 `asimdhp`）且 Net 開了 `use_fp16_storage` 時才成立。x86、舊 ARM、或 fp16 storage 關掉時進來的 blob 是 fp32，`Cast` 照 fp16 解讀，OCR 就靜默吐垃圾、不 crash。所以引擎（`Ocr.pickParam`）只在 `OcrConfig.ncnnMixed && ncnnFp16Storage && NcnnBackend.cpuSupportsFp16` 時載 `_mixed`、否則載原版；桌面腳本則完全不執行它。

### 坑 7：PE 不當輸入，trace 寬度就會漏進圖裡

`PositionalEncoding.forward` 用 `x.size(1)` 切自己的 buffer，trace 時那是常數。直接 trace → pnnx 的產出只在 trace 寬度正確、其他寬度全是垃圾——不報錯、只是字讀錯。2026-07 那次就是撞在這裡，OCR 才一直留在 ONNX Runtime 上直到 v4。匯出把編碼抽成 `in1`（見 OCR 段），`inputshape` 配一組距離很遠的 `inputshape2` 讓 attention 的 Reshape 保持動態，而 `verify()` 的寬度掃描就是回歸測試：把一條字條補白到九個不相干的寬度、每個都要讀出同樣的字。

### 比較小、但每個都花過時間的

- **`ncnn.Mat(ndarray)` 不會複製 buffer。** 傳一個暫存進去（`ncnn.Mat(np.ascontiguousarray(x))`）會讓它當場被 GC，於是你讀的是已釋放的記憶體。症狀很陰險：**同一顆模型跑兩次差 max|d| = 2.0**（整個值域），而且時好時壞。要用變數把 numpy 物件持有住。（腳本已標註那兩個變數「別簡化掉」。）
- **解析 ncnn param 時，Padding 的 `6=` 是 `per_channel_pad_data_size`、不是 `weight_data_size`。** 照後者算會整個位移、解出垃圾——而那個垃圾讀起來會很像「權重不一樣」。只有 Convolution / Deconvolution / InnerProduct 帶權重。
- **別用 `torch.randn` 驗證。** 雜訊不在去字模型的資料分布內，它的輸出本來就會亂跳，於是你會得到一堆看起來很糟、但毫無意義的數字。腳本用的是真漫畫頁 + 矩形擦除塊。
- **中間的 `.pt` 不是 bit-stable**（兩次 trace 得到 308,689,713 vs 308,689,649 B——zip metadata/timestamp），即使 ncnn 產出**是**逐位元相同的。永遠別拿 `.pt` 的雜湊當可重現性的訊號，只看 ncnn 產出。
- **`DBNet_resnet34.py` 裡的 `ImageMultiheadSelfAttention` 是死碼**——`TextDetection` 根本沒用到它，產出的 param 也證實沒有任何 attention 層。別跑去為 attention 的轉檔除錯。

## 上線

產出都落在 `parity/out/`（已 gitignore）。十個檔裡有兩個上線的檔名跟建出來的不一樣：

| 建出來 | 上線名 | `models.json` 角色 |
|---|---|---|
| `dbnet.ncnn.param` / `.bin` | **`dbnet_detect.ncnn.param` / `.bin`**——必須改名 | detector |
| `ocr_48px_ctc.ncnn.param` / `ocr_48px_ctc_mixed.ncnn.param` / `ocr_48px_ctc.ncnn.bin` | 同名——原名直用 | ocr |
| `mit_aot_fixed512.ncnn.param` / `.bin` | `mit_aot_fixed512.ncnn.param` / `.bin`——原名直用 | inpainter |
| `cartoonseg.ncnn.param` / `.bin` | 同名——原名直用 | charseg |
| `manga_seg_s.ncnn.param` / `.bin` | 同名——原名直用（腳本已把 ultralytics 的 `model.ncnn.*` 改好名） | charseg |

```bash
cp parity/out/dbnet/dbnet.ncnn.param /tmp/ship/dbnet_detect.ncnn.param
cp parity/out/dbnet/dbnet.ncnn.bin   /tmp/ship/dbnet_detect.ncnn.bin
```

偵測器改名是人工步驟，所以很容易漏。自備模型（BYOM）就算不改名也還是會認得——`ModelSet` 是 substring 比對（`.param` 含 `dbnet` → 偵測、`.param` 含 `aot` → 去字、`.param` 含 `ocr` → OCR——哪份 param 都行，引擎自己挑 `_mixed` 或原版；`manga_seg`／`cartoonseg` → 兩顆選配的夜讀分割器），`.bin` 則靠把副檔名換掉找同名檔——但 **release asset 一定要用 `models.json` 宣告的名字**，否則自動下載會失敗。

如果你發佈的權重跟現行的不同，請在同一個改動裡一起更新 `models.json` 的 `size` 與 `sha256`——manifest 跟檔案一起版本化，這正是那個檢查有意義的原因。

## 桌面重建驗不到的東西

講明白，免得有人白花一天：

- **效能與精度數字是裝置端量的。** 「比 int8 快 ~23%」、混合精度的 241/242，以及 [MODELS_zh.md](MODELS_zh.md) 裡每頁偵測／OCR 用時（9 頁：0.79 秒／1.25 秒）那組數字，都是在真機（SD 8 Gen 3）上量的，夜讀那組（yolo ~0.46 秒、cseg ~0.83 秒、每頁 6–25 秒）也是。**這條重建流程量不出來。**
- **混合精度 OCR param 在 x86 上跑不了。** 那裡沒有 fp16 storage，而它的 `Cast` 假設進來的是 fp16（坑 6）。腳本只 parse 與結構檢查它；實際跑的是原版 param。
- **這些腳本在 x86 上量到的時間是噪音。** 同一顆**逐位元相同**的 OCR 模型（退役的 int8 那顆）跑兩次，量到 1732 ms 與 3336 ms——同一個檔、~2× 的落差。x86 上看到的 fp32 vs int8「~29×」同樣是假象。**別從桌面跑的結果讀出任何速度結論。**
- **`out1` mask 的解析度隨平台而異，兩端都別寫死。** x86 上回來的是半解析（H/2 × W/2）、arm64 上回來的是全解析。引擎的做法是「配全解析上限的緩衝 + 由 JNI 回實際尺寸」動態讀（commit `7c62f78` 修的就是這個越界）。別讓桌面量到的結果說服你把尺寸寫死在任何一端。
