# `parity/` — 桌面驗證 harness

[English](README.md) ｜ 中文

不出貨。開發專用的 Python harness，跑跟 Kotlin `:engine` 同樣的 pipeline 階段，讓我們在信任它上機前，
先檢查裝置端移植跟參考（[manga-image-translator](https://github.com/zyddnys/manga-image-translator)，m-i-t）一致。

引擎把 m-i-t（Python/torch）重寫成 Kotlin/NCNN，這種移植沒辦法逐行 diff，所以正確性是「同輸入、近輸出」。
這些腳本產出那份參考輸出，grouping 還有自動化跨語言斷言。見 [`../docs/ARCHITECTURE_zh.md`](../docs/ARCHITECTURE_zh.md#兩半)。

---

## 設定

```bash
pip install -r parity/requirements.txt    # numpy, opencv-python, onnxruntime, pillow,
                                          # networkx, shapely（torch/opencc 只給工具用）
```

所有路徑集中在一處——**`parity/paths.py`**——而且跟機器有關的那些都**可用環境變數覆蓋**，
換機器 / CI / 公開後別人不用改腳本就能跑：

| 是什麼 | `paths.py` | env 覆蓋 | 預設 |
|--------|-----------|----------|------|
| ONNX 模型 | `MODELS` | —（repo 內） | `engine/src/main/assets/models` |
| OCR 字元表 | `ALPHABET` | `YAKU_ALPHABET` / `YAKU_OCR_CTC_DIR` | `/tmp/ocr-ctc/alphabet-all-v5.txt` |
| OCR checkpoint | `OCR_CKPT` | `YAKU_OCR_CKPT` / `YAKU_OCR_CTC_DIR` | `/tmp/ocr-ctc/ocr-ctc.ckpt` |
| m-i-t clone | `MIT_CLONE` | `YAKU_MIT_CLONE` | `/mnt/d/Gits/manga-image-translator` |
| 測試頁 + m-i-t 輸出 | `RAW_DIR` / `MIT_DIR` | `YAKU_TEST_DIR` | `~/OneDrive/Manga/yakuyomi/test/{raw,mit}` |
| API key | `API_KEYS` | —（repo 根、gitignored） | `api-keys.properties`（`DEEPSEEK_API_KEY=`） |

```bash
# 例：指到別的測試夾 + m-i-t clone，不用改腳本：
YAKU_TEST_DIR=~/manga-test YAKU_MIT_CLONE=~/src/mit python3 pipeline_parity.py raw/002.jpg
```

輸出落在 `parity/out/`（快取 JSON + 比對 PNG；gitignored）。

---

## Fixture（`parity/fixtures/`，入庫）

**刻意放進 repo** 的驗證素材，讓我們公開宣稱的數字可以從空白 clone 重新量出來：

- `faithful_boxes.json`——每一次 OCR 轉檔都拿來驗的 30 個文字行 quad：`export_ocr_ncnn.py` 用它們把出貨的 NCNN OCR
  對 ORT fp32 逐行比（30/30 相同），更早之前它們定義的是退役 int8 模型的 96.7% 那個數字。**凍結**：它由 `ctd_reference.py` 跑**已退役**的 comic-text-detector 產出，
  該模型已不在任何 models release 裡 ⇒ 重產不出來；而且凍結才對——這個數字要量的是「**OCR 模型對**
  在同一批 strip 上讀出的字是否一致」，不是偵測器的性質。來歷寫在檔案裡（`_provenance`）。
- 測試頁——`app-sandbox/src/main/assets/test/demo03.png`（舊名 `page.png`；commit `ea3e166` 只是
  **改名**、位元完全相同）。與上面那 30 框是一組。
- 字表——`engine/src/main/assets/models/alphabet-all-v5.txt`（與上游逐位元相同），`paths.ALPHABET`
  缺 ckpt 時自動退回這份 ⇒ 純解碼的腳本不必抓 ckpt zip。

重現這兩個檢查（都需要 fp32 ONNX 參考，見 `docs/BUILD_MODELS_zh.md`）：

```bash
python3 parity/export_ocr_ncnn.py --skip-export   # 出貨的 NCNN OCR vs ORT fp32：逐行文字 + 寬度掃描
python3 parity/ocr_parity.py                      # 退役的 int8 vs fp32：印出「逐行 exact match = N/30 = xx.x%」
```

實測：NCNN 文字 **30/30** 與 fp32 相同、寬度掃描全過。退役的 int8 模型是 **29/30 = 96.7%**（2026-07-16）；
它唯一不同的那行是低信心行（p=0.66）、且 int8 當時讀得**比較對**。真機的效能與精度宣稱（如「比 int8 快 ~23%」、
「241/242 行」）是**真機數字、桌面驗不出來**——`_mixed` param 在 x86 上更是根本跑不了。

---

## 有什麼

**端到端**
- `pipeline_parity.py <img…>`——整條 detect→OCR→group→translate→inpaint→typeset。
  主驅動；寫 `out/final_<name>.png` + 快取中間結果。（端到端仍跑退役的 ctd + LaMa、OCR 用 fp32 ONNX；出貨的
  DBNet/OCR/AOT 三顆 NCNN 模型走 per-stage 驗證——`export_*_ncnn.py` 轉檔比對、分組測試。）

**逐階段 parity**（跑/檢視單一階段）
- `ctd_reference.py [page]`——偵測：faithful（m-i-t 後處理）vs simplified，並排。
  凍在歷史：需要已退役的 comic-text-detector ONNX（見上面 Fixture）。
- `ocr_parity.py`——對凍結的 30 框用 ORT 做 48px CTC 辨識（fp32；退役的 int8 在的話順便印它的歷史 parity 數字）。
  出貨的 NCNN OCR 改由 `export_ocr_ncnn.py` 驗。空白 clone 可跑（fixture + repo 內字表）。
- `group_exp.py <name…>`——分組：我們的區域 vs m-i-t 的，畫成框。
- `translate_parity.py`——OCR 出的日文 → DeepSeek → 繁中。
- `merge_translate_parity.py`——先併行再翻。
- `inpaint_parity.py`——對區域跑 LaMa 去字。已凍結：LaMa 已從引擎退役、僅留凍結參考；現行 AOT-GAN
  去字由 `export_aot_ncnn.py`／`compare_inpaint.py` 產出/比對。
- `typeset_parity.py [v|h|auto]` / `retypeset.py <name…>`——排版（retypeset = 從快取重排、不重打 LLM；快速調版用）。

**規格本**（ground truth，從 m-i-t 複製——跟 `.upstream-ref` 同步）
- `mit_grouping.py`——m-i-t 的兩階段分組（`merge_bboxes_text_region`），自含。
- `ctd_reference.py`——也拉 m-i-t 的偵測後處理。

**工具**
- `export_ocr_onnx.py`——把 48px CTC checkpoint 匯出成 fp32 ONNX：NCNN 轉檔拿來驗證的桌面參考、不出貨（build-time，需 torch）。
- `quantize_ocr_int8.py`——把上面那顆 fp32 OCR ONNX 動態量化成 int8 → `ocr_int8.onnx`：models-v3 以前出貨的 OCR，
  OCR 改跑 NCNN 後退役；留著給 `export_ocr_ncnn.py` 驗證裡的 int8 那欄用。
- `export_ocr_ncnn.py`——從上游 ckpt 產出**出貨的** OCR：48px CTC 的 NCNN 檔（`ocr_48px_ctc.ncnn.param` + `ocr_48px_ctc_mixed.ncnn.param`
  共用一份 `.bin`；`write_mixed_param()` 衍生混合精度 param——backbone fp16、transformer fp32）。正弦位置編碼當第二個輸入
  （`in1`，T=floor(W/4)−1）而不是烤進圖裡，先前的嘗試在 trace 寬度以外全壞就是卡在這。對 ORT fp32 驗 30 條 fixture 字條
  與一組寬度掃描（CTC 文字相同），並把 JVM fixture 寫到 `engine/src/test/resources/ocr/`。
- `export_dbnet_ncnn.py`——從上游 ckpt 產出出貨的 DBNet 偵測器 NCNN 檔（`dbnet_detect.ncnn.param`/`.bin`）。
- `export_aot_ncnn.py`——從上游 ckpt 產出出貨的 AOT-GAN 去字 NCNN 檔（`mit_aot_fixed512.ncnn.param`/`.bin`）。
- `compare_inpaint.py`——去字模型×方法比較 + 計時；驗證出貨的 AOT-GAN 去字。
- `seg_validate.py`——在不同閾值下檢視偵測器的 `seg` 筆畫遮罩。
- `emit_grouping_fixture.py`——產生 Kotlin 分組測試 fixture（見下）。

**夜讀重繪（暗色模式原型）**
- `nightread.py <頁圖> [-o 夾]`——單頁一條龍：DBNet 偵測 → 三分區遮罩（氣泡/留白/畫面）→
  合成暗色閱讀頁。設計紅線（畫面絕不反相、氣泡＝深底亮字）、可調常數與三個修法
  （氣泡元件面積上限 / 無框頁型降級 / 留白格框感知）全在檔頭。輸出到 `out/nightread/`：
  `<頁名>_final.png` + `_cmp.png`（三聯：原圖｜成品｜遮罩視覺化）+ 各遮罩/regions json。
- `nightread_batch.py [頁名…]`——批次跑一組頁（預設 sandbox 11 張測試頁）、印白面積表、
  落 `nightread_stats.json`。頁名直接寫 `demo01` 這種（自動在 sandbox test 夾找）。

---

## 跨語言分組測試

唯一一個橫跨兩語言的自動化 parity 檢查：

```
emit_grouping_fixture.py                          # 桌面：偵測真實頁面、用 mit_grouping 分組、
   → engine/src/test/kotlin/.../GroupingFixture.kt #   把偵測到的行 + 期望區域 emit 成 Kotlin
                                                   #
gradlew :engine:testDebugUnitTest                 # 裝置端：把同樣的行餵給 Kotlin Grouping，
   → GroupingParityTest                            #   斷言區域（bbox ±2px）+ 角度（±1°）吻合
```

所以動了 Kotlin 分組（或重新同步 `mit_grouping.py`）會被自動抓到：改、重跑 `emit_grouping_fixture.py`、跑測試。
其餘階段仍靠目視驗證（拿 `out/*.png` 對 `…/test/mit/`）。

---

## 典型流程

1. `pipeline_parity.py raw/002.jpg raw/012.jpg`——端到端，目視 `out/final_*.png` vs `mit/`。
2. 只調版？改 `typeset_parity.py`、`retypeset.py 002 012`（不打 LLM）。
3. 動了分組？`emit_grouping_fixture.py` 然後 `:engine:testDebugUnitTest`。
4. 同步了 m-i-t？bump `mit_grouping.py` / `.upstream-ref`，重跑相關 parity、修到綠。
