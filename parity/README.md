# `parity/` — desktop validation harness

**English** · [中文](README_zh.md)

Not shipped. A developer-only Python harness that runs the same pipeline stages as the Kotlin
`:engine`, so we can check the on-device port matches the reference
([manga-image-translator](https://github.com/zyddnys/manga-image-translator), m-i-t) before trusting
it on a device.

The engine re-implements m-i-t (Python/torch) in Kotlin/ONNX, and that port can't be diffed line for
line, so correctness means "same input, close output". These scripts produce the reference output,
and for grouping an automated cross-language assertion. See
[`../docs/ARCHITECTURE.md`](../docs/ARCHITECTURE.md#the-two-halves).

---

## Setup

```bash
pip install -r parity/requirements.txt    # numpy, opencv-python, onnxruntime, pillow,
                                          # networkx, shapely  (torch/opencc only for tools)
```

All paths live in one place — **`parity/paths.py`** — and the machine-specific ones are
**env-overridable**, so a different machine / CI / public user can run without editing scripts:

| What | `paths.py` | env override | default |
|------|-----------|--------------|---------|
| ONNX models | `MODELS` | — (repo-relative) | `engine/src/main/assets/models` |
| OCR alphabet | `ALPHABET` | `YAKU_ALPHABET` / `YAKU_OCR_CTC_DIR` | `/tmp/ocr-ctc/alphabet-all-v5.txt` |
| OCR checkpoint | `OCR_CKPT` | `YAKU_OCR_CKPT` / `YAKU_OCR_CTC_DIR` | `/tmp/ocr-ctc/ocr-ctc.ckpt` |
| m-i-t clone | `MIT_CLONE` | `YAKU_MIT_CLONE` | `/mnt/d/Gits/manga-image-translator` |
| test pages + m-i-t outputs | `RAW_DIR` / `MIT_DIR` | `YAKU_TEST_DIR` | `~/OneDrive/Manga/yakuyomi/test/{raw,mit}` |
| API key | `API_KEYS` | — (repo root, gitignored) | `api-keys.properties` (`DEEPSEEK_API_KEY=`) |

```bash
# e.g. point at a different test dir + m-i-t clone, no script edits:
YAKU_TEST_DIR=~/manga-test YAKU_MIT_CLONE=~/src/mit python3 pipeline_parity.py raw/002.jpg
```

Outputs land in `parity/out/` (cached JSON + comparison PNGs; gitignored).

---

## Fixtures (`parity/fixtures/`, committed)

Verification material that is **in the repo on purpose**, so the numbers we publish can be
re-measured from a clean clone:

- `faithful_boxes.json` — the 30 text-line quads that define the OCR **int8-vs-fp32 CTC parity**
  figure quoted in `models.json` / `docs/MODELS.md`. Frozen: they came from `ctd_reference.py`
  running the **retired** comic-text-detector, which no longer ships in any models release, so
  they cannot be regenerated — and freezing them is what makes the number a stable measurement
  of the *OCR model pair* rather than of the detector. Provenance is inside the file (`_provenance`).
- Test page — `app-sandbox/src/main/assets/test/demo03.png` (was `page.png`; commit `ea3e166`
  **renamed** it, same bytes). Paired with the quads above.
- Alphabet — `engine/src/main/assets/models/alphabet-all-v5.txt` (byte-identical to upstream's),
  which `paths.ALPHABET` falls back to, so decode-only scripts need no checkpoint download.

Reproduce the parity number (needs both OCR models — see `docs/BUILD_MODELS.md`):

```bash
python3 parity/ocr_parity.py     # prints "逐行 exact match = N/30 = xx.x%"
```

Measured 2026-07-16: **29/30 = 96.7%**, matching the published figure. The one differing line is
low-confidence (p=0.66) and int8 is the *better* read there — so 96.7% is not a 3.3% quality loss.
Speed claims (e.g. "~3.6× on ARM") are **device numbers and cannot be verified on desktop**.

---

## What's here

**End-to-end**
- `pipeline_parity.py <img…>` — full chain detect→OCR→group→translate→inpaint→typeset for a page.
  The main driver; writes `out/final_<name>.png` + caches intermediates. (Still runs the retired ctd +
  LaMa end-to-end; the shipped DBNet/AOT are validated per-stage — int8 OCR parity, the grouping test,
  and the `export_*_ncnn.py` build/verify.)

**Per-stage parity** (run/inspect one stage)
- `ctd_reference.py [page]` — detection: faithful (m-i-t post-processing) vs simplified, side by side.
  Frozen in history: needs the retired comic-text-detector ONNX (see Fixtures above).
- `ocr_parity.py` — 48px CTC recognition on the frozen quads; with an int8 model present it also
  reports the published CTC parity figure. Runs from a clean clone (fixtures + repo alphabet).
- `group_exp.py <name…>` — grouping: our regions vs m-i-t's, drawn as boxes.
- `translate_parity.py` — OCR'd JP → DeepSeek → CHT.
- `merge_translate_parity.py` — line-merge then translate.
- `inpaint_parity.py` — LaMa text removal on regions. Frozen: LaMa is retired from the engine; the
  shipped AOT-GAN inpaint is built and compared via `export_aot_ncnn.py` / `compare_inpaint.py`.
- `typeset_parity.py [v|h|auto]` / `retypeset.py <name…>` — typesetting (retypeset = re-render from
  cache without re-calling the LLM; for tuning layout fast).

**Vendored spec** (ground truth, copied from m-i-t — keep in sync with `.upstream-ref`)
- `mit_grouping.py` — m-i-t's two-stage grouping (`merge_bboxes_text_region`), self-contained.
- `ctd_reference.py` — also pulls m-i-t's detection post-processing.

**Tools**
- `export_ocr_onnx.py` — export the 48px CTC checkpoint to ONNX (build-time, needs torch).
- `quantize_ocr_int8.py` — dynamic-quantize that fp32 OCR ONNX to int8 → `ocr_int8.onnx` (the shipped OCR weights).
- `export_dbnet_ncnn.py` — build the shipped DBNet detector NCNN files (`dbnet_detect.ncnn.param`/`.bin`) from the upstream ckpt.
- `export_aot_ncnn.py` — build the shipped AOT-GAN inpaint NCNN files (`mit_aot_fixed512.ncnn.param`/`.bin`) from the upstream ckpt.
- `compare_inpaint.py` — inpaint model × method comparison + timing; verifies the shipped AOT-GAN removal.
- `seg_validate.py` — inspect the detector's `seg` stroke mask at thresholds.
- `emit_grouping_fixture.py` — generate the Kotlin grouping test fixture (see below).

**Night-read rebuild (dark-mode prototype)**
- `nightread.py <page> [-o dir]` — single page end-to-end: DBNet detection → three-zone masks
  (bubbles / gutters / scene) → composed dark reading page. Design red lines (scene is never
  inverted; bubbles = dark fill + bright text), tunable constants and the three fixes
  (bubble component area cap / frameless-page downgrade / panel-aware gutters) are all in the
  file header. Outputs to `out/nightread/`: `<name>_final.png` + `_cmp.png` (triptych:
  original | result | mask viz) + per-mask PNGs / regions json.
- `nightread_batch.py [names…]` — run a page set (default: the 11 sandbox test pages), print
  the white-area table, write `nightread_stats.json`. Bare names like `demo01` resolve against
  the sandbox test dir.
- Two experiment switches, both already set to the chosen values — override only to reproduce
  the comparisons: `NIGHTREAD_CURVE` picks the scene tone curve (`lin8` = chosen: black stays
  black, paper white compressed to mid-grey; `d2`/`lin`/`knee` are the alternatives that lost
  the A/B) and `NIGHTREAD_AURA` picks how gutter fill keeps clear of bleed-page figures
  (`hard` = chosen: binary keep-out with a margin exemption; `glow` = distance-field gradient
  that follows the contour, rejected as style-adding).

---

## The cross-language grouping test

The one automated parity check spans both languages:

```
emit_grouping_fixture.py                          # desktop: detect real pages, group with mit_grouping,
   → engine/src/test/kotlin/.../GroupingFixture.kt #   emit detected lines + expected regions as Kotlin
                                                   #
gradlew :engine:testDebugUnitTest                 # device-side: feed the same lines to Kotlin Grouping,
   → GroupingParityTest                            #   assert regions (bbox ±2px) + angle (±1°) match
```

So a change to the Kotlin grouping (or a re-sync of `mit_grouping.py`) is caught automatically: edit,
re-run `emit_grouping_fixture.py`, run the test. Other stages are still validated visually (compare
`out/*.png` against `…/test/mit/`).

---

## Typical loop

1. `pipeline_parity.py raw/002.jpg raw/012.jpg` — end-to-end, eyeball `out/final_*.png` vs `mit/`.
2. Tuning layout only? edit `typeset_parity.py`, `retypeset.py 002 012` (no LLM call).
3. Touched grouping? `emit_grouping_fixture.py` then `:engine:testDebugUnitTest`.
4. Synced m-i-t? bump `mit_grouping.py` / `.upstream-ref`, re-run the relevant parity, fix to green.
