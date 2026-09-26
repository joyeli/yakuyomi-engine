# Building the models

English ｜ [中文](BUILD_MODELS_zh.md)

The three models the engine loads are our own conversions of [manga-image-translator](https://github.com/zyddnys/manga-image-translator)'s upstream checkpoints — there is no upstream distributable to point at, so we build and host them. [MODELS.md](MODELS.md) covers what they are and how to get them; this page covers how to rebuild them from the upstream checkpoints, and how to tell whether your rebuild came out right. The two optional character-segmentation models night reading uses are conversions as well — of third-party weights, not manga-image-translator's — and have [their own section](#character-segmentation--ncnn-night-reading) after the three rebuild paths.

Nothing here is a recipe you have to reconstruct by hand — each path is one script. What follows is the environment those scripts need, the criteria for judging their output, and the traps that make an unverified rebuild silently wrong.

> **You probably don't need this page.** The weights we built are downloadable and checksum-verified — [MODELS.md](MODELS.md) covers that, and it's what both the reader and the sandbox use. Come here if you want to audit our conversions, retarget a different upstream checkpoint, or rebuild under your own toolchain. To just *see the engine run*, go to the [repo README](../README.md#try-it); to integrate it, [engine/README.md](../engine/README.md).

## What "reproducible" means here

Read this before you compare any hashes, because the obvious check is wrong for one of the three models and unconfirmed for another.

**The sha256 values in [`models.json`](../models.json) are a distribution integrity check, not a reproducibility criterion.** They exist so the app can confirm the file it downloaded is the file we published. They are not a definition of "correctly rebuilt" — a rebuild can be numerically identical and still hash differently.

| Model | Criterion | Notes |
|---|---|---|
| Detector (DBNet) | **Bit-identical** — sha256 matches `models.json` | Verified: reproduces `9e6db2f8…` / `f57bdbed…` exactly, on a cold rerun |
| OCR (NCNN) | **Numerical equivalence** — all 30 fixture strips decode identically to the ORT fp32 reference, argmax agrees on every timestep, the width sweep is clean. The `models.json` hashes are what shipped | Same trace → pnnx path as DBNet, so the pinned toolchain is expected to reproduce the bytes, but that has not been re-checked on a cold rerun — judge by the criterion, not the hash |
| Inpaint (AOT-GAN) | **Numerical equivalence** — `out0` bit-identical + per-layer weight compare. **The sha256 will not match** | Expected and understood: pnnx's layer auto-naming and ordering differ. See below |
| Character segmentation (yolo / cseg, night reading) | **Numerical equivalence** — NCNN raw outputs against the source ONNX within fp16-storage tolerance, and the union character mask's IoU against the ONNX path over the test pages. The `models.json` hashes are what shipped | Not re-checked on a cold rerun; the yolo path runs the pnnx that ultralytics bundles, so don't expect the bytes to survive a version change — judge by the criterion |

**Why AOT's hash never matches.** Our rebuild produces `mit_aot_fixed512.ncnn.param` at 33,762 B against the released 33,810 B (−48 B), and a `.bin` of exactly the released size but different bytes. Both differences are pnnx version artifacts, and both were traced to the end:

- **param**: layer auto-naming (`conv_24` / `relu_0` / `reflectpad2d_40` here vs `conv_70` / `relu_6` / `pad_0` in the release), plus the release writing out Padding's defaults `5=0 6=0`. The op histogram diff is empty, and layer/blob counts match at 402/500.
- **bin**: 77% of bytes differ, but parsed layer by layer, the 76 weight tensors are the same set, each bit-identical — a newer pnnx just orders AOTBlock's parallel dilated branches and its fuse conv differently.

The check that matters is behavioural, and the script runs it once you give it the released weights to compare against: on a real page, `out0` is bit-identical at both s=512 and s=768 (`np.array_equal` true, max|d| = 0.0).

**Bit-identity is bound to the pinned versions.** The DBNet result above holds for torch 2.1.1 + pnnx 1.0.20260526 on x86 Linux. On other versions they will most likely degrade to numerical equivalence — **that is expected, not a failure**. Judge those rebuilds by the same tolerances listed per model below.

## Prerequisites

### Python environment

```bash
pip install -r parity/requirements.txt
```

The model-build section of that file is **pinned on purpose** — bit-for-bit reproduction depends on this exact set:

| Package | Version | Used by |
|---|---|---|
| torch | 2.1.1 | all three export paths |
| torchvision | 0.16.1 | DBNet — ResNet34 backbone |
| onnx | 1.17.0 | OCR — the retired int8 path only (`quantize_ocr_int8.py`) |
| onnxruntime | 1.23.0 | OCR — runs the fp32 ONNX reference the NCNN export is verified against (and the retired int8 path) |
| pnnx | 1.0.20260526 | torch → ncnn |
| ncnn | 1.0.20260526 | verification: load the exported model and compare forward |

Verified on Python 3.10.12 / numpy 1.26.4 / opencv 4.11, x86 Linux (WSL2).

**pnnx is a pip package, not a binary you have to build.** `pip install pnnx` gives you both the Python module (`import pnnx`, used by the AOT script) and a console script at `~/.local/bin/pnnx` (invoked as a subprocess by the DBNet and OCR scripts). Override the binary path with `YAKU_PNNX` if yours lands elsewhere.

### The upstream clone

All the export scripts read the model definitions out of a manga-image-translator clone rather than vendoring copies:

```bash
git clone https://github.com/zyddnys/manga-image-translator
export YAKU_MIT_CLONE=/path/to/manga-image-translator   # default: /mnt/d/Gits/manga-image-translator
```

**The scripts only read the clone — they make no changes to its files or git state.** They can't simply import it: upstream's `manga_translator/__init__` drags in translators → tiktoken → openai, and `detection/__init__` imports a `rusty_manga_image_translator` that isn't there. Each script gets around this in memory, by installing fake module stubs and package shells that carry a `__path__` but never execute the `__init__` body. The model classes themselves are plain torch modules and resolve fine that way.

### A note on `.upstream-ref`

There is a discrepancy worth stating plainly. [`.upstream-ref`](../.upstream-ref) pins `efdc229` (2026-07-01), but the clone these models were built against sits at `d5a3eee` (2026-05-24), and the scripts' `ported spec` headers say `@ d5a3eee` (`export_ocr_ncnn.py` reuses `export_ocr_onnx.py`'s loader, so it inherits that pin).

This does not affect rebuilds, and that was checked rather than assumed. Diffing `d5a3eee..efdc229` across the watched model paths:

- `detection/default_utils/DBNet_resnet34.py` — **no diff at all**
- `inpainting/inpainting_aot.py` and `ocr/model_48px_ctc.py` — **one line each**, and in both cases it is device dispatch in the *loader* class (`cuda`/`mps`/`xpu`), not in the `nn.Module` we export
- `detection/default.py` — the same one-line device dispatch
- Across `detection/`, `ocr/` and `inpainting/` as a whole, **no class or function signature changes**

So the architectures we trace are identical at either commit. The scripts' `@ d5a3eee` headers record what was actually built and verified; the pin is ahead of the clone by changes that are, for our purposes, no-ops (§4 tier three material — the kind of upstream drift we deliberately don't chase).

## Upstream checkpoints

**You don't have to fetch any of these by hand.** Each script downloads what it needs through one shared `fetch()` in [`parity/paths.py`](../parity/paths.py) and verifies it before use. Every hash below is copied from upstream's own `_MODEL_MAPPING` declaration, not invented here; all come from the [beta-0.3 release](https://github.com/zyddnys/manga-image-translator/releases/tag/beta-0.3).

| Upstream file | Size | sha256 (upstream-declared) | Fetched by |
|---|---|---|---|
| `detect-20241225.ckpt` | 308,380,176 B | `67ce1c4ed4793860f038c71189ba9630a7756f7683b1ee5afb69ca0687dc502e` | `export_dbnet_ncnn.py` |
| `inpainting.ckpt` | 22,785,303 B | `878d541c68648969bc1b042a6e997f3a58e49b6c07c5636ad55130736977149f` | `export_aot_ncnn.py` |
| `ocr-ctc.zip` | — | `fc61c52f7a811bc72c54f6be85df814c6b60f63585175db27cb94a08e0c30101` | `export_ocr_onnx.py` / `export_ocr_ncnn.py` (also unzips) |

Checkpoints are cached in `parity/out/ckpt/` (gitignored, so nothing large enters the repo). `fetch()` verifies sha256 on **every** run, not just after downloading; a file whose hash doesn't match is re-downloaded once and then refused rather than used. Downloads go through a `.part` file, so an interrupted run can't leave a truncated file masquerading as the real one. Point `YAKU_DET_CKPT` / `YAKU_INPAINT_CKPT` / `YAKU_OCR_CTC_DIR` at copies you already have to skip the downloads.

**The OCR checkpoint takes one extra step: it ships as a zip.** `export_ocr_onnx.py` (and `export_ocr_ncnn.py`, through the same loader) downloads `ocr-ctc.zip`, verifies it, and extracts `ocr-ctc.ckpt` + `alphabet-all-v5.txt` into `parity/out/ckpt/ocr-ctc/`. Note that **upstream declares a hash for the zip only** — the two extracted files have no upstream-declared hash, so the scripts don't invent one and pin it. Verifying the zip is what establishes provenance; a hash we computed ourselves could only prove the unzip didn't corrupt anything, which is a different claim. (For reference, what we observe locally: `ocr-ctc.ckpt` 169,075,247 B, `alphabet-all-v5.txt` 95,997 B / `c1295ae1…54da33`.)

`parity/paths.py` holds every path and env override in one place.

## Rebuilding the detector (DBNet)

```bash
python3 parity/export_dbnet_ncnn.py              # export + verify (~2-3 min; pnnx is the slow step)
python3 parity/export_dbnet_ncnn.py --skip-verify
```

Checkpoint verify → pull `TextDetection` out of the clone → `load_state_dict` (strict) → `model.eval()` → `torch.jit.trace` at `[1,3,1024,768]` → pnnx → ncnn.

**Output** — `parity/out/dbnet/dbnet.ncnn.param` + `.bin`. Expect, exactly:

```
dbnet.ncnn.param      13,392 B  sha256 9e6db2f8c6b0662ab00eb2100b3373d3c984a235eaac0e61c0b2a484ee1ff7b5
dbnet.ncnn.bin   153,010,556 B  sha256 f57bdbede7764a534c56e88be0269602259a7fcd47e54e8b7d954fd0fcc55c3d
```

Those are the `models.json` values. **The file name is not** — see [Shipping](#shipping-a-rebuild).

**Verification** runs automatically: it loads the output in ncnn, checks the blob contract (`in0` / `out0` / `out1`), and compares a forward pass against torch eager. The test page defaults to `app-sandbox/src/main/assets/test/ch34_006.jpg`, which is in the repo, so this works from a clean clone (`YAKU_DBNET_TESTPAGE` to override). Comparing against the *released* weights is an extra step and optional — point `YAKU_DBNET_REF` at a folder holding them if you have them; the script says it skipped rather than pretending otherwise. For this model you don't need them anyway: the sha256 check against `models.json` is the stronger statement.

Against torch eager, the known tolerance is:

- `out0` sigmoid(ch0): maxdiff 0.0034 (mean 4.5e-05, corr 0.9999997)
- `out1` mask: maxdiff 0.272 — a single outlier. Mean 6.9e-06, only 0.004% of pixels differ by >0.05, and thresholding at 0.5 flips ~3 of 196,608 pixels.

That is ncnn's fp16 storage rounding where sigmoid is steep, it has no practical effect on boxes or masks, and — since our rebuild is bit-identical to the release — **it is already present in the shipping model**. It is not something the rebuild introduces.

**Don't quantize this one.** int8 makes it emit no boxes at all, and isn't faster on ARM either. fp16 storage is why the detector alone is ~153 MB.

**The trace shape is not a runtime limit.** The network is fully convolutional — the output param contains only Convolution/Deconvolution/Pooling/Concat/Split/ReLU/BinaryOp, no Reshape or Interp — so it runs at any size. The 768×1024 rectangle matches what the engine actually feeds it, which also keeps clear of an ncnn heap-corruption bug on square inputs in the 832–992 band.

## Rebuilding the OCR model (48px CTC, NCNN mixed precision)

Two stages, no manual downloads. The first produces the fp32 ONNX the second is verified against; only the second produces what ships.

```bash
python3 parity/export_ocr_onnx.py                 # ckpt (auto-fetch + unzip) -> fp32 ONNX: the verification reference
python3 parity/export_ocr_ncnn.py                 # ckpt -> NCNN (plain + _mixed param, one .bin), + verify
python3 parity/export_ocr_ncnn.py --skip-export   # re-derive the _mixed param and re-verify from the existing outputs
python3 parity/export_ocr_ncnn.py --fixture       # ...and also write the JVM test fixture
```

**Stage 1** exports `OCR.forward` with opset 17 and dynamic axes N/W → `parity/out/ocr_48px_ctc.onnx`, 164,974,063 B, sha256 `3019b406…2c35d8`. Deterministic on a rerun; the torch version is what this stage's bytes hinge on. Nothing from it ships any more: it is the fp32 reference that stage 2's verification (and `ocr_parity.py`) run through onnxruntime on the desktop.

Both stages read the checkpoint zip — stage 2 traces from the checkpoint, not from the ONNX. The decode-only parity scripts need just the alphabet, and `paths.ALPHABET` falls back to the copy in the engine's assets (`engine/src/main/assets/models/alphabet-all-v5.txt`, bit-identical to upstream's) when the extracted one isn't there — so they run from a clean clone without fetching anything.

**Stage 2** is `export_ocr_ncnn.py`, in order:

1. Load `OCR` from the clone and the checkpoint, dropping the `pe.pe` buffers (the baked positional-encoding tables; loaded non-strict).
2. Measure the backbone's width downsampling at ten widths (64 … 1024) and match it against a set of candidate formulas — `T = ⌊W/4⌋ − 1` is the one that fits, recorded in `t_of_w.txt`; the engine's `sinusoidalPe` uses the same formula, and the script refuses to continue if no candidate fits.
3. Wrap the model (`build_wrapper`): the three encoder layers' `PositionalEncoding.forward` are patched to add an *input tensor* instead of a slice of their buffer, and the unused `color` head is dropped so the only output is `char_logits` (text colour comes from the cleaned background, not from the OCR). Fed the model's own PE table, the wrapper reproduces the original `char_logits` exactly (`max|Δ| = 0` is asserted), so it changes plumbing and nothing else.
4. `torch.jit.trace` at W = 256 with a freshly computed sinusoidal PE (`make_pe`, T = 63), then pnnx with `inputshape=[1,3,48,256],[1,63,320]` **and** `inputshape2=[1,3,48,1024],[1,255,320]`. The second shape is what makes pnnx keep the attention Reshapes' dimensions dynamic instead of baking in the trace width — the further apart the two shapes, the more reliably it infers which dimension varies.
5. `write_mixed_param()` derives `ocr_48px_ctc_mixed.ncnn.param` from the plain param, as text: every layer from the backbone's `Squeeze` onward, plus the `Split` on the PE input `in1`, gets `31=7` appended (ncnn's per-layer featmask: bit0 disables fp16 arithmetic, bit1 fp16 storage/packing, bit2 bf16 — bf16 is off globally anyway, masked so nobody can turn it on later), and one `Cast` layer (`0=2 1=1`, fp16 → fp32) is inserted between the last backbone convolution and the `Squeeze`. Only the header counts and those tails differ; the weight order does not, so both params read the same `.bin`.

**Why the PE is an input.** Upstream's `PositionalEncoding.forward` is `x + self.pe[:, :x.size(1)]`; under tracing `size(1)` is a constant, so pnnx bakes the slice at the trace width and every other width returns garbage. That wall is what kept OCR on ONNX Runtime through v3. With the encoding fed as `in1` — pure sinusoid, `pe[t, 2i] = sin(t / 10000^(2i/320))`, `pe[t, 2i+1] = cos(…)`, a few lines to compute on the device and shared by all three encoder layers — the graph holds only `x + in1`. The blob contract the engine's `ncnn_jni.cpp:ocrCtcNative` reads: `in0` = image `[3,48,W]` in `(x − 127.5)/127.5`, `in1` = PE `[T,320]`, `out0` = `char_logits [T,19264]` raw; greedy CTC and the top-1 log-softmax confidence are computed in JNI, the collapse and alphabet lookup in Kotlin.

**Why mixed precision.** On device (SD 8 Gen 3, 9 pages / 242 lines against an fp32 ground truth): all-fp16 is 27–32% faster but reads 219 — the transformer misreads small kana (なぃ, か6, だろぅ, やは自); all-fp32 reads 242 but is 32–45% slower; fp16 storage-only is 10× slower (every layer casts in and out). Backbone fp16 + transformer fp32 reads 241 (the one differing line is the ground truth's own misread) at ~23% less OCR time than the int8 ONNX model, and loads in ~0.3–0.4 s.

**Output** — `parity/out/ocr_ncnn/ocr_48px_ctc.ncnn.param` + `ocr_48px_ctc_mixed.ncnn.param` + `ocr_48px_ctc.ncnn.bin`. The names already match `models.json`; the shipped bytes are:

```
ocr_48px_ctc.ncnn.param          18,133 B  sha256 e701cfc5df9d3c55c9fd0a36725499d01d45c18e32946271fc26e580cd9901bd
ocr_48px_ctc_mixed.ncnn.param    18,438 B  sha256 32e298deca3acb8ba95897ca8cccbd028582ae42a697c3c05b4eaefe198309e3
ocr_48px_ctc.ncnn.bin        83,037,664 B  sha256 3e0a809441f5284871d18a3d757a7ec098c4b9ee64bdfd7ac776a84aa057c7a9
```

**Verification** runs automatically, from a clean clone, on the *plain* param with every fp16 option off — which is all an x86 CPU can do:

- **Real strips** — the same 30 frozen quads as ever (`faithful_boxes.json` on `demo03.png`): NCNN against the ORT fp32 reference, decoded text per line and argmax per timestep, plus the int8 ONNX as a third column when it is present. Measured: **text identical on 30/30, argmax identical on 30/30** (and 29/30 against int8 — the known line where int8 differs from fp32).
- **Width sweep** — the longest strip, right-padded with white to W+1, W+7, 300, 333, 512, 777, 1000, 1024 and 1500: every width must decode to the same text as ORT fp32 at that width. Measured: no width differs. This is the check that the trace width didn't leak into the graph; a failure here is trap 7.
- **`--fixture`** writes `engine/src/test/resources/ocr/` — strip 0's `in0`, its PE, the expected argmax indices and log-probs, and copies of both params — for the JVM tests: `OcrCtcParityTest` (the engine's `sinusoidalPe` and CTC collapse against numpy) and `OcrMixedParamTest` (the structural rules the mixed param must satisfy — trap 6).

**What the desktop cannot verify: the mixed param.** x86 has no fp16 storage, so `_mixed` can only be parsed and structurally checked here; whether it *computes* correctly is a device question (the 241/242 above). Trap 6 is why that is the only way it can go wrong.

**The retired int8 model.** Through v3 the shipped OCR was `ocr_int8.onnx`: the stage-1 ONNX dynamically quantized by `quantize_ocr_int8.py` (`quant_pre_process(skip_symbolic_shape=True)`, then `quantize_dynamic(weight_type=QUInt8)` — 43,625,294 B, sha256 `353e68a5…29fa4c5c`, bit-identical to the `models-v2` release; 29/30 on the strips, the "96.7% CTC parity" of old). The script and `ocr_parity.py` are kept so stage 2's int8 column can still be reproduced; nothing ships from them. Trap 4 records the quantizer's two non-obvious preconditions.

## Rebuilding the inpaint model (AOT-GAN)

```bash
python3 parity/export_aot_ncnn.py            # convert + verify (~1-2 min)
python3 parity/export_aot_ncnn.py --skip-ref # skip the comparison against released weights
```

Checkpoint (auto-download) → `AOTGenerator` from the clone → `load_state_dict` → `model.eval()` → **`my_layer_norm` monkey-patch** (see [trap 1](#trap-1-pnnx-cant-lower-torchstd--a-dead-model-with-no-error) — without it you get a dead model) → `torch.jit.trace` at 512 → `pnnx.convert(fp16=True, optlevel=2)` → ncnn.

If you have a real m-i-t install, note its models folder may only carry `lama_large_512px.ckpt` — `inpainting.ckpt` is a separate download, which the script handles.

**Output** — `parity/out/aot/mit_aot_fixed512.ncnn.param` (33,762 B) + `.bin` (11,366,088 B). The name already matches `models.json`; no rename needed.

**Judge it by the criterion, not the hash** — the sha256 will differ, for the reasons given [above](#what-reproducible-means-here). The script verifies the blob contract (`in0`/`in1`/`out0`) and compares against torch at 512 and 768. Against torch fp32 the tolerance is s=512 max|d| 0.0477 (mean 5.3e-4) and s=768 max|d| 0.1017 (mean 4.8e-4) — ordinary fp16 storage error, present in the released weights too.

This is the one model whose full criterion needs something a clean clone doesn't have: the released weights, to confirm **`out0` is bit-identical** and to run the layer-by-layer weight compare. Point `YAKU_REF_MODELS` at a folder containing them — download them from the `models-v2` release urls in [`models.json`](../models.json) — or pass `--skip-ref` and settle for the torch comparison above.

**`fixed512` in the name is the trace shape, not a limit.** AOT-GAN is fully convolutional; the engine runs it at **tile 768** (`InpainterConfig.tileSize`). The name is historical baggage — renaming would mean churning `models.json` and the release assets, which isn't worth it. One artifact of tracing at 512 is that the layer-norm element count is baked in as a constant (`mul_10 2=16384.0` / `div_11 2=16383.0` = 128×128). At 768 the true values would be 36864/36863, but this only shifts the Bessel factor from 1.0000271 to 1.0000610 — a ~3e-5 relative difference, and the reduction itself stays dynamic. The 768 output being bit-identical to the release is the practical proof.

## Character segmentation → NCNN (night reading)

Two optional models, used only by night reading — the device recipe is the union of their character masks (see [MODELS.md](MODELS.md#night-reading-models)). Unlike the three above, their sources are not manga-image-translator checkpoints, and the scripts do **not** fetch them: get the weights yourself — `manga_seg_s.pt` (plus its ONNX export, used as the verification reference) from Hugging Face [anonimkaq4/manga-page-element-segmentation](https://huggingface.co/anonimkaq4/manga-page-element-segmentation), `cartoonseg.onnx` from [Jakaline/CartoonSegmentationOnnx](https://huggingface.co/Jakaline/CartoonSegmentationOnnx) — and point the scripts at them. By default they look in `../yakuyomi-nightread/research/out/models/` (the night-reading research repo checked out next to this one); `YAKU_CSEG_ONNX`, `YAKU_YOLOSEG_PT` and `YAKU_YOLOSEG_ONNX` override that. The yolo export needs `ultralytics`, which `parity/requirements.txt` does not pin. Both scripts print the sha256 and size of what they wrote.

```bash
python3 parity/export_cseg_ncnn.py                # cut the ONNX graph -> pnnx -> ncnn, + three-tier verify
python3 parity/export_cseg_ncnn.py --fixture      # ...and write the JVM test fixture
python3 parity/export_cseg_ncnn.py --skip-export  # verify only, against the existing param/bin
python3 parity/export_yoloseg_ncnn.py             # ultralytics export format=ncnn -> ncnn, + verify
python3 parity/export_yoloseg_ncnn.py --fixture
python3 parity/export_yoloseg_ncnn.py --skip-export
```

**cseg** — CartoonSegmentation's RTMDet-Ins, from `cartoonseg.onnx`. The mmdeploy-exported ONNX carries NonMaxSuppression, TopK and the per-instance dynamic convolutions *inside* the graph, and ncnn has no layers for those. So the script cuts the graph (`onnx.utils.extract_model`) right after the **raw head outputs** — what is left is the CSPNeXt backbone + PAFPN neck + head, all plain Convolution / Swish / Pooling / Interp — fixes the input to `[1,3,640,640]`, and runs pnnx (its default `fp16=1`, i.e. fp16 storage; no quantization). Post-processing moves to Kotlin: `CsegPost` is a port of the script's numpy `postprocess()`, which follows mmdet 3.3's `RTMDetInsHead` — priors at `(col × stride, row × stride)` in row-major order, l/t/r/b box decode clamped to `[0, 640]`, score > 0.05 → NMS IoU 0.6 → at most 100 instances, then the dynamic mask head (relative coordinates + the 8 mask features → 1×1 convs 10→8→8→1) → bilinear ×8 to 640 → sigmoid. The pipeline keeps instances scoring > 0.3, thresholds at 0.5, crops the padding, nearest-neighbour resizes back to the page and unions. Blob contract (`engine/src/main/cpp/ncnn_jni.cpp:extractNative`; names and order are fixed): `in0` = `[3,640,640]` **BGR**, `(x − mean) / std` with mean `(103.53, 116.28, 123.675)` and std `(57.375, 57.12, 58.395)`, aspect-preserving resize to a 640 long edge, padded **bottom-right** with 114 before normalization (mmdet convention, not centred); `out0..out2` = `rtm_cls [1,H,W]` at strides 8/16/32 (single-class logit, sigmoid = score), `out3..out5` = `rtm_reg [4,H,W]` (l,t,r,b — relu, then × stride, gives pixels; the relu is done in Kotlin), `out6..out8` = `rtm_kernel [169,H,W]` (dynamic-conv weights `[80,64,8]` then biases `[8,8,1]`), `out9` = `mask_feat [8,80,80]`. Output: `parity/out/cseg/cartoonseg.ncnn.param` + `.bin`.

Verification runs in three tiers: (1) cut ONNX (ORT) vs NCNN — max|Δ| over the ten raw outputs, at fp16-storage tolerance (the logits reach ~40); (2) cut ONNX + numpy post-processing vs the **full** ONNX with its in-graph NMS — instances paired by box IoU, then mask-probability difference and binary IoU per pair; (3) on the test pages in `app-sandbox/src/main/assets/test/`, the **union mask** IoU across all three paths (full ONNX / ORT cut + post / NCNN + post); the script flags pages whose union IoU is low. `--fixture` writes `ch34_011`'s ten raw outputs (fp16) and the expected mask into `engine/src/test/resources/charseg/` for `CsegPostParityTest`, which checks the Kotlin post-processing pixel for pixel.

**yolo** — YOLO11-seg (`manga_seg_s.pt`, trained on MangaSeg / Manga109-s). Ultralytics' own exporter does the whole conversion — `YOLO(pt).export(format="ncnn", imgsz=1024, half=True)` — and it is pnnx underneath, the same route as DBNet and AOT: one step, fp16 storage, no quantization. The script copies the resulting `model.ncnn.param` / `.bin` to `parity/out/yoloseg/manga_seg_s.ncnn.param` + `.bin`. Blob contract (`NcnnBackend.extract`; fixed): `in0` = `[3,1024,1024]` **RGB**, `/255`, the ultralytics letterbox (aspect-preserving resize to a 1024 long edge, **centred** pad 114); `out0` = `[39,21504]`, per anchor `cx,cy,w,h` in 1024 coordinates + 3 class scores (0 = frame, 1 = speech_bubble, 2 = character) + 32 mask coefficients; `out1` = `[32,256,256]` mask prototypes. Post-processing (`YoloSegPost`, a port of the script's `postprocess()`, itself the research harness's `run_yoloseg_onnx` that produced the desktop guard-box numbers): character class only, score > 0.25 → NMS IoU 0.45 (greedy, descending score) → `sigmoid(coefficients · prototypes)` at 256×256 → crop to the box (`crop_mask`: rows `[int(y1), ceil(y2))`, columns likewise) → bilinear to 1024 → un-letterbox → bilinear to the page size → > 0.5 → union. Both bilinear steps are `cv2.resize INTER_LINEAR` (half-pixel centres, clamped edges); the Kotlin port evaluates only inside each box's support and lands on the same result.

Verification: NCNN vs ONNX (ORT) on `out0`'s score channels and on `out1`, then the **union mask** IoU of the two paths after the same numpy post-processing over the test pages. `--fixture` writes `ch34_011`'s `out0` / `out1` (fp16) and the expected mask into `engine/src/test/resources/charseg/` for `YoloSegPostParityTest`.

**Judge both by the criterion, not the hash.** Both went through pnnx like AOT (yolo through the copy ultralytics bundles) and neither's hash reproducibility has been established on a cold rerun; the `models.json` values are the bytes that shipped. The output names already match `models.json`, so they ship as-is.

## The traps

These are the reasons the models were, until now, not rebuildable by anyone but the person who first did it.

### Trap 1: pnnx can't lower `torch.std` — a dead model, with no error

`AOTBlock.my_layer_norm` uses `feat.std((2,3))`. pnnx 1.0.20260526 converts it into pnnx IR but **cannot lower it to an ncnn layer**: `layer torch.std not exists or registered` → `network graph not ready` → `find_blob_index_by_name in0/in1/out0 failed`, extract returns −1.

**It does not fail the build.** pnnx exits happily and writes a perfectly normal-looking `.param` (29,852 B) and `.bin`. The model is only discovered to be dead when something tries to load it. **Rebuild without verifying and you will ship a dead model.**

The fix, which the script applies at export time (in memory — the clone is not modified), is to monkey-patch `my_layer_norm` into the hand-expanded equivalent: mean → sub → `d*d` → mean → ×N ÷(N−1) → sqrt → +1e-9. Two details that are easy to get wrong:

- **`×N ÷(N−1)` is Bessel's correction.** `torch.std` defaults to `unbiased=True` — it does not divide by N. Drop this and your output is subtly biased.
- **Use `d*d`, not `d**2`**, so pnnx emits a BinaryOp mul, matching the release.

This is also evidence that the original conversion did exactly the same thing: the released param contains no `std`, but does contain that same expansion (`mean_87` / `mul_10 2=16384.0` / `div_11 2=16383.0` / `sqrt_12`), and its op histogram matches our rebuild item for item.

### Trap 2: DBNet's `out0` is raw logits — not sigmoid'd

`out0` ch0 is the shrink map as **raw logits**. Upstream applies sigmoid outside the model (`detection/default.py:23`, `db = db.sigmoid()`), and so does the engine (`Detector.kt:59`). If you "helpfully" fold sigmoid into the exported model, it gets applied twice and **every box is wrong**. Leave it out.

(ch1, the threshold map, *is* sigmoid'd inside the model. The asymmetry is upstream's, not ours.)

### Trap 3: `model.eval()` is a hard requirement, not hygiene

`DBHead.forward` branches on `self.training`: in train mode it emits an extra `binary_maps`, so `out0` becomes 3-channel and no longer matches the engine's interface. The DBNet script asserts `db.shape[1] == 2` to catch this.

The same applies to AOT-GAN for a different reason: `AOTGenerator.forward`'s training branch **omits the `clip(-1,1)`**, so the output range silently changes.

### Trap 4: the OCR quantizer needs two non-obvious preconditions

*(Retired int8 path — only matters if you rebuild `ocr_int8.onnx` for the comparison column.)*

1. **Constant-fold first.** In torch's exported graph, `layer4.5/conv1`'s weight arrives as `Conv <- Identity <- initializer`. ORT's Conv quantizer only recognises "input[1] is directly an initializer" and won't see through the Identity, so a bare `quantize_dynamic` dies with `ValueError: Expected onnx::Conv_1267 to be an initializer`. Exporting with `do_constant_folding=True` does *not* remove this one. `quant_pre_process` does (nodes 646 → 437, non-initializer Conv weights 1 → 0).
2. **`skip_symbolic_shape=True` is required.** Symbolic shape inference can't cope with the dynamic W: `Cannot determine if floor(floor(W/2)/2) - 1 < 0` → `Incomplete symbolic shape inference`. Dynamic quantization doesn't need shape inference anyway — we only want the constant folding.

### Trap 5: an ncnn `.param` and its `.bin` must ship as a matched pair

An ncnn `.bin` is just a linear stream of weights laid out in the `.param`'s layer order. A different pnnx version orders layers differently, so the `.bin` bytes change completely — even when every individual tensor is bit-identical. **Mixing a new `.param` with an old `.bin` does not error. It silently outputs all zeros** (text removal renders solid black).

Measured: `ours.param` + `release.bin` → 0.0, and `release.param` + `ours.bin` → 0.0, while each matched pair gives the same 513071.40625. In `models.json` the `.param` and `.bin` are two independent assets — **always replace both from the same conversion**, and remember the app may have cached the old one. The AOT script's `compare_weights()` exists to catch this class of mistake.

### Trap 6: a featmask changes how a layer computes, not what it receives

The mixed-precision OCR param marks the transformer layers `31=7` so they run fp32. The first version did only that, and crashed on device with a SIGSEGV inside `conv3x3s1_winograd43_fp16sa` — a *backbone* convolution, on a *different* page's thread, nowhere near the transformer.

The mechanism, verified against ncnn's `net.cpp`: `convert_layout` only casts an fp16 blob to fp32 when `opt.use_fp16_storage && !layer->support_fp16_storage`, and a masked layer's option already has `use_fp16_storage` turned off — so the condition is false, no cast happens, and the backbone's fp16 pack8 output flows into the fp32 layers as-is. `Squeeze` only reshapes, so nothing shows; `Permute` allocates its output with elemsize 2 but writes `w*h` floats through a `float*` — twice the buffer — and the heap overflow lands on whichever concurrent thread's Winograd workspace sits next door. Small (T=30 is about 19 KB), silent, and remote from its cause.

The fix is the explicit `Cast` (`0=2 1=1`) between the last backbone conv and the `Squeeze`: `Cast_arm` supports fp16 storage itself on asimdhp CPUs, so the Net leaves it alone; it consumes the fp16 pack8 blob and emits fp32 pack8, and `Squeeze` (a base layer, no packing) gets unpacked by the Net to pack1. The `Cast` itself carries no mask. `OcrMixedParamTest` guards the rule: every input of every masked layer must come from a masked layer, from `in1` (the PE, which the extractor injects as fp32), or from a `Cast`.

The corollary is the mixed param's precondition: its `Cast` declares its input to be fp16, which is true only when the CPU has fp16 storage (ARMv8.2 `asimdhp`) and the Net has `use_fp16_storage` on. On x86, on older ARM, or with fp16 storage disabled the incoming blob is fp32, the `Cast` reinterprets it, and OCR returns garbage without crashing. The engine (`Ocr.pickParam`) therefore loads `_mixed` only when `OcrConfig.ncnnMixed && ncnnFp16Storage && NcnnBackend.cpuSupportsFp16`, and the plain param otherwise; the desktop scripts never execute it at all.

### Trap 7: the OCR trace width leaks into the graph unless the PE is an input

`PositionalEncoding.forward` slices its buffer by `x.size(1)`, a constant under tracing. A straight trace → pnnx export therefore works at exactly the trace width and returns garbage at every other — no error, just wrong text. This is the wall the 2026-07 attempt hit, and why OCR stayed on ONNX Runtime until v4. The export lifts the encoding out as `in1` (see the OCR section), pairs `inputshape` with a distant `inputshape2` so the attention Reshapes stay dynamic, and the width sweep in `verify()` is the regression test: pad one strip to nine unrelated widths and demand the same text at each.

### Smaller ones, all of which have cost time

- **`ncnn.Mat(ndarray)` does not copy the buffer.** Passing a temporary (`ncnn.Mat(np.ascontiguousarray(x))`) lets it be garbage-collected immediately, and you read freed memory. The symptom is vicious: the same model, run twice, differing by max|d| = 2.0 (the entire value range), intermittently. Hold the numpy object in a variable. (The scripts flag these variables as not-to-be-simplified.)
- **When parsing an ncnn param, Padding's `6=` is `per_channel_pad_data_size`, not `weight_data_size`.** Reading it as the latter shifts everything and decodes garbage — which reads convincingly as "the weights are different". Only Convolution / Deconvolution / InnerProduct carry weights.
- **Don't verify with `torch.randn`.** Noise is outside the inpaint model's data distribution; its output legitimately flails, and you get bad-looking numbers that mean nothing. The script uses a real manga page with a rectangular erase block.
- **The intermediate `.pt` is not bit-stable** (two traces gave 308,689,713 vs 308,689,649 B — zip metadata/timestamps), even when the ncnn output *is* bit-identical. Never use the `.pt` hash as a reproducibility signal; judge only the ncnn output.
- **`ImageMultiheadSelfAttention` in `DBNet_resnet34.py` is dead code** — `TextDetection` never uses it, and the exported param confirms there are no attention layers. Don't go debugging attention conversion.

## Shipping a rebuild

Outputs land in `parity/out/` (gitignored). Two of the ten files ship under a different name than they're built with:

| Built | Ships as | `models.json` role |
|---|---|---|
| `dbnet.ncnn.param` / `.bin` | **`dbnet_detect.ncnn.param` / `.bin`** — rename required | detector |
| `ocr_48px_ctc.ncnn.param` / `ocr_48px_ctc_mixed.ncnn.param` / `ocr_48px_ctc.ncnn.bin` | same names — as-is | ocr |
| `mit_aot_fixed512.ncnn.param` / `.bin` | `mit_aot_fixed512.ncnn.param` / `.bin` — as-is | inpainter |
| `cartoonseg.ncnn.param` / `.bin` | same names — as-is | charseg |
| `manga_seg_s.ncnn.param` / `.bin` | same names — as-is (the script already renames ultralytics' `model.ncnn.*`) | charseg |

```bash
cp parity/out/dbnet/dbnet.ncnn.param /tmp/ship/dbnet_detect.ncnn.param
cp parity/out/dbnet/dbnet.ncnn.bin   /tmp/ship/dbnet_detect.ncnn.bin
```

The detector rename is a manual step and therefore easy to forget. Bring-your-own-model will still resolve the unrenamed file — `ModelSet` matches by substring (`.param` containing `dbnet` → detector, `.param` containing `aot` → inpainter, `.param` containing `ocr` → OCR, either param — the engine picks `_mixed` or plain itself; `manga_seg` / `cartoonseg` → the two optional night-reading segmenters) and finds the `.bin` by swapping the suffix — but a release asset must carry the name `models.json` declares, or auto-download fails.

If you publish weights that differ from the current ones, update `models.json`'s `size` and `sha256` in the same change — the manifest is versioned with the files, which is what keeps that check meaningful.

## What you cannot check from a desktop rebuild

Being explicit, so nobody burns a day trying:

- **Performance and precision numbers are device-side.** "~23% faster than int8", the 241/242 mixed-precision read rate, and the per-page detection / OCR times in [MODELS.md](MODELS.md) (0.79 s / 1.25 s over 9 pages) were measured on real hardware (SD 8 Gen 3), and so were the night-reading numbers (yolo ~0.46 s, cseg ~0.83 s, 6–25 s per page). They cannot be reproduced by this build process.
- **The mixed-precision OCR param cannot be executed on x86.** No fp16 storage there, and its `Cast` assumes fp16 input (trap 6). The scripts parse and structure-check it; the plain param is what they run.
- **x86 timings from these scripts are noise.** Two runs of a *bit-identical* OCR model (the retired int8 one) measured 1732 ms and 3336 ms — a ~2× spread on the same file. The fp32-vs-int8 "~29×" seen on x86 is likewise an artifact. Don't read any speed conclusion out of a desktop run.
- **`out1` mask resolution differs by platform and must not be hard-coded.** On x86 it comes back half-resolution (H/2 × W/2); on arm64 it comes back full-resolution. The engine allocates for the full-resolution worst case and reads the actual dimensions back from JNI (commit `7c62f78` fixed exactly this overrun). Don't let a desktop measurement talk you into fixing a size at either end.
