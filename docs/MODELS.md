# Models

English ｜ [中文](MODELS_zh.md)

The engine ships no model weights. It needs three models — a detector, an OCR model, and a text-removal (inpaint) model — supplied two ways: manually (bring your own model) or by auto-download from this repo's releases. All three run on NCNN and ship as `.param` + `.bin`; OCR has two `.param` files (plain and mixed-precision) over one `.bin` — six files in all. Both routes land in the same models folder; downstream resolution is identical. The engine has no ONNX Runtime dependency any more.

## The three models

| Role | Backend | File(s) | Size | License | Source |
|---|---|---|---|---|---|
| Detection | NCNN (fp16) | `dbnet_detect.ncnn.param` + `.bin` | ~153 MB | GPL-3.0 | DBNet (ResNet34 + DB head), the default detector of [manga-image-translator](https://github.com/zyddnys/manga-image-translator) |
| OCR | NCNN (mixed fp16/fp32) | `ocr_48px_ctc.ncnn.param` + `ocr_48px_ctc_mixed.ncnn.param` + `ocr_48px_ctc.ncnn.bin` | ~83 MB | GPL-3.0 | 48px CTC, converted here from [manga-image-translator](https://github.com/zyddnys/manga-image-translator) weights |
| Text removal | NCNN (fp16) | `mit_aot_fixed512.ncnn.param` + `.bin` | ~11 MB | GPL-3.0 | AOT-GAN from [manga-image-translator](https://github.com/zyddnys/manga-image-translator) |

**Backends.** All three run on NCNN (ARM NEON / Winograd kernels), on the CPU — GPU/NPU was tried and does not work for these models (NCNN's Vulkan path miscomputes the AOT-GAN; LiteRT can't compile them). v1's LaMa inpaint is retired and removed; AOT-GAN (manga-image-translator's inpaint) replaces it. OCR ran on ONNX Runtime (int8) up to v3; v4 moves it to NCNN and drops ONNX Runtime from the engine entirely.

**v4 OCR.** The 48px CTC model on NCNN, in **mixed precision**: the convolutional backbone (≈96% of the MACs) stays fp16, the transformer encoder and character head are forced to fp32 with ncnn's per-layer featmask (`31=7`) plus an explicit `Cast` layer between the two. The precision split is the whole point. On 9 pages / 242 lines against an fp32 ground truth: all-fp16 NCNN reads 219 lines and the old int8 ONNX model 223 — both misread small kana (なぃ, か6, だろぅ) — all-fp32 NCNN reads 242, and mixed reads 241 (the one differing line is the ground truth's own misread; mixed is right). OCR time is ~23% lower than the int8 model's (e.g. 1256 vs 1669 ms per page) and the model loads in ~0.3–0.4 s. The sinusoidal positional encoding is not traced into the graph but fed as a second input, so strips of any width work — the wall that had kept OCR on ONNX Runtime. Lines run 8 at a time on a single-threaded net that stays outside NCNN's global inference lock. The two `.param` files share the one `.bin`:

- `ocr_48px_ctc_mixed.ncnn.param` — the mixed-precision one, used by default. It is only valid on a CPU with ARMv8.2 fp16 (`asimdhp`) and with fp16 storage enabled: its `Cast` layer declares its input to be fp16, so anywhere else it would read fp32 data as fp16 and produce garbage silently.
- `ocr_48px_ctc.ncnn.param` — the plain one, every layer at the net's global precision. The engine (`Ocr.pickParam`) loads it whenever the mixed one can't be used — no fp16 on the CPU, `OcrConfig.ncnnFp16Storage` / `ncnnMixed` off, or the `_mixed` file missing — so a phone without fp16 still reads correctly, in fp32 and slower.

The fp16 `.bin` is ~83 MB against the retired int8 model's 44 MB; the precision, not the size, is what recovers the small kana.

**v3 detector.** comic-text-detector is retired and removed; DBNet (manga-image-translator's default detector) replaces it, reading 1.6–2.5× more text correctly on device. It is kept in fp16 storage — int8 quantization makes it emit no boxes at all and is no faster on ARM, so it is not used, which is why the detector alone is ~153 MB of the ~247 MB of on-device weights. Input is `resize_aspect` to 1024 padded to a multiple of 256; the resulting rectangular input also steers clear of an ncnn heap-corruption bug on square 832–992 inputs. On an SD 8 Gen 3, over 9 pages / 242 detected lines, detection averages 0.79 s per page and the v4 OCR 1.25 s per page (23% less than the v3 int8 OCR); all 242 lines are read, 241 identical to an fp32 reference.

Exact bytes and checksums are pinned in [`models.json`](../models.json):

```
dbnet_detect.ncnn.param            13392  sha256 9e6db2f8…ee1ff7b5
dbnet_detect.ncnn.bin          153010556  sha256 f57bdbed…fcc55c3d
ocr_48px_ctc.ncnn.param            18133  sha256 e701cfc5…cd9901bd
ocr_48px_ctc_mixed.ncnn.param      18438  sha256 32e298de…198309e3
ocr_48px_ctc.ncnn.bin           83037664  sha256 3e0a8094…a057c7a9
mit_aot_fixed512.ncnn.param        33810  sha256 f21ef860…ee7d32b5
mit_aot_fixed512.ncnn.bin       11366088  sha256 a52db45e…5e3560b6
```

These checksums are a distribution integrity check — they confirm the file you hold is the one we published. They are **not** a criterion for judging a rebuild: a rebuilt model can be numerically identical and still hash differently (the inpaint model always does). To rebuild any of these from the upstream checkpoints, see [BUILD_MODELS.md](BUILD_MODELS.md).

## Redistribution and licensing

These weights are all GPL-3.0. This project redistributes them under that license, with attribution to the sources above, purely as a convenience for auto-download. All three are our own NCNN conversions of manga-image-translator's weights (the OCR one with the positional encoding lifted out as an input, and a second, mixed-precision param derived from the first) — there is no upstream distributable to point at, so they are hosted here. If you prefer, obtain the original weights yourself from the sources and use bring-your-own-model. The full conversion path from upstream checkpoint to each of these files — scripts, pinned versions, and how to verify the result — is in [BUILD_MODELS.md](BUILD_MODELS.md).

## Getting the models

**Auto-download (reader).** The reader fetches every file listed in the manifest in one step, verifying each file's sha256 against [`models.json`](../models.json). Each entry carries its own url: the detector comes from the `models-v3` release, the three OCR files from `models-v4`, and text removal is unchanged and still served from `models-v2`. Same result as bring-your-own-model, just automated.

**Bring your own model (manual).** Put the files in the models folder you point the app at (every role needs the `.param` and its `.bin`; for OCR put both `.param` files and the `.bin` in). They are resolved by name and extension — a `.param` matching `dbnet` → detector, a `.param` matching `aot` → text removal, a `.param` matching `ocr` → OCR (either OCR param can be the entry point; the engine switches to `_mixed` or back on its own after checking the CPU). All three roles must resolve or the set is rejected. There is no fallback runtime: the ONNX Runtime OCR path, the ONNX detector/inpaint paths and LaMa are all removed, so a leftover `ocr_int8.onnx` from v3 is no longer used.

## Verification

`ModelDownloader.verify(models, dir)` returns, per role, whether the local file's size and sha256 match what we published. Because the manifest is versioned together with the files, the checksum is always correct — updating the weights means a new manifest version, not a stale hash. This confirms the file you hold is byte-for-byte the one we distribute.

## API

```kotlin
// engine — ModelDownloader
val models = ModelDownloader.fetchManifest()              // models.json -> List<RemoteModel>
ModelDownloader.ensure(models, destDir) { progress -> }   // download missing/mismatched, verify sha256
ModelDownloader.verify(models, destDir)                   // role -> ok (verify only, no download)
```

The engine only fetches, verifies, and writes files; where to download, when to trigger, and the progress UI belong to the reader — the same engine/reader split as the LLM model list (see [PROVIDERS.md](PROVIDERS.md)).
