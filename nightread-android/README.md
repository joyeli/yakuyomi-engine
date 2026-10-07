# `:nightread-android` — night reading on Android

English ｜ [中文](README_zh.md)

Night reading rebuilds a manga page so the page itself is dark: bubbles turn dark with light text, empty background goes black, and the characters are protected. The algorithm lives in the pure-Kotlin [`yakuyomi-nightread`](https://github.com/joyeli/yakuyomi-nightread) library. This module is the Android side that feeds it: it turns a `Bitmap` into the library's input, runs text detection and character segmentation on NCNN, and turns the result back into a `Bitmap`.

It does **not** depend on the translation engine. Its dependencies are [`:inference-core`](../inference-core/README.md) (NCNN, DBNet detection, grouping, model download) and the night-read library, both through `api`:

```
li.joye.yakuyomi:nightread ◀──api── :nightread-android ──api──▶ :inference-core
```

Group: `li.joye.yakuyomi:nightread-android`. Min SDK 26. arm64-v8a only.

## What is in it

| File | What |
|---|---|
| `NightReadRenderer.kt` | `NightReadRenderer` (the entry points), `NightReadStats` (timings), `UnionCharSegmenter`. |
| `CharSeg.kt` | `CharSegmenter` and the two NCNN segmenters: `YoloSegSegmenter` (YOLO11-seg `manga_seg_s`) and `CsegSegmenter` (CartoonSegmentation RTMDet-Ins), with their post-processing (`YoloSegPost`, `CsegPost`). |

The Kotlin package is `li.joye.yakuyomi.engine`, not `li.joye.yakuyomi.nightread`. That is historical: the code used to live in the engine module, and the reader app imports it by that package. The night-read library's own types (`NightReadParams`, `NightTier`, …) are in `li.joye.yakuyomi.nightread`.

## Adding it to an app

The engine repo is consumed as source through a Gradle composite build.

1. Add the engine repo as a git submodule. Use `--recursive`: it carries the night-read library as its own submodule.

   ```
   git submodule add https://github.com/joyeli/yakuyomi-engine.git yakuyomi-engine
   git submodule update --init --recursive
   ```

2. In `settings.gradle.kts`:

   ```kotlin
   includeBuild("yakuyomi-engine")
   ```

3. In the app's `build.gradle.kts`:

   ```kotlin
   dependencies {
       implementation("li.joye.yakuyomi:nightread-android:0.5.0") // the version is a placeholder; includeBuild substitutes the source
   }
   ```

   The app should build `arm64-v8a` only (`ndk { abiFilters += "arm64-v8a" }`); there is no inference backend for other ABIs.

Requirements: NDK `28.2.13676358` and CMake `3.22.1` are installed (`:inference-core` builds `libyakuyomi_ncnn.so` from source), and the app's `minSdk` is 26 or higher. The engine build and the night-read library build nested in it each look for the Android SDK on their own and do not read the app's `local.properties`: set `ANDROID_HOME`, or put a `local.properties` with `sdk.dir=...` in both `yakuyomi-engine/` and `yakuyomi-engine/yakuyomi-nightread/`.

## Models

Two roles from the engine's [`models.json`](../models.json):

| Role | Files | Size |
|---|---|---|
| `detector` | `dbnet_detect.ncnn.param` + `.bin` | ~153 MB |
| `charseg` | `manga_seg_s.ncnn.param` + `.bin`, `cartoonseg.ncnn.param` + `.bin` | ~147 MB |

The OCR and inpaint models are translation-only; night reading never loads them. To download just these two roles:

```kotlin
val dir = File(context.filesDir, "models")
val wanted = ModelDownloader.fetchManifest().filter { it.role == "detector" || it.role == "charseg" }
ModelDownloader.ensure(wanted, dir) { progress -> /* ModelProgress */ }
```

Both calls are `suspend`; run them in a coroutine. The app declares kotlinx-coroutines itself to do that (`:inference-core` uses it as an `implementation` dependency, so it is not on your compile classpath). `:inference-core` puts OkHttp and kotlinx-coroutines into the APK at runtime either way.

Or put the files in a local folder yourself. Pass paths; the nets load from the file into native memory. Do not read weights into the JVM heap with `readBytes()`.

The two segmentation models are **not** GPL-3.0 and carry their own terms: the YOLO11-seg weights (model card `license: other`, Ultralytics AGPL-3.0, "Copyrighted by Minshan Xie") and the CartoonSegmentation weights (no license stated upstream). Yakuyomi redistributes their NCNN conversions for research and non-commercial use and will take them down on a rights holder's request. Details: [docs/MODELS.md](../docs/MODELS.md#night-reading-models).

## Quick start

```kotlin
import li.joye.yakuyomi.engine.Detector
import li.joye.yakuyomi.engine.NightReadRenderer
import li.joye.yakuyomi.nightread.NightTier

Detector(dir.resolve("dbnet_detect.ncnn.param").path).use { detector ->
    val seg = NightReadRenderer.charSegmenter(
        dir.resolve("manga_seg_s.ncnn.param").path,
        dir.resolve("cartoonseg.ncnn.param").path,
    ) ?: error("no character-segmentation model")
    seg.use {
        detector.warmUp(); it.warmUp()   // first-time native init on one thread, before any concurrency
        val night: Bitmap = NightReadRenderer.render(page, detector, it, NightTier.L2.apply())
    }
}
```

`NightTier.L2.apply()` is what the reader calls "standard"; `NightTier.L3.apply()` is "more" (more background turns black). `renderTiers(...)` produces several tiers from one detection and one analysis, handing each to a callback.

`charSegmenter` takes the two `.param` paths (the `.bin` must sit next to each). Both → their union (the settled recipe); one → that one; none → `null`, night reading unavailable.

## Entry points

- `render(page, detector, charSeg, params)` — scale down if needed, detect, segment, rebuild, return a new `ARGB_8888` bitmap. Pages over `MAX_PIXELS` (3.5 MPx) are scaled down first, and **the output has the scaled size** (`NightReadStats.scaledTo` says so).
- `render(page, detect = …, segment = …, beforeRender = …)` — the same with your own lambdas around each inference, so you can add checkpoints (pause, yield). `beforeRender` returning `false` abandons the page and returns `null`.
- `renderTiers(...)` — several tiers in one pass. Each output bitmap is recycled as soon as the callback returns, so write it out inside the callback. A tier identical to the previous one arrives as `null`.
- `render(page, detection, charMask, params)` — when you already have a same-size `Detection` and character mask.

`extraLines` (text lines you know about from elsewhere, in page coordinates) and `inpaintMask` (a translated page's text-removal mask) are optional and only matter for translated pages.

## Threading and memory

- Warm the detector and the segmenter once, single-threaded, before running pages concurrently. Several pages hitting a freshly loaded net at once crashed on device.
- After that, several pages can run at once. Neither this module nor the library has shared mutable state. Multi-threaded NCNN forwards are serialized behind the core's global lock.
- Memory is the caller's job. A page in flight needs about 58 bytes of Java heap per pixel, roughly 150 MB for a typical page, so a 512 MB heap holds about two pages.
- `parallel: Executor?` (default `null`) runs independent branches of one page's analysis on your executor. Output is bit-identical. It raises a page's heap peak (budget at least 70 B/px instead of 58) and only pays off when that page is the only one in flight.
- `NcnnFlavor` decides how the nets are built. `DEFAULT` is right for an app that only does night reading. `NIGHT_LOCKED` / `NIGHT_FREE` and `NcnnLowPriorityHook` exist so the reader can make night reading yield to translation.
- The input bitmap is never recycled by these calls.

## Tests

`:nightread-android:testDebugUnitTest` runs 10 JVM tests: pixel-for-pixel checks of both segmenters' post-processing against Python fixtures (`src/test/resources/charseg/`), and whole-page checks that read the test pages from the night-read submodule (`../yakuyomi-nightread/nightread/src/test/resources/page`).
