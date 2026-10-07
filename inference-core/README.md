# `:inference-core` — shared on-device inference

English ｜ [中文](README_zh.md)

The part of Yakuyomi that both translation and night reading need. It holds the NCNN native layer, text detection, line grouping and the model downloader. Two modules sit on top of it and do not depend on each other:

```
li.joye.yakuyomi:nightread   (pure-Kotlin night-read library, nested includeBuild)
        ▲ api
:nightread-android ──api──▶ :inference-core ◀──api── :engine
   (night reading)            (libyakuyomi_ncnn.so)      (translation)
```

You normally do not depend on this module directly. Depend on [`:engine`](../engine/README.md) for translation or [`:nightread-android`](../nightread-android/README.md) for night reading; both pull this module in through `api`.

Group: `li.joye.yakuyomi:inference-core`. Min SDK 26. arm64-v8a only.

## What is in it

| File | What |
|---|---|
| `NcnnBackend.kt` + `src/main/cpp/ncnn_jni.cpp` | JNI to NCNN: loading and releasing nets, the global inference lock, the low-priority gate night reading uses, and the model-specific forwards. Builds `libyakuyomi_ncnn.so`. |
| `src/main/cpp/ncnn/` | Prebuilt NCNN (headers, `libncnn.a`, CMake config): arm64, SimpleOMP, no Vulkan. |
| `Detector.kt` | DBNet text detection: text-line quads plus a per-pixel stroke mask (`Detection`). |
| `DetectorConfig.kt` | Detection parameters. `EngineConfig.detector` in `:engine` uses it, and so does night reading. |
| `TextLine.kt`, `Grouping.kt` | Detected lines and m-i-t's two-stage grouping into regions (`TextRegion`). |
| `Geometry.kt`, `ImageOps.kt` | `Pt`, `RotRect` and the geometry/preprocessing helpers. |
| `ModelDownloader.kt` | Manifest-driven model download with sha256 checks. |
| `EngineTrace.kt` | Diagnostic trace hook around native calls. |

All Kotlin code here keeps the package `li.joye.yakuyomi.engine`. The JNI symbol names are derived from it (`Java_li_joye_yakuyomi_engine_NcnnBackend_*`), and the reader app imports the classes by that package, so the split into modules did not rename anything.

## One native library

`libyakuyomi_ncnn.so` is built here and nowhere else. NCNN is a static library; if a second module built its own `.so`, each would carry its own copy of NCNN, and so its own SimpleOMP worker pool and its own `pthread_once`. `NcnnBackend` creates the pool from a dedicated thread at normal priority and serializes multi-threaded forwards behind one lock; both rules would then only cover one of the two copies.

So every native entry point stays in this module, even the ones only one side uses:

| JNI entry (`NcnnBackend_…`) | Kotlin wrapper | Used by |
|---|---|---|
| `createNetNative`, `createNetExNative` | `createNet`, `createNetEx` | every model |
| `initThreadPoolNative` | private `ensureThreadPool` | `NcnnBackend` itself |
| `releaseNet` | `releaseNet` | every model |
| `detectDbnetNative` | `detectDbnet` | `Detector` (translation and night reading) |
| `cpuSupportsFp16Native` | `cpuSupportsFp16` | OCR param choice (`:engine`) |
| `ocrCtcNative` | `ocrCtc` | `Ocr` (`:engine`) |
| `inpaintAotNative` | `inpaintAot` | `Inpainter` (`:engine`) |
| `extractNative` | `extract` | character segmentation (`:nightread-android`) |

Night reading on its own therefore carries the two small OCR and inpaint functions it never calls. NCNN's layer code is linked in full either way.

`consumer-rules.pro` keeps the native method names through R8; it travels with this module to every app that uses it.

## Building

- NDK `28.2.13676358` and CMake `3.22.1` (`externalNativeBuild`). The reader app pins the same NDK so it can strip the library.
- Only `arm64-v8a` is built; the prebuilt NCNN is arm64 only.
- If you replace the prebuilt NCNN, build it with SimpleOMP (`NCNN_SIMPLEOMP=ON`) and without Vulkan, and make sure `ncnn.cmake` does not link a bare `pthread` (the NDK has no `libpthread`; `Threads::Threads` is fine).

## `@InternalEngineApi`

`NcnnBackend` and `Geometry` were `internal` before the split. Kotlin's `internal` does not cross Gradle modules, and `:engine` and `:nightread-android` both need them, so they are now public but marked `@InternalEngineApi`, an opt-in annotation at error level. The three engine modules opt in for the whole module. Outside code should not opt in: these are not a public API and can change at any time. `ImageOps` is still `internal`.

## Downloading models

`ModelDownloader` reads a manifest (default: [`models.json`](../models.json) on this repo's `main`) and makes sure the files are in a folder you pick, checking each sha256. Files already present with the right size and hash are skipped.

```kotlin
val remote = ModelDownloader.fetchManifest()
val dir = File(context.filesDir, "models")
ModelDownloader.ensure(remote.filter { it.role == "detector" }, dir) { progress -> /* ModelProgress */ }
```

The role is a plain string from the manifest (today `detector`, `ocr`, `inpainter`, `charseg`); filter by it to fetch only what you need. `ensure` returns a role → file map, but a role with several files (a `.param` and a `.bin`) keeps only the last one in that map, so look the files up by name in the folder instead. The app needs the `INTERNET` permission; this module declares none.

Load models from a local path. Do not read weights into the JVM heap with `readBytes()`: the heap is capped around 512 MB regardless of device RAM.

## `EngineTrace`

`EngineTrace.sink` is `null` by default, which costs nothing. Set it to receive one line before and after each native call (`xxx.enter` / `xxx.call` / `xxx.exit`). If the process dies inside native code, the last line names the call. The reader app writes these lines to its diagnostic log.
