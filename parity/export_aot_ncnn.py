#!/usr/bin/env python3
"""
把 m-i-t 的 AOT-GAN 去字模型（inpainting.ckpt）轉成 NCNN：ckpt → torch trace → pnnx → ncnn。
產出 = 引擎跑的 `mit_aot_fixed512.ncnn.param` + `.bin`（models.json 的 inpainter role）。

ported spec: manga_translator/inpainting/inpainting_aot.py:AOTGenerator @ d5a3eee
  上游 forward(img, mask) → torch.cat([mask, img], dim=1) → head → 10×AOTBlock → tail → clip(-1,1)

blob 契約（引擎 `inference-core/src/main/cpp/ncnn_jni.cpp:inpaintAotNative` 照這個吃，名字不可改）：
  in0  = img [3,s,s]  值域 [-1,1]、holes（mask=1 處）已歸零
  in1  = mask[1,s,s]  {0,1}，1 = 要擦掉重建
  out0 = img [3,s,s]  值域 [-1,1]（模型 eval 分支已 clip）
  ⚠️ in0/in1 的順序由 trace 的引數順序決定（forward(img, mask)）；param 裡會看到
     `Concat cat_0 2 1 in1 in0`＝上游的 cat([mask, img])，這是對的、不要「修正」成 in0 in1。

⚠️ 檔名的 `fixed512` 只是 trace 時餵的 shape，**不是**執行時的限制：
   AOT-GAN 是全卷積（fully-convolutional）⇒ ncnn 可以吃任意尺寸。
   引擎實際跑的是 **tile 768**（`InpainterConfig.tileSize=768`，真機 A/B 定的甜蜜點），
   不是 512。名字純歷史包袱，改名要連 models.json + release asset 一起換，不值得。
   本腳本最後會實測 512 與 768 兩種 shape，證明這件事。

⚠️ **param 與 bin 是「配對」的，不可分開換**：ncnn 的 bin 就是照 param 的層順序線性排的權重流。
   pnnx 版本不同會把同一張圖的層**排成不同順序** ⇒ bin 位元組大不同（實測 8.7M/11.4M B 不同、
   sha256 不同），但每層權重其實 bit-identical、只是順序換了 ⇒ **同一對 param+bin 算出來完全一樣**。
   混用（新 param + 舊 bin）不會報錯，而是**安靜吐全 0**（去字結果整片黑）。
   ⇒ models.json 裡 .param 與 .bin 是兩個 asset，**永遠要同一次轉檔的產物一起換**。

重現性結論（2026-07-16 實跑，torch 2.1.1 + pnnx 20260526）：
   本腳本產出 vs 現行 release 權重 → **out0 bit-identical**（真頁、s=512 與 768 都是）。
   sha256 不同、param 差 48 B（純 pnnx 版本差異：層自動命名 conv_24 vs conv_70、
   release 的 Padding 多寫了預設值 5=0 6=0）。⇒ 逐位元不同、數值等價，可安心重建。

用法：
    python3 parity/export_aot_ncnn.py            # 轉檔 + 驗證
    python3 parity/export_aot_ncnn.py --skip-ref # 略過與現有 release 模型的比對
ckpt 不在本機時會自動從 m-i-t beta-0.3 release 下載並驗 sha256（見 paths.py 的 YAKU_INPAINT_CKPT）。
輸出落在 parity/out/aot/（gitignore），要上 release 再自己複製出去。
"""
import argparse
import importlib.util
import os
import sys
import types

import numpy as np
import torch

# 集中路徑 + 共用的下載/驗 hash（三支轉換腳本同一支 fetch），見 paths.py
from paths import INPAINT_CKPT, MIT_CLONE as MIT, OUT as _OUTDIR, SANDBOX_TEST, fetch, sha256_of

OUT_DIR = os.path.join(_OUTDIR, "aot")
STEM = "mit_aot_fixed512"          # 產出檔名 stem（models.json / release asset 用這個）
TRACE_SIZE = 512                   # trace shape（見上方 fixed512 註解）
ENGINE_TILE = 768                  # 引擎實跑 shape（InpainterConfig.tileSize）

# 上游 AotInpainter._MODEL_MAPPING['model'] @ d5a3eee（url + hash 都照抄，別自己編）
CKPT_URL = "https://github.com/zyddnys/manga-image-translator/releases/download/beta-0.3/inpainting.ckpt"
CKPT_SHA256 = "878d541c68648969bc1b042a6e997f3a58e49b6c07c5636ad55130736977149f"

# 對照用（現行 release 的權重）；--skip-ref 或檔案不在就跳過
# 與現有 release 權重的比對＝**選配**（空白 clone 沒有它）：沒指到就只跟 torch 參考比。
REF_DIR = os.environ.get("YAKU_REF_MODELS", "")


def ensure_ckpt():
    """確保 inpainting.ckpt 在本機且 hash 對；缺檔（或 hash 不符）就從上游 release 下載。

    ⚠️ 這顆一定得下載：使用者的 m-i-t 模型夾只有 lama_large_512px.ckpt、沒有 inpainting.ckpt。
    """
    return fetch(CKPT_URL, INPAINT_CKPT, CKPT_SHA256, label="inpainting.ckpt")


def load_AOTGenerator():
    """
    只取 AOTGenerator nn.Module。

    為什麼要繞：inpainting_aot.py 頂部 `from .inpainting_lama_mpe import LamaMPEInpainter`
    → 拖進 manga_translator/__init__.py（translators → tiktoken/openai 整包）+ 缺的
    rusty_manga_image_translator。我們只要那個純 torch 的 generator。
    手法照抄 /tmp/dbnet/mit_check.py：假 module stub + package shell
    （建 module 帶 __path__ 但「不執行」__init__ body）。純 in-memory，不碰使用者 clone。
    """
    sys.path.insert(0, MIT)

    def _pkg_shell(name, path):
        spec = importlib.util.spec_from_file_location(
            name, os.path.join(path, "__init__.py"), submodule_search_locations=[path])
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod        # 故意不 exec_module ⇒ __init__ body 永不跑
        return mod

    _pkg_shell("manga_translator", os.path.join(MIT, "manga_translator"))
    _pkg_shell("manga_translator.inpainting", os.path.join(MIT, "manga_translator", "inpainting"))
    # inpainting_aot.py 只用到 LamaMPEInpainter 當基底類別（AotInpainter 用；我們不碰）
    mpe = types.ModuleType("manga_translator.inpainting.inpainting_lama_mpe")
    mpe.LamaMPEInpainter = object
    sys.modules["manga_translator.inpainting.inpainting_lama_mpe"] = mpe

    spec = importlib.util.spec_from_file_location(
        "manga_translator.inpainting.inpainting_aot",
        os.path.join(MIT, "manga_translator", "inpainting", "inpainting_aot.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["manga_translator.inpainting.inpainting_aot"] = mod
    spec.loader.exec_module(mod)
    _patch_layer_norm(mod)
    return mod.AOTGenerator


def _patch_layer_norm(mod):
    """
    ⚠️ 匯出用改寫（§4 第三層知情偏離，數值等價、非行為改動）：
    把 my_layer_norm 裡的 `feat.std(...)` 換成手刻等價式。

    為什麼非改不可：pnnx（20260526）轉得出 pnnx IR 的 torch.std，但**下不到 ncnn 層**
    （`layer torch.std not exists or registered` → 整張圖壞掉、extract out0 回 -1）。
    現行 release 的 param 裡也**沒有** std，而是同一組展開（mean→sub→x*x→mean→×N→÷(N-1)
    →sqrt→+1e-9，見 param 的 mean_87/mul_10/div_11/sqrt_12），⇒ 當初那次轉檔做的是同一件事。

    ×N ÷(N-1) = Bessel 修正（torch.std 預設 unbiased=True，不是除以 N）。漏掉這步會偏。
    monkey-patch 模組層級的 my_layer_norm；AOTBlock.forward 是呼叫時才查 global ⇒ 吃得到。
    不碰使用者的 clone 檔案。
    """
    def my_layer_norm(feat):
        mean = feat.mean((2, 3), keepdim=True)
        d = feat - mean
        var = (d * d).mean((2, 3), keepdim=True)      # 用 d*d 不用 d**2 ⇒ pnnx 出 BinaryOp mul
        n = feat.shape[2] * feat.shape[3]
        # ⚠️ n 在 trace 時會被烤成常數（512 trace ⇒ n=16384）。跑 768 時真值是 36864、
        # 但這只影響 Bessel 係數 N/(N-1)：16384/16383=1.0000610 vs 36864/36863=1.0000271
        # ⇒ 相對誤差 ~3e-5，對 [-1,1] 的輸出可忽略（下方 s=768 的驗證有實測背書）。
        std = torch.sqrt(var * n / (n - 1)) + 1e-9
        return 5 * (2 * d / std - 1)

    # 先自證等價再換掉（別默默改數學）
    x = torch.randn(1, 8, 16, 16) * 3 + 1
    ref, got = mod.my_layer_norm(x), my_layer_norm(x)
    err = (ref - got).abs().max().item()
    assert err < 1e-3, f"my_layer_norm 改寫不等價：max|d|={err}"
    print(f"my_layer_norm 改寫（避開 pnnx 的 torch.std 缺口）：與上游等價 max|d|={err:.2e}")
    mod.my_layer_norm = my_layer_norm


def build_model(ckpt):
    """AOTGenerator + 載權重。ckpt 是 {'model': state_dict}（照上游 AotInpainter._load）。"""
    AOTGenerator = load_AOTGenerator()
    model = AOTGenerator()
    sd = torch.load(ckpt, map_location="cpu")
    sd = sd["model"] if "model" in sd else sd
    res = model.load_state_dict(sd)
    print(f"load_state_dict OK（missing={len(res.missing_keys)} unexpected={len(res.unexpected_keys)}）")
    model.eval()   # ⚠️ 必須 eval：AOTGenerator.forward 的 training 分支不含 clip(-1,1)
    return model


def export(model):
    """trace → pnnx → ncnn param/bin。回傳 (param, bin) 路徑。"""
    import pnnx

    os.makedirs(OUT_DIR, exist_ok=True)
    s = TRACE_SIZE
    img = torch.randn(1, 3, s, s).clamp(-1, 1)
    mask = (torch.rand(1, 1, s, s) > 0.7).float()
    img = img * (1 - mask)                        # holes 歸零＝引擎餵法（見 blob 契約）

    with torch.no_grad():
        y = model(img, mask)
    print(f"torch forward OK: img{tuple(img.shape)} mask{tuple(mask.shape)} → out{tuple(y.shape)} "
          f"range[{y.min():.3f},{y.max():.3f}]")

    pt = os.path.join(OUT_DIR, f"{STEM}.pt")
    ts = torch.jit.trace(model, (img, mask))      # 兩個引數 ⇒ pnnx 給 in0=img、in1=mask
    ts.save(pt)

    # pnnx 產一堆中間檔（.pnnx.*/ _pnnx.py/ .ncnn.py），都丟 OUT_DIR；我們只要 .ncnn.param/.bin
    # fp16=True＝pnnx 預設，bin 存 fp16（11MB vs fp32 22MB），現行 release 權重也是這樣轉的
    pnnx.convert(pt, inputs=[img, mask], fp16=True, optlevel=2, device="cpu")

    param = os.path.join(OUT_DIR, f"{STEM}.ncnn.param")
    binf = os.path.join(OUT_DIR, f"{STEM}.ncnn.bin")
    for f in (param, binf):
        if not os.path.exists(f):
            raise SystemExit(f"pnnx 沒吐出 {f}")
    print(f"pnnx → {param} ({os.path.getsize(param):,} B)\n"
          f"       {binf} ({os.path.getsize(binf):,} B)")
    return param, binf


def check_blob_names(param):
    """param 開頭必須是 in0/in1、結尾必須有 out0 — 引擎 JNI 認的是名字。"""
    lines = [l.split() for l in open(param).read().splitlines()[2:] if l.strip()]
    inputs = [l[1] for l in lines if l[0] == "Input"]
    # out0 是某層的輸出 blob（不是最後一個 token — 尾巴是 0=… 參數），所以掃整行
    produces_out0 = [l[1] for l in lines if "out0" in l]
    ok = inputs == ["in0", "in1"] and len(produces_out0) == 1
    print(f"blob 契約：Input={inputs}、out0 由 {produces_out0} 產出 → "
          f"{'✓ 與 JNI 相符' if ok else '✗ 與 JNI 不符！'}")
    return ok


def _read_weights(param, binf):
    """
    照 param 的層順序把 bin 解成 [(層名, 權重 ndarray)]。
    ncnn bin = 純權重流：每個帶權重的層 = 4B tag（0x01306B47=fp16、0=fp32）+ 資料（+bias）。
    ⚠️ 只有 Convolution/Deconvolution/InnerProduct 帶權重；Padding 的 `6=` 是 per_channel_pad
       **不是** weight_data_size（照它算會整個位移、解出垃圾 — 踩過）。
    """
    WEIGHTED = {"Convolution", "Deconvolution", "InnerProduct"}
    data = open(binf, "rb").read()
    off, out = 0, []
    for line in open(param).read().splitlines()[2:]:
        t = line.split()
        if not t or t[0] not in WEIGHTED:
            continue
        d = {k.split("=")[0]: k.split("=")[1] for k in t[4:] if "=" in k and not k.startswith("-")}
        n = int(d["6"])
        tag = int.from_bytes(data[off:off + 4], "little"); off += 4
        if tag == 0x01306B47:
            arr = np.frombuffer(data, np.float16, n, off).astype(np.float32); off += n * 2
        elif tag == 0:
            arr = np.frombuffer(data, np.float32, n, off).copy(); off += n * 4
        else:
            raise RuntimeError(f"{os.path.basename(binf)}: 未知 tag {hex(tag)} @ {t[1]}（param/bin 不配對？）")
        if int(d.get("5", 0)) == 1:      # 5=1 ⇒ 後面接 bias（out_ch 個 fp32）
            off += int(d["0"]) * 4
        out.append((t[1], arr))
    if off != len(data):
        raise RuntimeError(f"{os.path.basename(binf)}: 解析用掉 {off} B ≠ 檔案 {len(data)} B（param/bin 不配對？）")
    return out


def compare_weights(param_a, bin_a, param_b, bin_b):
    """兩次轉檔的權重是否同一組（可能只是層順序不同）。回傳 (是否等價, 說明)。"""
    try:
        A, B = _read_weights(param_a, bin_a), _read_weights(param_b, bin_b)
    except RuntimeError as e:
        return False, f"解析失敗：{e}"
    if len(A) != len(B):
        return False, f"層數不同 {len(A)} vs {len(B)}"
    in_order = all(a[1].shape == b[1].shape and np.array_equal(a[1], b[1]) for a, b in zip(A, B))
    if in_order:
        return True, f"{len(A)} 層權重逐層 bit-identical、順序也相同"
    ka = sorted(hash(a[1].tobytes()) for a in A)
    kb = sorted(hash(b[1].tobytes()) for b in B)
    if ka == kb:
        return True, (f"{len(A)} 層權重是同一組（multiset bit-identical），但**層順序不同**"
                      f"（pnnx 版本差異）⇒ bin 位元組會不同、算出來一樣；param+bin 必須配對使用")
    return False, "權重集合不同 — 這是真的不一樣，別發布"


def ncnn_forward(param, binf, img, mask):
    """用 ncnn 跑一次 forward。img[1,3,s,s]/mask[1,1,s,s] 是 torch tensor → 回 numpy [3,s,s]。"""
    import ncnn

    net = ncnn.Net()
    net.opt.use_vulkan_compute = False          # 純 CPU（GPU Vulkan 對這顆算錯，見 memory）
    net.load_param(param)
    net.load_model(binf)
    # ⚠️ ncnn.Mat(ndarray) 是「包住」那塊 buffer、不複製 ⇒ numpy 物件必須有 python 參考活著。
    # 直接塞暫存（ncnn.Mat(np.ascontiguousarray(...))）會讓 buffer 當場被回收 → 讀到已釋放記憶體
    # → 輸出是垃圾（實測同一顆模型跑兩次 max|d|=2.0＝整個值域）。踩過，別「簡化」掉這兩個變數。
    img_np = np.ascontiguousarray(img[0].numpy())           # [3,s,s]
    mask_np = np.ascontiguousarray(mask[0].numpy())         # [1,s,s]
    im = ncnn.Mat(img_np)
    mk = ncnn.Mat(mask_np)
    ex = net.create_extractor()
    ex.input("in0", im)
    ex.input("in1", mk)
    ret, out = ex.extract("out0")
    if ret != 0:
        raise RuntimeError(f"ncnn extract out0 失敗 ret={ret}")
    arr = np.array(out).reshape(out.c, out.h, out.w).copy()
    del ex
    net.clear()
    return arr


def sample_inputs(s):
    """
    真漫畫頁 + 矩形擦除塊 = 貼近引擎實跑的輸入。
    刻意不用 torch.randn：雜訊不在這顆模型的資料分布內，去字輸出本來就會亂跳，
    拿它比對容易得到「看起來很糟但其實無意義」的數字。找不到測試頁才退回合成漸層。
    """
    page = None
    if os.path.isdir(SANDBOX_TEST):
        imgs = sorted(f for f in os.listdir(SANDBOX_TEST) if f.lower().endswith((".jpg", ".png")))
        if imgs:
            import cv2
            bgr = cv2.imread(os.path.join(SANDBOX_TEST, imgs[0]))
            if bgr is not None:
                rgb = cv2.cvtColor(cv2.resize(bgr, (s, s)), cv2.COLOR_BGR2RGB)
                page = torch.from_numpy(rgb.astype(np.float32).transpose(2, 0, 1))[None] / 127.5 - 1.0
    if page is None:
        g = torch.linspace(-1, 1, s)
        page = (g[None, :] * g[:, None]).expand(1, 3, s, s).clone()
    mask = torch.zeros(1, 1, s, s)
    mask[:, :, s // 5:s // 3, s // 6:s // 2] = 1.0    # 一塊「要擦掉的字」
    return page * (1 - mask), mask                    # holes 歸零＝引擎餵法


def verify(model, param, binf, skip_ref):
    """① ncnn vs torch（trace shape 512）② tile 768 真的能跑 ③ vs 現行 release 權重。"""
    ok = True

    for s in (TRACE_SIZE, ENGINE_TILE):
        img, mask = sample_inputs(s)
        with torch.no_grad():
            ref = model(img, mask)[0].numpy()
        got = ncnn_forward(param, binf, img, mask)
        if got.shape != ref.shape:
            print(f"✗ s={s}: shape 不符 ncnn{got.shape} vs torch{ref.shape}")
            ok = False
            continue
        d = np.abs(got - ref)
        # 容差寬：bin 是 fp16 存 ⇒ 對 [-1,1] 輸出，~1e-2 級誤差是正常的
        tag = "trace shape" if s == TRACE_SIZE else f"引擎實跑 tile（全卷積 ⇒ 非 {TRACE_SIZE} 也能跑）"
        good = d.mean() < 2e-2
        ok &= good
        print(f"{'✓' if good else '✗'} s={s}（{tag}）：ncnn vs torch  "
              f"max|d|={d.max():.4f} mean|d|={d.mean():.5f}")

    if skip_ref:
        return ok
    rp = os.path.join(REF_DIR, f"{STEM}.ncnn.param")
    rb = os.path.join(REF_DIR, f"{STEM}.ncnn.bin")
    if not (os.path.exists(rp) and os.path.exists(rb)):
        print(f"（跳過 release 權重比對：{REF_DIR} 找不到 {STEM}.ncnn.*）")
        return ok

    print(f"\n— 與現行 release 權重比對（{REF_DIR}）—")
    for f, r in ((param, rp), (binf, rb)):
        a, b = os.path.getsize(f), os.path.getsize(r)
        same = sha256_of(f) == sha256_of(r)
        print(f"  {os.path.basename(f)}: 我們={a:,} B  release={b:,} B  "
              f"（差 {a - b:+,} B）sha256 {'相同' if same else '不同（預期：pnnx 版本 ⇒ 層順序/命名不同）'}")
    # bin 位元組不同不代表權重不同 — 逐層比對值，證明「只是順序不同、權重一樣」
    same_w, note = compare_weights(param, binf, rp, rb)
    print(f"  {'✓' if same_w else '⚠️'} 權重逐層比對：{note}")
    for s in (TRACE_SIZE, ENGINE_TILE):
        img, mask = sample_inputs(s)
        a = ncnn_forward(param, binf, img, mask)
        b = ncnn_forward(rp, rb, img, mask)
        d = np.abs(a.astype(np.float64) - b.astype(np.float64))
        good = d.mean() < 2e-2
        ok &= good
        print(f"  {'✓' if good else '✗'} out0 數值（s={s}）：max|d|={d.max():.3e} mean|d|={d.mean():.3e}"
              f"{'（bit-identical）' if np.array_equal(a, b) else ''} "
              f"→ {'容差內等價' if good else '不等價！'}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-ref", action="store_true", help="不與現有 release 權重比對")
    args = ap.parse_args()

    ckpt = ensure_ckpt()
    model = build_model(ckpt)
    param, binf = export(model)
    ok = check_blob_names(param)
    ok &= verify(model, param, binf, args.skip_ref)
    print(f"\n{'✅ 全過' if ok else '❌ 有項目沒過'}：{param}\n{'':13}{binf}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
