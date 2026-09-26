#!/usr/bin/env python3
"""
把 48px CTC OCR 轉成 NCNN：ckpt → 包一層（PE 改成輸入、拿掉 color 頭）→ torch.jit.trace → pnnx（動態寬）→ ncnn。
產出 = `ocr_48px_ctc.ncnn.param` + `.bin`（models.json 的 ocr 角色，取代 ORT 的 ocr_int8.onnx）。

寬度牆與繞法：
  上游 `PositionalEncoding.forward` 是 `x + self.pe[:, :x.size(1)]`——trace 時 `size(1)` 是常數，pnnx 把 PE 切片
  烤死在 trace 寬度，換寬就是垃圾（2026-07 就是這樣撞死、才讓 OCR 留 ORT）。這裡把 **PE 當第二個輸入**餵進去
  （三層 encoder 共用同一張、值是純正弦、Kotlin 幾行算得出來），圖裡只剩 `x + in1`；再配 pnnx 的
  `inputshape2`，讓 attention 的 Reshape 維度保持動態。`color` 輸出沒人用（文字色由去字後背景判），拿掉。

blob 契約（引擎 `ncnn_jni.cpp:ocrCtcNative` 照這個吃）：
  in0 = image [3,48,W]  值域 (x−127.5)/127.5、右側白邊由呼叫端補（同 ORT 路徑的 stripToTensor）
  in1 = pe    [T,320]   T = floor(W/4) − 1（backbone 實測歸納、export() 逐寬檢查；W<8 無效）
                        pe[t, 2i] = sin(t / 10000^(2i/320))、pe[t, 2i+1] = cos(同)
  out0 = char_logits [T,19264]  raw logits；greedy CTC（blank=0、收合重複）與 top-1 log_softmax 信心在 JNI 算完
        只回 idx[T] + logp[T]，Kotlin 端收合＋查字表（同 ORT 路徑的 ctcDecodeArr 語意）。

驗證：30 個凍結字條（parity/fixtures/faithful_boxes.json on demo03，同 ocr_parity.py）在 ORT fp32、ORT int8、
NCNN 三條路的 CTC 讀出文字逐行比；另掃一組非典型寬度（奇數、極短、極長）證明沒有寬度牆。
`--fixture` 把一條字條的 in0/in1 與期望 idx/logp 存進 engine/src/test/resources/ocr/ 給 JVM 測試。

用法：
    python3 parity/export_ocr_ncnn.py              # 轉檔 + 驗證
    python3 parity/export_ocr_ncnn.py --skip-export
"""
import argparse
import hashlib
import json
import math
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import ALPHABET, OUT as _OUT, PNNX, ROOT  # noqa: E402
import export_ocr_onnx as E   # noqa: E402  ensure_ocr_ctc / load_OCR / CKPT

OUTDIR = os.path.join(_OUT, "ocr_ncnn")
TRACED = os.path.join(OUTDIR, "ocr_48px_ctc.pt")
PARAM = os.environ.get("YAKU_OCR_NCNN_PARAM", os.path.join(OUTDIR, "ocr_48px_ctc.ncnn.param"))
BIN = os.environ.get("YAKU_OCR_NCNN_BIN", os.path.join(OUTDIR, "ocr_48px_ctc.ncnn.bin"))
MIXED_PARAM = os.path.join(OUTDIR, "ocr_48px_ctc_mixed.ncnn.param")   # 同一份 .bin；多了 per-layer featmask + 一個 Cast
FP32_ONNX = os.path.join(_OUT, "ocr_48px_ctc.onnx")
INT8_ONNX = os.path.join(_OUT, "ocr_int8.onnx")
FIXTURE_DIR = os.path.join(ROOT, "engine", "src", "test", "resources", "ocr")

D_MODEL = 320
TEXT_H = 48
BLANK = 0
W0, W1 = 256, 1024          # trace 寬與 pnnx inputshape2 的第二寬（差越大、動態維度推得越準）


def t_of_w(w):
    """backbone 的寬度下採樣：實測歸納（verify 逐寬檢查）。"""
    return t_of_w.fn(w)


t_of_w.fn = lambda w: w // 4 - 1   # 實測歸納（見 export() 的 backbone 掃描），main() 會再驗一次

T_CANDIDATES = {
    "floor(W/4)-1": lambda w: w // 4 - 1,
    "floor(W/4)": lambda w: w // 4,
    "ceil(W/4)": lambda w: (w + 3) // 4,
    "floor(W/4)+1": lambda w: w // 4 + 1,
    "floor(W/8)": lambda w: w // 8,
}


def make_pe(t):
    pos = np.arange(t, dtype=np.float32)[:, None]
    div = np.exp(np.arange(0, D_MODEL, 2, dtype=np.float32) * (-math.log(10000.0) / D_MODEL))
    pe = np.zeros((t, D_MODEL), np.float32)
    pe[:, 0::2] = np.sin(pos * div)
    pe[:, 1::2] = np.cos(pos * div)
    return pe


# ── 包裝：PE 當輸入、只出 char_logits ───────────────────────────────────────
def build_wrapper(model):
    import torch
    import torch.nn as nn

    class OcrForNcnn(nn.Module):
        def __init__(self, ocr):
            super().__init__()
            self.ocr = ocr
            self.pes = [layer.pe for layer in ocr.encoders.layers]
            for pe in self.pes:
                pe.forward = (lambda mod: (lambda x, offset=0: x + mod.pe_in))(pe)

        def forward(self, img, pe):
            for m in self.pes:
                m.pe_in = pe            # [1,T,320]，trace 記到的是「加上輸入張量」
            feats = self.ocr.backbone(img).squeeze(2).permute(0, 2, 1)
            feats = self.ocr.encoders(feats)
            return self.ocr.char_pred(self.ocr.char_pred_norm(feats))

    return OcrForNcnn(model).eval()


def write_mixed_param(src=None, dst=None):
    """
    混合精度 param：backbone（96% MACs 的卷積）維持 fp16、**transformer + char_pred 全 fp32**，同一份 .bin。
    真機 A/B（2026-09-21）：fp16 全開快 27–32% 但 transformer 零星讀錯（「偉いな山」「だろ立」）、fp32 全關 81/81 對但慢
    32–45%；fp16 storage-only 反而慢 10×（每層來回 cast）。

    做法＝ncnn 的 per-layer featmask（param key 31，net.cpp get_masked_option：bit0 關 fp16 arithmetic、bit1 關
    fp16 storage/packed、bit2 關 bf16）：從 Squeeze（backbone 輸出）起的每一層、加上 PE 輸入 in1 的 Split，都標 `31=7`
    （bf16 目前全域關、但 mask 順手關掉免得日後有人打開就走 bf16）。
    ⚠️ featmask 只管「這層自己怎麼算」，**不管進來的 blob**：net.cpp convert_layout 的 fp16→fp32 分支條件是
    `opt.use_fp16_storage && !layer->support_fp16_storage`，masked 層的 opt 已把 use_fp16_storage 關掉 → 條件不成立、
    不 cast → conv 吐出的 fp16 pack8 blob 一路被當 fp32 用：Squeeze 只 reshape 沒事，**Permute 用 elemsize=2 建 top 卻以
    float* 寫 w*h 個元素（兩倍）→ 堆積溢出**（T=30 約 19KB），砸到別的並發執行緒的 winograd workspace → 真機 SIGSEGV
    出現在無辜的 conv3x3s1_winograd43_fp16sa（第一版就是這樣死的；tombstone 符號化 + 五視角對 ncnn 原始碼驗證，2026-09-26）。
    所以在 conv 輸出與 Squeeze 之間**手動插一個 `Cast` 層**（0=2 fp16 → 1=1 fp32；Cast_arm 在 asimdhp 上自己
    support_fp16_storage，Net 不會預先動它、它吃 fp16 pack8 吐 fp32 pack8；Squeeze 是 base 層無 packing，Net 會自動 unpack
    到 pack1）。Cast 本身不標 mask。engine 的 OcrCtcParityTest 有結構測試守這條規則（masked 層的每個輸入都要來自
    masked 層、in1 或 Cast）。

    ⚠️ 這份 param 只在「有 fp16 storage 的 CPU（arm82 asimdhp）且 Net opt.use_fp16_storage 開」時正確：x86 桌面／舊 ARM／
    fp16 storage 關掉時 blob 是 fp32，Cast 卻宣告 fp16→fp32 → 靜默吐垃圾（不 crash、OCR 全錯）。桌面 verify 只驗得了 parse；
    引擎端由 Ocr.kt 依 cfg.ncnnFp16Storage && NcnnBackend.cpuSupportsFp16 決定載 mixed 還是原 param（同 bin）。
    """
    src = src or PARAM
    dst = dst or MIXED_PARAM
    lines = open(src, encoding="utf-8").read().splitlines()
    n_layer, n_blob = (int(v) for v in lines[1].split())
    out = []
    in1_blobs = set()
    masked = 0
    after_backbone = False
    for line in lines[2:]:
        f = line.split()
        typ, n_in, n_out = f[0], int(f[2]), int(f[3])
        ins, outs = f[4:4 + n_in], f[4 + n_in:4 + n_in + n_out]
        if typ == "Input" and f[1] == "in1":
            in1_blobs.add(outs[0])
        if typ == "Squeeze":
            after_backbone = True
            cast_out = ins[0] + "_fp32"
            out.append(f"Cast                     cast_backbone_fp32       1 1 {ins[0]} {cast_out} 0=2 1=1")
            line = line.replace(f" {ins[0]} ", f" {cast_out} ", 1)
            n_layer += 1
            n_blob += 1
        mask = after_backbone or (typ == "Split" and any(b in in1_blobs for b in ins))
        if mask and typ != "Input":
            line += " 31=7"
            masked += 1
        out.append(line)
    text = "\n".join([lines[0], f"{n_layer} {n_blob}"] + out) + "\n"
    open(dst, "w", encoding="utf-8").write(text)
    print(f"混合精度 param → {os.path.basename(dst)}（{masked} 層標 31=7：Squeeze 起全部 + in1 的 Split；"
          f"conv→Squeeze 間插 Cast fp16→fp32；表頭 {n_layer} 層 {n_blob} blob；.bin 共用）")
    return dst


def export():
    import torch
    E.ensure_ocr_ctc()
    os.makedirs(OUTDIR, exist_ok=True)
    OCR = E.load_OCR()
    dictionary = [s[:-1] for s in open(ALPHABET, encoding="utf-8").readlines()]
    model = OCR(dictionary, 768)
    sd = torch.load(E.CKPT, map_location="cpu")
    sd = sd["model"] if "model" in sd else sd
    for k in list(sd):
        if k.endswith("pe.pe"):
            sd.pop(k)
    model.load_state_dict(sd, strict=False)
    model.eval()
    # backbone 下採樣實測 → t_of_w
    with torch.no_grad():
        ts = {w: model.backbone(torch.zeros(1, 3, TEXT_H, w)).shape[-1] for w in (64, 100, 255, 256, 257, 258, 259, 300, 1000, 1024)}
    print("backbone T(W)：", ts)
    for cand_name, cand in T_CANDIDATES.items():
        if all(cand(w) == t for w, t in ts.items()):
            t_of_w.fn = cand
            print(f"  → T(W) = {cand_name}")
            break
    else:
        raise SystemExit("T(W) 對不上任何候選公式，看上面實測值另外歸納")
    with open(os.path.join(OUTDIR, "t_of_w.txt"), "w") as f:
        f.write(cand_name + "\n")

    img0 = torch.zeros(1, 3, TEXT_H, W0)
    t0 = t_of_w(W0)
    pe_buf = model.encoders.layers[0].pe.pe[:, :t0].clone()      # 模型自己的正弦表
    pe0 = torch.from_numpy(make_pe(t0))[None]                     # 引擎端要重算的那張
    print(f"make_pe vs 模型 pe buffer max|Δ| = {float((pe_buf - pe0).abs().max()):.1e}（float32 捨入；logits 會放大到 ~1e-2、argmax 不受影響）")
    with torch.no_grad():
        ref = model(img0)[0]          # 包裝前算：build_wrapper 會 monkeypatch 原模型的 pe.forward
    wrapper = build_wrapper(model)
    with torch.no_grad():
        got = wrapper(img0, pe_buf)
        d = float((ref - got).abs().max())
        print(f"包裝（餵模型自己的 pe）vs 原模型 char_logits max|Δ| = {d:.2e}（應為 0）")
        assert d == 0.0, "包裝改變了計算圖"
        traced = torch.jit.trace(wrapper, (img0, pe0))
    traced.save(TRACED)
    t0, t1 = t_of_w(W0), t_of_w(W1)
    cmd = [PNNX, os.path.basename(TRACED),
           f"inputshape=[1,3,{TEXT_H},{W0}],[1,{t0},{D_MODEL}]",
           f"inputshape2=[1,3,{TEXT_H},{W1}],[1,{t1},{D_MODEL}]"]
    print("pnnx：", " ".join(cmd))
    r = subprocess.run(cmd, cwd=OUTDIR, capture_output=True, text=True)
    if r.returncode != 0 or not os.path.exists(PARAM):
        print(r.stdout[-3000:], r.stderr[-3000:])
        raise SystemExit(f"pnnx 失敗（rc={r.returncode}）")
    layers = {}
    with open(PARAM, encoding="utf-8") as f:
        for line in list(f)[2:]:
            t = line.split()[0]
            layers[t] = layers.get(t, 0) + 1
    print(f"  → {os.path.basename(PARAM)} {os.path.getsize(PARAM):,} B、{os.path.basename(BIN)} {os.path.getsize(BIN):,} B")
    print(f"  層：{layers}")
    write_mixed_param()


# ── 推論 ──────────────────────────────────────────────────────────────────────
def ncnn_run(x, t):
    """x=[3,48,W] float → char_logits [T,dict]。"""
    import ncnn
    net = getattr(ncnn_run, "net", None)
    if net is None:
        net = ncnn.Net()
        net.opt.use_fp16_packed = False
        net.opt.use_fp16_storage = False
        net.opt.use_fp16_arithmetic = False
        net.opt.use_bf16_storage = False
        net.opt.use_vulkan_compute = False
        net.load_param(PARAM)
        net.load_model(BIN)
        ncnn_run.net = net
    # ⚠️ ncnn.Mat(ndarray) 不複製、只引用 numpy 記憶體 → 兩個輸入陣列都要活到 extract 結束
    #    （曾把 make_pe(t) 當暫時值直接塞進去：被釋放的記憶體推論中被部分覆寫 → 「大多對、零星錯」且不可重現）
    x_in = np.ascontiguousarray(x, dtype=np.float32)
    pe_in = np.ascontiguousarray(make_pe(t), dtype=np.float32)
    ex = net.create_extractor()
    ex.input("in0", ncnn.Mat(x_in))
    ex.input("in1", ncnn.Mat(pe_in))
    rc, m = ex.extract("out0")
    if rc != 0:
        raise SystemExit(f"ncnn extract 失敗 rc={rc}（W={x.shape[-1]} T={t}）")
    out = np.array(m).copy()
    del ex
    return out.reshape(-1, out.shape[-1])


def ort_run(path, x):
    import onnxruntime as ort
    cache = getattr(ort_run, "cache", {})
    sess = cache.get(path)
    if sess is None:
        sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        cache[path] = sess
        ort_run.cache = cache
    return sess.run(["char_logits"], {"image": x[None]})[0][0]


def verify(fixture=False):
    import cv2
    import ocr_parity as P
    dictionary = [s[:-1] for s in open(ALPHABET, encoding="utf-8").readlines()]
    img = cv2.cvtColor(cv2.imread(P.SANDBOX_PAGE), cv2.COLOR_BGR2RGB)
    boxes = json.load(open(P.FAITHFUL_BOXES, encoding="utf-8"))
    boxes = boxes["boxes"] if isinstance(boxes, dict) and "boxes" in boxes else boxes
    strips = P.build_strips(img, boxes)
    print(f"\n=== 驗證：{len(strips)} 條字條 ===")
    same_fp32 = same_int8 = argmax_same = 0
    have_int8 = os.path.exists(INT8_ONNX)
    rows = []
    for s in strips:
        x = s["x"][0]                       # [3,48,W]
        w = x.shape[-1]
        t = t_of_w(w)
        ref = ort_run(FP32_ONNX, x)
        got = ncnn_run(x, t)
        if got.shape[0] != ref.shape[0]:
            print(f"  ⚠️ W={w}：T 不合（ORT {ref.shape[0]} vs NCNN {got.shape[0]}），t_of_w 錯")
            continue
        txt_ref, p_ref = P.ctc_decode(ref, dictionary)[:2]
        txt_got, p_got = P.ctc_decode(got, dictionary)[:2]
        am = float((ref.argmax(1) == got.argmax(1)).mean())
        argmax_same += am == 1.0
        same_fp32 += txt_ref == txt_got
        line = f"  W={w:4d} T={t:3d} argmax同 {am:.3f}  fp32「{txt_ref}」 ncnn「{txt_got}」"
        if have_int8:
            txt_i8 = P.ctc_decode(ort_run(INT8_ONNX, x), dictionary)[0]
            same_int8 += txt_i8 == txt_got
            line += f" int8「{txt_i8}」"
        rows.append((w, t, am, txt_ref, txt_got))
        print(line)
    n = len(rows)
    print(f"NCNN vs ORT fp32：文字逐行相同 {same_fp32}/{n}、argmax 全同 {argmax_same}/{n}" + (f"；vs ORT int8 相同 {same_int8}/{n}" if have_int8 else ""))
    # 寬度牆掃描：同一條字條左右補白到各種寬，讀出文字要一樣
    s = max(strips, key=lambda s: len(P.ctc_decode(ort_run(FP32_ONNX, s["x"][0]), dictionary)[0]))
    x = s["x"][0]
    base = P.ctc_decode(ort_run(FP32_ONNX, x), dictionary)[0]
    print(f"\n=== 寬度牆掃描（字條「{base}」，原寬 {x.shape[-1]}，右補白到各寬）===")
    bad = 0
    for w in (x.shape[-1] + 1, x.shape[-1] + 7, 300, 333, 512, 777, 1000, 1024, 1500):
        if w < x.shape[-1]:
            continue
        xp = np.ones((3, TEXT_H, w), np.float32)
        xp[:, :, :x.shape[-1]] = x
        got = P.ctc_decode(ncnn_run(xp, t_of_w(w)), dictionary)[0]
        ref = P.ctc_decode(ort_run(FP32_ONNX, xp), dictionary)[0]
        ok = got == ref
        bad += not ok
        print(f"  W={w:4d} T={t_of_w(w):3d}  {'✓' if ok else '✗'}  ncnn「{got}」 fp32「{ref}」")
    print("寬度牆：", "無" if bad == 0 else f"{bad} 個寬度讀出不同")
    if fixture:
        os.makedirs(FIXTURE_DIR, exist_ok=True)
        s = strips[0]
        x = s["x"][0]
        t = t_of_w(x.shape[-1])
        logits = ort_run(FP32_ONNX, x)
        idx = logits.argmax(1)
        lse = np.log(np.exp(logits - logits.max(1, keepdims=True)).sum(1)) + logits.max(1)
        logp = logits[np.arange(len(idx)), idx] - lse
        with open(os.path.join(FIXTURE_DIR, "strip0_in0.bin"), "wb") as f:
            f.write(np.ascontiguousarray(x).astype(np.float32).tobytes())
        with open(os.path.join(FIXTURE_DIR, "strip0_pe.bin"), "wb") as f:
            f.write(make_pe(t).tobytes())
        import shutil
        shutil.copy(PARAM, os.path.join(FIXTURE_DIR, "ocr_48px_ctc.ncnn.param"))            # 18KB：結構測試比對用
        shutil.copy(MIXED_PARAM, os.path.join(FIXTURE_DIR, "ocr_48px_ctc_mixed.ncnn.param"))
        with open(os.path.join(FIXTURE_DIR, "strip0_meta.txt"), "w", encoding="utf-8") as f:
            f.write(f"w {x.shape[-1]}\nt {t}\ntext {P.ctc_decode(logits, dictionary)[0]}\n")
            f.write("idx " + " ".join(map(str, idx.tolist())) + "\n")
            f.write("logp " + " ".join(f"{v:.5f}" for v in logp.tolist()) + "\n")
        print(f"fixture → {FIXTURE_DIR}")


def main():
    ap = argparse.ArgumentParser(description="48px CTC OCR → NCNN 轉檔與驗證")
    ap.add_argument("--skip-export", action="store_true")
    ap.add_argument("--fixture", action="store_true")
    a = ap.parse_args()
    if not a.skip_export:
        export()
    else:
        name = open(os.path.join(OUTDIR, "t_of_w.txt")).read().strip()
        t_of_w.fn = T_CANDIDATES[name]
        write_mixed_param()
    for f in (PARAM, BIN):
        print(f"sha256 {os.path.basename(f)} = {hashlib.sha256(open(f, 'rb').read()).hexdigest()}  ({os.path.getsize(f):,} B)")
    verify(fixture=a.fixture)


if __name__ == "__main__":
    main()
