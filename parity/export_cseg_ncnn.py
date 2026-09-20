#!/usr/bin/env python3
"""
把 CartoonSegmentation 的人物分割模型（RTMDet-Ins，cartoonseg.onnx）轉成 NCNN：
ONNX 切圖（去掉圖內的 NMS／動態遮罩頭）→ pnnx → ncnn，後處理改由 Kotlin 做。
產出 = 夜讀用的 `cartoonseg.ncnn.param` + `.bin`（models.json 的 charseg 角色）。

為什麼要切：mmdeploy 匯出的 ONNX 把 NonMaxSuppression、TopK 與逐實例的動態卷積都放在圖裡，
NCNN 沒有這些層。切在**原始頭輸出**之後，剩下的 backbone（CSPNeXt）＋ neck（PAFPN）＋ head 全是
Convolution／Swish／Pooling／Interp 這類標準層，pnnx 一次轉過。

blob 契約（引擎 `engine/src/main/cpp/ncnn_jni.cpp:extractNative` 照這個吃，名字與順序不可改）：
  in0  = [3,640,640] float，**BGR**，(x − mean) / std，mean=(103.53,116.28,123.675) std=(57.375,57.12,58.395)
         等比縮到長邊 640、**右下角** pad（mmdet 慣例、非置中），pad 值 114（在正規化之前填）。
  out0..out2 = rtm_cls   [1,H,W]   三層 stride 8/16/32（H=80/40/20），單類別 logit → sigmoid 是分數
  out3..out5 = rtm_reg   [4,H,W]   l,t,r,b 距離；**relu 後 × stride** 才是像素距離（relu 在 Kotlin 做）
  out6..out8 = rtm_kernel[169,H,W] 動態卷積參數：weights [80,64,8] 再 biases [8,8,1]
  out9       = mask_feat [8,80,80] 遮罩原型特徵（stride 8）

後處理規格（mmdet 3.3 `RTMDetInsHead._predict_by_feat_single` / `_mask_predict_by_feat_single`，
本檔 `postprocess()` 是 numpy 參考實作，Kotlin `CsegPost` 照它移植、JVM 測試逐像素比對）：
  先驗   prior(x,y) = (col × stride, row × stride)，offset 0；展平序 = row-major（idx = row×W + col）
  解碼   x1 = px − l, y1 = py − t, x2 = px + r, y2 = py + b，夾到 [0,640]
  門檻   score > 0.05（ONNX 內 NMS 的 score_threshold）→ NMS IoU 0.6 → 最多 100 個
  遮罩   相對座標 = (prior − grid8) / (stride × 8)，grid8 = 80×80 的 (col×8,row×8)；
         cat[相對座標(2), mask_feat(8)] → 1×1 動態卷積 10→8 relu → 8→8 relu → 8→1
         → 雙線性 ×8 到 640（align_corners=False）→ sigmoid = 機率
  管線用法（research/charmask.py run_cseg）：score > 0.3 的實例、機率 > 0.5、裁掉 pad、最近鄰放回原尺寸、取聯集

驗證三層：
  (1) 切圖 ONNX（ORT）vs NCNN（ncnn python）：十個原始輸出的 max|Δ|（fp16 storage 容差）
  (2) 切圖 + numpy 後處理 vs **完整 ONNX**（ORT，含圖內 NMS）：實例配對後遮罩機率差、二值 IoU
  (3) 對 11 張測試頁：三條路（完整 ONNX／ORT 切圖＋後處理／NCNN＋後處理）的**聯集遮罩** IoU

輸出落在 parity/out/cseg/（gitignore）；`--fixture` 另外把 ch34_011 的十個原始輸出（fp16）與期望遮罩
寫進 engine/src/test/resources/charseg/，給 Kotlin 後處理的 JVM parity 測試。

用法：
    python3 parity/export_cseg_ncnn.py              # 轉檔 + 三層驗證
    python3 parity/export_cseg_ncnn.py --fixture    # 另外產 JVM 測試 fixture
    python3 parity/export_cseg_ncnn.py --skip-export  # 只跑驗證（用既有 param/bin）
來源 ONNX：預設 $YAKU_CSEG_ONNX，缺省找 ../yakuyomi-nightread/research/out/models/cartoonseg.onnx。
"""
import argparse
import glob
import hashlib
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import OUT as _OUT, PNNX, ROOT  # noqa: E402

SRC = os.environ.get(
    "YAKU_CSEG_ONNX",
    os.path.join(os.path.dirname(ROOT), "yakuyomi-nightread", "research", "out", "models", "cartoonseg.onnx"),
)
OUTDIR = os.path.join(_OUT, "cseg")
CUT = os.path.join(OUTDIR, "cartoonseg.onnx")            # 切圖後的 ONNX（pnnx 依此檔名命名輸出）
PARAM = os.path.join(OUTDIR, "cartoonseg.ncnn.param")
BIN = os.path.join(OUTDIR, "cartoonseg.ncnn.bin")
PAGES = sorted(glob.glob(os.path.join(ROOT, "app-sandbox", "src", "main", "assets", "test", "*.*")))
FIXTURE_DIR = os.path.join(ROOT, "engine", "src", "test", "resources", "charseg")

SIZE = 640
STRIDES = (8, 16, 32)
MEAN = np.array([103.53, 116.28, 123.675], np.float32)   # BGR
STD = np.array([57.375, 57.12, 58.395], np.float32)
SCORE_THR = 0.05      # ONNX 內 NMS 的 score_threshold
NMS_IOU = 0.6
MAX_PER_IMG = 100
PIPELINE_SCORE = 0.3  # research/charmask.py run_cseg 的門檻
MASK_THR = 0.5
HEAD_OUTPUTS = (
    [f"/bbox_head/rtm_cls.{i}/Conv_output_0" for i in range(3)]
    + [f"/bbox_head/rtm_reg.{i}/Conv_output_0" for i in range(3)]
    + [f"/bbox_head/rtm_kernel.{i}/Conv_output_0" for i in range(3)]
    + ["/bbox_head/mask_head/projection/Conv_output_0"]
)
WEIGHT_NUMS = (80, 64, 8)   # (8+2)×8, 8×8, 8×1
BIAS_NUMS = (8, 8, 1)


# ── 前處理（與 research/charmask.py run_cseg 逐位元同式）──────────────────────
def preprocess(bgr):
    import cv2
    h, w = bgr.shape[:2]
    s = SIZE / max(h, w)
    nh, nw = int(round(h * s)), int(round(w * s))
    canvas = np.full((SIZE, SIZE, 3), 114, np.uint8)
    canvas[:nh, :nw] = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_AREA)
    x = ((canvas.astype(np.float32) - MEAN) / STD).transpose(2, 0, 1)
    return np.ascontiguousarray(x), nh, nw


# ── 切圖 + pnnx ────────────────────────────────────────────────────────────────
def export():
    import onnx
    from onnx import utils
    os.makedirs(OUTDIR, exist_ok=True)
    print(f"切圖：{SRC}")
    utils.extract_model(SRC, CUT, ["input"], HEAD_OUTPUTS)
    m = onnx.load(CUT)
    for d, v in zip(m.graph.input[0].type.tensor_type.shape.dim, (1, 3, SIZE, SIZE)):
        d.dim_value = v
        d.dim_param = ""
    onnx.save(m, CUT)
    ops = {}
    for n in m.graph.node:
        ops[n.op_type] = ops.get(n.op_type, 0) + 1
    print(f"  {len(m.graph.node)} 節點：{ops}")
    cmd = [PNNX, os.path.basename(CUT), f"inputshape=[1,3,{SIZE},{SIZE}]"]   # 預設 fp16=1（bin 存 fp16）、不量化
    print("pnnx：", " ".join(cmd))
    r = subprocess.run(cmd, cwd=OUTDIR, capture_output=True, text=True)
    if r.returncode != 0 or not os.path.exists(PARAM):
        print(r.stdout[-2000:], r.stderr[-2000:])
        raise SystemExit(f"pnnx 失敗（rc={r.returncode}）")
    layers = {}
    with open(PARAM, encoding="utf-8") as f:
        for line in list(f)[2:]:
            t = line.split()[0]
            layers[t] = layers.get(t, 0) + 1
    print(f"  → {os.path.basename(PARAM)} {os.path.getsize(PARAM):,} B、{os.path.basename(BIN)} {os.path.getsize(BIN):,} B")
    print(f"  層：{layers}")


# ── 推論 ──────────────────────────────────────────────────────────────────────
def ort_cut(x):
    import onnxruntime as ort
    sess = ort_cut.sess = getattr(ort_cut, "sess", None) or ort.InferenceSession(CUT, providers=["CPUExecutionProvider"])
    names = [o.name for o in sess.get_outputs()]
    outs = sess.run(None, {"input": x[None]})
    assert names == HEAD_OUTPUTS, names
    return [o[0] for o in outs]


def ort_full(x):
    import onnxruntime as ort
    sess = ort_full.sess = getattr(ort_full, "sess", None) or ort.InferenceSession(SRC, providers=["CPUExecutionProvider"])
    dets, labels, masks = sess.run(None, {"input": x[None]})
    return dets[0], masks[0]


def ncnn_forward(x):
    import ncnn
    net = ncnn_forward.net = getattr(ncnn_forward, "net", None)
    if net is None:
        net = ncnn.Net()
        # x86 的 ncnn python 開 fp16 算術會出 NaN（DBNet 驗證腳本也關）；權重仍是 bin 裡的 fp16
        net.opt.use_fp16_packed = False
        net.opt.use_fp16_storage = False
        net.opt.use_fp16_arithmetic = False
        net.opt.use_bf16_storage = False
        net.opt.use_vulkan_compute = False
        net.load_param(PARAM)
        net.load_model(BIN)
        ncnn_forward.net = net
    ex = net.create_extractor()
    ex.input("in0", ncnn.Mat(x))
    outs = []
    for i in range(10):
        rc, m = ex.extract(f"out{i}")
        if rc != 0:
            raise SystemExit(f"ncnn extract out{i} 失敗 rc={rc}")
        outs.append(np.array(m))
    return outs


# ── numpy 後處理（＝Kotlin CsegPost 的規格）─────────────────────────────────────
def sigmoid(v):
    return 1.0 / (1.0 + np.exp(-v))


def nms(boxes, scores, iou_thr):
    order = np.argsort(-scores, kind="stable")
    keep = []
    while order.size:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(boxes[i, 0], boxes[rest, 0])
        yy1 = np.maximum(boxes[i, 1], boxes[rest, 1])
        xx2 = np.minimum(boxes[i, 2], boxes[rest, 2])
        yy2 = np.minimum(boxes[i, 3], boxes[rest, 3])
        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        a = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
        b = (boxes[rest, 2] - boxes[rest, 0]) * (boxes[rest, 3] - boxes[rest, 1])
        iou = inter / np.maximum(a + b - inter, 1e-9)
        order = rest[iou <= iou_thr]
    return np.array(keep, dtype=np.int64)


def upsample8_bilinear(m):
    """F.interpolate(scale_factor=8, mode='bilinear', align_corners=False)：來源座標 = (dst+0.5)/8 − 0.5。"""
    h, w = m.shape
    H, W = h * 8, w * 8
    ys = (np.arange(H, dtype=np.float32) + 0.5) / 8 - 0.5
    xs = (np.arange(W, dtype=np.float32) + 0.5) / 8 - 0.5
    ys = np.clip(ys, 0, h - 1)
    xs = np.clip(xs, 0, w - 1)
    y0 = np.floor(ys).astype(int)
    x0 = np.floor(xs).astype(int)
    y1 = np.minimum(y0 + 1, h - 1)
    x1 = np.minimum(x0 + 1, w - 1)
    wy = (ys - y0).astype(np.float32)[:, None]
    wx = (xs - x0).astype(np.float32)[None, :]
    top = m[y0][:, x0] * (1 - wx) + m[y0][:, x1] * wx
    bot = m[y1][:, x0] * (1 - wx) + m[y1][:, x1] * wx
    return top * (1 - wy) + bot * wy


def postprocess(outs, score_thr=SCORE_THR, iou_thr=NMS_IOU, max_per_img=MAX_PER_IMG):
    """十個原始輸出 → (dets[N,5]=x1,y1,x2,y2,score（640 座標）, masks[N,640,640] 機率)。"""
    cls, reg, ker, mask_feat = outs[0:3], outs[3:6], outs[6:9], outs[9]
    boxes_l, scores_l, kernels_l, priors_l, strides_l = [], [], [], [], []
    for lv, stride in enumerate(STRIDES):
        c = cls[lv][0]                           # [H,W]
        h, w = c.shape
        yy, xx = np.mgrid[0:h, 0:w]
        px = (xx * stride).astype(np.float32).reshape(-1)
        py = (yy * stride).astype(np.float32).reshape(-1)
        score = sigmoid(c.reshape(-1).astype(np.float32))
        d = np.maximum(reg[lv], 0).reshape(4, -1).astype(np.float32) * stride   # relu × stride
        x1 = np.clip(px - d[0], 0, SIZE)
        y1 = np.clip(py - d[1], 0, SIZE)
        x2 = np.clip(px + d[2], 0, SIZE)
        y2 = np.clip(py + d[3], 0, SIZE)
        boxes_l.append(np.stack([x1, y1, x2, y2], 1))
        scores_l.append(score)
        kernels_l.append(ker[lv].reshape(169, -1).T.astype(np.float32))          # [HW,169]
        priors_l.append(np.stack([px, py], 1))
        strides_l.append(np.full(h * w, stride, np.float32))
    boxes = np.concatenate(boxes_l)
    scores = np.concatenate(scores_l)
    kernels = np.concatenate(kernels_l)
    priors = np.concatenate(priors_l)
    strides = np.concatenate(strides_l)
    sel = np.nonzero(scores > score_thr)[0]
    if sel.size == 0:
        return np.zeros((0, 5), np.float32), np.zeros((0, SIZE, SIZE), np.float32)
    keep = sel[nms(boxes[sel], scores[sel], iou_thr)][:max_per_img]

    # 動態卷積遮罩頭
    mf = mask_feat.astype(np.float32)             # [8,80,80]
    _, mh, mw = mf.shape
    gy, gx = np.mgrid[0:mh, 0:mw]
    gx8 = (gx * STRIDES[0]).astype(np.float32)
    gy8 = (gy * STRIDES[0]).astype(np.float32)
    masks = np.zeros((keep.size, SIZE, SIZE), np.float32)
    for n, i in enumerate(keep):
        k = kernels[i]
        w0 = k[0:80].reshape(8, 10)
        w1 = k[80:144].reshape(8, 8)
        w2 = k[144:152].reshape(1, 8)
        b0, b1, b2 = k[152:160], k[160:168], k[168:169]
        rel_x = (priors[i, 0] - gx8) / (strides[i] * 8)
        rel_y = (priors[i, 1] - gy8) / (strides[i] * 8)
        feat = np.concatenate([rel_x[None], rel_y[None], mf], 0).reshape(10, -1)   # [10, HW]
        x = np.maximum(w0 @ feat + b0[:, None], 0)
        x = np.maximum(w1 @ x + b1[:, None], 0)
        x = (w2 @ x + b2[:, None]).reshape(mh, mw)
        masks[n] = sigmoid(upsample8_bilinear(x))
    dets = np.concatenate([boxes[keep], scores[keep, None]], 1)
    return dets, masks


def union_mask(dets, masks, nh, nw, h, w, score=PIPELINE_SCORE):
    """管線用法：score>0.3 的實例、機率>0.5、裁 pad、最近鄰放回原尺寸、聯集（同 run_cseg）。"""
    import cv2
    out = np.zeros((h, w), np.uint8)
    for d, m in zip(dets, masks):
        if d[4] <= score:
            continue
        mm = (m[:nh, :nw] > MASK_THR).astype(np.uint8)
        out[cv2.resize(mm, (w, h), interpolation=cv2.INTER_NEAREST) > 0] = 1
    return out


# ── 驗證 ──────────────────────────────────────────────────────────────────────
def iou(a, b):
    a = a > 0
    b = b > 0
    u = np.logical_or(a, b).sum()
    return 1.0 if u == 0 else float(np.logical_and(a, b).sum() / u)


def match(dets_a, dets_b):
    """用框 IoU 配對兩組實例（≥0.7），回 [(ia, ib)]。"""
    pairs = []
    used = set()
    for ia, a in enumerate(dets_a):
        best, bj = 0.0, -1
        for ib, b in enumerate(dets_b):
            if ib in used:
                continue
            xx1, yy1 = max(a[0], b[0]), max(a[1], b[1])
            xx2, yy2 = min(a[2], b[2]), min(a[3], b[3])
            inter = max(0, xx2 - xx1) * max(0, yy2 - yy1)
            ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
            v = inter / ua if ua > 0 else 0
            if v > best:
                best, bj = v, ib
        if best >= 0.7:
            pairs.append((ia, bj))
            used.add(bj)
    return pairs


def verify():
    import cv2
    print(f"\n=== 驗證（{len(PAGES)} 頁）===")
    worst_raw = 0.0
    rows = []
    for p in PAGES:
        bgr = cv2.imread(p, cv2.IMREAD_COLOR)
        if bgr is None:
            continue
        name = os.path.splitext(os.path.basename(p))[0]
        h, w = bgr.shape[:2]
        x, nh, nw = preprocess(bgr)
        ref = ort_cut(x)
        got = ncnn_forward(x)
        # (1) 原始輸出
        d_raw = max(float(np.abs(r - g.reshape(r.shape)).max()) for r, g in zip(ref, got))
        worst_raw = max(worst_raw, d_raw)
        # (2) 後處理 vs 完整 ONNX
        dets_f, masks_f = ort_full(x)
        dets_r, masks_r = postprocess(ref)
        dets_n, masks_n = postprocess([g.reshape(r.shape) for r, g in zip(ref, got)])
        pairs = match(dets_f[dets_f[:, 4] > PIPELINE_SCORE], dets_r[dets_r[:, 4] > PIPELINE_SCORE])
        nf = int((dets_f[:, 4] > PIPELINE_SCORE).sum())
        nr = int((dets_r[:, 4] > PIPELINE_SCORE).sum())
        prob_d = 0.0
        bin_iou = 1.0
        if pairs:
            fsel = dets_f[:, 4] > PIPELINE_SCORE
            rsel = dets_r[:, 4] > PIPELINE_SCORE
            for ia, ib in pairs:
                mf_, mr_ = masks_f[fsel][ia], masks_r[rsel][ib]
                prob_d = max(prob_d, float(np.abs(mf_ - mr_).max()))
                bin_iou = min(bin_iou, iou(mf_ > MASK_THR, mr_ > MASK_THR))
        # (3) 聯集遮罩三條路
        u_f = union_mask(dets_f, masks_f, nh, nw, h, w)
        u_r = union_mask(dets_r, masks_r, nh, nw, h, w)
        u_n = union_mask(dets_n, masks_n, nh, nw, h, w)
        rows.append((name, d_raw, nf, nr, len(pairs), prob_d, bin_iou, iou(u_f, u_r), iou(u_f, u_n)))
        print(f"  {name:12s} 原始 max|Δ| {d_raw:6.3f} | 實例(>0.3) 完整 {nf:2d} 切圖 {nr:2d} 配對 {len(pairs):2d} "
              f"機率 max|Δ| {prob_d:.3f} 二值 IoU≥{bin_iou:.3f} | 聯集 IoU：切圖 {iou(u_f, u_r):.4f} NCNN {iou(u_f, u_n):.4f}")
    print(f"原始輸出 vs ORT 最差 max|Δ| = {worst_raw:.3f}（fp16 storage；logit 值域到 ~40）")
    bad = [r for r in rows if r[7] < 0.98 or r[8] < 0.97]
    if bad:
        print("⚠️ 聯集遮罩 IoU 偏低的頁：", [r[0] for r in bad])
    return rows


def write_fixture(page="ch34_011"):
    import cv2
    p = next(q for q in PAGES if os.path.splitext(os.path.basename(q))[0] == page)
    bgr = cv2.imread(p, cv2.IMREAD_COLOR)
    h, w = bgr.shape[:2]
    x, nh, nw = preprocess(bgr)
    outs = [o.astype(np.float16) for o in ort_cut(x)]      # fp16 存檔：kernel 169×8400 太大
    os.makedirs(FIXTURE_DIR, exist_ok=True)
    with open(os.path.join(FIXTURE_DIR, f"{page}_raw.bin"), "wb") as f:
        for o in outs:
            f.write(np.ascontiguousarray(o).tobytes())
    # 期望＝對**同一份 fp16 捨入後的值**做後處理，Kotlin 讀到的就是這份
    dets, masks = postprocess([o.astype(np.float32) for o in outs])
    u = union_mask(dets, masks, nh, nw, h, w)
    cv2.imwrite(os.path.join(FIXTURE_DIR, f"{page}_expected.png"), u * 255)
    with open(os.path.join(FIXTURE_DIR, f"{page}_meta.txt"), "w", encoding="utf-8") as f:
        f.write(f"w {w}\nh {h}\nnw {nw}\nnh {nh}\nshapes " + ";".join("x".join(map(str, o.shape)) for o in outs) + "\n")
        for d in dets:
            f.write("det " + " ".join(f"{v:.4f}" for v in d) + "\n")
    print(f"fixture → {FIXTURE_DIR}（{page}：{len(dets)} 個實例，聯集 {int(u.sum())} px）")


def main():
    ap = argparse.ArgumentParser(description="cartoonseg → NCNN 轉檔與驗證")
    ap.add_argument("--skip-export", action="store_true")
    ap.add_argument("--fixture", action="store_true")
    a = ap.parse_args()
    if not a.skip_export:
        export()
    for f in (PARAM, BIN):
        print(f"sha256 {os.path.basename(f)} = {hashlib.sha256(open(f, 'rb').read()).hexdigest()}  ({os.path.getsize(f):,} B)")
    verify()
    if a.fixture:
        write_fixture()


if __name__ == "__main__":
    main()
