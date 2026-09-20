#!/usr/bin/env python3
"""
把夜讀用的 yoloseg（YOLO11-seg，manga109 訓練的 manga_seg_s.pt）轉成 NCNN：ultralytics export → ncnn。
產出 = `manga_seg_s.ncnn.param` + `.bin`（models.json 的 charseg-yolo 角色）。

ultralytics 的 `format=ncnn` 底層就是 pnnx（同 DBNet／AOT 的轉法），一鍵成功、fp16 storage、不量化。

blob 契約（引擎 `NcnnBackend.extract` 照這個吃，名字與順序不可改）：
  in0  = [3,1024,1024] float，**RGB**，/255，ultralytics letterbox：等比縮到長邊 1024、**置中** pad 114
  out0 = [39,21504]  每個 anchor：cx,cy,w,h（1024 座標）+ 3 類分數（0=frame 1=speech_bubble 2=character）+ 32 個遮罩係數
  out1 = [32,256,256] 遮罩 prototypes

後處理規格（＝research/charmask.py `run_yoloseg_onnx`，桌面守護框數字的來源；本檔 `postprocess()` 是同一套的
numpy 版，Kotlin `YoloSegPost` 照它移植、JVM 測試逐像素比）：
  篩   只取 character 類且分數 > 0.25 → NMS IoU 0.45（貪婪、分數遞減）
  遮罩 sigmoid(係數 · prototypes) [256×256] → 裁到 bbox（crop_mask：行 [int(y1), ceil(y2))、列同）
       → 雙線性放到 1024 → 去 letterbox → 雙線性放到原尺寸 → > 0.5 → 聯集
  ⚠️ 兩段雙線性都是 cv2.resize INTER_LINEAR（半像素中心、邊界夾住）；Kotlin 只在框的支撐區內算，結果相同。

驗證：NCNN vs ONNX（ORT）的 out0 分數通道／out1；兩條路 + numpy 後處理的**聯集遮罩** IoU（12 頁）。
`--fixture` 把 ch34_011 的 out0/out1（fp16）與期望遮罩寫進 engine/src/test/resources/charseg/。

用法：
    python3 parity/export_yoloseg_ncnn.py              # 轉檔 + 驗證
    python3 parity/export_yoloseg_ncnn.py --fixture
    python3 parity/export_yoloseg_ncnn.py --skip-export
來源：$YAKU_YOLOSEG_PT / $YAKU_YOLOSEG_ONNX，缺省找 ../yakuyomi-nightread/research/out/models/。
"""
import argparse
import glob
import hashlib
import os
import shutil
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import OUT as _OUT, ROOT  # noqa: E402

_MODELS = os.path.join(os.path.dirname(ROOT), "yakuyomi-nightread", "research", "out", "models")
PT = os.environ.get("YAKU_YOLOSEG_PT", os.path.join(_MODELS, "manga_seg_s.pt"))
ONNX = os.environ.get("YAKU_YOLOSEG_ONNX", os.path.join(_MODELS, "manga_seg_s.onnx"))
OUTDIR = os.path.join(_OUT, "yoloseg")
# 驗證可指向別的 param/bin（例如 ncnn2int8 的產物）：YAKU_YOLOSEG_NCNN_PARAM / _BIN
PARAM = os.environ.get("YAKU_YOLOSEG_NCNN_PARAM", os.path.join(OUTDIR, "manga_seg_s.ncnn.param"))
BIN = os.environ.get("YAKU_YOLOSEG_NCNN_BIN", os.path.join(OUTDIR, "manga_seg_s.ncnn.bin"))
PAGES = sorted(glob.glob(os.path.join(ROOT, "app-sandbox", "src", "main", "assets", "test", "*.*")))
FIXTURE_DIR = os.path.join(ROOT, "engine", "src", "test", "resources", "charseg")

SIZE = 1024
CONF = 0.25
IOU = 0.45
CHAR_CLS = 2
MASK_THR = 0.5


def preprocess(bgr):
    import cv2
    h, w = bgr.shape[:2]
    r = min(SIZE / h, SIZE / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    top, left = (SIZE - nh) // 2, (SIZE - nw) // 2
    canvas = np.full((SIZE, SIZE, 3), 114, np.uint8)
    canvas[top:top + nh, left:left + nw] = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
    x = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return np.ascontiguousarray(x.transpose(2, 0, 1)), nh, nw, top, left


def export():
    from ultralytics import YOLO
    os.makedirs(OUTDIR, exist_ok=True)
    tmp = os.path.join(OUTDIR, "manga_seg_s.pt")
    shutil.copy(PT, tmp)
    out = YOLO(tmp).export(format="ncnn", imgsz=SIZE, half=True)   # 底層 pnnx；half=fp16 storage
    src_dir = out if os.path.isdir(out) else os.path.dirname(out)
    shutil.copy(os.path.join(src_dir, "model.ncnn.param"), PARAM)
    shutil.copy(os.path.join(src_dir, "model.ncnn.bin"), BIN)
    print(f"→ {os.path.basename(PARAM)} {os.path.getsize(PARAM):,} B、{os.path.basename(BIN)} {os.path.getsize(BIN):,} B")


def ort_run(x):
    import onnxruntime as ort
    sess = ort_run.sess = getattr(ort_run, "sess", None) or ort.InferenceSession(ONNX, providers=["CPUExecutionProvider"])
    o0, o1 = sess.run(None, {sess.get_inputs()[0].name: x[None]})
    return o0[0], o1[0]


def ncnn_run(x):
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
    ex = net.create_extractor()
    ex.input("in0", ncnn.Mat(x))
    _, o0 = ex.extract("out0")
    _, o1 = ex.extract("out1")
    o0 = np.array(o0)
    o1 = np.array(o1)
    return o0.reshape(-1, o0.shape[-1]), o1


def nms(boxes_xywh, scores, iou_thr):
    """cv2.dnn.NMSBoxes 語意：分數遞減貪婪，IoU > thr 者剔除。boxes = x,y,w,h。"""
    x1 = boxes_xywh[:, 0]
    y1 = boxes_xywh[:, 1]
    x2 = x1 + boxes_xywh[:, 2]
    y2 = y1 + boxes_xywh[:, 3]
    area = boxes_xywh[:, 2] * boxes_xywh[:, 3]
    order = np.argsort(-scores, kind="stable")
    keep = []
    while order.size:
        i = order[0]
        keep.append(i)
        rest = order[1:]
        if rest.size == 0:
            break
        iw = np.clip(np.minimum(x2[i], x2[rest]) - np.maximum(x1[i], x1[rest]), 0, None)
        ih = np.clip(np.minimum(y2[i], y2[rest]) - np.maximum(y1[i], y1[rest]), 0, None)
        inter = iw * ih
        iou = inter / np.maximum(area[i] + area[rest] - inter, 1e-9)
        order = rest[iou <= iou_thr]
    return np.array(keep, dtype=np.int64)


def postprocess(o0, o1, nh, nw, top, left, h, w):
    """out0 [39,N] + out1 [32,mh,mw] → 原尺寸聯集遮罩（uint8 0/1）。＝run_yoloseg_onnx。"""
    import cv2
    pred = o0.T.astype(np.float32)
    protos = o1.astype(np.float32)
    nc = pred.shape[1] - 4 - protos.shape[0]
    scores = pred[:, 4:4 + nc]
    cls = scores.argmax(1)
    sc = scores.max(1)
    keep = (sc > CONF) & (cls == CHAR_CLS)
    mask = np.zeros((h, w), np.uint8)
    if not keep.any():
        return mask, 0
    boxes = pred[keep, :4]
    xywh = np.stack([boxes[:, 0] - boxes[:, 2] / 2, boxes[:, 1] - boxes[:, 3] / 2, boxes[:, 2], boxes[:, 3]], 1)
    idx = nms(xywh, sc[keep], IOU)
    coeff = pred[keep][idx, 4 + nc:]
    mh, mw = protos.shape[1:]
    mm = 1.0 / (1.0 + np.exp(-(coeff @ protos.reshape(protos.shape[0], -1))))
    mm = mm.reshape(-1, mh, mw).astype(np.float32)
    for j, bi in enumerate(idx):
        bx = boxes[bi]
        x1 = (bx[0] - bx[2] / 2) * mw / SIZE
        x2 = (bx[0] + bx[2] / 2) * mw / SIZE
        y1 = (bx[1] - bx[3] / 2) * mh / SIZE
        y2 = (bx[1] + bx[3] / 2) * mh / SIZE
        m1 = np.zeros((mh, mw), np.float32)
        m1[max(0, int(y1)):int(np.ceil(y2)), max(0, int(x1)):int(np.ceil(x2))] = 1
        full = cv2.resize(mm[j] * m1, (SIZE, SIZE), interpolation=cv2.INTER_LINEAR)
        sub = full[top:top + nh, left:left + nw]
        mask[cv2.resize(sub, (w, h), interpolation=cv2.INTER_LINEAR) > MASK_THR] = 1
    return mask, len(idx)


def iou_of(a, b):
    u = np.logical_or(a > 0, b > 0).sum()
    return 1.0 if u == 0 else float(np.logical_and(a > 0, b > 0).sum() / u)


def verify():
    import cv2
    print(f"\n=== 驗證（{len(PAGES)} 頁）===")
    for p in PAGES:
        bgr = cv2.imread(p, cv2.IMREAD_COLOR)
        if bgr is None:
            continue
        name = os.path.splitext(os.path.basename(p))[0]
        h, w = bgr.shape[:2]
        x, nh, nw, top, left = preprocess(bgr)
        a0, a1 = ort_run(x)
        b0, b1 = ncnn_run(x)
        d_score = float(np.abs(a0[4:7] - b0[4:7]).max())
        d_proto = float(np.abs(a1 - b1).mean())
        ma, na = postprocess(a0, a1, nh, nw, top, left, h, w)
        mb, nb = postprocess(b0, b1, nh, nw, top, left, h, w)
        print(f"  {name:12s} 分數 max|Δ| {d_score:.3f}  proto mean|Δ| {d_proto:.4f} | 實例 ORT {na:2d} NCNN {nb:2d} | 聯集 IoU {iou_of(ma, mb):.4f}")


def write_fixture(page="ch34_011"):
    import cv2
    p = next(q for q in PAGES if os.path.splitext(os.path.basename(q))[0] == page)
    bgr = cv2.imread(p, cv2.IMREAD_COLOR)
    h, w = bgr.shape[:2]
    x, nh, nw, top, left = preprocess(bgr)
    a0, a1 = ort_run(x)
    o0 = a0.astype(np.float16)
    o1 = a1.astype(np.float16)
    os.makedirs(FIXTURE_DIR, exist_ok=True)
    with open(os.path.join(FIXTURE_DIR, f"{page}_yolo_raw.bin"), "wb") as f:
        f.write(np.ascontiguousarray(o0).tobytes())
        f.write(np.ascontiguousarray(o1).tobytes())
    mask, n = postprocess(o0.astype(np.float32), o1.astype(np.float32), nh, nw, top, left, h, w)
    cv2.imwrite(os.path.join(FIXTURE_DIR, f"{page}_yolo_expected.png"), mask * 255)
    with open(os.path.join(FIXTURE_DIR, f"{page}_yolo_meta.txt"), "w", encoding="utf-8") as f:
        f.write(f"w {w}\nh {h}\nnw {nw}\nnh {nh}\ntop {top}\nleft {left}\n"
                f"shapes {'x'.join(map(str, o0.shape))};{'x'.join(map(str, o1.shape))}\ninstances {n}\n")
    print(f"fixture → {FIXTURE_DIR}（{page}：{n} 個實例，聯集 {int(mask.sum())} px）")


def main():
    ap = argparse.ArgumentParser(description="yoloseg → NCNN 轉檔與驗證")
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
