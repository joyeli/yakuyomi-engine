#!/usr/bin/env python3
"""
comic-text-detector parity harness (CLAUDE.md §7).

對一張頁跑 comictextdetector.pt.onnx，畫出並比較：
  綠 = 上游「完整」後處理（直接 import m-i-t 的 SegDetectorRepresenter：輪廓→minAreaRect→unclip）
  紅 = 本專案 M0c「簡化」後處理（二值化→連通元件→軸對齊 bbox），與當時 engine/Detector.kt 的 ctd 版同演算法（163ee2b 移除；現在的 Detector.kt 在 :inference-core、是 DBNet）

用法：python3 parity/ctd_reference.py [頁圖]（預設 paths.SANDBOX_PAGE = 測試頁 demo03.png）

⚠ 這支要的 comictextdetector.pt.onnx 是**已退役**的偵測器（commit 163ee2b 起 DBNet 是唯一偵測器），
  該模型不在任何 models release 裡 ⇒ 這支等於凍在歷史。它產出的 faithful_boxes.json 已經
  **凍結成 parity/fixtures/faithful_boxes.json**（OCR parity 數字就是對那 30 框定義的），
  所以重跑這支不是驗證 parity 的必要條件，別為了它去找退役模型。
輸出：parity/out/{faithful,simplified,compare}.png 與 faithful_boxes.json（§7 基準）
"""
import os, sys, json, importlib.util
import numpy as np
import cv2
import onnxruntime as ort

from paths import ROOT, OUT, MODELS, MIT_CLONE, SANDBOX_PAGE  # 集中路徑，見 paths.py
MODEL = os.path.join(MODELS, "comictextdetector.pt.onnx")
MIT_DB = os.path.join(MIT_CLONE, "manga_translator/detection/ctd_utils/utils/db_utils.py")
INPUT_SIZE = 1024
THRESH = 0.3       # SegDetectorRepresenter(thresh=0.3)
BOX_THRESH = 0.6   # ctd.py 外部過濾
MIN_SIDE = 3

os.makedirs(OUT, exist_ok=True)
img_path = sys.argv[1] if len(sys.argv) > 1 else SANDBOX_PAGE


def import_seg_rep():
    spec = importlib.util.spec_from_file_location("ctd_db_utils", MIT_DB)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.SegDetectorRepresenter


def letterbox(im, new_shape=(INPUT_SIZE, INPUT_SIZE), color=(0, 0, 0)):
    # 對齊 ctd_utils/utils/imgproc_utils.py:letterbox（auto=False, padding 全在右/下）
    h, w = im.shape[:2]
    r = min(new_shape[0] / h, new_shape[1] / w)
    nw, nh = round(w * r), round(h * r)
    dw, dh = new_shape[1] - nw, new_shape[0] - nh
    if (w, h) != (nw, nh):
        im = cv2.resize(im, (nw, nh), interpolation=cv2.INTER_LINEAR)
    im = cv2.copyMakeBorder(im, 0, dh, 0, dw, cv2.BORDER_CONSTANT, value=color)
    return im, r, dw, dh


def simplified_boxes(prob, im_w, im_h):
    """本專案 M0c：二值化→連通元件→軸對齊 bbox（與 Detector.kt.boxesFromProbMap 同邏輯）。"""
    hh, ww = prob.shape
    binary = (prob > THRESH).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    boxes = []
    for i in range(1, n):
        x, y, w, h, _ = stats[i]
        score = float(prob[y:y + h, x:x + w][labels[y:y + h, x:x + w] == i].mean())
        if score < BOX_THRESH or min(w, h) < MIN_SIDE:
            continue
        boxes.append((round(x / ww * im_w), round(y / hh * im_h),
                      round((x + w) / ww * im_w), round((y + h) / hh * im_h), score))
    return boxes


def main():
    SegDetectorRepresenter = import_seg_rep()
    img = cv2.imread(img_path)  # BGR
    if img is None:
        sys.exit(f"讀不到圖：{img_path}")
    im_h, im_w = img.shape[:2]
    print(f"image: {img_path}  {im_w}x{im_h}")

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    lb, r, dw, dh = letterbox(rgb)
    blob = cv2.dnn.blobFromImage(lb, scalefactor=1 / 255.0, size=(INPUT_SIZE, INPUT_SIZE))  # NCHW, 不 swap → RGB

    sess = ort.InferenceSession(MODEL, providers=["CPUExecutionProvider"])
    blk, seg, det = sess.run(["blk", "seg", "det"], {"images": blob.astype(np.float32)})
    print(f"outputs: blk{blk.shape} seg{seg.shape} det{det.shape}  letterbox r={r:.4f} pad=({dw},{dh})")

    # 去 letterbox padding（ctd.py 同樣裁掉右/下）
    det = det[..., :det.shape[2] - dh, :det.shape[3] - dw]
    prob = det[0, 0]

    # 綠：完整後處理
    seg_rep = SegDetectorRepresenter(thresh=THRESH)
    boxes_batch, scores_batch = seg_rep(None, det, height=im_h, width=im_w)
    faithful = [(b, float(s)) for b, s in zip(boxes_batch[0], scores_batch[0]) if s > BOX_THRESH]

    # 紅：簡化後處理
    simple = simplified_boxes(prob, im_w, im_h)

    print(f"faithful(綠)={len(faithful)} 框 ; simplified(紅)={len(simple)} 框")

    im_f = img.copy()
    for b, s in faithful:
        cv2.polylines(im_f, [np.array(b, np.int32).reshape(-1, 1, 2)], True, (0, 255, 0), 3)
    im_s = img.copy()
    for (x0, y0, x1, y1, s) in simple:
        cv2.rectangle(im_s, (x0, y0), (x1, y1), (0, 0, 255), 3)
    compare = np.hstack([im_f, im_s])

    cv2.imwrite(os.path.join(OUT, "faithful.png"), im_f)
    cv2.imwrite(os.path.join(OUT, "simplified.png"), im_s)
    cv2.imwrite(os.path.join(OUT, "compare.png"), compare)
    with open(os.path.join(OUT, "faithful_boxes.json"), "w") as f:
        json.dump({"image": os.path.basename(img_path), "w": im_w, "h": im_h,
                   "boxes": [{"quad": np.array(b).tolist(), "score": s} for b, s in faithful]},
                  f, ensure_ascii=False, indent=2)
    print(f"輸出 → {OUT}/(faithful|simplified|compare).png, faithful_boxes.json")


if __name__ == "__main__":
    main()
