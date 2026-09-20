#!/usr/bin/env python3
"""
用 NCNN（fp16 或 int8）的人物分割模型，把測試頁的人物遮罩落成 `<頁名>_char.png`，
給 yakuyomi-nightread 的桌面批次（`NIGHTREAD_CHARMASK=<此輸出夾>`）跑守護框＋白泡。

這是 int8 的**管線級**驗收：模型級的 IoU 看不出「遮罩過度覆蓋讓紅線指標變好看」（守護框只問
有沒有塗到人物、不問該塗的有沒有塗），所以一定要把遮罩餵進夜讀、看白泡與亮區。

配方＝yolo ∪ cseg（2026-09-21 定案）。兩顆各自的 param/bin 由 env 指定（預設 parity/out 的 fp16）：
  YAKU_CSEG_NCNN_PARAM / YAKU_CSEG_NCNN_BIN、YAKU_YOLOSEG_NCNN_PARAM / YAKU_YOLOSEG_NCNN_BIN

用法：python3 parity/charseg_ncnn_masks.py -o <輸出夾> [--only cseg|yolo]
"""
import argparse
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import export_cseg_ncnn as C   # noqa: E402
import export_yoloseg_ncnn as Y   # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--outdir", required=True)
    ap.add_argument("--only", choices=("cseg", "yolo"), default=None)
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    print(f"cseg: {C.PARAM}\nyolo: {Y.PARAM}")
    for p in C.PAGES:
        bgr = cv2.imread(p, cv2.IMREAD_COLOR)
        if bgr is None:
            continue
        name = os.path.splitext(os.path.basename(p))[0]
        h, w = bgr.shape[:2]
        u = np.zeros((h, w), np.uint8)
        nc = ny = 0
        if a.only != "yolo":
            x, nh, nw = C.preprocess(bgr)
            outs = C.ncnn_forward(x)
            ref_shapes = [(1, 80, 80), (1, 40, 40), (1, 20, 20), (4, 80, 80), (4, 40, 40), (4, 20, 20),
                          (169, 80, 80), (169, 40, 40), (169, 20, 20), (8, 80, 80)]
            outs = [o.reshape(s) for o, s in zip(outs, ref_shapes)]
            dets, masks = C.postprocess(outs)
            u |= C.union_mask(dets, masks, nh, nw, h, w)
            nc = int((dets[:, 4] > C.PIPELINE_SCORE).sum()) if len(dets) else 0
        if a.only != "cseg":
            x, nh, nw, top, left = Y.preprocess(bgr)
            o0, o1 = Y.ncnn_run(x)
            m, ny = Y.postprocess(o0, o1, nh, nw, top, left, h, w)
            u |= m
        cv2.imwrite(os.path.join(a.outdir, f"{name}_char.png"), u * 255)
        print(f"  {name:12s} cseg 實例 {nc:2d}  yolo 實例 {ny:2d}  人物像素 {100 * u.mean():5.1f}%")
    print(f"→ {a.outdir}")


if __name__ == "__main__":
    main()
