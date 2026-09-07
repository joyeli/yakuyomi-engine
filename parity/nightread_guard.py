#!/usr/bin/env python3
"""nightread_guard.py — 夜讀紅線的數值化驗收：前景保護區框內「原白像素被塗黑」的比例。

背景：目視驗證（包括我自己三輪裁圖目檢）在這個尺度上系統性不可靠——審查員小組兩輪都抓到
我看過說「完整」的臉/白鬍被填黑。紅線「絕不塗錯」必須變成可量測的測試。

輸入：
  parity/nightread_guard.json —— 每頁一組保護框（臉/手/皮膚/白衣/白髮），原圖像素座標，
      由審查員標註（每頁兩位、取聯集）；可手動增修。
用法：
  python3 nightread_guard.py out/nr_b26 [out/nr_b19 ...]   # 逐版本印違規表
違規判定：框內「原圖 ≥ WHITE_TH 的像素」在成品中 < DARK 的比例 > VIOL_FRAC ⇒ 該框違規。
  （框故意貼身；門檻留餘裕給描邊/字等合法暗像素）
"""
import glob
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths  # noqa: E402

GUARD_JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nightread_guard.json")
WHITE_TH = 235
DARK = 60
VIOL_FRAC = 0.15      # 框內原白像素被塗黑 > 15% ⇒ 違規


def load_guard():
    with open(GUARD_JSON, encoding="utf-8") as f:
        return json.load(f)


def evaluate(outdir, guard, verbose=True):
    """回傳 (違規框數, 總框數, 逐頁明細)。"""
    total, viol = 0, 0
    rows = []
    for page, boxes in guard.items():
        src = sorted(glob.glob(os.path.join(paths.SANDBOX_TEST, page + ".*")))
        fin_p = os.path.join(outdir, f"{page}_final.png")
        if not src or not os.path.exists(fin_p):
            continue
        g = cv2.imread(src[0], 0)
        fin = cv2.imread(fin_p, 0)
        for b in boxes:
            x0, y0, x1, y1 = b["x0"], b["y0"], b["x1"], b["y1"]
            x0, y0 = max(0, x0), max(0, y0)
            x1, y1 = min(g.shape[1], x1), min(g.shape[0], y1)
            if x1 <= x0 or y1 <= y0:
                continue
            wb = g[y0:y1, x0:x1] >= WHITE_TH
            if wb.sum() < 50:
                continue
            frac = float((fin[y0:y1, x0:x1][wb] < DARK).mean())
            total += 1
            bad = frac > VIOL_FRAC
            viol += bad
            rows.append((page, b["label"], b["kind"], frac, bad))
    if verbose:
        print(f"== {outdir}: 違規 {viol}/{total} 框 ==")
        for page, label, kind, frac, bad in rows:
            if bad:
                print(f"  ✗ {page:9s} {kind:4s} {label:28s} 塗黑 {frac*100:5.1f}%")
    return viol, total, rows


def main():
    guard = load_guard()
    dirs = sys.argv[1:] or [os.path.join(paths.OUT, "nightread")]
    print(f"保護框 {sum(len(v) for v in guard.values())} 個 / {len(guard)} 頁；門檻：框內原白像素塗黑 > {VIOL_FRAC*100:.0f}%")
    for d in dirs:
        evaluate(d, guard)


if __name__ == "__main__":
    main()
