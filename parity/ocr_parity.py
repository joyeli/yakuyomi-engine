#!/usr/bin/env python3
"""
OCR parity：對凍結的 30 個文字行 quad 跑 48px CTC OCR ONNX，讀出日文；
給了 int8 模型就順便算出 models.json / docs/MODELS.md 宣稱的「CTC parity vs fp32」數字。

裁切複製自 manga_translator/utils/generic.py:sort_pnts + Quadrilateral.get_transformed_region @ d5a3eee
CTC 解碼複製自 model_48px_ctc.py:decode_ctc_top1（greedy, blank=0, 收合重複+去blank）

素材（皆入庫 ⇒ 空白 clone 可跑）：
  頁圖   app-sandbox/src/main/assets/test/demo03.png（paths.SANDBOX_PAGE；舊名 page.png，
         commit ea3e166 只是改名、位元相同）
  quad   parity/fixtures/faithful_boxes.json（paths.FAITHFUL_BOXES；30 框，來歷見該檔 _provenance）
  字表   engine/src/main/assets/models/alphabet-all-v5.txt（paths.ALPHABET 缺 ckpt 時自動退回這份）

模型（不入庫，兩顆都要自己重建）：
  fp32   python3 parity/export_ocr_onnx.py     → parity/out/ocr_48px_ctc.onnx（需 ckpt）
  int8   python3 parity/quantize_ocr_int8.py   → parity/out/ocr_int8.onnx
  ⚠ int8 ONNX 已不出貨（models-v4 起 OCR 走 NCNN 混合精度，release 不再掛 ocr_int8.onnx），
    這支的 int8 對照只剩歷史／參考用途；要跑就自己用 quantize_ocr_int8.py 重建。

用法：
  python3 parity/ocr_parity.py                       # 有 int8 就自動一起跑並印 parity
  python3 parity/ocr_parity.py --fp32 A.onnx --int8 B.onnx
  python3 parity/ocr_parity.py --int8 ""             # 只跑 fp32、不比對

⚠ 這支只驗「同一批 strip 上 int8 與 fp32 讀出的文字是否逐行相同」。**效能數字（如 ARM 快 3.6×）
  桌面驗不出來**——x86 上逐位元相同的同一顆模型都能測到 ~2× 落差（純噪音），要真機才算數。
"""
import os, json, glob, argparse
import numpy as np
import cv2
import onnxruntime as ort

from paths import ROOT, OUT, ALPHABET, MIT_CLONE, SANDBOX_PAGE, FAITHFUL_BOXES  # 集中路徑，見 paths.py
FP32 = os.path.join(OUT, "ocr_48px_ctc.onnx")
INT8 = os.path.join(OUT, "ocr_int8.onnx")
TEXT_H, BLANK = 48, 0


def sort_pnts(pts):
    pts = np.array(pts, dtype=np.float64)
    pv = (pts[:, None] - pts[None]).reshape((16, -1))
    lsv = pv[np.argsort(np.linalg.norm(pv, axis=1))[[8, 10]]]
    if (lsv[0] * lsv[1]).sum() < 0:
        lsv[0] = -lsv[0]
    struc = np.abs(lsv.mean(axis=0))
    is_v = struc[0] <= struc[1]
    if is_v:
        pts = pts[np.argsort(pts[:, 1])]
        pts = pts[[*np.argsort(pts[:2, 0]), *(np.argsort(pts[2:, 0])[::-1] + 2)]]
        return pts, True
    pts = pts[np.argsort(pts[:, 0])]
    out = np.zeros_like(pts)
    out[[0, 3]] = sorted(pts[[0, 1]], key=lambda p: p[1])
    out[[1, 2]] = sorted(pts[[2, 3]], key=lambda p: p[1])
    return out, False


def transformed_region(img, pts, direction, th):
    p = pts.astype(np.float32)
    l1a, l1b = (p[0] + p[1]) / 2, (p[2] + p[3]) / 2
    l2a, l2b = (p[1] + p[2]) / 2, (p[3] + p[0]) / 2
    ratio = np.linalg.norm(l1b - l1a) / max(np.linalg.norm(l2b - l2a), 1e-6)
    src = pts.astype(np.int64).copy()
    im_h, im_w = img.shape[:2]
    x1, y1 = max(int(src[:, 0].min()), 0), max(int(src[:, 1].min()), 0)
    x2, y2 = min(int(src[:, 0].max()), im_w), min(int(src[:, 1].max()), im_h)
    crop = img[y1:y2, x1:x2]
    src[:, 0] -= x1
    src[:, 1] -= y1
    if direction == 'h':
        h, w = max(int(th), 2), max(int(round(th / ratio)), 2)
    else:
        w, h = max(int(th), 2), max(int(round(th * ratio)), 2)
    dst = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], np.float32)
    M, _ = cv2.findHomography(src.astype(np.float32), dst, cv2.RANSAC, 5.0)
    if M is None:
        return None
    region = cv2.warpPerspective(crop, M, (w, h))
    if direction == 'v':
        region = cv2.rotate(region, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return region


def check_color(region_img):
    """彩色文字判定（對齊 m-i-t utils/bubble.py:check_color）：>10 個非灰階像素 → True。"""
    img = region_img.astype(np.float32)
    gray = img @ np.array([0.299, 0.587, 0.114], np.float32)
    d = ((img - gray[..., None]) ** 2).sum(-1)
    return int((d > 100).sum()) > 10


def is_ignore(region_img, ignore_bubble=0):
    """SFX/非氣泡文字判定（對齊 m-i-t utils/bubble.py:is_ignore）：
       邊框 2px 黑比例落在 [ignore_bubble, 100-ignore_bubble] 之間（混色＝非乾淨氣泡），或彩色文字 → 跳過。
       ignore_bubble 有效範圍 1–50（其他值＝關閉）。"""
    if ignore_bubble < 1 or ignore_bubble > 50:
        return False
    _, b = cv2.threshold(region_img, 127, 255, cv2.THRESH_BINARY)
    h, w = b.shape[:2]
    val0 = total = 0
    for sl in (b[0:2, 0:w], b[h - 2:h, 0:w], b[2:h - 2, 0:2], b[2:h - 2, w - 2:w]):
        val0 += int((sl.ravel() == 0).sum()); total += sl.size
    ratio = round(val0 / total, 6) * 100 if total else 0
    if ignore_bubble <= ratio <= 100 - ignore_bubble:
        return True
    return check_color(region_img)


def ctc_decode(logits, dictionary, colors=None):
    """greedy CTC（blank=0、收合重複+去blank）。
    傳入 colors（[T,6]＝fg_rgb+bg_rgb，未 clamp）則回傳 (text, prob, fg, bg)；
    色＝保留的非空白 char 對應 timestep 取色、clip 0..1、整行平均 ×255（對齊 model_48px_ctc.decode_ctc_top1）。"""
    lp = logits - logits.max(1, keepdims=True)
    lp = lp - np.log(np.exp(lp).sum(1, keepdims=True))  # log_softmax
    idx = lp.argmax(1)
    chars, probs, last = [], [], BLANK
    fgs, bgs = [], []
    for t in range(len(idx)):
        c = int(idx[t])
        if c != last and c != BLANK:
            ch = dictionary[c]
            sp = ch == '<SP>'
            chars.append(' ' if sp else ch)
            probs.append(lp[t, c])
            if colors is not None and not sp:
                cv = np.clip(colors[t], 0.0, 1.0)
                fgs.append(cv[:3]); bgs.append(cv[3:6])
        last = c
    text = ''.join(chars)
    prob = float(np.exp(np.mean(probs))) if probs else 0.0
    if colors is None:
        return text, prob
    fg = tuple(int(round(v * 255)) for v in (np.mean(fgs, axis=0) if fgs else (0.0, 0.0, 0.0)))
    bg = tuple(int(round(v * 255)) for v in (np.mean(bgs, axis=0) if bgs else (1.0, 1.0, 1.0)))
    return text, prob, fg, bg


def find_font():
    for c in (glob.glob(os.path.join(MIT_CLONE, "fonts/*.[to][tc][cf]")) +
              ["/mnt/c/Windows/Fonts/YuGothM.ttc", "/mnt/c/Windows/Fonts/msgothic.ttc",
               "/mnt/c/Windows/Fonts/meiryo.ttc", "/mnt/c/Windows/Fonts/msmincho.ttc"]):
        if os.path.exists(c):
            return c
    return None


def build_strips(img, boxes):
    """quad → 48px strip（每框一條，含方向）。兩顆模型吃同一批 ⇒ 差異只可能來自模型本身。"""
    strips = []
    for i, b in enumerate(boxes):
        try:
            pts, is_v = sort_pnts(b["quad"])
            direction = 'v' if is_v else 'h'
            region = transformed_region(img, pts, direction, TEXT_H)
            if region is None or region.shape[1] < 2:
                continue
            x = np.transpose((region.astype(np.float32) - 127.5) / 127.5, (2, 0, 1))[None]
            strips.append({"i": i, "dir": direction, "x": x, "quad": b["quad"]})
        except Exception as e:
            print(f"[{i:2d}] strip error: {e}")
    return strips


def run_ocr(model_path, strips, dictionary, label):
    """對整批 strip 跑一顆模型 → 每條的 (text, prob)。回傳 dict: strip index → result。"""
    sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    out = {}
    for s in strips:
        try:
            char_logits, _ = sess.run(["char_logits", "color"], {"image": s["x"]})
            text, prob = ctc_decode(char_logits[0], dictionary)
        except Exception as e:
            print(f"[{s['i']:2d}] {label} error: {e}")
            continue
        out[s["i"]] = {"i": s["i"], "dir": s["dir"], "prob": round(prob, 3), "text": text,
                       "quad": s["quad"]}
    return out


def report_parity(fp32_res, int8_res):
    """models.json / docs 宣稱的「CTC parity vs fp32」＝逐行 exact match 率。
    ⚠ 這是「讀出的字是否一模一樣」，**不等於品質損失**——不同的那行可能 int8 讀得比較對。"""
    keys = sorted(set(fp32_res) | set(int8_res))
    same = [k for k in keys if fp32_res.get(k, {}).get("text") == int8_res.get(k, {}).get("text")]
    diff = [k for k in keys if k not in same]
    pct = 100.0 * len(same) / len(keys) if keys else 0.0
    print(f"\n=== CTC parity：int8 vs fp32 ===")
    print(f"逐行 exact match = {len(same)}/{len(keys)} = {pct:.1f}%")
    for k in diff:
        f, q = fp32_res.get(k, {}), int8_res.get(k, {})
        print(f"  [{k:2d}] fp32 p={f.get('prob')} {f.get('text')!r}"
              f"  ≠  int8 p={q.get('prob')} {q.get('text')!r}")
    return {"lines": len(keys), "match": len(same), "parity_pct": round(pct, 2),
            "diff": [{"i": k, "fp32": fp32_res.get(k, {}).get("text"),
                      "fp32_prob": fp32_res.get(k, {}).get("prob"),
                      "int8": int8_res.get(k, {}).get("text"),
                      "int8_prob": int8_res.get(k, {}).get("prob")} for k in diff]}


def main():
    ap = argparse.ArgumentParser(description="48px CTC OCR parity（fp32 讀字；給 int8 就順便算 parity）")
    ap.add_argument("--page", default=SANDBOX_PAGE, help="頁圖（預設 repo 內 demo03.png）")
    ap.add_argument("--boxes", default=FAITHFUL_BOXES, help="quad fixture（預設 parity/fixtures/）")
    ap.add_argument("--fp32", default=FP32, help="fp32 ONNX（export_ocr_onnx.py 產）")
    ap.add_argument("--int8", default=INT8, help="int8 ONNX（quantize_ocr_int8.py 產）；'' = 不比對")
    args = ap.parse_args()

    for p, what in ((args.page, "頁圖"), (args.boxes, "quad fixture"), (ALPHABET, "alphabet")):
        if not os.path.exists(p):
            raise SystemExit(f"缺{what}：{p}")
    if not os.path.exists(args.fp32):
        raise SystemExit(f"缺 fp32 模型：{args.fp32}\n  先跑：python3 parity/export_ocr_onnx.py（需上游 ocr-ctc ckpt）")

    os.makedirs(OUT, exist_ok=True)   # 空白 clone 沒有 parity/out
    dictionary = [s[:-1] for s in open(ALPHABET, encoding="utf-8").readlines()]
    img = cv2.cvtColor(cv2.imread(args.page), cv2.COLOR_BGR2RGB)
    boxes = json.load(open(args.boxes, encoding="utf-8"))["boxes"]
    print(f"page  : {args.page}")
    print(f"boxes : {args.boxes}（{len(boxes)} 框）")
    print(f"fp32  : {args.fp32}")

    strips = build_strips(img, boxes)
    fp32_res = run_ocr(args.fp32, strips, dictionary, "fp32")
    for r in (fp32_res[k] for k in sorted(fp32_res)):
        print(f"[{r['i']:2d}] {r['dir']} p={r['prob']:.2f}  {r['text']}")

    parity = None
    if args.int8:
        if os.path.exists(args.int8):
            print(f"\nint8  : {args.int8}")
            int8_res = run_ocr(args.int8, strips, dictionary, "int8")
            parity = report_parity(fp32_res, int8_res)
        else:
            print(f"\n（跳過 parity：找不到 int8 模型 {args.int8}）")

    # 讀出的字：沿用舊行為只留非空行（給 merge/translate 那幾支接手用）
    results = [fp32_res[k] for k in sorted(fp32_res) if fp32_res[k]["text"].strip()]
    json.dump(results, open(os.path.join(OUT, "ocr_results.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    if parity:
        json.dump(parity, open(os.path.join(OUT, "ocr_parity.json"), "w", encoding="utf-8"),
                  ensure_ascii=False, indent=2)
        print(f"parity → {OUT}/ocr_parity.json")

    font = find_font()
    print(f"\nfont: {font}")
    if font:
        try:
            from PIL import Image, ImageDraw, ImageFont
            pim = Image.fromarray(img).convert("RGB")
            dr = ImageDraw.Draw(pim)
            fnt = ImageFont.truetype(font, 22)
            for r in results:
                q = np.array(r["quad"], np.int32)
                dr.line([tuple(map(int, p)) for p in q] + [tuple(map(int, q[0]))], fill=(255, 0, 0), width=3)
                x0, y0 = int(q[:, 0].min()), int(q[:, 1].min())
                dr.text((x0, max(0, y0 - 24)), r["text"], fill=(0, 110, 255), font=fnt)
            pim.save(os.path.join(OUT, "ocr_overlay.png"))
            print(f"overlay → {OUT}/ocr_overlay.png")
        except Exception as e:
            print(f"render skipped: {e}")
    print(f"讀出 {len(results)} 行")


if __name__ == "__main__":
    main()
