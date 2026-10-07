#!/usr/bin/env python3
"""產生 Kotlin grouping parity 測試 fixture：偵測 002/012 的文字行 quads + mit_grouping 的期望區域 bbox，
寫成 engine 的 test 原始檔。讓 JVM 單元測試驗證 Kotlin Grouping 與已驗證的 mit_grouping.py 同輸出（§7）。"""
import os, sys
import numpy as np, cv2

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pipeline_parity as pp
from mit_grouping import Quadrilateral, merge_bboxes_text_region
from collections import Counter

from paths import RAW_DIR as RAW  # 集中路徑，見 paths.py
DST = os.path.join(os.path.dirname(HERE),
                   "inference-core/src/test/kotlin/li/joye/yakuyomi/engine/GroupingFixture.kt")
PAGES = ["002", "012"]


def regions_for(res, W, H):
    quads = []
    for r in res:
        fg = r.get("fg") or (0, 0, 0)
        bg = r.get("bg") or (255, 255, 255)
        quads.append(Quadrilateral(np.array(r["quad"], float), r["text"], 0.9,
                                   tuple(int(c) for c in fg), tuple(int(c) for c in bg)))
    out = []
    for txtlns, _, _ in merge_bboxes_text_region(quads, W, H):
        x0 = min(t.aabb.x for t in txtlns); y0 = min(t.aabb.y for t in txtlns)
        x1 = max(t.aabb.x + t.aabb.w for t in txtlns); y1 = max(t.aabb.y + t.aabb.h for t in txtlns)
        angle = float(np.degrees(np.mean([t.angle for t in txtlns])) - 90)
        if abs(angle) < 3:
            angle = 0.0
        out.append((int(x0), int(y0), int(x1), int(y1), round(angle)))
    return out


def main():
    det = __import__("onnxruntime").InferenceSession(
        f"{pp.MODELS}/comictextdetector.pt.onnx", providers=["CPUExecutionProvider"])
    ocr = __import__("onnxruntime").InferenceSession(
        f"{pp.MODELS}/ocr_48px_ctc.onnx", providers=["CPUExecutionProvider"])
    seg_rep = pp.import_seg_rep()(thresh=0.3)
    dic = [s[:-1] for s in open(pp.ALPHABET, encoding="utf-8").readlines()]

    blocks = []
    for name in PAGES:
        rgb = cv2.cvtColor(cv2.imread(os.path.join(RAW, f"{name}.jpg")), cv2.COLOR_BGR2RGB)
        H, W = rgb.shape[:2]
        quads = pp.detect(det, seg_rep, rgb)
        res = pp.ocr_all(ocr, dic, rgb, quads)
        regions = regions_for(res, W, H)

        lines_kt = []
        for r in res:
            pts = ", ".join(f"Pt({p[0]:.1f}f, {p[1]:.1f}f)" for p in r["quad"])
            lines_kt.append(f"            listOf({pts})")
        regs_kt = ",\n".join(f"            intArrayOf({x0}, {y0}, {x1}, {y1}, {ang})"
                             for (x0, y0, x1, y1, ang) in regions)
        blocks.append((name, ",\n".join(lines_kt), regs_kt, len(res), len(regions)))
        print(f"{name}: {len(res)} 行 → {len(regions)} 區（期望）")

    with open(DST, "w", encoding="utf-8") as f:
        f.write("package li.joye.yakuyomi.engine\n\n")
        f.write("// 自動產生（parity/emit_grouping_fixture.py）；勿手改。\n")
        f.write("// 偵測自 test/raw 的文字行 quads + mit_grouping.py 期望區域 bbox（§7 parity）。\n")
        f.write("internal object GroupingFixture {\n")
        f.write("    class Page(val name: String, val lines: List<List<Pt>>, val regions: List<IntArray>)\n\n")
        f.write("    val pages: List<Page> = listOf(\n")
        for i, (name, lines_kt, regs_kt, nl, nr) in enumerate(blocks):
            f.write(f'        Page(\n            "{name}",\n')
            f.write("            listOf(\n")
            f.write(",\n".join(f"        {ln}" for ln in lines_kt.split(",\n")) + "\n")
            f.write("            ),\n")
            f.write("            listOf(\n")
            f.write(",\n".join(f"        {rg}" for rg in regs_kt.split(",\n")) + "\n")
            f.write("            ),\n")
            f.write("        )" + ("," if i < len(blocks) - 1 else "") + "\n")
        f.write("    )\n}\n")
    print(f"→ {DST}")


if __name__ == "__main__":
    main()
