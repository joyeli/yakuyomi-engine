#!/usr/bin/env python3
"""nightread.py — 夜讀重繪（單頁）：DBNet 偵測 → 三分區遮罩 → 合成暗色閱讀頁。

這是什麼：把白底漫畫頁「重繪」成適合夜間閱讀的暗色頁（非濾鏡、非反相），
桌面原型（scratchpad dm_detect/dm_v4/dm_art/dm_final 系列）的正式收斂版。
三分區處理：
  留白（貼頁邊白＝頁邊距/格溝）→ 填深 BG + 格框描亮；
  氣泡內部                     → 深底 BG、文字筆畫畫亮 INK（原圖墨度當 alpha、
                                 邊緣天然抗鋸齒）、輪廓描亮；
  畫面（其餘）                 → D2：高光滾降 LUT（單調、保序）+ 自適應墨線增亮
                                 （只在局部背景偏暗處拉筆畫、cap<紙白 ⇒ 不反相）。

設計紅線（不可違反）：畫面絕不反相（只允許單調映射壓暗）；框白填深、字反白
（氣泡＝深底亮字）。最壞情況只是「某區沒變暗」，絕不出現負片畫面。

相對原型的三個修法（2026-08 校準，11 張測試頁量測定閾）：
  修法1  氣泡白元件「整頁面積上限」——併入白色連通元件前先查它在整頁的佔比
         （BUBBLE_COMP_MAX_FRAC）與「相對文字窗的局部性」（BUBBLE_LOCAL_K），
         超標＝不是氣泡（格內背景白）不併；改成整顆元件併入（不裁窗截斷），
         原型的 regrow 事後補救隨之移除。滅：demo01 黑塊、demo02 灰縫、
         ch34_014 方塊化。
  修法2  頁型判別降級——長直格框線（形態學開運算）太少＝無框/白背景頁
         （demo04/05 這型），命中則背景不填深、只重繪氣泡，畫面照 D2 壓暗。
         判準：H/V 線各 ≥ FRAME_MIN_EACH 且合計 ≥ FRAME_MIN_SUM（px/千像素；
         校準：有框頁最弱 demo02=1.19/5.77，無框頁最強 demo04=0.86/3.36）。
  修法3  留白遮罩格框感知——貼頁邊白元件逐顆分類：厚芯（距離變換 > CORE_R）
         大量「深入頁內」（距頁邊 > deep_px）＝格內白（如出血格的天空），
         或「深入頁內且包住線稿」（小洞內墨密度 ≥ DEEP_INK_RATIO）＝白包畫，
         皆改判畫面（D2 壓暗、不填深）。校準：真留白網絡 coreDeep 0.00–0.08，
         問題格（ch34_006 老人格 0.58 / ch34_010 第1格 0.40 / demo06 教堂 0.91）。
  修法4  純白背景填黑＋前景白描邊（貼紙式立體化）——「不承載調子的純白背景」
         （修法3 的格內白元件；frameless 頁則是 ≥WHITE_TH 的大面積背景白元件）
         不再只壓暗：背景 W 填 BG、格內容照 D2、dilate(F,STROKE_OBJ)∩W 畫白
         描邊把前景從黑底抬出來（與氣泡「深底亮字」同語彙）。figure/ground
         分離靠連通性：被墨線封閉的白（臉/衣服）不與 W 連通、天然保留。
         安全網（sticker_metrics，分離可疑 ⇒ 整顆退回上輪壓暗降級）：
         figFrac（前景佔比過低＝整格都被當背景，如 demo05 合格紙、demo02
         人群格）、thinFrac（W 細碎佔比過高＝鬍鬚型交界，前景白已連進背景，
         如 ch34_006 白鬍老人格）、textOn（語意證據＝文字以 tight bbox 真的
         壓在該元件上才算；eaten 超標或小元件須有證據才填黑——40px 窗版
         textCov 只留給漏併氣泡閘，防鄰格文字湊假證據吃掉前景白，
         如 ch34_010 披肩）。校準見 STICKER_* 常數行內註記。

偵測路徑＝export_dbnet_ncnn.build_model（m-i-t TextDetection @ .upstream-ref、
detect-20241225.ckpt，paths.fetch 自動下載+驗 sha256）torch eager 前向 ＋
m-i-t dbnet_utils.SegDetectorRepresenter 後處理 ＋ mit_grouping 兩階段區域合併
——與引擎同款前處理（長邊 1024、pad 右下到 256 倍數、/127.5-1）。

用法：
  python3 nightread.py <頁圖> [-o 輸出夾]      # 預設輸出 parity/out/nightread/
輸出（皆帶頁名前綴）：_regions.json / _seg.png / _bubble.png / _gutter.png /
  _final.png / _cmp.png（三聯：原圖｜成品｜遮罩視覺化）。
批次（多頁 + 白面積表）用 nightread_batch.py。
"""
import argparse
import importlib.util
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import export_dbnet_ncnn as ex                                    # noqa: E402
from mit_grouping import Quadrilateral, merge_bboxes_text_region  # noqa: E402
import paths                                                      # noqa: E402

OUT_DEFAULT = os.path.join(paths.OUT, "nightread")

# ── 設計常數（真機 A/B 要調的旋鈕全在這）─────────────────────────────
# 顏色/位準
BG = 16                # 深底（留白/氣泡底）
INK = 240              # 亮字
STROKE = 3             # 描邊半徑（格框/氣泡輪廓描亮 band）
# 畫面 D2 曲線
DIM_FLOOR, DIM_CEIL = 30, 140   # 壓暗值域（紙白 255 → 140）
ROLLOFF_G = 0.55                # 高光滾降 y = floor+(ceil-floor)*x^g（g<1 凹）
GLOW_STRENGTH = 55              # 自適應墨線增亮強度
GLOW_CAP = 112                  # 增亮上限（< DIM_CEIL ⇒ 線永遠比紙暗、不反相）
# 偵測/遮罩
SEG_TH = 0.12          # 筆畫遮罩二值化閾（引擎 Config.segThreshold 同款）
WHITE_TH = 235         # 「白」的灰階下限（氣泡白/留白共用同一份連通元件）
BUBBLE_PAD = 40        # 文字區 bbox 外擴的搜尋窗
WHITE_MEASURE_TH = 200 # 白面積統計閾（回報用，不進演算法）
GUTTER_MIN_AREA_FRAC = 0.0006   # 留白元件最小面積（整頁佔比）
# 修法1：氣泡白元件上限
BUBBLE_COMP_MAX_FRAC = 0.07     # 元件整頁佔比上限（6–8% 帶，取中偏上）
BUBBLE_LOCAL_K = 4.0            # 元件面積 ≤ K × 文字搜尋窗面積（局部性）
# 修法2：頁型判別（長直格框線，px/千像素）
FRAME_LINE_L_DIV = 5            # 線長 = min(W,H)//DIV（至少 60px）
FRAME_DARK_TH = 100             # 「框線暗」灰階上限
FRAME_MIN_EACH = 1.0            # H、V 各自下限
FRAME_MIN_SUM = 4.5             # H+V 合計下限
# 修法3：留白元件分類（gutter vs 格內白）
CORE_R = 26                     # 厚芯：距離變換 > CORE_R（真格溝半寬遠小於此）
DEEP_EDGE_FRAC = 0.07           # deep_px = max(64, 0.07*min(W,H))（頁邊距帶寬）
IN_PANEL_CORE_FRAC = 0.15       # 規則A：厚芯佔元件 ≥ 此
IN_PANEL_CORE_DEEP = 0.25       #        且厚芯深入頁內比例 ≥ 此 ⇒ 格內白
DEEP_INK_DEEP = 0.5             # 規則B：厚芯深入比例 ≥ 此
DEEP_INK_RATIO = 0.02           #        且小洞內墨/元件面積 ≥ 此 ⇒ 白包畫
HOLE_MAX_FRAC = 0.01            # 「小洞」上限（整頁佔比；大洞＝整格，不算包線稿）
INK_DARK_TH = 128               # 洞內「墨」灰階上限
# 修法4：純白背景填黑＋前景白描邊（貼紙式）
STROKE_OBJ_FRAC = 0.0035        # 描邊半徑（× min(W,H)，clamp 下二行）
STROKE_OBJ_MIN, STROKE_OBJ_MAX = 4, 7
STROKE_OBJ_V = 220              # 描邊亮度（略低於 INK＝與氣泡字位準區分）
STICKER_MIN_FRAC = 0.01         # frameless 背景白元件最小整頁佔比（大面積才算背景）
FIG_NOISE_AREA = 40             # F 小噪點面積門檻（px）：不描邊、整顆併入背景填深
FIG_NOISE_CLOSE = 11            # 噪點「在 W 內」判定：comp 閉運算核（覆蓋到的孤點才吞）
# 修法4 安全網（figure/ground 分離可疑 ⇒ 該元件退回上輪壓暗降級）
STICKER_FIG_MIN = 0.10          # 前景佔 bbox 比例下限（過低＝整格都被當背景）
STICKER_FIG_MAX = 0.85          # 上限（過高＝根本沒分出背景）
STICKER_THIN_R = 4              # W 細碎判定半徑（距離變換 ≤ R ＝細白絲）
STICKER_THIN_MAX = 0.45         # W 細碎佔比上限（整體細碎＝背景根本破碎）
STICKER_CHROMA_MAX = 6.0        # 元件平均彩度（max−min 通道）上限：淡彩水彩底
                                # 灰階會 ≥235 但不是「純白」，整顆不進候選（彩頁不毀）
                                # （校準：demo04 淡彩 2.5–15.4、demo05 11.7/18.3、
                                #   純黑白頁全 ≤2.0；demo04 最低那顆 2.5 另由 textCov 擋）
STICKER_EATEN_R = 5             # 「疑似被吃前景白」判定：細白（dist ≤ R）…
STICKER_EATEN_DENS = 0.35       # …且局部墨密度（blur 15×15，墨＝<WHITE_TH）≥ 此
                                # ＝密集筆畫縫隙白（白鬍/髮絲 vs 花窗速寫都長這樣）
STICKER_EATEN_MAX = 0.06        # eaten 佔 W 比例軟上限：超標＝可能有白色前景物連進
                                # 背景（ch34_006 白鬍老人格 0.073）…
STICKER_TEXT_BG_MIN = 0.10      # …除非文字確實壓在這片白上（textOn）≥ 此＝作者把它
                                # 當背景寫字的語意證據（demo06 教堂 0.167 / 金髮女孩格
                                # 0.228 手寫字直接落在背景白 ⇒ 放行；老人格 0.000、
                                # ch34_010 披肩 0.023 ⇒ 無證據）
# 修法5：閉合格背景擢升（貼框白元件 → sticker 候選）
FRAME_HUG_DILATE = 5            # 元件外擴後與格線遮罩的交集算「貼框」
FRAME_HUG_THICK = 3.0           # 格線名目厚度（px）：交集數 → 貼框長度的除數
FRAME_HUG_MIN = 0.25            # 貼框長度 / bbox 周長 下限（背景沿格框跑；前景白衣只點狀碰框）
FRAME_HUG_STRONG = 0.40         # 強貼框：背景證據足夠強 → 走放寬門（見 sticker_plan 兩級制）
PROMOTED_TEXTON_MAX = 0.30      # 強貼框仍拒的文字覆蓋上限：真有大量文字壓在上面＝旁白框，
                                # 填黑會把字變描邊糊 → 留灰（demo06 教堂 textOn 0.167 是可接受上界的參照）
# 批1（2026-08-26 使用者拍板「先大片白、複雜區塊第二批」）：
CORE_NECK_R = 12                # 寬域核心：開運算半徑（切斷臉/白衣連進背景的線稿缺口窄頸）
CORE_RECOVER_R = 9              # 核心確定後往墨線邊回收的測地半徑（貼線稿、免留白圈）
FAINT_OF_F_MAX = 0.62           # F 像素中淡色(>FAINT_G)佔比上限：群眾/建築淡速寫背景 → 推遲批2
FAINT_G = 160
# 偽泡（開口氣泡/字壓背景救回）：
PB_COV_MAX = 0.85               # 氣泡遮罩蓋率低於此的 text region 才啟動偽泡
PB_NECK_R = 7                   # 偽泡切頸（擋下巴縫/泡尾缺口；泡內行間白不受影響）
PB_GROW_FRAC = 0.35             # 生長距離上限＝max(text bbox 邊) × 此值
                                # ⚠️ 0.60 時 ch34_010 左下開口泡長進相連的外套白、把手塗黑（真凶
                                # 是偽泡非貼紙——驗屍 2026-08-26）；收緊後 demo02 大字框仍蓋滿
# 批1.5（2026-08-26 使用者兩案）：
# 手/外套漏填（ch34_010 左下案）：測地/直線比——背景從格框「直直就到」（比≈1）、
# 衣料/皮膚要繞過人物墨線障礙才到（比高）。人物殼 closing 版已證蓋不住寬開衣料白、廢棄。
GEO_RATIO_MAX = 1.6             # 測地距離 ≤ 直線距離×此 才視為背景
GEO_SLACK = 40                  # 加法餘裕（px）：近框處比值不穩定的緩衝
# 人頭一致化（demo02 案）：元件級指標已證不可分（feat 被鄰墨污染、ringDark 與正常頁衣料
# 重疊）→ 改構圖層「暗區地圖」：粗尺度上暗色主導的帶（群眾帶/已填格）內，小白元件一律
# 填深+內緣亮。主角臉防護＝面積上限＋暗區要求（臉大、通常也不在暗帶）。
HARMONIZE_ZONE_CELL = 16        # 暗區地圖降採樣尺度
HARMONIZE_ZONE_DARK = 0.45      # 粗胞暗(<60)佔比 ≥ 此 ⇒ 暗區
HARMONIZE_IN_ZONE = 0.6         # 亮島落在暗區內的比例下限
HARMONIZE_AREA_MAX = 0.004      # 亮島整頁佔比上限（群眾人頭尺度；主角臉更大 → 排除）
HARMONIZE_COLLAR_INK = 0.30     # 亮島外環細墨密度上限：鬍鬚/密集髮絲（ch34_006）高 → 排除；
                                # 空白人頭只有單條輪廓線、低 → 放行
# 留白填深的人物灰暈（ch34_010 左下案＝出血式無框特寫：外套白與頁白連續且輪廓開放，
# 像素層無界 → 唯一安全解＝gutter 填色避開大型人物墨結構周圍，人物旁留灰暈）：
AURA_MIN_INK_AREA = 2500        # 「大型人物墨」門檻（px）；格線/氣泡輪廓先排除不算
AURA_R = 30                     # 灰暈半徑：距人物墨此距離內的留白不填
AURA_FRAME_EXEMPT = 45          # 距格線此範圍內的留白豁免灰暈（正常格間留白照填；hard 模式用）
AURA_BORDER_EXEMPT = 60         # 距頁邊此範圍內的留白豁免灰暈（hard 模式用）
# ★ 拍板 hard（2026-09-05 使用者 A/B 目檢）：邊界 crisp、黑就是黑。glow 保留備查（見下），
# 它把安全妥協變成「夜景輪廓光」但會在人物衣料上疊一層原作沒有的漸層 ⇒ 風格添加，不採用。
AURA_MODE = os.environ.get("NIGHTREAD_AURA", "hard")  # hard=二值+豁免帶（定案）；glow=距離場漸層
AURA_GLOW_R0 = 8                # glow：距人物墨 ≤R0 全保留場景調
AURA_GLOW_R1 = 42               # glow：≥R1 全 BG；中間線性淡入（距離場＝天然跟隨輪廓、無鋸齒）
STICKER_TEXTON_PAD = 8          # textOn 的文字 bbox 外擴（語意證據要「真的壓在元件上」
                                # ⇒ 只容 bbox 抖動的小 pad；BUBBLE_PAD 40px 窗會把鄰格
                                # 文字掃進相鄰白元件湊假證據——ch34_010 披肩 pad40=0.207
                                # vs pad8=0.023、demo06 教堂 pad40=0.232 vs pad8=0.167）
STICKER_SMALL_AREA = 0.02       # 「小元件」整頁佔比門檻：小於此的白元件必須有 textOn
                                # 語意證據才可填黑（真正的格背景白都大：正當 ACCEPT 最小
                                # ch34_006 天空 0.035；ch34_010 披肩衣料 0.0105 ⇒ 退回。
                                # eaten 軟上限對平滑衣料白這型前景是盲區，面積補上）
STICKER_EATEN_HARD = 0.30       # eaten 硬上限：細碎過半＝分離無意義，一律退回
STICKER_TEXT_MAX = 0.55         # 文字窗（region bbox+BUBBLE_PAD）蓋住 W 的比例上限：
                                # 過高＝這顆白其實是漏併的氣泡，交還壓暗、不當背景
                                # （此閘保留 40px 窗：量的是「字＋周邊白」的氣泡構形；
                                #   換 tight 值會讓 demo02/03/04 的漏併氣泡掉下 0.55）
# 修法4 區域級保護（_sticker_protect：accept 後的第二道網，逐團不填不描、留 D2）
STICKER_NECK_R = 8              # 窄頸半徑：附屬白＝只能經寬 < 2R 縫隙抵達的 W（白鬍/
                                # 髮絲經筆畫縫隙連進背景就是這型；geodesic 重建切下）
STICKER_CORE_MIN = 0.03         # 開放背景核最小佔比（× W 面積）：erode 後殘核 ≥ 此
                                # 才當背景種子（白臉額頭的小殘核不算背景）
STICKER_PROTECT_EATEN_MIN = 0.002  # 附屬白連通團含 eaten ≥ max(100px, 此×W面積) 才保護
                                # （乾淨格的窄框縫也是附屬白、但無 eaten ⇒ 照填）
STICKER_PROTECT_DILATE = 6      # 保護團外擴（px）：蓋住鬍鬚邊緣的過渡帶

_model = None
_dbnet_utils = None


def _load_dbnet_utils():
    """m-i-t dbnet_utils 無相對 import，以檔案載入。"""
    p = os.path.join(ex.MIT, "manga_translator/detection/default_utils/dbnet_utils.py")
    spec = importlib.util.spec_from_file_location("mit_dbnet_utils", p)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["mit_dbnet_utils"] = mod
    spec.loader.exec_module(mod)
    return mod


def get_model():
    """DBNet（torch eager）＋後處理模組，模組級快取（批次只載一次）。"""
    global _model, _dbnet_utils
    if _model is None:
        ex.fetch_ckpt()                       # paths.fetch：缺檔下載 + sha256 驗證
        _model = ex.build_model()
        _dbnet_utils = _load_dbnet_utils()
    return _model, _dbnet_utils


# ── 偵測 ────────────────────────────────────────────────────────────

def detect(img_bgr):
    """DBNet 前向 + m-i-t 後處理 + mit_grouping 區域合併。

    回傳 (lines, regions, seg)：文字行 Quadrilateral、區域 dict（bbox/angle/lines）、
    seg 筆畫二值遮罩（bool、原圖解析度）。
    """
    import torch
    model, du = get_model()
    H, W = img_bgr.shape[:2]
    chw, inW, inH, ratio = ex.preprocess(img_bgr)      # 引擎同款前處理
    th_, tw_ = int(round(H * ratio)), int(round(W * ratio))
    with torch.no_grad():
        db, mask = model(torch.from_numpy(chw[None]))
    db = db.sigmoid().numpy()                          # m-i-t default.py：模型外 sigmoid
    mask = mask.numpy()[0, 0]

    rep = du.SegDetectorRepresenter(thresh=0.5, box_thresh=0.7, unclip_ratio=2.3)
    boxes, scores = rep({"shape": [(inH, inW)]}, db)
    boxes, scores = boxes[0], scores[0]
    lines = []
    if boxes.size:
        idx = boxes.reshape(boxes.shape[0], -1).sum(axis=1) > 0
        for pts, sc in zip(boxes[idx].astype(np.float64), np.asarray(scores)[idx]):
            q = pts / ratio                            # pad 在右下 ⇒ 除 ratio 即原圖座標
            q[:, 0] = np.clip(q[:, 0], 0, W - 1)
            q[:, 1] = np.clip(q[:, 1], 0, H - 1)
            if cv2.contourArea(q.astype(np.float32)) > 16:
                lines.append(Quadrilateral(q.astype(int), "", float(sc)))

    regions = []
    for txtlns, _, _ in merge_bboxes_text_region(list(lines), W, H):
        x0 = int(min(t.aabb.x for t in txtlns)); y0 = int(min(t.aabb.y for t in txtlns))
        x1 = int(max(t.aabb.x + t.aabb.w for t in txtlns))
        y1 = int(max(t.aabb.y + t.aabb.h for t in txtlns))
        ang = float(np.degrees(np.mean([t.angle for t in txtlns])) - 90)
        if abs(ang) < 3:
            ang = 0.0
        regions.append({
            "bbox": [x0, y0, x1, y1],
            "angle": round(ang, 1),
            "lines": [{"quad": t.pts.tolist(), "score": round(float(t.prob), 4)}
                      for t in txtlns],
        })

    # seg 筆畫遮罩：半解析 → canvas → 裁 pad → 原圖 → 閾值（對齊 seg_validate/引擎）
    m_canvas = cv2.resize(mask, (inW, inH), interpolation=cv2.INTER_LINEAR)
    m_full = cv2.resize(m_canvas[:th_, :tw_], (W, H), interpolation=cv2.INTER_LINEAR)
    seg = m_full > SEG_TH
    return lines, regions, seg


# ── 遮罩：白元件分類（修法2/3）＋氣泡（修法1）───────────────────────

def frame_line_mask(g):
    """長直格框線遮罩：暗像素對「長水平/垂直線」形態學開運算＝只留貼直的長線（格框）。
    回傳 (lh, lv) 兩個 0/1 uint8。修法2 的頁型判別與修法5 的貼框擢升共用。"""
    H, W = g.shape
    dark = (g < FRAME_DARK_TH).astype(np.uint8)
    L = max(60, min(W, H) // FRAME_LINE_L_DIV)
    lh = cv2.morphologyEx(dark, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (L, 1)))
    lv = cv2.morphologyEx(dark, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, L)))
    return lh, lv


def page_is_frameless(g):
    """修法2：長直格框線存在性。回傳 (frameless, h_px_per_k, v_px_per_k)。
    無框/白背景頁（demo04/05 型）兩向都近零。"""
    lh, lv = frame_line_mask(g)
    hk, vk = 1000.0 * lh.mean(), 1000.0 * lv.mean()
    frameless = not (min(hk, vk) >= FRAME_MIN_EACH and hk + vk >= FRAME_MIN_SUM)
    return frameless, hk, vk


def _hole_ink_ratio(comp_u8, g):
    """元件「小洞內墨」/元件面積：白元件包住的線稿量（修法3 規則B 的訊號）。

    填洞（1px 零邊框 + 從外 floodFill）→ 洞＝沒被外部填到的非元件像素；
    只計小洞（< HOLE_MAX_FRAC 頁面；大洞＝被留白環住的整格，不是包線稿）。
    """
    ff = np.pad(comp_u8, 1)
    m = np.zeros((ff.shape[0] + 2, ff.shape[1] + 2), np.uint8)
    cv2.floodFill(ff, m, (0, 0), 2)
    holes = (ff[1:-1, 1:-1] == 0).astype(np.uint8)     # 非元件且外部填不到＝洞
    hn, hlab, hstats, _ = cv2.connectedComponentsWithStats(holes, 8)
    ink = 0
    for j in range(1, hn):
        if hstats[j, cv2.CC_STAT_AREA] < HOLE_MAX_FRAC * g.size:
            ink += int((g[hlab == j] < INK_DARK_TH).sum())
    return ink / max(int(comp_u8.sum()), 1)


def classify_white_components(g):
    """整頁白（>=WHITE_TH）連通元件一次算完，供留白與氣泡共用。

    回傳 (lab, stats, gutter_ids, panel_ids)：
      gutter_ids＝判定為留白（頁邊距/格溝）的元件 → 填深；
      panel_ids ＝貼頁邊但屬「格內白」的元件（修法3）→ 當畫面壓暗、氣泡也不併。
    """
    H, W = g.shape
    white = (g >= WHITE_TH).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(white, 8)
    dist = cv2.distanceTransform(white, cv2.DIST_L2, 5)   # 白內距最近非白（元件間互不影響）
    deep_px = max(64, int(round(DEEP_EDGE_FRAC * min(W, H))))
    min_area = int(g.size * GUTTER_MIN_AREA_FRAC)

    gutter_ids, panel_ids = set(), set()
    for i in range(1, n):
        a = int(stats[i, cv2.CC_STAT_AREA])
        if a < min_area:
            continue
        x, y, cw, ch = (stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP],
                        stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT])
        if not (x <= 2 or y <= 2 or x + cw >= W - 2 or y + ch >= H - 2):
            continue                                    # 不貼頁邊 ⇒ 非留白候選
        comp = (lab == i)
        core = comp & (dist > CORE_R)                   # 厚芯：比格溝半寬還厚的部分
        core_frac = core.sum() / a
        if core.any():
            ys, xs = np.nonzero(core)
            edge_d = np.minimum(np.minimum(xs, W - 1 - xs), np.minimum(ys, H - 1 - ys))
            core_deep = float((edge_d > deep_px).mean())  # 厚芯深入頁內（非頁邊距帶）比例
        else:
            core_deep = 0.0
        # 修法3：規則A＝厚芯大量深入頁內（出血格天空）；規則B＝深入且包住線稿（白包畫）
        in_panel = (core_frac >= IN_PANEL_CORE_FRAC and core_deep >= IN_PANEL_CORE_DEEP)
        if not in_panel and core_deep >= DEEP_INK_DEEP:
            in_panel = _hole_ink_ratio(comp.astype(np.uint8), g) >= DEEP_INK_RATIO
        (panel_ids if in_panel else gutter_ids).add(i)
    return lab, stats, gutter_ids, panel_ids


def build_bubble_mask(g, regions, seg, lab, stats, excluded_ids):
    """氣泡內部遮罩（修法1）：每文字區 bbox+BUBBLE_PAD 窗內，找「貼著（外擴後）
    文字筆畫」的白色連通元件，通過守門則整顆併入（不裁窗 ⇒ 無截斷方塊，
    原型 regrow 補救移除）。守門（不併＝該區只保留筆畫，安全降級）：
      整頁佔比 ≤ BUBBLE_COMP_MAX_FRAC（格內背景白太大，不是氣泡）
      面積 ≤ BUBBLE_LOCAL_K × 搜尋窗（局部性：氣泡跟它的字同尺度）
      不在 excluded_ids（留白/格內白元件）
    """
    H, W = g.shape
    seg_u8 = seg.astype(np.uint8) * 255
    seg_dil = cv2.dilate(seg_u8, np.ones((9, 9), np.uint8))  # 筆畫外擴→碰得到氣泡白底
    bubble = np.zeros((H, W), bool)
    merged, rejected = set(), set()
    for r in regions:
        x0, y0, x1, y1 = r["bbox"]
        cx0, cy0 = max(0, x0 - BUBBLE_PAD), max(0, y0 - BUBBLE_PAD)
        cx1, cy1 = min(W, x1 + BUBBLE_PAD), min(H, y1 + BUBBLE_PAD)
        win_area = (cx1 - cx0) * (cy1 - cy0)
        lab_c = lab[cy0:cy1, cx0:cx1]
        touch = np.unique(lab_c[(seg_dil[cy0:cy1, cx0:cx1] > 0) & (lab_c > 0)])
        for i in touch:
            if i in merged:
                continue
            a = int(stats[i, cv2.CC_STAT_AREA])
            if (a > BUBBLE_COMP_MAX_FRAC * g.size or a > BUBBLE_LOCAL_K * win_area
                    or i in excluded_ids):
                rejected.add(int(i))
                continue
            bubble |= lab == i
            merged.add(int(i))
        bubble[y0:y1, x0:x1] |= seg[y0:y1, x0:x1]       # 區內筆畫本身一定算氣泡內容
    return bubble, merged, rejected


# ── 修法4：純白背景填黑＋前景白描邊（貼紙式）────────────────────────

def _comp_window(g, lab, stats, i, margin):
    """元件 bbox 外擴 margin 的工作窗：回傳 (x0,y0,x1,y1, sub_g, comp_bool)。"""
    H, W_ = g.shape
    x, y, cw, ch = (stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP],
                    stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT])
    x0, y0 = max(0, x - margin), max(0, y - margin)
    x1, y1 = min(W_, x + cw + margin), min(H, y + ch + margin)
    return x0, y0, x1, y1, g[y0:y1, x0:x1], (lab[y0:y1, x0:x1] == i)


def _sticker_eaten(sub, comp):
    """「疑似被吃前景白」遮罩：細白（dist ≤ EATEN_R）且局部墨密度 ≥ EATEN_DENS。

    白鬍/髮絲這類前景白透過筆畫縫隙連進背景 W 時，就長這個樣（密集筆畫的
    縫隙白）；背景密細節（教堂花窗速寫）也會命中——兩者統計上分不開，
    後續一律走區域級保護（不填黑、留 D2），見 _sticker_protect。
    """
    dist = cv2.distanceTransform(comp.astype(np.uint8), cv2.DIST_L2, 5)
    dens = cv2.blur((sub < WHITE_TH).astype(np.float32), (15, 15))
    return (comp & (dist <= STICKER_EATEN_R) & (dens >= STICKER_EATEN_DENS))


def _sticker_protect(eaten, comp):
    """區域級保護遮罩＝「窄頸附屬白 ∧ 含 eaten」（真 figure/ground 分離）。

    開放背景核＝erode(W, NECK_R) 後的大殘核（≥ CORE_MIN × W；白臉額頭殘核小、
    不算背景）；由核作 3×3 遮罩膨脹的 geodesic 重建（小核不會跳過 ≥1px 墨線）
    → 重建到不了的 W ＝只能經寬 < 2×NECK_R 窄縫抵達的「物件附屬白」（白鬍臉
    整片，含不算 eaten 的寬白叢——之前純形態學聚團蓋不住的就是這塊）。
    附屬白連通區含 eaten 夠多才保護（乾淨格的窄框縫無 eaten ⇒ 照填）。
    半解析度重建（數百次 3×3 迭代，省 4 倍）。
    """
    area = max(int(comp.sum()), 1)
    h, w = comp.shape
    hh, hw = max(1, h // 2), max(1, w // 2)
    comp_h = cv2.resize(comp.astype(np.uint8), (hw, hh), interpolation=cv2.INTER_NEAREST)
    eaten_h = cv2.resize(eaten.astype(np.uint8), (hw, hh), interpolation=cv2.INTER_NEAREST)
    r = max(2, STICKER_NECK_R // 2)
    ke = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1,) * 2)
    core = cv2.erode(comp_h, ke)
    nn, ll, ss, _ = cv2.connectedComponentsWithStats(core, 8)
    area_h = max(int(comp_h.sum()), 1)
    seed = np.zeros_like(comp_h)
    for j in range(1, nn):
        if ss[j, cv2.CC_STAT_AREA] >= STICKER_CORE_MIN * area_h:
            seed[ll == j] = 1
    if not seed.any():                                  # 沒有開放背景核＝全窄碎
        return np.ones((h, w), bool)                    # 全保護（極端保守）
    k3 = np.ones((3, 3), np.uint8)
    recon, prev = seed, -1
    for _ in range(4000):
        recon = cv2.dilate(recon, k3) & comp_h
        cnt = int(cv2.countNonZero(recon))
        if cnt == prev:
            break
        prev = cnt
    appendage = (comp_h > 0) & (recon == 0)
    nn, ll, _, _ = cv2.connectedComponentsWithStats(appendage.astype(np.uint8), 8)
    need = max(100, STICKER_PROTECT_EATEN_MIN * area) / 4.0   # 半解析度像素數 ÷4
    protect_h = np.zeros_like(comp_h)
    for j in range(1, nn):
        blob = ll == j
        if int(eaten_h[blob].sum()) >= need:
            protect_h[blob] = 1
    if not protect_h.any():
        return np.zeros((h, w), bool)
    protect = cv2.resize(protect_h, (w, h), interpolation=cv2.INTER_NEAREST)
    kd = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * STICKER_PROTECT_DILATE + 1,) * 2)
    return cv2.dilate(protect, kd) > 0


def sticker_metrics(g, img_bgr, lab, stats, i, text_rects, text_rects_on):
    """單顆白背景元件的 figure/ground 診斷（修法4 安全網用）。

    在元件 bbox（外擴 8px）內量：
      figFrac ＝前景佔比（暗像素 ∪ 被 W 封閉的白＝臉/衣服）；
      thinFrac＝W 細碎佔比（距離變換 ≤ STICKER_THIN_R；整體破碎會爆高）；
      chroma  ＝W 平均彩度（max−min 通道）：淡彩水彩底非「純白」的鐵證；
      eaten   ＝疑似被吃前景白佔 W 比例（_sticker_eaten）；
      protect ＝區域級保護區佔 W 比例（審計/報表用，實際遮罩 paint 時重算）；
      textCov ＝文字窗（bbox+BUBBLE_PAD）蓋住 W 的比例（過高＝漏併的氣泡）；
      textOn  ＝tight 文字 bbox（+STICKER_TEXTON_PAD）∩ W 的比例＝文字「真的
                壓在這個白元件上」的語意證據（TEXT_BG_MIN 放行只認這個；40px
                窗會把鄰格文字掃進相鄰元件湊假證據，ch34_010 披肩案）；
      rough   ＝周長²/(4π·面積)（圓=1；輪廓破碎度，審計參考）。
    """
    H, W_ = g.shape
    x0, y0, x1, y1, sub, comp = _comp_window(g, lab, stats, i, 8)
    comp_u8 = comp.astype(np.uint8)
    area = max(int(comp.sum()), 1)

    ff = np.pad(comp_u8, 1)                             # 零邊框 → 外部一定連通
    mk = np.zeros((ff.shape[0] + 2, ff.shape[1] + 2), np.uint8)
    cv2.floodFill(ff, mk, (0, 0), 2)
    enclosed = ff[1:-1, 1:-1] == 0                      # 非 W 且外部到不了＝被 W 封閉
    fig = (sub < WHITE_TH) | enclosed
    fig_frac = float(fig.mean())

    dist = cv2.distanceTransform(comp_u8, cv2.DIST_L2, 5)
    thin_frac = float(((dist <= STICKER_THIN_R) & comp).sum() / area)

    sub_c = img_bgr[y0:y1, x0:x1].astype(np.int16)
    chroma = float((sub_c.max(axis=2) - sub_c.min(axis=2))[comp].mean())

    eaten = _sticker_eaten(sub, comp)
    eaten_frac = float(eaten.sum() / area)
    protect_frac = float((_sticker_protect(eaten, comp) & comp).sum() / area)

    tmask = np.zeros((H, W_), np.uint8)
    for rx0, ry0, rx1, ry1 in text_rects:
        cv2.rectangle(tmask, (rx0, ry0), (rx1, ry1), 1, -1)
    text_cov = float(tmask[y0:y1, x0:x1][comp].mean()) if area else 0.0

    tmask_on = np.zeros((H, W_), np.uint8)
    for rx0, ry0, rx1, ry1 in text_rects_on:
        cv2.rectangle(tmask_on, (rx0, ry0), (rx1, ry1), 1, -1)
    text_on = float(tmask_on[y0:y1, x0:x1][comp].mean()) if area else 0.0

    cnts, _ = cv2.findContours(comp_u8, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    perim = float(sum(cv2.arcLength(c, True) for c in cnts))
    rough = perim * perim / (4.0 * np.pi * area)

    bx, by = int(stats[i, cv2.CC_STAT_LEFT]), int(stats[i, cv2.CC_STAT_TOP])
    return {"comp": int(i),
            "bbox": [bx, by, bx + int(stats[i, cv2.CC_STAT_WIDTH]),
                     by + int(stats[i, cv2.CC_STAT_HEIGHT])],
            "areaFrac": round(area / g.size, 4), "figFrac": round(fig_frac, 3),
            "thinFrac": round(thin_frac, 3), "chroma": round(chroma, 1),
            "eatenFrac": round(eaten_frac, 4), "protectFrac": round(protect_frac, 4),
            "textCov": round(text_cov, 3), "textOn": round(text_on, 3),
            "rough": round(rough, 1)}


def sticker_plan(g, img_bgr, lab, stats, gutter_ids, panel_ids, frameless, regions):
    """挑修法4 目標元件並過安全網。回傳 (accept_ids, audit)。

    目標＝「不承載調子的純白背景」：有框頁＝修法3 的格內白（panel_ids）；
    frameless 頁＝貼頁邊白元件（修法3 只在乎有框頁的 gutter/panel 之分，這裡
    兩類都收）中整頁佔比 ≥ STICKER_MIN_FRAC 的大面積背景。
    安全網（任一不過＝不進 accept ⇒ 該元件維持上輪行為：有框頁 panelwhite＝
    D2 壓暗；frameless＝背景保留，絕不毀畫面）：
      chroma  > CHROMA_MAX ＝淡彩/彩頁底（灰階 ≥235 但非純白）→ 不動；
      textCov > TEXT_MAX   ＝其實是漏併的氣泡（40px 窗量氣泡構形）；
      figFrac 出界 / thinFrac 過高 ＝ 沒分出前景 或 背景本身破碎；
      eaten   > EATEN_HARD ＝細碎過半、分離無意義，一律退回；
      eaten   > EATEN_MAX 且 textOn < TEXT_BG_MIN ＝疑有前景白連進背景、又無
              「作者把它當背景寫字」的語意證據（ch34_006 白鬍老人格）→ 退回；
              textOn ≥ TEXT_BG_MIN 放行（demo06 教堂速寫底 0.167）；
      areaFrac < SMALL_AREA 且 textOn < TEXT_BG_MIN ＝小元件又無文字語意證據
              → 退回（真格背景白都大；平滑的前景衣料白 eaten 量不到，
              ch34_010 右下格披肩 0.0105/textOn 0.023 走這條退回）。
    語意證據一律用 textOn（tight bbox+TEXTON_PAD）：文字要「真的壓在這個白
    元件上」才算數；textCov 的 40px 窗會把鄰格文字掃進相鄰元件湊假證據。
    eaten 沒爆但局部聚團（白鬍臉這型）＝ paint_sticker 的區域級保護處理
    （該團塊不填黑、留 D2），不整顆退回——見 _sticker_protect 常數註記。
    """
    H, W_ = g.shape
    def _rects(pad):
        return [(max(0, r["bbox"][0] - pad), max(0, r["bbox"][1] - pad),
                 min(W_ - 1, r["bbox"][2] + pad), min(H - 1, r["bbox"][3] + pad))
                for r in regions]
    text_rects = _rects(BUBBLE_PAD)            # textCov：漏併氣泡閘（窗語意）
    text_rects_on = _rects(STICKER_TEXTON_PAD)  # textOn：語意證據（壓在元件上）
    if frameless:
        cand = {i for i in (gutter_ids | panel_ids)
                if stats[i, cv2.CC_STAT_AREA] >= STICKER_MIN_FRAC * g.size}
        hug = {}
    else:
        cand = set(panel_ids)
        # 修法5：閉合格背景擢升——未列管（非 gutter 非 panel）的大白元件，若「貼格線長度 /
        # bbox 周長」夠高＝背景沿格框跑（閉合格內的天空/空白背景被格線封住、永遠進不了
        # 修法3 的 gutter/panel 分類），擢升進候選、照走下面同一套安全網。前景白（臉/白衣）
        # 是獨立元件、只點狀碰框 → hug 低、天然不擢升。ch34_014 型（22% 頁面積）的主修。
        lh, lv = frame_line_mask(g)
        frame = (lh | lv).astype(np.uint8)
        kd = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (FRAME_HUG_DILATE * 2 + 1,) * 2)
        hug = {}
        min_area = GUTTER_MIN_AREA_FRAC * g.size
        listed = gutter_ids | panel_ids
        for i in range(1, stats.shape[0]):
            if i in listed or stats[i, cv2.CC_STAT_AREA] < min_area:
                continue
            x, y, w_, h_ = (stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP],
                            stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT])
            pad = FRAME_HUG_DILATE + 1
            x0, y0 = max(0, x - pad), max(0, y - pad)
            x1, y1 = min(g.shape[1], x + w_ + pad), min(g.shape[0], y + h_ + pad)
            comp = (lab[y0:y1, x0:x1] == i).astype(np.uint8)
            contact = int((cv2.dilate(comp, kd) & frame[y0:y1, x0:x1]).sum())
            hug_len = contact / FRAME_HUG_THICK
            frac = hug_len / max(1.0, 2.0 * (w_ + h_))
            hug[i] = round(float(frac), 3)
            if frac >= FRAME_HUG_MIN:
                cand.add(i)
    accept, audit, promoted = set(), [], set()
    for i in sorted(cand):
        met = sticker_metrics(g, img_bgr, lab, stats, i, text_rects, text_rects_on)
        if i in hug:
            met["frameHug"] = hug[i]
        if hug.get(i, 0.0) >= FRAME_HUG_STRONG:
            # 修法5 兩級制——強貼框（背景證據極強）走放寬門：
            # · 跳過 textCov（pad40 窗對大背景是鄰泡污染；真文字壓上用 textOn 擋）
            # · 跳過 eaten 中段門與小面積門（窄頸類危險交給批1 核心填色的幾何保護）
            # · 保留 fig/thin/chroma/eatenHARD 四道硬底線
            # · 批1 淡色門：F 像素中淡色佔比高＝群眾/建築淡速寫背景，填黑會變漂浮碎片
            #   → 推遲批2（ch34_014 底格群眾實測 faintOfF 高、中排乾淨背景低）
            x_, y_, w2, h2 = (stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP],
                              stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT])
            sub_g = g[y_:y_ + h2, x_:x_ + w2]
            fpx = sub_g[sub_g < WHITE_TH]
            met["faintOfF"] = round(float((fpx > FAINT_G).mean()), 3) if fpx.size else 0.0
            ok = (STICKER_FIG_MIN <= met["figFrac"] <= STICKER_FIG_MAX
                  and met["thinFrac"] <= STICKER_THIN_MAX
                  and met["chroma"] <= STICKER_CHROMA_MAX
                  and met["eatenFrac"] <= STICKER_EATEN_HARD
                  and met["textOn"] <= PROMOTED_TEXTON_MAX
                  and met["faintOfF"] <= FAINT_OF_F_MAX)
            if ok:
                promoted.add(i)
        else:
            ok = (STICKER_FIG_MIN <= met["figFrac"] <= STICKER_FIG_MAX
                  and met["thinFrac"] <= STICKER_THIN_MAX
                  and met["chroma"] <= STICKER_CHROMA_MAX
                  and met["textCov"] <= STICKER_TEXT_MAX
                  and met["eatenFrac"] <= STICKER_EATEN_HARD
                  and (met["eatenFrac"] <= STICKER_EATEN_MAX
                       or met["textOn"] >= STICKER_TEXT_BG_MIN)
                  and (met["areaFrac"] >= STICKER_SMALL_AREA
                       or met["textOn"] >= STICKER_TEXT_BG_MIN))
        met["accept"] = bool(ok)
        audit.append(met)
        if ok:
            accept.add(i)
    return accept, audit, promoted


def geodesic_grow(seed, within, iters, step=5):
    """測地生長：seed 在 within 內反覆 3×3 膨脹 iters 次（≈ 距離 px）。step 批次化省時。"""
    k = np.ones((3, 3), np.uint8)
    cur = (seed & within).astype(np.uint8)
    w8 = within.astype(np.uint8)
    done = 0
    while done < iters:
        n = min(step, iters - done)
        cur = cv2.dilate(cur, k, iterations=n) & w8
        done += n
    return cur > 0


def broad_core_fill(comp, seeds, neck_r=CORE_NECK_R, recover_r=CORE_RECOVER_R):
    """批1 核心填色：comp（白元件）先開運算切窄頸 → 只留寬闊區；由 seeds 所在的
    寬闊連通塊出發（臉/白衣經細縫連入背景 → 在頸口被切開、到不了）；最後往墨線邊
    做小半徑測地回收（貼合線稿、不留白圈）。回傳實際要填的遮罩。"""
    ku = comp.astype(np.uint8)
    ko = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * neck_r + 1,) * 2)
    core = cv2.morphologyEx(ku, cv2.MORPH_OPEN, ko)
    n, lb = cv2.connectedComponents(core, 8)
    ids = np.unique(lb[seeds & (core > 0)])
    ids = ids[ids > 0]
    if ids.size == 0:
        return np.zeros_like(comp, bool)
    filled = np.isin(lb, ids)
    return geodesic_grow(filled, comp, recover_r, step=3)


def paint_sticker(out, g, lab, stats, accept, bubble, core_ids=(), frame=None):
    """修法4 合成：W 填深、前景描白邊（dilate(F, r) ∩ W）、W 內孤立小噪點吞掉、
    eaten 聚團區域級保護（不填黑、原樣留 D2）。

    F＝bbox 內非白（< WHITE_TH）且非氣泡的內容（墨線/調子/SFX）；被墨線封閉
    的白（臉/衣服）不與 W 連通、照 D2 保留，其輪廓墨線屬 F ⇒ 描邊自然沿輪廓。
    小噪點（面積 < FIG_NOISE_AREA 且被 comp 閉運算覆蓋＝孤懸 W 中）不描邊、
    直接併入背景填深；氣泡輪廓由 paint_bubbles 自己描，不在此重複。
    保護區（_sticker_protect：白鬍臉/密集髮絲/花窗速寫這型）整團不填不描，
    維持 D2 壓暗 ⇒ 前景白物件連進背景也吃不掉（ch34_006 白鬍老人格）。
    """
    H, W_ = g.shape
    r = int(np.clip(round(STROKE_OBJ_FRAC * min(H, W_)), STROKE_OBJ_MIN, STROKE_OBJ_MAX))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    kc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (FIG_NOISE_CLOSE,) * 2)
    for i in sorted(accept):
        x0, y0, x1, y1, sub, comp = _comp_window(g, lab, stats, i, r + 2)
        if i in core_ids and frame is not None:
            # 批1 核心填色（擢升元件）：只填「從格框種子出發、不擠過窄頸」的寬闊區。
            # 臉/白衣即使因線稿缺口與背景同元件，也在頸口被切斷 ⇒ 幾何保護、非門檻保護。
            fr = frame[y0:y1, x0:x1] > 0
            kd = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (FRAME_HUG_DILATE * 2 + 1,) * 2)
            seeds = (cv2.dilate(fr.astype(np.uint8), kd) > 0) & comp
            fill = broad_core_fill(comp, seeds)
            if fill.any():
                # 批1.5 測地比刪填：geo＝從格框種子在白域內走的距離；euc＝到格線的直線距離。
                # 背景 geo≈euc；衣料/皮膚要繞過人物墨線/氣泡障礙才到 → geo≫euc → 刪。
                # （人物殼 closing 版蓋不住寬開的外套白、已廢棄。）
                geo = np.full(comp.shape, np.float32(1e9))
                cur = (seeds & comp)
                k3 = np.ones((3, 3), np.uint8)
                d, STEP = 0, 6
                geo[cur] = 0
                while cur.any() and d < 4000:
                    grown = (cv2.dilate(cur.astype(np.uint8), k3, iterations=STEP) > 0) & comp
                    new_px = grown & (geo == 1e9)
                    d += STEP
                    if not new_px.any():
                        break
                    geo[new_px] = d
                    cur = grown
                fr8 = (~fr).astype(np.uint8)
                euc = cv2.distanceTransform(fr8, cv2.DIST_L2, 3)
                fill = fill & (geo <= GEO_RATIO_MAX * euc + GEO_SLACK)
            if not fill.any():
                continue
        else:
            fill = comp
        f_raw = ((sub < WHITE_TH) & ~bubble[y0:y1, x0:x1]).astype(np.uint8)
        nn, ll, ss, _ = cv2.connectedComponentsWithStats(f_raw, 8)
        keep = np.zeros(nn, bool)
        if nn > 1:
            keep[1:] = ss[1:, cv2.CC_STAT_AREA] >= FIG_NOISE_AREA
        f_main = keep[ll]                               # 清完噪點的前景（描邊來源）
        fill_closed = cv2.morphologyEx(fill.astype(np.uint8), cv2.MORPH_CLOSE, kc) > 0
        noise = (ll > 0) & ~keep[ll] & fill_closed      # 孤懸填色區內的小噪點 → 併入背景
        protect = _sticker_protect(_sticker_eaten(sub, comp), comp)
        band = (cv2.dilate(f_main.astype(np.uint8), k) > 0) & fill & ~protect
        o = out[y0:y1, x0:x1]
        o[(fill | noise) & ~protect] = BG
        o[band] = STROKE_OBJ_V
    return out


# ── 合成（畫面 D2 / 留白 / 氣泡）────────────────────────────────────

def lut_rolloff(floor=DIM_FLOOR, ceil=DIM_CEIL, gpow=ROLLOFF_G):
    """高光滾降 LUT：y = floor + (ceil-floor)*x^g。嚴格單調 ⇒ 保序、零負片感；
    亮部（紙白）壓得重、中暗部（網點/陰影）對比留得比線性多。"""
    x = np.arange(256, dtype=np.float32) / 255.0
    return np.clip(floor + (ceil - floor) * np.power(x, gpow), 0, 255).astype(np.uint8)


# ── 畫面曲線變體（2026-08-25 起：「只換白、不抬黑」實驗，NIGHTREAD_CURVE 選）──
# 使用者對 D2 的回饋＝「說不出的怪」：floor=30 全域抬黑 + g=0.55 凹曲線把暗部抬得特別兇
# （原墨 26 → 61），整頁發灰=「濁」。以下變體共同原則：黑保持黑（或近黑）、白仍壓 140，
# 動態範圍 110 → ~140。全部嚴格單調（保序鐵則不動）。
KNEE_X = 64        # knee 變體：此值以下完全保真（墨線原樣）
KNEE_END_SLOPE = 0.15  # 尾端斜率（紙白附近壓平）

def lut_linear(floor=0, ceil=DIM_CEIL):
    """全線性：y = floor + (ceil-floor)·x/255。黑→floor、白→ceil，相對關係完全保留。"""
    x = np.arange(256, dtype=np.float32) / 255.0
    return np.clip(floor + (ceil - floor) * x, 0, 255).astype(np.uint8)


def lut_knee(knee=KNEE_X, ceil=DIM_CEIL, end_slope=KNEE_END_SLOPE):
    """暗部保真 + 高光滾降：x ≤ knee 恆等（墨線一階不動＝對比最大化）；knee 以上
    monotone cubic Hermite (knee,knee,斜率1) → (255,ceil,斜率 end_slope)。
    單調性（Fritsch–Carlson）：割線 (ceil-knee)/(255-knee)=0.398，兩端斜率 1、0.15
    皆 ≤ 3×割線 ⇒ 全段單調。網點是抖點非平滑漸層，中段壓縮不致 banding。"""
    x = np.arange(256, dtype=np.float32)
    y = x.copy()
    t = np.clip((x - knee) / (255.0 - knee), 0.0, 1.0)
    h00 = 2 * t**3 - 3 * t**2 + 1
    h10 = t**3 - 2 * t**2 + t
    h01 = -2 * t**3 + 3 * t**2
    h11 = t**3 - t**2
    span = 255.0 - knee
    hy = h00 * knee + h10 * span * 1.0 + h01 * ceil + h11 * span * end_slope
    y = np.where(x > knee, hy, y)
    return np.clip(y, 0, 255).astype(np.uint8)


SCENE_CURVES = {
    "d2":   lambda: lut_rolloff(),
    "lin":  lambda: lut_linear(0),
    "lin8": lambda: lut_linear(8),   # 極暗保護：OLED 黑碎顧慮的最小抬升（8 遠小於舊 30）
    "knee": lambda: lut_knee(),
}
SCENE_CURVE = os.environ.get("NIGHTREAD_CURVE", "lin8")  # 2026-08-25 使用者目檢拍板 lin 系（黑實、對比大）


def lut_scene():
    return SCENE_CURVES[SCENE_CURVE]()


def ink_line_mask(g, seg=None, bh_ksize=7, bh_gain=45.0, dark_lo=40, dark_hi=185):
    """軟性墨線遮罩 0..1：blackhat（細暗線構）×暗度權重 ∪ DBNet seg。
    實心黑塊內部為 0（只認邊緣細線 ⇒ 增亮不掉大塊黑的對比）。"""
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (bh_ksize, bh_ksize))
    bh = cv2.morphologyEx(g, cv2.MORPH_BLACKHAT, k).astype(np.float32)
    soft = np.clip(bh / bh_gain, 0.0, 1.0)
    soft *= np.clip((dark_hi - g.astype(np.float32)) / (dark_hi - dark_lo), 0.0, 1.0)
    if seg is not None:
        soft = np.maximum(soft, seg.astype(np.float32))
    return soft


def ink_glow(dimmed, ink_soft, strength=GLOW_STRENGTH, cap=GLOW_CAP, bg_thr=95, bg_sigma=8.0):
    """自適應墨線增亮：只在「局部背景偏暗」處把筆畫往亮拉，夾在 cap 之下
    （cap < 紙白位準 DIM_CEIL ⇒ 線永遠比紙暗、不反相）。"""
    out = dimmed.astype(np.float32)
    gain = np.float32(strength) * ink_soft
    bg = cv2.GaussianBlur(dimmed, (0, 0), bg_sigma).astype(np.float32)
    gain *= np.clip((bg_thr - bg) / bg_thr, 0.0, 1.0)
    lifted = np.minimum(out + gain, np.maximum(out, np.float32(cap)))
    return np.clip(lifted, 0, 255).astype(np.uint8)


def scene_final(g, seg):
    """畫面區最終處理：場景曲線 LUT（NIGHTREAD_CURVE 選、預設 d2）+ 自適應墨線增亮。"""
    return ink_glow(lut_scene()[g], ink_line_mask(g, seg))


def ink_alpha(g, gain):
    """原圖墨度（1-亮度）×gain 夾 [0,1]：把墨線「轉亮」時的 alpha（邊緣抗鋸齒）。"""
    return np.clip((1.0 - g.astype(np.float32) / 255.0) * gain, 0.0, 1.0)


def paint_gutter(out, g, gutter, frame=None, bubble=None):
    """留白填深 + 格框描亮 + 人物灰暈。

    灰暈（批1.5，ch34_010 左下案）：出血式無框特寫的人物白衣與頁白連續且輪廓開放 →
    修法3 正確判 gutter、但整顆塗掉會吞掉衣料/手。像素層無界 ⇒ 安全解＝**避開大型
    人物墨結構周圍 AURA_R 的白不填**（格線/氣泡輪廓排除不算人物）。代價＝人物旁一圈
    灰暈（失敗方向＝不夠暗，合紅線）；封閉輪廓的一般留白離人物墨遠、不受影響。"""
    fill = gutter.copy()
    if frame is not None:
        ink = (g < WHITE_TH).astype(np.uint8)
        k7 = np.ones((15, 15), np.uint8)
        excl = cv2.dilate(frame.astype(np.uint8), k7) > 0
        if bubble is not None:
            excl |= cv2.dilate(bubble.astype(np.uint8), k7) > 0
        ink[excl] = 0
        n, lb, st, _ = cv2.connectedComponentsWithStats(ink, 8)
        big = np.zeros(n, bool)
        if n > 1:
            big[1:] = st[1:, cv2.CC_STAT_AREA] >= AURA_MIN_INK_AREA
        figm = big[lb]
        if figm.any() and AURA_MODE == "glow":
            # 發光式灰暈：距離場漸層——人物輪廓旁保留場景調、隨距離淡入 BG。距離場天然
            # 跟隨輪廓（等距線＝輪廓偏移）⇒ 無鋸齒；不需豁免帶（b18 的代價在漸層下只剩
            # 一圈柔光）。fill 本身不縮，改在最後把 gutter 區的值做 lerp。
            dist = cv2.distanceTransform((~figm).astype(np.uint8), cv2.DIST_L2, 5)
            alpha = np.clip((dist - AURA_GLOW_R0) / float(AURA_GLOW_R1 - AURA_GLOW_R0), 0.0, 1.0)
            scene_g = out.copy()                     # 進來時＝場景曲線後的值
            out_f = out
            m = fill
            out_f[m] = scene_g[m] * (1.0 - alpha[m]) + np.float32(BG) * alpha[m]
            k = np.ones((STROKE * 2 + 1,) * 2, np.uint8)
            band = (cv2.dilate(fill.astype(np.uint8), k) > 0) & ~fill
            a2 = ink_alpha(g, 1.6)
            out_f[band] = np.maximum(out_f[band], BG + a2[band] * (INK - BG))
            return out_f
        if figm.any():
            ka = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (AURA_R * 2 + 1,) * 2)
            aura = cv2.dilate(figm.astype(np.uint8), ka) > 0
            # 灰暈只管「深入無框區」：距格線/頁邊近的留白＝正常格間留白，豁免（否則
            # 出血人物碰個邊就吃掉整條 margin，全站 +5~10pt 亮區，b18 實測教訓）
            fd = cv2.distanceTransform((frame == 0).astype(np.uint8), cv2.DIST_L2, 3)
            H2, W2 = g.shape
            yy, xx = np.mgrid[0:H2, 0:W2]
            bd = np.minimum(np.minimum(yy, H2 - 1 - yy), np.minimum(xx, W2 - 1 - xx))
            aura &= (fd > AURA_FRAME_EXEMPT) & (bd > AURA_BORDER_EXEMPT)
            fill = fill & ~aura
    out[fill] = BG
    k = np.ones((STROKE * 2 + 1,) * 2, np.uint8)
    band = (cv2.dilate(fill.astype(np.uint8), k) > 0) & ~fill
    a = ink_alpha(g, 1.6)
    out[band] = np.maximum(out[band], BG + a[band] * (INK - BG))
    return out


def paint_bubbles(out, g, bubble, seg, text_pad=2):
    """氣泡重繪：內部填深、文字畫亮（墨度 alpha）、輪廓描亮。"""
    out[bubble] = BG
    kt = np.ones((text_pad * 2 + 1,) * 2, np.uint8)
    text = (cv2.dilate((seg & bubble).astype(np.uint8), kt) > 0) & bubble
    out[text] = np.maximum(out[text], BG + ink_alpha(g, 1.4)[text] * (INK - BG))
    ko = np.ones((STROKE * 2 + 1,) * 2, np.uint8)
    band = (cv2.dilate(bubble.astype(np.uint8), ko) > 0) & ~bubble
    out[band] = np.maximum(out[band], BG + ink_alpha(g, 1.6)[band] * (INK - BG))
    return out


def build_pseudo_bubbles(g, regions, bubble):
    """偽泡（demo02 型救回）：氣泡遮罩蓋率 < PB_COV_MAX 的 text region（開口氣泡＝泡內白
    流出去與留白/背景連通而被氣泡遮罩拒收；或字直接寫在背景/留白上），從「字底下的白」
    出發在輕切頸後的白域內測地生長（距離上限＝bbox 尺度）→ 得到泡狀填色遮罩，交
    paint_bubbles 同工法（填深＋亮字＋描邊）。輕切頸擋下巴縫這類小缺口，免得長進臉。"""
    H, W_ = g.shape
    white = (g >= WHITE_TH)
    ko = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * PB_NECK_R + 1,) * 2)
    w_cut = cv2.morphologyEx(white.astype(np.uint8), cv2.MORPH_OPEN, ko)
    w_cut = geodesic_grow(w_cut > 0, white, PB_NECK_R, step=3)   # 回收切掉的邊緣
    pb = np.zeros((H, W_), bool)
    for r_ in regions:
        x0, y0, x1, y1 = r_["bbox"]
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(W_, x1), min(H, y1)
        if x1 <= x0 or y1 <= y0:
            continue
        win = bubble[y0:y1, x0:x1]
        if win.size == 0 or win.mean() >= PB_COV_MAX:
            continue
        cap = int(PB_GROW_FRAC * max(x1 - x0, y1 - y0))
        pad = cap + PB_NECK_R + 2
        wx0, wy0 = max(0, x0 - pad), max(0, y0 - pad)
        wx1, wy1 = min(W_, x1 + pad), min(H, y1 + pad)
        seed = np.zeros((wy1 - wy0, wx1 - wx0), bool)
        seed[y0 - wy0:y1 - wy0, x0 - wx0:x1 - wx0] = True
        grown = geodesic_grow(seed & w_cut[wy0:wy1, wx0:wx1],
                              w_cut[wy0:wy1, wx0:wx1], cap)
        pb[wy0:wy1, wx0:wx1] |= grown
    return pb & ~bubble


def harmonize_enclosed_whites(out, g, lab, stats, skip_mask):
    """批1.5 人頭一致化（demo02 案，構圖層）：暗區地圖內的「殘餘亮島」→ 填深＋內緣描亮。
    亮島＝白(原圖)∧仍亮(成品)∧非已處理——**不是**白元件：群眾人頭白常與背景白同元件、
    背景被偽泡塗過後元件級 skip 會整顆跳過（b15/b16 失效原因），殘餘亮島把已塗部分
    切掉後獨立評估。防護：面積上限（主角臉大）＋外環細墨密度（鬍鬚/密髮 → 排除）。"""
    H, W_ = g.shape
    dark = (out < 60).astype(np.float32)
    ch, cw = max(1, H // HARMONIZE_ZONE_CELL), max(1, W_ // HARMONIZE_ZONE_CELL)
    coarse = cv2.resize(dark, (cw, ch), interpolation=cv2.INTER_AREA)
    coarse = cv2.GaussianBlur(coarse, (0, 0), 2.0)
    zone = cv2.resize((coarse >= HARMONIZE_ZONE_DARK).astype(np.uint8),
                      (W_, H), interpolation=cv2.INTER_NEAREST) > 0
    if not zone.any():
        return out
    resid = ((g >= WHITE_TH) & (out >= 110) & ~skip_mask).astype(np.uint8)
    n, lb, st, _ = cv2.connectedComponentsWithStats(resid, 8)
    kc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
    for i in range(1, n):
        if st[i, cv2.CC_STAT_AREA] / g.size > HARMONIZE_AREA_MAX or st[i, cv2.CC_STAT_AREA] < 150:
            continue
        x, y, w2, h2 = st[i, :4]
        x0, y0 = max(0, x - 8), max(0, y - 8)
        x1, y1 = min(W_, x + w2 + 8), min(H, y + h2 + 8)
        isl = lb[y0:y1, x0:x1] == i
        if (zone[y0:y1, x0:x1] & isl).mean() < HARMONIZE_IN_ZONE:
            continue
        cu = isl.astype(np.uint8)
        collar = (cv2.dilate(cu, kc) > 0) & ~isl
        gw = g[y0:y1, x0:x1]
        fine = (gw < 200) & (gw > 40)          # 細墨（鬍鬚/髮絲調子；排除純黑實塊）
        if collar.any() and float(fine[collar].mean()) > HARMONIZE_COLLAR_INK:
            continue
        o = out[y0:y1, x0:x1]
        o[isl] = BG
        edge = isl & ~(cv2.erode(cu, np.ones((5, 5), np.uint8)) > 0)
        o[edge] = np.maximum(o[edge], np.float32(STROKE_OBJ_V))
    return out


def compose(g, gutter, bubble, seg, frameless, lab=None, stats=None, sticker=(),
            core_ids=(), frame=None, regions=None):
    """整頁合成：D2 畫面 →（有框頁才）留白填深 → 修法4 貼紙式背景 → 氣泡重繪。"""
    out = scene_final(g, seg).astype(np.float32)
    if not frameless:                                   # 修法2：無框頁背景不填深
        out = paint_gutter(out, g, gutter, frame=frame, bubble=bubble)
    if sticker:                                         # 修法4：純白背景填黑＋前景白描邊
        out = paint_sticker(out, g, lab, stats, sticker, bubble,
                            core_ids=core_ids, frame=frame)
    out = paint_bubbles(out, g, bubble, seg)
    if regions is not None:                             # 偽泡：開口泡/字壓背景/字壓留白救回
        pb = build_pseudo_bubbles(g, regions, bubble)
        if pb.any():
            out = paint_bubbles(out, g, pb, seg)
        skip = bubble | pb | gutter
    else:
        skip = bubble | gutter
    if lab is not None:                                 # 批1.5：浮在黑裡的空白人頭一致化
        out = harmonize_enclosed_whites(out, g, lab, stats, skip)
    return np.clip(out, 0, 255).astype(np.uint8)


# ── 出圖/IO ─────────────────────────────────────────────────────────

def _label(img, text, bar_h=48, scale=0.9):
    """圖上加白底黑字標籤列（cv2.putText 無 CJK ⇒ 英文標籤）。"""
    im = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if img.ndim == 2 else img.copy()
    bar = np.full((bar_h, im.shape[1], 3), 255, np.uint8)
    cv2.putText(bar, text, (10, int(bar_h * 0.7)), cv2.FONT_HERSHEY_SIMPLEX,
                scale, (0, 0, 0), 2, cv2.LINE_AA)
    return np.vstack([bar, im])


def _hcat(cols, sep_w=6, sep_v=128):
    h = max(c.shape[0] for c in cols)
    padded = []
    for c in cols:
        if c.shape[0] < h:
            c = cv2.copyMakeBorder(c, 0, h - c.shape[0], 0, 0,
                                   cv2.BORDER_CONSTANT, value=(255, 255, 255))
        padded.append(c)
    sep = np.full((h, sep_w, 3), sep_v, np.uint8)
    row = padded[0]
    for c in padded[1:]:
        row = np.hstack([row, sep, c])
    return row


def mask_viz(img_bgr, gutter, panel_scene, bubble, seg, regions, sticker_mask=None):
    """遮罩視覺化：留白=黃、修法3格內白(降級)=橘、修法4貼紙背景=藍、氣泡=綠、
    筆畫=紅、區域框=洋紅。"""
    viz = img_bgr.copy()
    layers = [(gutter, (0, 200, 200)), (panel_scene, (0, 128, 255)),
              (bubble, (0, 160, 0))]
    if sticker_mask is not None:
        layers.append((sticker_mask, (255, 120, 40)))
    for m, col in layers:
        viz[m] = (viz[m] * 0.5 + np.array(col) * 0.5).astype(np.uint8)
    viz[seg] = (0, 0, 255)
    for r in regions:
        x0, y0, x1, y1 = r["bbox"]
        cv2.rectangle(viz, (x0, y0), (x1, y1), (255, 0, 255), 2)
    return viz


def run_page(page_path, outdir=OUT_DEFAULT, col_w=1000):
    """單頁一條龍：偵測 → 遮罩 → 合成 → 落檔。回傳統計 dict（批次表用）。"""
    name = os.path.splitext(os.path.basename(page_path))[0]
    os.makedirs(outdir, exist_ok=True)
    img = cv2.imread(page_path)                        # 彩頁也吃（偵測吃 BGR）
    assert img is not None, page_path
    g = cv2.imread(page_path, cv2.IMREAD_GRAYSCALE)
    H, W = g.shape

    lines, regions, seg = detect(img)
    frameless, hk, vk = page_is_frameless(g)
    lab, stats, gutter_ids, panel_ids = classify_white_components(g)
    bubble, merged, rejected = build_bubble_mask(
        g, regions, seg, lab, stats, gutter_ids | panel_ids)
    sticker, audit, promoted = sticker_plan(g, img, lab, stats, gutter_ids, panel_ids,
                                            frameless, regions)
    gutter_show = gutter_ids - sticker if frameless else gutter_ids
    gutter = np.isin(lab, sorted(gutter_show)) if gutter_show else np.zeros((H, W), bool)
    panel_show = panel_ids - sticker
    panel_scene = np.isin(lab, sorted(panel_show)) if panel_show else np.zeros((H, W), bool)
    sticker_mask = np.isin(lab, sorted(sticker)) if sticker else np.zeros((H, W), bool)
    lhm, lvm = frame_line_mask(g)
    final = compose(g, gutter, bubble, seg, frameless, lab, stats, sticker,
                    core_ids=promoted, frame=(lhm | lvm), regions=regions)

    pref = os.path.join(outdir, name)
    with open(f"{pref}_regions.json", "w", encoding="utf-8") as f:
        json.dump({
            "image": page_path, "width": W, "height": H,
            "pageType": "frameless" if frameless else "framed",
            "frameLines": {"h": round(hk, 3), "v": round(vk, 3)},
            "detector": "DBNet (m-i-t default @ .upstream-ref, detect-20241225.ckpt) "
                        "torch eager; text_th=0.5 box_th=0.7 unclip=2.3; "
                        "regions=mit_grouping.merge_bboxes_text_region",
            "sticker": audit,                          # 修法4 逐元件審計（含安全網量測）
            "lines": [{"quad": t.pts.tolist(), "score": round(float(t.prob), 4)}
                      for t in lines],
            "regions": regions,
        }, f, ensure_ascii=False, indent=1)
    cv2.imwrite(f"{pref}_seg.png", seg.astype(np.uint8) * 255)
    cv2.imwrite(f"{pref}_bubble.png", bubble.astype(np.uint8) * 255)
    cv2.imwrite(f"{pref}_gutter.png", gutter.astype(np.uint8) * 255)
    cv2.imwrite(f"{pref}_final.png", final)

    wb = float((g >= WHITE_MEASURE_TH).mean())
    wa = float((final >= WHITE_MEASURE_TH).mean())
    cols = []
    for im, t in ((img, f"{name} original"),
                  (final, f"night rebuild ({'FRAMELESS: bubbles only' if frameless else 'framed'})"),
                  (mask_viz(img, gutter, panel_scene, bubble, seg, regions, sticker_mask),
                   "masks: gutter=y panelwhite=o sticker=b bubble=g seg=r")):
        s = col_w / im.shape[1]
        im2 = cv2.resize(im, (col_w, int(im.shape[0] * s)), interpolation=cv2.INTER_AREA)
        cols.append(_label(im2, t, bar_h=46, scale=0.8))
    cv2.imwrite(f"{pref}_cmp.png", _hcat(cols))

    st = {"page": name, "pageType": "frameless" if frameless else "framed",
          "regions": len(regions), "whiteBefore": round(wb, 4), "whiteAfter": round(wa, 4),
          "gutterFrac": round(float(gutter.mean()), 4),
          "panelWhiteFrac": round(float(panel_scene.mean()), 4),
          "stickerFrac": round(float(sticker_mask.mean()), 4),
          "stickerComps": len(sticker), "stickerFallback": len(audit) - len(sticker),
          "stickerAudit": audit,
          "bubbleFrac": round(float(bubble.mean()), 4),
          "bubbleCompsMerged": len(merged), "bubbleCompsRejected": len(rejected),
          "cmp": f"{pref}_cmp.png"}
    print(f"[{name}] {st['pageType']}  regions={st['regions']}  "
          f"white {wb:.3f}->{wa:.3f}  gutter={st['gutterFrac']:.3f}  "
          f"panelWhite={st['panelWhiteFrac']:.3f}  sticker={st['stickerFrac']:.3f}"
          f"({len(sticker)}/{len(audit)})  bubble={st['bubbleFrac']:.3f}  "
          f"comps merged/rejected={len(merged)}/{len(rejected)}", flush=True)
    for a in audit:
        print(f"    sticker comp={a['comp']:4d} bbox={a['bbox']} area={a['areaFrac']:.3f} "
              f"fig={a['figFrac']:.3f} thin={a['thinFrac']:.3f} chroma={a['chroma']:5.1f} "
              f"eaten={a['eatenFrac']:.4f} protect={a['protectFrac']:.4f} "
              f"textCov={a['textCov']:.3f} textOn={a['textOn']:.3f} rough={a['rough']:6.1f} "
              f"-> {'ACCEPT' if a['accept'] else 'fallback'}", flush=True)
    return st


def main():
    ap = argparse.ArgumentParser(description="夜讀重繪（單頁）")
    ap.add_argument("page", help="頁圖路徑")
    ap.add_argument("-o", "--outdir", default=OUT_DEFAULT)
    a = ap.parse_args()
    run_page(a.page, a.outdir)


if __name__ == "__main__":
    main()
