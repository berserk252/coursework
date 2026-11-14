"""
Векторное поле внутри автоматически найденных зон дороги и точные маски
неподвижного переднего плана. Только оптический поток и фон, без YOLO.
Независимый анализ YOLO + ByteTrack запускается через track_traffic.py.

    ┌──────────────────────────┬──────────────────────────┐
    │ видео + стрелки поля     │ скорость по частям V(t)  │
    │ + части дороги (пунктир) │ скорость в дырке/вокруг  │
    │ + дырки (полигоны)       │ div и |rot| в частях     │
    ├──────────────────────────┼──────────────────────────┤
    │ дивергенция              │ ротор                    │
    │ (область анализа и дырки │ (то же)                  │
    │  обведены)               │                          │
    └──────────────────────────┴──────────────────────────┘

По умолчанию зоны определяются по первым 10 с движения каждой сцены.
Прежние ручные JSON используются только при --road_mode manual.

Поле -- очищенный оптический поток (flow_filters.FlowCleaner), обнулённый вне
частей дороги; дивергенция и ротор -- divergence_curl.DivCurl; дырки --
flow_holes.HoleDetector.

Выход -- каждый вид в своей папке, имя файла = имя видео:
    combined/      все 4 части вместе
    video_arrows/  видео + стрелки + части дороги + дырки
    divergence/    дивергенция в частях дороги
    curl/          ротор в частях дороги
    graphs/        графики
    stats/         CSV по кадрам
    field_stats/   CSV: средние div и ротор в дырке, вне дырки и по частям дороги
    holes/         CSV дырок: кадр, номер, часть, сколько нет движения, полигон
    snapshots/     кадр combined в момент первого "ДТП?" (или первой дырки)

Пример:
    python flow_polygons.py --video ../traffic_flow_dataset/5_avariya/view4.mp4 --out_dir out
"""

import argparse
import csv
import glob
import json
import math
import os
import time

import cv2
import numpy as np

from divergence_curl import DivCurl, default_cell
from flow_canvas import Text, field_canvas, fmt, line_plot, num, video_panel
from flow_filters import FlowCleaner, SceneGuard
from bg_model import BackgroundModel
from spatial_support import SpatialSupport
from flow_trace import TraceWriter
from flow_holes import HoleDetector, pool_mean
from foreground_objects import ForegroundObjects, FlowRegionConfirmation, stable_output_regions
from auto_road import scene_plans
from flow_stats import SpeedMeter
from road_parts import RoadParts, part_color

OUTPUTS = ("combined", "video_arrows", "divergence", "curl", "graphs")
BG_COLOR = (150, 150, 150)       # дырка, совпавшая с фоном (пустой асфальт) -- серый
HOLE_COLOR = (0, 140, 255)       # дырка -- оранжевый
INC_COLOR = (0, 0, 230)          # "ДТП?" -- красный
STATIONARY_COLOR = HOLE_COLOR   # неподвижный передний план, без утверждения о ДТП
NAN = float("nan")


class PartSpeed:
    """Скорость потока по каждой части дороги: медиана |v| по движущимся точкам
    части, px/с исходного видео, затем медиана за последнюю секунду."""

    def __init__(self, fps, scale, labels, n_parts, window_s=1.0, min_px=30):
        self.fps, self.scale = float(fps), float(scale)
        self.masks = [labels == p for p in range(n_parts)]
        self.window = max(1, round(fps * window_s))
        self.min_px = min_px
        self.win = [[] for _ in range(n_parts)]
        self.hist = [[] for _ in range(n_parts)]
        self.v = [NAN] * n_parts

    def update(self, c, valid=True):
        for p, m in enumerate(self.masks):
            if valid:
                mv = c.moving & m
                vf = float(np.median(c.mag_clean[mv])) * self.fps / self.scale if mv.sum() >= self.min_px else NAN
                self.win[p] = (self.win[p] + [vf])[-self.window:]
                vals = [x for x in self.win[p] if math.isfinite(x)]
                self.v[p] = float(np.median(vals)) if vals else 0.0
            else:
                self.v[p] = NAN
            self.hist[p].append(self.v[p])


def part_field_stats(dc: DivCurl, cell_part, n_parts):
    """Средняя дивергенция и средний |ротор| по активным клеткам каждой части, 1/с."""
    out = []
    for p in range(n_parts):
        if dc.active is None:
            out.append((NAN, NAN))
            continue
        sel = dc.active & (cell_part == p)
        out.append((float(dc.div[sel].mean()), float(np.abs(dc.rot[sel]).mean())) if sel.any() else (NAN, NAN))
    return out


def mean_stats(div, rot, sel):
    """Средние div, ротор, |ротор| по клеткам sel и их число (NAN, если клеток нет)."""
    if not sel.any():
        return [NAN, NAN, NAN, 0]
    r = rot[sel]
    return [float(div[sel].mean()), float(r.mean()), float(np.abs(r).mean()), int(sel.sum())]


def region_field_stats(dc: DivCurl, dc_part, n_parts, hole_g):
    """div и ротор отдельно: в дырке, по остальному потоку дороги, и то же по каждой части.
    Вне дырок -- активные клетки поля (там недавно было движение); в дырке движения нет по
    определению, поэтому берутся все её клетки; hole_g -- маска дырок на сетке поля."""
    if dc.active is None:
        return [NAN, NAN, NAN, 0] * (2 + 2 * n_parts)
    act = dc.active & (dc_part >= 0)
    out = mean_stats(dc.div, dc.rot, (dc_part >= 0) & hole_g) + mean_stats(dc.div, dc.rot, act & ~hole_g)
    for p in range(n_parts):
        out += (mean_stats(dc.div, dc.rot, act & (dc_part == p) & ~hole_g)
                + mean_stats(dc.div, dc.rot, (dc_part == p) & hole_g))
    return out


def main_hole(holes):
    """Дырка, о которой пишем на графике: самая давняя (дольше всех нет движения)."""
    return max(holes, key=lambda h: h.idle_s) if holes else None


def hole_label(h):
    if h.kind == "stationary":
        return f"стоящий объект {h.idle_s:.0f} с", f"stationary {h.idle_s:.0f}s"
    if h.bg_match:
        return "как фон: пустой асфальт, тревоги нет", "matches background"
    if h.bg_valid is False:
        return "фон ещё не накоплен: кандидат", "background not ready: candidate"
    if h.incident:
        return f"ДТП? нет движения {h.idle_s:.0f} с", f"ACCIDENT? {h.idle_s:.0f}s"
    v = f"{h.v_in:.1f}" if math.isfinite(h.v_in) else "-"
    return f"дырка: нет движения {h.idle_s:.0f} с, v = {v} px/с", f"hole {h.idle_s:.0f}s v={v}"


def draw_holes(img, holes, txt, k=1.0, labels=True, fill=True):
    """Слабая однократная заливка и тонкие внешние контуры точных масок.

    Пересечения не затемняются повторно; фоновые просветы не закрашиваются.
    Объединение здесь влияет только на рисунок, исходные маски и статистику
    не меняет. k -- масштаб полигонов без пиксельной маски (холсты div/rot).
    """
    groups = {}
    for h in holes:
        color = STATIONARY_COLOR if h.kind == "stationary" else BG_COLOR if h.bg_match else INC_COLOR if h.incident else HOLE_COLOR
        priority = 0 if h.bg_match else 2 if h.incident else 1
        mask = np.zeros(img.shape[:2], np.uint8)
        if h.mask is not None:
            mask = cv2.resize(h.mask.astype(np.uint8), (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
        elif h.polygons:
            cv2.fillPoly(mask, [np.round(p * k).astype(np.int32) for p in h.polygons], 1)
        key = (priority, color)
        if key not in groups:
            groups[key] = mask > 0
        else:
            groups[key] |= mask > 0

    # При пересечении красная тревога имеет приоритет над кандидатом и фоном.
    # Для каждого пикселя выбирается один цвет, после чего альфа применяется раз.
    chosen = np.zeros(img.shape[:2], np.int8)
    colors = {}
    for index, (key, mask) in enumerate(sorted(groups.items()), 1):
        chosen[mask] = index
        colors[index] = key[1]
    if fill and colors:
        over = img.copy()
        for index, color in colors.items():
            over[chosen == index] = color
        cv2.addWeighted(over, 0.12, img, 0.88, 0, dst=img)
    for index, color in colors.items():
        contours, _ = cv2.findContours((chosen == index).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = [cv2.approxPolyDP(c, max(.6, 1.5 * img.shape[1] / 640), True) for c in contours]
        cv2.drawContours(img, contours, -1, color, 1, cv2.LINE_AA)

    if not labels:
        return
    for h in holes:
        color = STATIONARY_COLOR if h.kind == "stationary" else BG_COLOR if h.bg_match else INC_COLOR if h.incident else HOLE_COLOR
        polys = [np.round(p * k).astype(np.int32) for p in h.polygons]
        if polys:
            x, y, w, hh = cv2.boundingRect(np.vstack(polys))
            text, alt = hole_label(h)
            size = 14 if h.incident else 12
            tw = txt.draw(np.zeros((1, 1, 3), np.uint8), text, 0, 0, size=size)
            tx = int(min(max(2, x), img.shape[1] - tw - 4))
            ty = y - size - 8 if y - size - 8 > 2 else y + hh + 4
            cv2.rectangle(img, (tx - 3, ty - 2), (tx + tw + 3, ty + size + 5), (255, 255, 255), -1)
            txt.draw(img, text, tx, ty, size=size, color=color, alt=alt)


def legend_box(img, lines, txt, x=6, y=None, size=11):
    """Небольшая легенда на белой подложке в левом нижнем углу."""
    y = img.shape[0] - 6 - (size + 6) * len(lines) if y is None else y
    for text, color in lines:
        w = txt.draw(np.zeros((1, 1, 3), np.uint8), text, 0, 0, size=size)
        cv2.rectangle(img, (x - 3, y - 2), (x + w + 3, y + size + 4), (255, 255, 255), -1)
        txt.draw(img, text, x, y, size=size, color=color)
        y += size + 6


def make_overlay(rp: RoadParts, outside: np.ndarray, holes, W, txt):
    """Для холстов div/rot: вне частей дороги -- бледно (анализа там нет),
    части -- пунктиром, дырки -- контуром."""
    def ov(plot):
        ph, pw = plot.shape[:2]
        o = cv2.resize(outside.astype(np.uint8), (pw, ph), interpolation=cv2.INTER_NEAREST) > 0
        plot[o] = (0.45 * plot[o] + 0.55 * 238).astype(np.uint8)
        rp.draw(plot, txt, thickness=1, labels=True, size=11)
        draw_holes(plot, holes, txt, k=pw / W, labels=False, fill=False)
        legend_box(plot, [("пунктир — область анализа (части дороги)", (60, 60, 60))]
                   + ([("контур — дырка в поле", HOLE_COLOR)] if holes else []), txt)
    return ov


FS_REST = (90, 90, 90)           # остальной поток -- серый


def field_series(fh, n_parts, k):
    """Ряды для графика: k = 0 div, 1 rot. В дырке, по остальному потоку и по частям дороги."""
    ser = [([v[0 * 4 + k] for v in fh], INC_COLOR, "дырка"),
           ([v[1 * 4 + k] for v in fh], FS_REST, "остальное")]
    ser += [([v[(2 + 2 * p) * 4 + k] for v in fh], part_color(p), f"часть {p + 1}") for p in range(n_parts)]
    return ser


def graphs_canvas(P_W, P_H, sm: SpeedMeter, ps: PartSpeed, fh, n_parts, holes, txt):
    img = np.full((P_H, P_W, 3), 255, np.uint8)
    left, right = 52, P_W - 12
    free = P_H - 128
    h1, h2 = int(free * 0.36), int(free * 0.28)
    r1 = (left, 26, right, 26 + h1)
    ys = r1[3] + 14
    r2 = (left, ys + 34, right, ys + 34 + h2)
    r3 = (left, r2[3] + 26, right, P_H - 26)
    series = [(ps.hist[p], part_color(p), f"V{p + 1}") for p in range(len(ps.hist))]
    line_plot(img, r1, sm.t, series, "Скорость по частям дороги V, px/с", "Speed by road part, px/s",
              txt, xlabels=False)
    # строка состояния: дырки
    x = left
    inc = any(h.incident for h in holes)
    if holes:
        mh = main_hole(holes)
        color = INC_COLOR if inc else HOLE_COLOR
        x += txt.draw(img, f"дырок в поле: {len(holes)}", x, ys - 7, size=12, color=color, alt="holes:") + 14
        txt.draw(img, f"нет движения {mh.idle_s:.0f} с", x, ys - 7, size=12, color=color, alt=f"{mh.idle_s:.0f}s")
    else:
        txt.draw(img, "дырок в поле нет", x, ys - 7, size=12, color=(90, 90, 90), alt="no holes")
    if inc:
        txt.draw(img, "ДТП?", right, ys - 9, size=15, color=INC_COLOR, alt="ACCIDENT?", anchor="rt")
    # r2 и r3: средние div и ротор -- в дырке, по остальному потоку и по каждой части дороги
    line_plot(img, r2, sm.t, field_series(fh, n_parts, 0),
              "Дивергенция: в дырке, в остальном потоке, по частям, 1/с", "div: hole, rest of flow, parts, 1/s",
              txt, xlabels=False, y_sym=True)
    line_plot(img, r3, sm.t, field_series(fh, n_parts, 1),
              "Ротор: в дырке, в остальном потоке, по частям, 1/с", "rot: hole, rest of flow, parts, 1/s",
              txt, y_sym=True)
    return img


def process(video, out_root, rel, args, txt):
    stem = os.path.splitext(os.path.basename(video))[0]
    ppath = RoadParts.path_for(video, args.parts_dir)
    automatic = args.road_mode == "auto"
    plans = scene_plans(video, args.road_seconds, args.max_width, args.seconds) if automatic else []
    if not automatic and not os.path.exists(ppath):
        print(f"пропуск (нет разметки {ppath}): {video}")
        return
    rp = plans[0][1] if automatic else RoadParts.load(ppath)
    if not automatic and len(rp) < 1:
        print(f"пропуск (в разметке нет частей): {video}")
        return
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    ok, frame = cap.read()
    if not ok:
        print(f"пропуск (не читается): {video}")
        return
    scale = min(1.0, args.max_width / frame.shape[1]) if args.max_width > 0 else 1.0

    def prep(f):
        return cv2.resize(f, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale != 1.0 else f

    frame = prep(frame)
    H, W = frame.shape[:2]
    H -= H % 2
    W -= W % 2
    frame = frame[:H, :W]
    prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    labels = rp.label_map(W, H)
    road = labels >= 0
    road_f = road.astype(np.float32)
    n_parts = max(1, max(len(p[1]) for p in plans)) if automatic else len(rp)
    scene_id = 0
    if automatic:
        calibration_dir = os.path.join(out_root, "auto_road", rel, stem)
        os.makedirs(calibration_dir, exist_ok=True)
        for i, (start, parts, metadata) in enumerate(plans):
            parts.save(os.path.join(calibration_dir, f"scene_{i:03d}.json"))
        with open(os.path.join(calibration_dir, "calibration.json"), "w") as stream:
            json.dump([dict(metadata, scene_id=i, start_frame=start, parts=len(parts))
                       for i, (start, parts, metadata) in enumerate(plans)], stream, ensure_ascii=False, indent=2)
        preview = frame.copy()
        rp.draw(preview, txt, fill_alpha=.15)
        cv2.imwrite(os.path.join(calibration_dir, "preview.png"), preview)

    cleaner = FlowCleaner(egomotion=bool(args.egomotion), road_mask=False)
    dc = DivCurl(cell=default_cell(W, args.cell), memory=args.memory, fps=fps)
    frac = np.stack([pool_mean((labels == p).astype(np.float32), dc.cell) for p in range(n_parts)])
    dc_part = np.where(frac.max(axis=0) >= 0.5, frac.argmax(axis=0), -1)
    sm = SpeedMeter(fps, scale=scale, strict_road=True)
    ps = PartSpeed(fps, scale, labels, n_parts)
    guard = SceneGuard(fps)
    guard.update(prev_gray)
    hd = HoleDetector(fps, labels, scale, incident_s=args.incident_s, fill_px=0, soft_grow=False, grow_px=24)
    bgm = BackgroundModel(fps, road) if args.bg else None
    occupancy = ForegroundObjects((H, W), fps)
    spatial = SpatialSupport((H, W))
    confirmation=FlowRegionConfirmation(args.incident_s)
    trace = TraceWriter(os.path.join(out_root, 'trace_cache', rel, stem), video, fps, (H, W),
                        n_parts, plans or [(0, rp, {})], scale) if getattr(args, 'trace_cache', False) else None
    k_frz = np.ones((hd.cell + 1, hd.cell + 1), np.uint8)
    hdiv, hrot, h_in, h_ring = [], [], [], []
    fh = []                                      # по кадрам: средние div/rot по областям (field_stats)
    n_cuts = 0
    t_first_inc = t_first_hole = NAN
    snap_done = None

    writers = {}
    for name in OUTPUTS:
        if name not in args.outputs_set:
            continue
        d = os.path.join(out_root, name, rel)
        os.makedirs(d, exist_ok=True)
        size = (2 * W, 2 * H) if name == "combined" else (W, H)
        w = cv2.VideoWriter(os.path.join(d, f"{stem}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
        if not w.isOpened():
            raise ValueError(f"Failed writer: {d}/{stem}.mp4")
        writers[name] = w
    d = os.path.join(out_root, "stats", rel)
    os.makedirs(d, exist_ok=True)
    f_csv = open(os.path.join(d, f"{stem}.csv"), "w", newline="")
    table = csv.writer(f_csv)
    table.writerow(["frame", "time_s", "valid", "cut", "speed_px_s"]
                   + [f"speed_part{p + 1}_px_s" for p in range(n_parts)]
                   + ["moving_area", "div_mean_per_s", "rot_mean_abs_per_s"]
                   + sum([[f"div_part{p + 1}_per_s", f"rot_abs_part{p + 1}_per_s"] for p in range(n_parts)], [])
                       + ["holes", "hole_area_frac", "hole_idle_max_s", "hole_v_in_px_s", "hole_v_ring_px_s", "incident",
                          "scene_id", "road_area_px", "road_mode",'bg_recovered_scene_px'])
    d = os.path.join(out_root, "holes", rel)
    os.makedirs(d, exist_ok=True)
    f_holes = open(os.path.join(d, f"{stem}.csv"), "w", newline="")
    th = csv.writer(f_holes)
    th.writerow(["frame", "time_s", "hole_id", "part", "idle_s", "age_s", "area_px", "v_in_px_s",
                 "v_ring_px_s", "ring_ratio", "incident", "bg_diff", "bg_match", "bg_valid", "bg_coverage", "polygon_xy_source_px",
                 "kind", "bg_removed_frac", "polygon_rings_source_px", "scene_id", "bg_inferred_frac",
                 'observed_s','confirmation_s'])

    d = os.path.join(out_root, "field_stats", rel)
    os.makedirs(d, exist_ok=True)
    f_fs = open(os.path.join(d, f"{stem}.csv"), "w", newline="")
    tf = csv.writer(f_fs)
    scopes = ["hole", "rest"] + sum([[f"part{p + 1}", f"part{p + 1}_hole"] for p in range(n_parts)], [])
    tf.writerow(["frame", "time_s", "valid", "holes"]
                + sum([[f"{sc}_div_per_s", f"{sc}_rot_per_s", f"{sc}_rot_abs_per_s", f"{sc}_cells"] for sc in scopes], []))
    fs_sum = {}                                  # сводка по кадрам, где есть дырка
    n_bg_cut = 0

    limit = int(args.seconds * fps) if args.seconds > 0 else 0
    snapshot_frames = {round(float(x) * fps) for x in args.snapshot_times.split(",") if x.strip()}
    idx = 0
    road_px = max(int(road.sum()), 1)
    progress_at = time.monotonic()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        idx += 1
        if limit and idx > limit:
            break
        frame = prep(frame)[:H, :W]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        t = idx / fps
        cut, valid = guard.update(gray)
        if cut:
            cleaner = FlowCleaner(egomotion=bool(args.egomotion), road_mask=False)
            dc.reset()
            sm.reset()
            n_cuts += 1
            if automatic:
                scene_id = max(i for i, plan in enumerate(plans) if plan[0] <= idx)
                rp = plans[scene_id][1]
                labels = rp.label_map(W, H)
                road, road_f = labels >= 0, (labels >= 0).astype(np.float32)
                road_px = max(int(road.sum()), 1)
                frac = np.stack([pool_mean((labels == p).astype(np.float32), dc.cell) for p in range(n_parts)])
                dc_part = np.where(frac.max(axis=0) >= .5, frac.argmax(axis=0), -1)
                hd = HoleDetector(fps, labels, scale, incident_s=args.incident_s, fill_px=0, soft_grow=False, grow_px=24)
                bgm = BackgroundModel(fps, road) if args.bg else None
                occupancy.reset()
                ps.masks = [labels == p for p in range(n_parts)]
                ps.win = [[] for _ in range(n_parts)]
            else:
                scene_id += 1
        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        valid = valid and not cut
        c = cleaner.update(flow, gray, prev_gray)
        c.road_w = road_f                       # анализ только внутри частей дороги
        if valid:
            dc.update(c)
        sm.update(c, t, valid=valid)
        ps.update(c, valid=valid)
        keep = None
        if bgm is not None and valid:
            def keep():
                return bgm.empty_cells(gray, hd.cell, hd.gh, hd.gw, args.bg_diff_frac, still_ref=~(c.moving > 0))
        st = hd.update(c, t, valid=valid, cut=cut, div=dc.div if valid else None, keep=keep)
        holes = list(st.holes)
        base_holes = holes
        if bgm is not None:
            if cut:
                bgm.reset()
                confirmation.reset()
                spatial.reset()
            frozen = np.isfinite(hd.hole_since)
            frozen = cv2.dilate(cv2.resize(frozen.astype(np.uint8), (hd.gw * hd.cell, hd.gh * hd.cell),
                                           interpolation=cv2.INTER_NEAREST), k_frz) > 0
            fz = np.zeros((H, W), bool)
            fz[: frozen.shape[0], : frozen.shape[1]] = frozen[:H, :W]
            if valid:
                bgm.update(gray, c.moving > 0, fz,frame=frame)
                occupancy.forget_background(bgm.newly_recovered)
                confirmation.forget_background(bgm.newly_recovered)
            foreground, known = bgm.foreground(gray, still_ref=~fz & ~(c.moving > 0))
            objects = occupancy.update(gray, bgm.bg, foreground, c.moving, labels, t,
                                       hd.cell, hd.gh, hd.gw, cut=cut, valid=valid, frame=frame, known=known)
            holes = confirmation.update(holes,objects,labels,hd.cell,t,foreground,known) if valid else []
            base_holes = stable_output_regions(holes, occupancy.confirm_s)
            holes = spatial.update(base_holes, gray, bgm.bg, t, frame=frame, known=known,
                                   foreground=foreground, pixel_age=occupancy.age, scene_id=scene_id) if valid else []
            for hole in holes:
                hole.v_in = float(c.mag_clean[hole.mask].mean()) * fps / scale
        real = [h for h in holes if not h.bg_match]          # кандидаты, не опровергнутые фоном
        holes_all, holes = holes, real
        mh = main_hole(holes)
        hdiv.append(dc.stats["div_mean"] if valid else NAN)
        hrot.append(dc.stats["rot_mean_abs"] if valid else NAN)
        h_in.append(mh.v_in if mh is not None else NAN)
        h_ring.append(mh.v_ring if mh is not None else NAN)
        if holes and math.isnan(t_first_hole):
            t_first_hole = t
        incident = any(h.incident for h in holes)
        if incident and math.isnan(t_first_inc):
            t_first_inc = t

        pstats = part_field_stats(dc, dc_part, n_parts)
        hole_g = np.zeros(dc.div.shape if dc.div is not None else (1, 1), bool)
        if holes and dc.div is not None:
            hm = np.zeros((H, W), np.uint8)
            for h in holes:
                if h.mask is not None:
                    hm[h.mask] = 1
                else:
                    cv2.fillPoly(hm, h.polygons, 1)
            hole_g = cv2.resize(hm.astype(np.float32), (hole_g.shape[1], hole_g.shape[0]),
                                interpolation=cv2.INTER_AREA) >= 0.15   # клетка поля (24 px) больше маленькой дырки
        fs = region_field_stats(dc, dc_part, n_parts, hole_g) if valid else [NAN, NAN, NAN, 0] * (2 + 2 * n_parts)
        fh.append(fs)
        tf.writerow([idx, f"{t:.3f}", int(valid), len(holes)]
                    + [f"{v:.6f}" if isinstance(v, float) else v for v in fs])
        if holes and valid:
            for j, v in enumerate(fs):
                if isinstance(v, float) and math.isfinite(v):
                    a = fs_sum.setdefault(j, [0.0, 0])
                    a[0] += v
                    a[1] += 1
        table.writerow([idx, f"{t:.3f}", int(valid), int(cut), f"{sm.v:.3f}"]
                       + [f"{v:.3f}" for v in ps.v]
                       + [f"{sm.area:.5f}", f"{dc.stats['div_mean']:.6f}", f"{dc.stats['rot_mean_abs']:.6f}"]
                       + sum([[f"{a:.6f}", f"{b:.6f}"] for a, b in pstats], [])
                       + [len(holes), f"{sum(h.area_px for h in holes) / road_px:.5f}",
                          f"{max((h.idle_s for h in holes), default=0.0):.1f}",
                          f"{mh.v_in:.3f}" if mh is not None else "", f"{mh.v_ring:.3f}" if mh is not None else "",
                          int(incident), scene_id, int(road.sum()), args.road_mode,
                          int(bgm.recovered.sum()) if bgm is not None else 0])
        for h in holes_all:
            pts = ";".join(" ".join(f"{v / scale:.0f}" for v in xy) for poly in h.polygons for xy in poly)
            th.writerow([idx, f"{t:.3f}", h.id, h.part + 1, f"{h.idle_s:.1f}", f"{h.age_s:.1f}", h.area_px,
                         f"{h.v_in:.3f}", f"{h.v_ring:.3f}", f"{h.ring_ratio:.2f}", int(h.incident),
                         f"{h.bg_diff:.3f}", int(h.bg_match),
                         "" if h.bg_valid is None else int(h.bg_valid), f"{h.bg_coverage:.3f}", pts,
                         h.kind, f"{h.bg_removed_frac:.4f}",
                         json.dumps([np.round(p / scale).astype(int).tolist() for p in h.polygons], separators=(",", ":")), scene_id,
                         f"{h.bg_inferred_frac:.4f}",f'{h.observed_s:.3f}',f'{h.confirmation_s:.3f}'])

        if trace is not None:
            trace.add(idx, t, dc, base_holes, holes_all, c.mag_clean, known, foreground, occupancy.age,
                      bgm.bg, scene_id, sm, ps, fh,
                      dict(valid=valid, cut=cut, speed=sm.v, part_speeds=list(ps.v), moving_area=sm.area,
                           part_field=pstats, road_area_px=int(road.sum()), road_mode=args.road_mode,
                           bg_recovered_scene_px=int(bgm.recovered.sum())), gray=gray)
        prev_gray = gray
        if time.monotonic() - progress_at >= 30:
            for stream in (f_csv, f_holes, f_fs):
                stream.flush()
            print(f"PROGRESS {stem}: frame={idx} time_s={t:.3f}", flush=True)
            progress_at = time.monotonic()
        event_snap = (snap_done is None and holes) or (snap_done == "hole" and incident)
        want_snap = event_snap or idx in snapshot_frames
        if want_snap and bgm is not None and getattr(args,'debug_masks',False):
            debug_dir=os.path.join(out_root,'diagnostics',rel,stem)
            os.makedirs(debug_dir,exist_ok=True)
            path=os.path.join(debug_dir,f'{idx:06d}')
            cv2.imwrite(path+'_background.jpg',np.clip(bgm.bg,0,255).astype(np.uint8))
            cv2.imwrite(path+'_known.png',known.astype(np.uint8)*255)
            np.savez_compressed(path+'.npz',frame=idx,time_s=t,gray=gray,background=bgm.bg,
                variance=bgm.var,count=bgm.cnt,foreground=foreground,known=known,pixel_age=occupancy.age,
                recovered=bgm.recovered,
                moving=c.moving,labels=labels,
                flow_cores=np.stack([h.core if h.core is not None else h.cells for h in st.holes]) if st.holes else np.empty((0,hd.gh,hd.gw),bool),
                hole_masks=np.stack([h.mask for h in holes_all]) if holes_all else np.empty((0,H,W),bool),
                hole_incident=np.array([h.incident for h in holes_all]),
                hole_ids=np.array([h.id for h in holes_all]),hole_idle=np.array([h.idle_s for h in holes_all]),
                hole_observed_s=np.array([h.observed_s for h in holes_all]),
                hole_anchors=np.stack([h.incident_anchor if h.incident_anchor is not None else np.zeros((H,W),bool) for h in holes_all]) if holes_all else np.empty((0,H,W),bool),
                hole_bg_diff=np.array([h.bg_diff for h in holes_all]),
                hole_bg_coverage=np.array([h.bg_coverage for h in holes_all]))
        if not writers and not want_snap:
            continue
        va = video_panel(frame, dc, txt)
        rp.draw(va, txt, thickness=1, size=11)
        draw_holes(va, holes_all, txt, labels=False)     # только полигоны, без надписей
        outside = ~road
        parts = {
            "video_arrows": va,
            "graphs": graphs_canvas(W, H, sm, ps, fh, n_parts, holes, txt),
            "divergence": field_canvas(W, H, frame, dc, "div", txt, edges=bool(args.edges),
                                       overlay=make_overlay(rp, outside, holes, W, txt)),
            "curl": field_canvas(W, H, frame, dc, "rot", txt, edges=bool(args.edges),
                                 overlay=make_overlay(rp, outside, holes, W, txt)),
        }
        out = np.vstack([np.hstack([parts["video_arrows"], parts["graphs"]]),
                         np.hstack([parts["divergence"], parts["curl"]])])
        cv2.line(out, (W, 0), (W, 2 * H), (180, 180, 180), 1)
        cv2.line(out, (0, H), (2 * W, H), (180, 180, 180), 1)
        parts["combined"] = out
        for name, w in writers.items():
            w.write(parts[name])
        if want_snap:
            d = os.path.join(out_root, "snapshots", rel)
            os.makedirs(d, exist_ok=True)
            if idx in snapshot_frames:
                cv2.imwrite(os.path.join(d, f"{stem}_кадр_{idx}_{t:.1f}с.png"), out)
            if event_snap:
                kind = "ДТП" if incident else "дырка"
                cv2.imwrite(os.path.join(d, f"{stem}_{kind}_{t:.0f}с.png"), out)
                snap_done = "inc" if incident else "hole"

    cap.release()
    if trace is not None:
        trace.close()
    for w in writers.values():
        w.release()
    f_csv.close()
    f_holes.close()
    f_fs.close()
    if fs_sum:
        names = ["div", "rot", "|rot|"]
        def avg(sc, k):
            a = fs_sum.get(sc * 4 + k)
            return f"{a[0] / a[1]:.4f}" if a else "-"
        print("  среднее за кадры с дыркой (div / rot / |rot|, 1/с):")
        for sc, nm in enumerate(scopes):
            print(f"    {nm}: " + " / ".join(avg(sc, k) for k in range(3)))
    msg = f"готово: {os.path.join(rel, stem)}  ({idx} кадров, склеек: {n_cuts}"
    if not math.isnan(t_first_hole):
        msg += f", первая дырка на {t_first_hole:.1f} с"
    if not math.isnan(t_first_inc):
        msg += f", ДТП? впервые на {t_first_inc:.1f} с"
    print(msg + ")")


def main():
    ap = argparse.ArgumentParser(description="Оптический поток, фон и полигоны без YOLO; зоны дороги определяются автоматически.")
    ap.add_argument("--video", required=True, help="Файл или папка (все .mp4 в ней и подпапках)")
    ap.add_argument("--parts_dir", default=os.path.join("..", "разметка_дорог"), help="Папка с разметкой частей дороги")
    ap.add_argument("--road_mode", choices=("auto", "manual"), default="auto", help="auto: зоны из движения; manual: прежние JSON для сравнения")
    ap.add_argument("--road_seconds", type=float, default=10, help="Окно автоматической калибровки каждой сцены, с")
    ap.add_argument("--out_dir", default="polygons_out")
    ap.add_argument("--max_width", type=int, default=640, help="Ширина обработки и одной части видео, px")
    ap.add_argument("--cell", type=int, default=24, help="Клетка поля div/rot при ширине 640 px")
    ap.add_argument("--memory", type=float, default=0.9, help="Сглаживание поля во времени, 0..1")
    ap.add_argument("--egomotion", type=int, default=1)
    ap.add_argument("--edges", type=int, default=1, help="1 = серые контуры кадра на холстах")
    ap.add_argument("--incident_s", type=float, default=20.0, help="Сколько секунд нет движения в дырке до \"ДТП?\"")
    ap.add_argument("--bg", type=int, default=1, help="1 = дырку, совпавшую с долгим фоном дороги, не считать (тревоги нет)")
    ap.add_argument("--bg_diff_frac", type=float, default=0.25,
                    help="Дырка совпадает с фоном, если отличных от него точек меньше этой доли")
    ap.add_argument("--seconds", type=float, default=0, help="Обработать только первые N секунд")
    ap.add_argument("--snapshot_times", default="", help="Моменты снимков через запятую, с; работают и при --outputs none")
    ap.add_argument('--debug_masks',type=int,default=0,help='Сохранить фон и численные маски в моменты снимков')
    ap.add_argument('--trace_cache', type=int, default=0, help='Сохранить точные промежуточные данные для повторной проверки полигонов')
    ap.add_argument("--outputs", default="all",
                    help="Какие видео писать через запятую: " + ",".join(OUTPUTS) + "; all -- все, none -- только CSV")
    ap.add_argument("--rel", default=".", help="Для одного файла: подпапка внутри каждой папки результата")
    args = ap.parse_args()
    if args.trace_cache and not args.bg:
        ap.error('--trace_cache требует --bg 1')
    if args.outputs == "all":
        args.outputs_set = set(OUTPUTS)
    elif args.outputs == "none":
        args.outputs_set = set()
    else:
        args.outputs_set = {x.strip() for x in args.outputs.split(",") if x.strip()}
        bad = args.outputs_set - set(OUTPUTS)
        if bad:
            ap.error(f"--outputs: неизвестные имена {sorted(bad)}")
    if os.environ.get("CV_THREADS"):
        cv2.setNumThreads(int(os.environ["CV_THREADS"]))

    txt = Text()
    if os.path.isdir(args.video):
        videos = sorted(glob.glob(os.path.join(args.video, "**", "*.mp4"), recursive=True))
        for v in videos:
            if args.road_mode == "auto" or os.path.exists(RoadParts.path_for(v, args.parts_dir)):
                process(v, args.out_dir, os.path.relpath(os.path.dirname(v), args.video), args, txt)
    else:
        process(args.video, args.out_dir, args.rel, args, txt)


if __name__ == "__main__":
    main()
