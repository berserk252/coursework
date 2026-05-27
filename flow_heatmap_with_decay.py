import argparse
import math
import cv2
import numpy as np


def draw_pro_arrow(img, p1, p2, color=(0, 255, 0), thickness=2, head_len=10.0,
                   head_wid_ratio=0.55, outline_color=(0, 0, 0), outline_extra=1):
    """
    Аккуратная стрелка как в профессиональных flow-визуализациях:
    - чёрная обводка под основной линией для контраста на любом фоне,
    - линия с anti-aliasing,
    - наконечник = залитый треугольник (а не "галочка" cv2.arrowedLine).
    """
    x1, y1 = p1
    x2, y2 = p2
    dx, dy = x2 - x1, y2 - y1
    length = math.hypot(dx, dy)
    if length < 1.5:
        return
    # ограничим длину наконечника, чтобы при коротких векторах он не "съедал" линию
    h_len = min(head_len, length * 0.45)
    h_wid = h_len * head_wid_ratio

    # точка, где линия "уходит" под наконечник (база треугольника)
    ux, uy = dx / length, dy / length
    bx, by = x2 - ux * h_len, y2 - uy * h_len
    # перпендикуляр
    px, py = -uy, ux
    left  = (int(round(bx + px * h_wid)), int(round(by + py * h_wid)))
    right = (int(round(bx - px * h_wid)), int(round(by - py * h_wid)))
    tip   = (int(round(x2)), int(round(y2)))
    base  = (int(round(bx)), int(round(by)))

    # 1) тонкая чёрная обводка тела стрелки
    cv2.line(img, (int(round(x1)), int(round(y1))), base,
             outline_color, thickness + 2 * outline_extra, cv2.LINE_AA)
    # 2) сама зелёная линия
    cv2.line(img, (int(round(x1)), int(round(y1))), base,
             color, thickness, cv2.LINE_AA)
    # 3) залитый треугольник-наконечник (с обводкой)
    pts = np.array([[tip, left, right]], dtype=np.int32)
    cv2.fillPoly(img, pts, outline_color, lineType=cv2.LINE_AA)
    # уменьшим треугольник немного и зальём цветом — получится "обводка + заливка"
    inset = max(1, outline_extra)
    cx = (tip[0] + left[0] + right[0]) / 3.0
    cy = (tip[1] + left[1] + right[1]) / 3.0
    def shrink(pt):
        vx, vy = pt[0] - cx, pt[1] - cy
        l = math.hypot(vx, vy)
        if l <= inset:
            return (int(round(cx)), int(round(cy)))
        k = (l - inset) / l
        return (int(round(cx + vx * k)), int(round(cy + vy * k)))
    pts2 = np.array([[shrink(tip), shrink(left), shrink(right)]], dtype=np.int32)
    cv2.fillPoly(img, pts2, color, lineType=cv2.LINE_AA)


def heatmap_smooth_gradient(speed_map: np.ndarray, low: float = 0.5, high: float = 3.0):
    """
    Плавный градиент по СКОРОСТИ движения:
      speed <= low     -> красный  (0,   0,   255)  — медленное движение
      speed == mid     -> синий    (255, 0,   0)    — среднее
      speed >= high    -> зелёный  (0,   255, 0)    — быстрое
    Между значениями — плавная интерполяция (red->blue->green).
    След накапливается во времени снаружи (через decay), здесь только
    превращаем "затухающую скорость" в цвет.
    Возвращает (heat_bgr uint8, alpha_map float32 [0..1]).
    """
    s = speed_map.astype(np.float32)

    # нормализованный t в [0..1] по диапазону [low..high]
    t = np.clip((s - low) / max(high - low, 1e-6), 0.0, 1.0)

    # нижняя половина t: red(0,0,255) -> blue(255,0,0)
    # верхняя половина t: blue(255,0,0) -> green(0,255,0)
    upper = t >= 0.5
    t_up = (t - 0.5) * 2.0
    t_lo = t * 2.0

    B = np.where(upper, 255.0 * (1.0 - t_up), 255.0 * t_lo)
    G = np.where(upper, 255.0 * t_up,         0.0)
    R = np.where(upper, 0.0,                  255.0 * (1.0 - t_lo))

    heat = np.stack([B, G, R], axis=-1).astype(np.uint8)

    # alpha: плавно от 0 (нет движения) до 0.55 (быстро); pow<1 — быстрее
    # появляется и медленно гаснет.
    alpha_norm = np.clip(s / max(high, 1e-6), 0.0, 1.0)
    alpha = (alpha_norm ** 0.6) * 0.55
    return heat, alpha


def alpha_blend(frame_bgr: np.ndarray, heat_bgr: np.ndarray, alpha_map: np.ndarray) -> np.ndarray:
    a = alpha_map[..., None].astype(np.float32)
    out = frame_bgr.astype(np.float32) * (1.0 - a) + heat_bgr.astype(np.float32) * a
    return np.clip(out, 0, 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser(
        description=(
            "ONE PASS: generates 3 opticalflow videos (90/95/98), 5 heatmaps, "
            "and 3 combined overlays (video+heatmap(nodecay)+opticalflow 90/95/98)."
        )
    )
    ap.add_argument("--video", default="video.webm", help="Input video path")
    ap.add_argument("--fps", type=float, default=30.0, help="Output FPS")
    ap.add_argument("--step", type=int, default=60, help="Grid cell size (px) for arrows")
    ap.add_argument("--scale", type=float, default=5.0, help="Arrow length scale (px per flow unit)")
    ap.add_argument(
        "--mag_thresh",
        type=float,
        default=0.0,
        help="Motion threshold for HEATMAP accumulation. 0 = accumulate all motion.",
    )
    ap.add_argument(
        "--presence_thresh",
        type=float,
        default=0.15,
        help="Threshold for 'any motion' mask in HEATMAP (in flow magnitude units).",
    )
    ap.add_argument(
        "--arrow_thresh",
        type=float,
        default=1.5,
        help="Motion threshold for drawing arrows (keeps arrows readable).",
    )
    ap.add_argument(
        "--hi_ema",
        type=float,
        default=0.95,
        help="Smoothing for heatmap normalization (0..1). Higher = more stable colors.",
    )
    ap.add_argument(
        "--max_frames",
        type=int,
        default=0,
        help="0 = process full video; otherwise stop after N frames (for quick tests).",
    )
    ap.add_argument(
        "--preview_seconds",
        type=float,
        default=0.0,
        help="If >0, process only first N seconds (fast preview).",
    )
    ap.add_argument(
        "--every_n",
        type=int,
        default=1,
        help="Process every Nth frame (speed-up preview; output fps stays same).",
    )
    ap.add_argument(
        "--resize",
        type=float,
        default=1.0,
        help="Resize factor for faster preview, e.g. 0.5 (outputs smaller videos).",
    )
    ap.add_argument("--out_dir", default=".", help="Output directory")
    args = ap.parse_args()

    out_dir = args.out_dir.rstrip("/")

    # Requested decays / outputs
    # Trail decay для стрелок — выше значит дольше остаются на асфальте.
    OPT_DECAYS = [0.97, 0.99, 0.995]  # 3 opticalflow videos
    # User requested 5 heatmaps; your old script had 4 (nodecay/90/95/98),
    # so we add a 99% option as the 5th.
    HEAT_DECAYS = [None, 0.90, 0.95, 0.98, 0.99]  # 5 heatmap videos

    cap = cv2.VideoCapture(args.video)
    ret, first = cap.read()
    if not ret:
        raise ValueError("Cannot read video")

    if args.resize != 1.0:
        first = cv2.resize(
            first,
            (int(first.shape[1] * args.resize), int(first.shape[0] * args.resize)),
            interpolation=cv2.INTER_AREA if args.resize < 1.0 else cv2.INTER_LINEAR,
        )

    H, W = first.shape[:2]
    prev_gray = cv2.cvtColor(first, cv2.COLOR_BGR2GRAY)

    # Сетка из ячеек step×step. На каждой ячейке живёт одна стрелка
    # с собственным "состоянием" (вектор), которое плавно тянется к
    # текущему движению и плавно сжимается к нулю в покое.
    cell = max(8, int(args.step))
    grid_cols = W // cell
    grid_rows = H // cell
    # Координаты центров клеток (в пикселях)
    grid_cx = (np.arange(grid_cols) * cell + cell // 2).astype(np.int32)
    grid_cy = (np.arange(grid_rows) * cell + cell // 2).astype(np.int32)
    # Состояние стрелки на каждый OPT_DECAYS — отдельная "память" сжатия
    arrow_states = [np.zeros((grid_rows, grid_cols, 2), dtype=np.float32)
                    for _ in OPT_DECAYS]

    # Accumulators
    heat_accums = [np.zeros((H, W), dtype=np.float32) for _ in HEAT_DECAYS]
    # Smoothed normalization scale per heatmap to avoid fast color flicker/extinguish
    heat_hi_ema = [0.0 for _ in HEAT_DECAYS]
    # Per-pixel "decayed peak speed" — на нём строится цвет: red(slow)/blue(mid)/green(fast).
    heat_speed = [np.zeros((H, W), dtype=np.float32) for _ in HEAT_DECAYS]

    # Writers (names exactly like older scripts + new combined)
    heat_names = [
        "heatmap_nodecay.mp4",
        "heatmap_90.mp4",
        "heatmap_95.mp4",
        "heatmap_98.mp4",
        "heatmap_99.mp4",
    ]
    flow_names = ["opticalflow90.mp4", "opticalflow95.mp4", "opticalflow98.mp4"]
    combined_names = [
        "heatmap_nodecay_opticalflow90.mp4",
        "heatmap_nodecay_opticalflow95.mp4",
        "heatmap_nodecay_opticalflow98.mp4",
    ]

    heat_writers = []
    for n in heat_names:
        p = f"{out_dir}/{n}"
        w = cv2.VideoWriter(p, cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (W, H))
        if not w.isOpened():
            raise ValueError(f"Failed writer: {p}")
        heat_writers.append(w)

    flow_writers = []
    for n in flow_names:
        p = f"{out_dir}/{n}"
        w = cv2.VideoWriter(p, cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (W, H))
        if not w.isOpened():
            raise ValueError(f"Failed writer: {p}")
        flow_writers.append(w)

    comb_writers = []
    for n in combined_names:
        p = f"{out_dir}/{n}"
        w = cv2.VideoWriter(p, cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (W, H))
        if not w.isOpened():
            raise ValueError(f"Failed writer: {p}")
        comb_writers.append(w)

    # Arrow styling — bright green
    arrow_color = (0, 255, 0)
    arrow_thickness = 2
    arrow_tip = 0.35
    arrow_min_mag = float(args.arrow_thresh)
    # минимальная видимая длина "вытянутой" стрелки в пикселях; ниже
    # рисуем только маленький кружок-индикатор клетки (покой)
    arrow_visible_px = 4.0
    rest_dot_color = (0, 200, 0)
    # Скорость "вытягивания" при появлении движения (доля к target за кадр)
    extend_alpha = 0.55

    # Per-frame multiplicative decay для heat_speed на каждый heatmap.
    # Чем ближе к 1.0, тем медленнее цвет уползает с green -> blue -> red.
    # nodecay (None) -> ничего не затухает (накапливается max).
    speed_decay_map = {None: None, 0.90: 0.97, 0.95: 0.985, 0.98: 0.992, 0.99: 0.996}

    frame_idx = 0
    if args.every_n < 1:
        raise ValueError("--every_n must be >= 1")
    seconds_limit_frames = (
        int(round(args.preview_seconds * args.fps))
        if args.preview_seconds and args.preview_seconds > 0
        else 0
    )
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # skip early for speed
        if args.every_n > 1 and (frame_idx % args.every_n) != 0:
            frame_idx += 1
            continue
        if args.resize != 1.0:
            frame = cv2.resize(
                frame,
                (W, H),
                interpolation=cv2.INTER_AREA if args.resize < 1.0 else cv2.INTER_LINEAR,
            )
        frame_idx += 1
        if args.max_frames and frame_idx > args.max_frames:
            break
        if seconds_limit_frames and frame_idx > seconds_limit_frames:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
        )

        mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
        if args.mag_thresh > 0:
            mag[mag < args.mag_thresh] = 0.0

        # motion presence mask ("любое движение = зелёный")
        presence_now = (mag >= float(args.presence_thresh)).astype(np.float32)

        # --- HEATMAPS (5 videos), цвет = decayed peak SPEED -> red/blue/green
        for i, decay in enumerate(HEAT_DECAYS):
            if decay is None:
                heat_accums[i] += mag
            else:
                heat_accums[i] = heat_accums[i] * float(decay) + mag

            sd = speed_decay_map[decay]
            if sd is None:
                # nodecay: запоминаем максимальную скорость навсегда
                heat_speed[i] = np.maximum(heat_speed[i], mag)
            else:
                # затухание скорости: текущая = max(decayed_old, mag).
                # cмягчаем дрожание: ниже presence_thresh не обновляем.
                fresh = np.where(mag >= float(args.presence_thresh), mag, 0.0)
                heat_speed[i] = np.maximum(heat_speed[i] * float(sd), fresh)

            heat, alpha = heatmap_smooth_gradient(heat_speed[i])
            vis = alpha_blend(frame, heat, alpha)
            heat_writers[i].write(vis)

        # --- FLOW overlays (shared arrows; trails per decay)
        # --- Сеточные стрелки: усреднённый flow по каждой клетке ---
        # Берём верхний-левый кусок кадра кратный размеру клетки и блок-усреднение
        # через INTER_AREA даёт средний flow по каждой ячейке.
        flow_crop = flow[: grid_rows * cell, : grid_cols * cell]
        fx_grid = cv2.resize(flow_crop[..., 0], (grid_cols, grid_rows),
                             interpolation=cv2.INTER_AREA)
        fy_grid = cv2.resize(flow_crop[..., 1], (grid_cols, grid_rows),
                             interpolation=cv2.INTER_AREA)
        mag_grid = np.sqrt(fx_grid * fx_grid + fy_grid * fy_grid)
        active = mag_grid >= arrow_min_mag

        # Для каждого decay-варианта — обновляем состояние стрелок и рисуем
        arrows_per_decay = []
        for di, decay in enumerate(OPT_DECAYS):
            st = arrow_states[di]
            # Активные клетки: быстро тянемся к новому вектору
            st[..., 0] = np.where(active,
                                  st[..., 0] * (1 - extend_alpha) + fx_grid * extend_alpha,
                                  st[..., 0] * float(decay))
            st[..., 1] = np.where(active,
                                  st[..., 1] * (1 - extend_alpha) + fy_grid * extend_alpha,
                                  st[..., 1] * float(decay))

            arrows_layer = np.zeros_like(frame)
            arrows_mask_layer = np.zeros((H, W), dtype=np.uint8)

            for jy in range(grid_rows):
                for ix in range(grid_cols):
                    cx = int(grid_cx[ix]); cy = int(grid_cy[jy])
                    avx = float(st[jy, ix, 0]); avy = float(st[jy, ix, 1])
                    L = math.hypot(avx, avy) * args.scale
                    if L >= arrow_visible_px:
                        nx = int(round(cx + avx * args.scale))
                        ny = int(round(cy + avy * args.scale))
                        nx = max(0, min(W - 1, nx))
                        ny = max(0, min(H - 1, ny))
                        cv2.arrowedLine(
                            arrows_layer, (cx, cy), (nx, ny),
                            arrow_color, arrow_thickness,
                            line_type=cv2.LINE_AA, tipLength=arrow_tip,
                        )
                        cv2.line(arrows_mask_layer, (cx, cy), (nx, ny),
                                 255, arrow_thickness + 1)
                    else:
                        # покой — маленький кружок в центре клетки (исходное положение)
                        cv2.circle(arrows_layer, (cx, cy), 1, rest_dot_color, -1, cv2.LINE_AA)
                        arrows_mask_layer[cy, cx] = 255

            arrows_per_decay.append((arrows_layer, arrows_mask_layer))

        # Базовая (для combined) — берём средний вариант (decay 0.99 = индекс 1)
        arrows, arrows_mask = arrows_per_decay[1]

        # для combined берём heatmap с decay 0.95 (HEAT_DECAYS index 2) — он
        # как раз даёт fade ~3 сек: видим плавный переход green -> blue -> red.
        comb_heat, comb_alpha = heatmap_smooth_gradient(heat_speed[2])
        comb_heat_overlay = alpha_blend(frame, comb_heat, comb_alpha)

        def overlay_arrows(base, arrows_layer, mask_layer):
            out = base.copy()
            m = mask_layer > 0
            out[m] = arrows_layer[m]
            return out

        # per-decay flow + combined (сеточные стрелки, без trails-линий)
        for i, d in enumerate(OPT_DECAYS):
            arrows_layer, mask_layer = arrows_per_decay[i]
            flow_vis = overlay_arrows(frame, arrows_layer, mask_layer)
            flow_writers[i].write(flow_vis)

            comb = overlay_arrows(comb_heat_overlay, arrows_layer, mask_layer)
            comb_writers[i].write(comb)

        prev_gray = gray

    cap.release()
    for w in heat_writers:
        w.release()
    for w in flow_writers:
        w.release()
    for w in comb_writers:
        w.release()


if __name__ == "__main__":
    main()

