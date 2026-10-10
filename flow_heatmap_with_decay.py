import argparse
import math
import os
import cv2
import numpy as np

from divergence_curl import DivCurl, default_cell
from flow_canvas import Text, make_corner, paste_corner
from flow_filters import FlowCleaner, SceneGuard
from flow_stats import IncidentRule, SpeedMeter
from flow_stopped import StoppedDetector


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


def relative_gradient(t: np.ndarray, presence: np.ndarray, weight=None,
                      alpha_max: float = 0.55, floor: float = 0.0):
    """Цвет по ОТНОСИТЕЛЬНОЙ скорости t в [0..1]: red(медленно) -> blue -> green(быстро).

    Отличие от прежней heatmap_smooth_gradient: там шкала была абсолютной
    (low=0.5, high=3.0 пикселя за кадр) и одинаковой для всего кадра. Но
    из-за перспективы у дальнего края дороги машины идут 0.04 px/кадр, а у
    ближнего 6.18 -- разброс измерен в 73 раза на одном ролике и в 431 раз
    на другом. Любая абсолютная шкала красит всю дальнюю половину кадра
    красным независимо от того, есть там затор или нет.

    Здесь t = текущая скорость / собственная опорная скорость ЭТОЙ точки,
    поэтому красный означает "здесь едут медленнее, чем обычно едут здесь".
    Это заменяет гомографию: масштаб учитывается сам собой.

    Прозрачность берётся из presence -- доли времени с движением, поэтому
    пустой асфальт остаётся прозрачным, сколько бы ни шло видео.
    """
    t = np.clip(t, 0.0, 1.0).astype(np.float32)
    # Присутствие нормируем по своему же 99-му перцентилю -- робастная
    # нормализация, как в Heatmap.py. Без неё карта с большой памятью
    # (decay 0.99, ~1000 кадров) остаётся почти прозрачной первые полминуты:
    # среднее просто не успевает вырасти. Проверено: на 8-секундном отрезке
    # без нормировки окрашено 0% площади, с нормировкой -- видно сразу.
    p = np.asarray(presence, np.float32)
    live = p[p > 0]
    hi = float(np.percentile(live, 99)) if live.size else 0.0
    presence = (p / hi) if hi > 1e-9 else np.zeros_like(p)
    # Веса (маска дороги, уверенность, плавный старт) умножаются ПОСЛЕ
    # нормировки. Если умножить до, нормировка по перцентилю их отменяет:
    # маска на старте даёт 0.1, перцентиль тоже становится 0.1, и после
    # деления карта снова в полную силу. Так и было: на кадре 7 окраска
    # прыгала с 4% до 20% за один кадр.
    wgt = None if weight is None else np.clip(np.asarray(weight, np.float32), 0.0, 1.0)
    # Порог присутствия. Прозрачность считается как presence ** 0.6, а
    # степень меньше единицы УСИЛИВАЕТ малые значения: 0.05 ** 0.6 = 0.17.
    # Точка, где что-то мелькнуло 5% времени (размытие от соседней полосы,
    # остаток дрожания), получала заметную окраску. Ниже порога -- ноль,
    # выше -- плавный рост от нуля, без ступеньки.
    if floor > 0:
        presence = np.clip((presence - floor) / (1.0 - floor), 0.0, 1.0)
    upper = t >= 0.5
    t_up = (t - 0.5) * 2.0
    t_lo = t * 2.0
    B = np.where(upper, 255.0 * (1.0 - t_up), 255.0 * t_lo)
    G = np.where(upper, 255.0 * t_up, 0.0)
    R = np.where(upper, 0.0, 255.0 * (1.0 - t_lo))
    heat = np.stack([B, G, R], axis=-1).astype(np.uint8)
    alpha = np.clip(presence, 0.0, 1.0) ** 0.6 * alpha_max
    # Вес -- ПОСЛЕ гамма-коррекции. До неё он усиливается: 0.1 ** 0.6 = 0.25,
    # то есть маска на 10% давала четверть прозрачности. На пустой дороге это
    # незаметно, а на плотной пробке, где окрашено полкадра, был скачок 11
    # на кадре 7 при фоне исходника 5-7.
    if wgt is not None:
        alpha = alpha * wgt
    return heat, alpha.astype(np.float32)


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
    ap.add_argument("--fps", type=float, default=0.0,
                    help="FPS выходных видео. 0 = как у исходного видео.")
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
        default=0.0,
        help="Порог для стрелок в px/кадр. 0 = подбирать автоматически "
             "(по шумовому полу и доле от скорости в кадре) -- так стрелки "
             "появляются и на дальнем плане, где машины идут медленнее 1 px/кадр.",
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
    ap.add_argument(
        "--legacy",
        action="store_true",
        help="Старое поведение (удержание пика + абсолютная шкала). Нужно, "
             "чтобы показать сравнение 'до/после' в работе.",
    )
    ap.add_argument(
        "--deadband_rel",
        type=float,
        default=0.02,
        help="Порог не ниже этой доли от p95 скорости в кадре. Срезает "
             "размытие Farneback вокруг машин (оно пропорционально их скорости).",
    )
    ap.add_argument(
        "--deadband_k",
        type=float,
        default=2.0,
        help="Порог не ниже k * шумовой пол (оценивается автоматически).",
    )
    ap.add_argument(
        "--coh_beta",
        type=float,
        default=0.7,
        help="Память фильтра постоянства направления (0..1). 0.7 ~ 3 кадра.",
    )
    ap.add_argument(
        "--coh_min",
        type=float,
        default=0.7,
        help="Минимальная согласованность направления в точке (0..1). "
             "0 = фильтр выключен.",
    )
    ap.add_argument(
        "--change_thresh",
        type=float,
        default=6.0,
        help="Порог изменения яркости между кадрами (0..255). Где яркость не "
             "изменилась, там движения нет. 0 = фильтр выключен.",
    )
    ap.add_argument(
        "--presence_floor",
        type=float,
        default=0.12,
        help="Доля присутствия (0..1), ниже которой точка не окрашивается. "
             "Отсекает редкие случайные попадания.",
    )
    ap.add_argument(
        "--road_mask",
        type=int,
        default=1,
        help="1 = отбрасывать колебательное движение (деревья, кусты, блики): "
             "к трафику оно не относится.",
    )
    ap.add_argument(
        "--ref_decay",
        type=float,
        default=0.9995,
        help="Забывание опорной скорости для каждой точки. Ближе к 1 -- норма "
             "держится дольше и затор не успевает стать нормой.",
    )
    ap.add_argument(
        "--egomotion",
        type=int,
        default=1,
        help="1 = вычитать движение камеры (дрон, панорама, тряска). "
             "Включается само только там, где реально уменьшает фон.",
    )
    ap.add_argument(
        "--ema_alpha",
        type=float,
        default=1.0,
        help="Бегущее среднее потока по компонентам u, v: "
             "F = a*F_new + (1-a)*F_old. 1.0 = выключено.",
    )
    ap.add_argument(
        "--arrow_gate",
        type=int,
        default=1,
        help="1 = рисовать стрелку только в клетке, где после очистки есть "
             "движение и которая лежит на дороге (без стрелок на деревьях).",
    )
    ap.add_argument(
        "--panel",
        type=int,
        default=1,
        help="1 = в правом верхнем углу каждого видео панель: дивергенция, "
             "ротор и средняя скорость потока.",
    )
    ap.add_argument(
        "--stats_csv",
        type=int,
        default=1,
        help="1 = писать flow_stats.csv: по кадрам скорость потока, её падение, "
             "средние и экстремумы дивергенции и ротора.",
    )
    ap.add_argument(
        "--max_width",
        type=int,
        default=0,
        help="Если кадр шире -- уменьшить до этой ширины (0 = не трогать). "
             "Удобно для папки с видео разного размера.",
    )
    ap.add_argument(
        "--outputs",
        default="heat,flow,combined",
        help="Какие видео писать, через запятую: heat (5 heatmap), flow "
             "(3 opticalflow), combined (3 heatmap+opticalflow).",
    )
    ap.add_argument("--out_dir", default=".", help="Output directory")
    args = ap.parse_args()
    if os.environ.get("CV_THREADS"):
        cv2.setNumThreads(int(os.environ["CV_THREADS"]))

    out_dir = args.out_dir.rstrip("/")
    os.makedirs(out_dir, exist_ok=True)

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
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if args.fps <= 0:
        args.fps = src_fps
    if args.max_width and first.shape[1] * args.resize > args.max_width:
        args.resize = args.max_width / first.shape[1]

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
    # --step задан для полного кадра; при --max_width/--resize сетку стрелок
    # уменьшаем вместе с кадром, иначе на 640 px стрелок вдвое меньше
    cell = max(8, int(round(args.step * min(1.0, args.resize))))
    grid_cols = W // cell
    grid_rows = H // cell
    # Координаты центров клеток (в пикселях)
    grid_cx = (np.arange(grid_cols) * cell + cell // 2).astype(np.int32)
    grid_cy = (np.arange(grid_rows) * cell + cell // 2).astype(np.int32)
    # Состояние стрелки на каждый OPT_DECAYS — отдельная "память" сжатия
    arrow_states = [np.zeros((grid_rows, grid_cols, 2), dtype=np.float32)
                    for _ in OPT_DECAYS]

    # Состояния карт.
    #
    # Что убрано: heat_accums (накопление mag с затуханием) считался каждый
    # кадр для всех пяти вариантов и НИГДЕ не использовался -- цвет брался
    # из heat_speed. Пять полнокадровых массивов float32 обновлялись впустую.
    # heat_hi_ema был объявлен и не использован ни разу.
    #
    # Что изменено по существу: heat_speed был удержанием пика
    #     heat_speed = max(heat_speed * sd, mag)
    # то есть пиксель запоминал максимальную скорость, когда-либо через него
    # прошедшую, и забывал её за сотни кадров (sd=0.996 -- затухание в e раз
    # за 250 кадров). Любой пиксель, через который проехала машина быстрее
    # 3 px/кадр, оставался зелёным ~10 секунд, и за минуту карта заливалась
    # целиком. Измерено на 8 секундах этого же ролика: ярко окрашенная
    # площадь росла 14% -> 32% -> 43%, монотонно.
    #
    # Стало: нормированное бегущее среднее скорости, обновляемое ТОЛЬКО там,
    # где сейчас есть движение (иначе пустые кадры тянут среднее к нулю),
    # плюс отдельная карта присутствия для прозрачности.
    heat_speed = [np.zeros((H, W), dtype=np.float32) for _ in HEAT_DECAYS]
    heat_presence = [np.zeros((H, W), dtype=np.float32) for _ in HEAT_DECAYS]
    # Опорная скорость СВОЯ ДЛЯ КАЖДОЙ ТОЧКИ: верхняя огибающая наблюдённых
    # скоростей. Вверх реагирует сразу, вниз забывает медленно, поэтому
    # затор не успевает переопределить норму.
    heat_ref = [np.zeros((H, W), dtype=np.float32) for _ in HEAT_DECAYS]
    # Сколько раз обновлялось среднее скорости в каждой точке -- нужно для
    # поправки на смещение (см. ниже в цикле).
    heat_count = [np.zeros((H, W), dtype=np.float32) for _ in HEAT_DECAYS]
    frames_seen = 0

    nf = 0.0                      # оценка шумового пола, px/кадр
    # Вся очистка потока (камера, шум, дрожание, размытие, деревья) --
    # в flow_filters.py, общая для всех скриптов
    cleaner = FlowCleaner(
        egomotion=bool(args.egomotion),
        ema_alpha=float(args.ema_alpha),
        deadband_k=float(args.deadband_k),
        deadband_rel=float(args.deadband_rel),
        coh_beta=float(args.coh_beta),
        coh_min=float(args.coh_min),
        change_thresh=float(args.change_thresh),
        road_mask=bool(args.road_mask),
    )

    # Дивергенция, ротор и средняя скорость потока -- для панели в углу и CSV
    use_stats = not args.legacy and (args.panel or args.stats_csv)
    if use_stats:
        divcurl = DivCurl(cell=default_cell(W), memory=0.9, fps=src_fps / args.every_n)
        sm = SpeedMeter(src_fps / args.every_n, scale=args.resize)
        sd = StoppedDetector(src_fps / args.every_n)
        inc = IncidentRule(src_fps / args.every_n)
        txt = Text()
    corner = None
    if use_stats and args.stats_csv:
        import csv
        stats_file = open(f"{out_dir}/flow_stats.csv", "w", newline="")
        stats_table = csv.writer(stats_file)
        stats_table.writerow(["frame", "time_s", "valid", "cut", "speed_px_s", "speed_frame_px_s",
                              "speed_base_px_s", "drop", "moving_area"]
                             + [k + "_per_s" for k in DivCurl.STAT_KEYS]
                             + ["active_cells", "stopped_frac", "occupied", "stopped_max_age_s", "incident"])

    def put(img):
        return paste_corner(img, corner) if corner is not None else img

    class _Skip:
        """Видео, которое не просили (--outputs): кадры не пишутся."""
        def write(self, img):
            pass

        def release(self):
            pass

    wanted = {x.strip() for x in args.outputs.split(",") if x.strip()}
    unknown = wanted - {"heat", "flow", "combined"}
    if unknown or not wanted:
        ap.error(f"--outputs: неизвестные имена {sorted(unknown)}; допустимо heat, flow, combined")
    guard = SceneGuard(src_fps / args.every_n)
    guard.update(prev_gray)

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
        if "heat" not in wanted:
            heat_writers.append(_Skip())
            continue
        p = f"{out_dir}/{n}"
        w = cv2.VideoWriter(p, cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (W, H))
        if not w.isOpened():
            raise ValueError(f"Failed writer: {p}")
        heat_writers.append(w)

    flow_writers = []
    for n in flow_names:
        if "flow" not in wanted:
            flow_writers.append(_Skip())
            continue
        p = f"{out_dir}/{n}"
        w = cv2.VideoWriter(p, cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (W, H))
        if not w.isOpened():
            raise ValueError(f"Failed writer: {p}")
        flow_writers.append(w)

    comb_writers = []
    for n in combined_names:
        if "combined" not in wanted:
            comb_writers.append(_Skip())
            continue
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
    arrow_visible_px = max(2.0, 4.0 * min(1.0, args.resize))
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
        cut, valid = guard.update(gray)
        if cut and not args.legacy:
            # Монтажная склейка (заставка, смена ракурса, концовка): всё, что
            # накоплено о прежней сцене, к новой не относится -- сбрасываем.
            cleaner = FlowCleaner(
                egomotion=bool(args.egomotion), ema_alpha=float(args.ema_alpha),
                deadband_k=float(args.deadband_k), deadband_rel=float(args.deadband_rel),
                coh_beta=float(args.coh_beta), coh_min=float(args.coh_min),
                change_thresh=float(args.change_thresh), road_mask=bool(args.road_mask))
            for arr in heat_speed + heat_presence + heat_ref + heat_count + arrow_states:
                arr[...] = 0.0
            frames_seen = 0
            if use_stats:
                divcurl.reset()
                sm.reset()
                inc.reset()
        valid = valid and not cut
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
        )

        if args.legacy:
            # старое поведение -- только для сравнения "до/после"
            mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
            if args.mag_thresh > 0:
                mag[mag < args.mag_thresh] = 0.0
            mag_clean = mag
            moving = mag >= float(args.presence_thresh)
            road_w = np.ones((H, W), np.float32)
        else:
            # камера -> мёртвая зона -> постоянство направления -> изменение
            # яркости -> маска дороги; подробности и замеры -- в flow_filters.py
            c = cleaner.update(flow, gray, prev_gray)
            flow, mag, mag_clean, moving, road_w = c.flow, c.mag, c.mag_clean, c.moving, c.road_w
            nf = cleaner.noise
            if use_stats:
                if valid:
                    divcurl.update(c)
                # стоящие машины (застой): поток их не видит, детектор -- по отличию от фона
                st = sd.update(frame, c.moving, c.road_w, cut=cut, valid=valid)
                sm.update(c, frame_idx / src_fps, valid=valid, occupied=st.occupied)
                inc.update(sm, st, frame_idx / src_fps)       # правило "ДТП?"
                if args.panel:
                    corner = make_corner(frame, divcurl, sm, txt, st=st, inc=inc)
                if args.stats_csv:
                    stats_table.writerow(
                        [frame_idx, f"{frame_idx / src_fps:.3f}", int(valid), int(cut), f"{sm.v:.3f}",
                         f"{sm.v_frame:.3f}", f"{sm.base:.3f}", f"{sm.drop:.4f}", f"{sm.area:.5f}"]
                        + [f"{divcurl.stats[k]:.6f}" for k in DivCurl.STAT_KEYS]
                        + [divcurl.stats["active_cells"], f"{st.frac:.5f}", int(st.occupied), f"{st.max_age:.1f}",
                           int(inc.active)])

        moving_f = moving.astype(np.float32)
        frames_seen += 1

        # --- HEATMAPS (5 videos): цвет = скорость ОТНОСИТЕЛЬНО нормы этой точки
        for i, decay in enumerate(HEAT_DECAYS):
            if args.legacy:
                sd = speed_decay_map[decay]
                if sd is None:
                    heat_speed[i] = np.maximum(heat_speed[i], mag)
                else:
                    fresh = np.where(mag >= float(args.presence_thresh), mag, 0.0)
                    heat_speed[i] = np.maximum(heat_speed[i] * float(sd), fresh)
                heat, alpha = heatmap_smooth_gradient(heat_speed[i])
                heat_writers[i].write(alpha_blend(frame, heat, alpha))
                continue

            # nodecay трактуем как очень долгую память, а не как бесконечную:
            # бесконечная сумма насыщается всегда, вопрос только времени
            d = 0.999 if decay is None else float(decay)

            # среднее скорости обновляем только там, где сейчас есть движение
            heat_speed[i] = np.where(
                moving, (1.0 - d) * mag_clean + d * heat_speed[i], heat_speed[i]
            )
            heat_count[i] = heat_count[i] + moving_f
            # присутствие -- нормированное среднее, значит в 0..1 и без насыщения
            heat_presence[i] = (1.0 - d) * moving_f + d * heat_presence[i]
            # опорная скорость точки: вверх сразу, вниз очень медленно
            heat_ref[i] = np.maximum(heat_ref[i] * float(args.ref_decay), mag_clean)

            # ПОПРАВКА НА СМЕЩЕНИЕ (bias correction, как в оптимизаторе Adam).
            # Бегущее среднее, стартующее с нуля, после n обновлений равно
            #     E_n = (1 - d^n) * v
            # то есть занижено ровно в (1 - d^n) раз. При d=0.95 на первом
            # кадре это 0.05*v, а опорная скорость уже равна v -- отношение
            # 0.05, "очень медленно", красный. Измерено: до 51% кадра красного
            # в первые 0.1 с, гаснет за ~1.3 с (на heatmap_99 -- за 3+ с).
            # Деление на (1 - d^n) убирает смещение точно, с первого кадра.
            n = heat_count[i]
            speed_c = np.where(n > 0, heat_speed[i] / np.maximum(1.0 - d ** n, 1e-6), 0.0)
            pres_c = heat_presence[i] / max(1.0 - d ** frames_seen, 1e-6)

            t = speed_c / np.maximum(heat_ref[i], 1e-6)
            # новые точки проявляются за несколько кадров, а не вспыхивают
            confidence = np.clip(n / 5.0, 0.0, 1.0)
            heat, alpha = relative_gradient(t, pres_c, weight=road_w * confidence,
                                            floor=float(args.presence_floor))
            heat_writers[i].write(put(alpha_blend(frame, heat, alpha)))

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
        if args.legacy or arrow_min_mag > 0:
            active = mag_grid >= arrow_min_mag
        else:
            # Порог для стрелок тоже относительный. Абсолютные 1.5 px/кадр
            # означали, что на дальнем плане стрелок не будет никогда
            # (там машины идут 0.04-0.5 px/кадр), а на ближнем стрелки
            # рисуются и на размытии вокруг машины.
            thr = max(3.0 * nf, 0.10 * float(np.percentile(mag_grid, 95.0)))
            active = mag_grid >= max(thr, 1e-6)
            if args.arrow_gate:
                # Порог по средней скорости клетки пропускал деревья (они
                # реально качаются) и дрожание на разметке: все фильтры выше
                # чистили только heatmap, а стрелки брались из сырого потока.
                # Теперь клетка со стрелкой должна содержать хотя бы 1%
                # очищенного движения и лежать на дороге.
                moving_grid = cv2.resize(
                    moving[: grid_rows * cell, : grid_cols * cell].astype(np.float32),
                    (grid_cols, grid_rows), interpolation=cv2.INTER_AREA)
                road_grid = cv2.resize(
                    road_w[: grid_rows * cell, : grid_cols * cell],
                    (grid_cols, grid_rows), interpolation=cv2.INTER_AREA)
                active = active & (moving_grid >= 0.01) & (road_grid >= 0.5)

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
        if args.legacy:
            comb_heat, comb_alpha = heatmap_smooth_gradient(heat_speed[2])
        else:
            dc = float(HEAT_DECAYS[2])
            nc = heat_count[2]
            comb_speed = np.where(nc > 0, heat_speed[2] / np.maximum(1.0 - dc ** nc, 1e-6), 0.0)
            comb_pres = heat_presence[2] / max(1.0 - dc ** frames_seen, 1e-6)
            comb_t = comb_speed / np.maximum(heat_ref[2], 1e-6)
            comb_heat, comb_alpha = relative_gradient(
                comb_t, comb_pres, weight=road_w * np.clip(nc / 5.0, 0.0, 1.0),
                floor=float(args.presence_floor))
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
            flow_writers[i].write(put(flow_vis))

            comb = overlay_arrows(comb_heat_overlay, arrows_layer, mask_layer)
            comb_writers[i].write(put(comb))

        prev_gray = gray

    cap.release()
    for w in heat_writers:
        w.release()
    for w in flow_writers:
        w.release()
    for w in comb_writers:
        w.release()
    if use_stats and args.stats_csv:
        stats_file.close()


if __name__ == "__main__":
    main()

