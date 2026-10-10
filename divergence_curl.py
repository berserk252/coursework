"""
Дивергенция и ротор векторного поля трафика.

Поле V(x, y) = (u, v) -- очищенный оптический поток (flow_filters.FlowCleaner:
без движения камеры, шума, дрожания, размытия и деревьев).

    div V = du/dx + dv/dy        rot V = dv/dx - du/dy

Производные -- центральными разностями (np.gradient), шаг h = размер клетки.

ПОЧЕМУ НЕ ПО ПИКСЕЛЯМ, А ПО КЛЕТКАМ
На уровне пикселей каждая машина -- это пятно скорости на нулевом фоне.
На переднем краю пятна u падает до нуля (div < 0), на заднем растёт от нуля
(div > 0), и каждая машина рисуется "диполем" -- это граница машины, а не
поведение потока. Поэтому поле усредняем по клеткам cell x cell ВМЕСТЕ с
нулями фона:

    q = средняя скорость в клетке = скорость машин * доля клетки, занятая машинами

Это плотность потока транспорта (машин в единицу времени через сечение), и
для неё действует уравнение неразрывности:

    d(плотность)/dt = -div q

То есть div q < 0 -- машин в зону въезжает больше, чем выезжает: НАКОПЛЕНИЕ,
образование затора. div q > 0 -- зона разгружается, затор рассасывается.
Затем поле сглаживается во времени (--memory), чтобы проезд одной машины
не давал вспышку.

ОГОВОРКА ПРО ПЕРСПЕКТИВУ (проверено на реальных видео)
Это поле в пикселях кадра, а не на плане дороги. Машина, уезжающая от камеры,
в кадре замедляется и уменьшается -- поток по её полосе "сходится" (div < 0)
даже при свободном движении; приезжающая -- "расходится". Кроме того, клетка
24 px и память ~10 кадров не до конца убирают "диполь" каждой машины (спереди
сходится, сзади расходится). Поэтому мгновенный знак div сам по себе ещё не
означает затор: признаком служит ИЗМЕНЕНИЕ div в клетке относительно её же
обычного значения на этом видео.

ЗНАК РОТОРА
Ось y на изображении направлена вниз, поэтому rot > 0 -- вращение ПО часовой
стрелке на экране, rot < 0 -- против. Большой |rot| -- поток поворачивает:
перестроения, объезд препятствия, развязка, круговое движение.

Выход:
    divergence.mp4   красный = сходится (накопление), синий = расходится
    curl.mp4         красный = по часовой, синий = против часовой
    div_curl.csv     по кадрам: средние и экстремумы div и rot по дороге

Пример:
    python divergence_curl.py --video ../newVideos/1900-151662242.mp4 --resize 0.5
"""

import argparse
import csv
import os

import cv2
import numpy as np

from flow_filters import FlowCleaner, SceneGuard


def pool(field: np.ndarray, cell: int) -> np.ndarray:
    """Среднее по клеткам cell x cell, нули фона входят в среднее."""
    h, w = field.shape[:2]
    gh, gw = h // cell, w // cell
    f = field[: gh * cell, : gw * cell]
    return f.reshape(gh, cell, gw, cell, -1).mean(axis=(1, 3))


def divergence(g: np.ndarray, h: float) -> np.ndarray:
    return np.gradient(g[..., 0], h, axis=1) + np.gradient(g[..., 1], h, axis=0)


def curl(g: np.ndarray, h: float) -> np.ndarray:
    return np.gradient(g[..., 1], h, axis=1) - np.gradient(g[..., 0], h, axis=0)


def colorize(val: np.ndarray, scale: float, neg_color, pos_color, alpha_max: float = 0.6):
    """Двухцветная карта: цвет -- знак, прозрачность -- величина."""
    t = np.clip(np.abs(val) / max(scale, 1e-12), 0.0, 1.0)
    heat = np.where((val < 0)[..., None], np.array(neg_color, np.float32),
                    np.array(pos_color, np.float32))
    return heat, (t ** 0.7 * alpha_max).astype(np.float32)


def to_frame(grid: np.ndarray, cell: int, H: int, W: int) -> np.ndarray:
    """Клетки -> пиксели кадра, плавно (а не квадратами)."""
    gh, gw = grid.shape[:2]
    up = cv2.resize(grid.astype(np.float32), (gw * cell, gh * cell), interpolation=cv2.INTER_CUBIC)
    out = np.zeros((H, W) + up.shape[2:], np.float32)
    out[: up.shape[0], : up.shape[1]] = up
    return out


def blend(frame: np.ndarray, heat: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    a = alpha[..., None]
    return np.clip(frame.astype(np.float32) * (1 - a) + heat * a, 0, 255).astype(np.uint8)


def draw_arrows(img, g, cell, scale, active):
    for jy in range(g.shape[0]):
        for ix in range(g.shape[1]):
            if not active[jy, ix]:
                continue
            u, v = g[jy, ix]
            L = np.hypot(u, v) * scale
            if L < 3:
                continue
            cx, cy = ix * cell + cell // 2, jy * cell + cell // 2
            cv2.arrowedLine(img, (cx, cy), (int(round(cx + u * scale)), int(round(cy + v * scale))),
                            (255, 255, 255), 1, cv2.LINE_AA, tipLength=0.3)


def legend(img, lines):
    for k, (text, color) in enumerate(lines):
        y = 22 + 22 * k
        cv2.putText(img, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)


class DivCurl:
    """Дивергенция и ротор очищенного поля по клеткам, со сглаживанием во времени.

        dc = DivCurl(cell=24, memory=0.9)
        c = cleaner.update(flow, gray, prev_gray)     # FlowCleaner
        dc.update(c)
        dc.div, dc.rot      -- карты (Hc, Wc)
        dc.grid             -- поле клеток (Hc, Wc, 2)
        dc.cover, dc.active -- доля движения в клетке и клетки, где оно недавно было
        dc.stats            -- средние и экстремумы по активным клеткам
        dc.scale_div, dc.scale_rot -- шкалы для цвета
    """

    STAT_KEYS = ("div_mean", "div_min", "div_max", "rot_mean_abs", "rot_max_abs")

    def __init__(self, cell: int = 24, memory: float = 0.9, fps: float = 1.0):
        """fps -- частота кадров: div и rot переводятся из 1/кадр в 1/с, чтобы
        видео с разным fps были сравнимы. fps=1 -- оставить 1/кадр."""
        self.cell = int(cell)
        self.memory = float(memory)
        self.fps = float(fps)
        self.reset()

    def reset(self) -> None:
        """Сброс после монтажной склейки: старое поле к новой сцене не относится."""
        self.state = None     # поле клеток, сглаженное во времени
        self.cover = None     # доля движения в клетке, сглаженная во времени
        self.scale_div = self.scale_rot = self.scale_arrow = 0.0
        self.grid = self.div = self.rot = self.active = None
        self.stats = dict.fromkeys(self.STAT_KEYS, 0.0)
        self.stats["active_cells"] = 0

    @staticmethod
    def _envelope(prev: float, x: float) -> float:
        """Шкала цвета: вниз медленно (0.995 за кадр), вверх -- не скачком, а на
        15% разницы за кадр. Раньше шкала прыгала вверх сразу, и один выброс
        (склейка, заставка) делал все следующие кадры бледными на десятки секунд."""
        if prev <= 0:
            return x
        if x > prev:
            return prev + 0.15 * (x - prev)
        return max(x, 0.995 * prev)

    def update(self, c) -> "DivCurl":
        cell, m = self.cell, self.memory
        # Деревья здесь нужно убрать из самих измерений: качание даёт
        # большой ротор. Маска плавная, поэтому скачка при её появлении нет.
        field = c.field() * c.road_w[..., None]
        grid = pool(field, cell)
        cov = pool((c.moving * c.road_w)[..., None].astype(np.float32), cell)[..., 0]
        self.state = grid if self.state is None else m * self.state + (1 - m) * grid
        self.cover = cov if self.cover is None else m * self.cover + (1 - m) * cov

        g = cv2.GaussianBlur(self.state, (0, 0), 1.0)
        self.grid = g
        self.div = divergence(g, cell) * self.fps
        self.rot = curl(g, cell) * self.fps

        # статистика только по клеткам, где недавно было движение
        self.active = self.cover >= 0.02
        if self.active.any():
            dv, rv = self.div[self.active], self.rot[self.active]
            vals = (dv.mean(), dv.min(), dv.max(), np.abs(rv).mean(), np.abs(rv).max())
            self.stats = {k: float(v) for k, v in zip(self.STAT_KEYS, vals)}
            # шкала цвета -- верхняя огибающая p99: вверх сразу, вниз медленно,
            # иначе цвета прыгают от кадра к кадру
            self.scale_div = self._envelope(self.scale_div, float(np.percentile(np.abs(dv), 99)))
            self.scale_rot = self._envelope(self.scale_rot, float(np.percentile(np.abs(rv), 99)))
            speed = np.linalg.norm(g, axis=2)[self.active]
            self.scale_arrow = self._envelope(self.scale_arrow, float(np.percentile(speed, 95)))
        else:
            self.stats = dict.fromkeys(self.STAT_KEYS, 0.0)
        self.stats["active_cells"] = int(self.active.sum())
        return self


def default_cell(width: int, base: int = 24) -> int:
    """Размер клетки по ширине кадра: 24 px при 640, пропорционально для других.
    Одно правило для flow_canvas.py и flow_heatmap_with_decay.py, чтобы их CSV совпадали."""
    return max(12, int(round(base * width / 640)))


def main():
    ap = argparse.ArgumentParser(description="Дивергенция и ротор поля трафика (видео + CSV).")
    ap.add_argument("--video", default="video.webm")
    ap.add_argument("--out_dir", default=".")
    ap.add_argument("--cell", type=int, default=24, help="Размер клетки, px. Больше -- глаже поле.")
    ap.add_argument("--memory", type=float, default=0.9,
                    help="Сглаживание поля клеток во времени, 0..1. 0.9 ~ 10 кадров.")
    ap.add_argument("--resize", type=float, default=1.0)
    ap.add_argument("--max_frames", type=int, default=0)
    ap.add_argument("--egomotion", type=int, default=1)
    ap.add_argument("--road_mask", type=int, default=1)
    ap.add_argument("--arrows", type=int, default=1, help="1 = рисовать поле клеток стрелками.")
    args = ap.parse_args()
    out_dir = args.out_dir.rstrip("/")
    os.makedirs(out_dir, exist_ok=True)

    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    ret, frame = cap.read()
    if not ret:
        raise ValueError("Cannot read video")

    def prep(f):
        if args.resize != 1.0:
            f = cv2.resize(f, None, fx=args.resize, fy=args.resize,
                           interpolation=cv2.INTER_AREA if args.resize < 1.0 else cv2.INTER_LINEAR)
        return f

    frame = prep(frame)
    H, W = frame.shape[:2]
    cell = int(args.cell)
    prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cleaner = FlowCleaner(egomotion=bool(args.egomotion), road_mask=bool(args.road_mask))

    writers = {}
    for name in ("divergence", "curl"):
        p = f"{out_dir}/{name}.mp4"
        w = cv2.VideoWriter(p, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
        if not w.isOpened():
            raise ValueError(f"Failed writer: {p}")
        writers[name] = w
    csv_file = open(f"{out_dir}/div_curl.csv", "w", newline="")
    table = csv.writer(csv_file)
    table.writerow(["frame", "time_s", "div_mean", "div_min", "div_max",
                    "rot_mean_abs", "rot_max_abs", "active_cells"])

    dc = DivCurl(cell=cell, memory=float(args.memory), fps=fps)
    guard = SceneGuard(fps)
    guard.update(prev_gray)
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        if args.max_frames and frame_idx > args.max_frames:
            break
        frame = prep(frame)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        cut, valid = guard.update(gray)
        if cut:
            cleaner = FlowCleaner(egomotion=bool(args.egomotion), road_mask=bool(args.road_mask))
            dc.reset()
        c = cleaner.update(flow, gray, prev_gray)

        if valid and not cut:
            dc.update(c)
        div, rot, g, cover, active = dc.div, dc.rot, dc.grid, dc.cover, dc.active
        st = dc.stats
        table.writerow([frame_idx, f"{frame_idx / fps:.3f}"]
                       + [f"{st[k]:.6f}" for k in DivCurl.STAT_KEYS] + [st["active_cells"]])

        # видимость: только там, где есть движение, без резкой границы
        vis_w = to_frame(np.clip(cover / 0.05, 0, 1), cell, H, W)
        arrow_k = 0.8 * cell / dc.scale_arrow if dc.scale_arrow > 0 else 0.0

        for name, val, scale, neg, pos, text in (
            ("divergence", div, dc.scale_div, (0, 0, 255), (255, 0, 0),
             [("div < 0: converging (accumulation)", (80, 80, 255)),
              ("div > 0: diverging (dissolving)", (255, 140, 60))]),
            ("curl", rot, dc.scale_rot, (255, 0, 0), (0, 0, 255),
             [("rot > 0: clockwise", (80, 80, 255)),
              ("rot < 0: counter-clockwise", (255, 140, 60))]),
        ):
            heat, alpha = colorize(to_frame(val, cell, H, W), scale, neg, pos)
            out = blend(frame, heat, alpha * vis_w)
            if args.arrows and arrow_k > 0:
                draw_arrows(out, g, cell, arrow_k, active)
            legend(out, text)
            writers[name].write(out)

        prev_gray = gray

    cap.release()
    for w in writers.values():
        w.release()
    csv_file.close()


if __name__ == "__main__":
    main()
