"""
Видео "кадр + учебные графики поля", всё синхронно:

    ┌─────────────────────┬─────────────────────┐
    │ исходное видео      │ средняя скорость    │
    │ + стрелки поля      │ потока V(t),        │
    │                     │ div и |rot| по t    │
    ├─────────────────────┼─────────────────────┤
    │ белый холст:        │ белый холст:        │
    │ дивергенция         │ ротор               │
    └─────────────────────┴─────────────────────┘

Холсты нарисованы как в учебнике: стрелки -- поле скоростей по клеткам,
цвет -- величина (палитра RdBu, белый = 0), шкала справа. Серым -- контуры
кадра, чтобы было видно, где дорога.

    div < 0  красный -- машины накапливаются      div > 0  синий -- разъезжаются
    rot > 0  красный -- по часовой                rot < 0  синий -- против часовой

Поле -- очищенный оптический поток (flow_filters.FlowCleaner), дивергенция и
ротор -- divergence_curl.DivCurl, скорость потока -- flow_stats.SpeedMeter.

Выход -- каждый вид в своей папке, имя файла = имя исходного видео:
    combined/      все 4 части вместе
    video_arrows/  исходное видео + стрелки поля
    divergence/    белый холст: дивергенция
    curl/          белый холст: ротор
    graphs/        графики: средняя скорость потока V(t), div, |rot|
    corner/        исходное видео, мини-холсты div и rot в правом верхнем углу
    stats/         CSV по кадрам: скорость, падение скорости, доля движения,
                   средние и экстремумы div и rot
Для папки с видео структура подпапок (классы) повторяется внутри каждой.

Пример:
    python flow_canvas.py --video ../newVideos/1900-151662242.mp4 --out_dir out
"""

import argparse
import csv
import glob
import math
import os

import cv2
import numpy as np

from divergence_curl import DivCurl, default_cell
from flow_filters import FlowCleaner, SceneGuard
from flow_stats import IncidentRule, SpeedMeter
from flow_stopped import StoppedDetector

# ColorBrewer RdBu (11 цветов): от тёмно-красного через белый к тёмно-синему
_RDBU = ["#67001f", "#b2182b", "#d6604d", "#f4a582", "#fddbc7", "#f7f7f7",
         "#d1e5f0", "#92c5de", "#4393c3", "#2166ac", "#053061"]


def _rdbu_lut() -> np.ndarray:
    rgb = np.array([[int(h[i:i + 2], 16) for i in (1, 3, 5)] for h in _RDBU], np.float32)
    x = np.linspace(0, 1, len(_RDBU))
    t = np.linspace(0, 1, 256)
    lut = np.stack([np.interp(t, x, rgb[:, k]) for k in (2, 1, 0)], axis=1)  # BGR
    return lut.astype(np.uint8)


LUT = _rdbu_lut()


def colormap(t: np.ndarray) -> np.ndarray:
    """t в [-1, 1]: -1 красный, 0 белый, +1 синий."""
    idx = np.clip((t + 1.0) * 127.5, 0, 255).astype(np.uint8)
    return LUT[idx]


class Text:
    """Подписи по-русски через PIL (если есть шрифт), иначе латиницей через OpenCV."""

    FONTS = [
        "/usr/share/fonts/TTF/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/noto/NotoSans-Regular.ttf",
        "/usr/share/fonts/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "/Library/Fonts/Arial.ttf",
    ]

    def __init__(self):
        self.cache = {}
        self.font_path = None
        try:
            from PIL import Image, ImageDraw, ImageFont  # noqa: F401
            self._pil = (Image, ImageDraw, ImageFont)
            paths = list(self.FONTS)
            try:
                import matplotlib
                paths.insert(0, os.path.join(matplotlib.get_data_path(), "fonts/ttf/DejaVuSans.ttf"))
            except ImportError:
                pass
            self.font_path = next((p for p in paths if os.path.exists(p)), None)
        except ImportError:
            self._pil = None

    def _render(self, text, alt, size):
        key = (text, alt, size)
        if key in self.cache:
            return self.cache[key]
        if len(self.cache) > 3000:      # подписи с числами меняются каждый кадр
            self.cache.clear()
        if self._pil and self.font_path:
            Image, ImageDraw, ImageFont = self._pil
            font = ImageFont.truetype(self.font_path, size)
            l, t, r, b = font.getbbox(text)
            im = Image.new("L", (r - l + 2, b - t + 2), 0)
            ImageDraw.Draw(im).text((1 - l, 1 - t), text, fill=255, font=font)
            mask = np.asarray(im, np.float32) / 255.0
        else:
            s = alt or text.encode("ascii", "replace").decode()
            fs = size / 30.0
            (w, h), base = cv2.getTextSize(s, cv2.FONT_HERSHEY_SIMPLEX, fs, 1)
            m = np.zeros((h + base + 2, w + 2), np.uint8)
            cv2.putText(m, s, (1, h + 1), cv2.FONT_HERSHEY_SIMPLEX, fs, 255, 1, cv2.LINE_AA)
            mask = m.astype(np.float32) / 255.0
        self.cache[key] = mask
        return mask

    def draw(self, img, text, x, y, size=14, color=(0, 0, 0), alt=None, anchor="lt"):
        """anchor: l/c/r по горизонтали + t/b по вертикали. Возвращает ширину надписи."""
        m = self._render(text, alt, size)
        h, w = m.shape
        if anchor[0] == "c":
            x -= w // 2
        elif anchor[0] == "r":
            x -= w
        if anchor[1] == "b":
            y -= h
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(img.shape[1], x + w), min(img.shape[0], y + h)
        if x1 <= x0 or y1 <= y0:
            return w
        mm = m[y0 - y:y1 - y, x0 - x:x1 - x, None]
        reg = img[y0:y1, x0:x1].astype(np.float32)
        img[y0:y1, x0:x1] = (reg * (1 - mm) + np.array(color, np.float32) * mm).astype(np.uint8)
        return w


def num(img, s, x, y, size=0.38, color=(60, 60, 60), anchor="l"):
    """Числа -- через OpenCV, они ASCII и меняются каждый кадр."""
    (w, h), _ = cv2.getTextSize(s, cv2.FONT_HERSHEY_SIMPLEX, size, 1)
    if anchor == "r":
        x -= w
    elif anchor == "c":
        x -= w // 2
    cv2.putText(img, s, (int(x), int(y + h // 2)), cv2.FONT_HERSHEY_SIMPLEX, size, color, 1, cv2.LINE_AA)


def nice_ticks(lo, hi, n=4):
    span = hi - lo
    if span <= 0:
        return [lo]
    raw = span / n
    mag = 10 ** math.floor(math.log10(raw))
    step = min((s * mag for s in (1, 2, 2.5, 5, 10) if s * mag >= raw), default=10 * mag)
    start = math.ceil(lo / step) * step
    return [start + k * step for k in range(int((hi - start) / step + 1e-9) + 1)]


def sym_ticks(hi):
    """Подписи симметричной оси: -s, 0, s, где s -- наибольшее "круглое" число <= hi.
    nice_ticks на симметричной оси примерно в трети случаев давал одну подпись "0"."""
    if hi <= 0:
        return [0.0]
    mag = 10 ** math.floor(math.log10(hi))
    s = max(m * mag for m in (1, 2, 2.5, 5) if m * mag <= hi)
    return [-s, 0.0, s]


def robust_top(values):
    """Верх оси графика без выбросов: забор Тьюки q3 + 3*IQR, но не выше максимума.
    Раньше ось бралась по максимуму всей истории, и один выброс (заставка
    видео, склейка) навсегда сжимал настоящие данные в нижние 5-10% графика."""
    a = np.asarray(values, np.float64)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return 0.0
    mx = float(a.max())
    if a.size >= 20:
        q1, q3 = np.percentile(a, [25, 75])
        mx = min(mx, float(q3 + 3 * (q3 - q1)))
    return mx


def fmt(v):
    if v is None or not math.isfinite(v):
        return "-"
    if v == 0:
        return "0"
    if abs(v) >= 100 or abs(v) < 0.01:
        return f"{v:.0f}" if abs(v) >= 100 else f"{v:.1e}"
    return f"{v:.3g}"


# ---------------------------------------------------------------------------
#  Холст поля (дивергенция / ротор)
# ---------------------------------------------------------------------------

def field_canvas(P_W, P_H, frame, dc: DivCurl, which: str, txt: Text, edges=True, compact=False, st=None):
    """compact=True -- маленький холст для угла кадра: короткий заголовок,
    без подзаголовка и подписей шкалы. st -- состояние StoppedDetector:
    стоящие машины закрашиваются серым (у них поток 0, ротор и дивергенция
    их не видят)."""
    img = np.full((P_H, P_W, 3), 255, np.uint8)
    if which == "div":
        val, scale, sign = dc.div, dc.scale_div, 1.0
        title, alt = "Дивергенция", "Divergence"
        sub, sub_alt = "красный — поток сходится, синий — расходится", "red: converging, blue: diverging"
    else:
        val, scale, sign = dc.rot, dc.scale_rot, -1.0
        title, alt = "Ротор", "Curl"
        sub, sub_alt = "красный — по часовой, синий — против часовой", "red: clockwise, blue: counter-clockwise"

    if compact:
        fs = max(10, P_W // 13)
        top, cbw, pad = fs + 6, 14, 3
        short = "Дивергенция" if which == "div" else "Ротор"
        txt.draw(img, short, P_W // 2, 2, size=fs, alt="div" if which == "div" else "rot", anchor="ct")
    else:
        top, cbw, pad = 44, 70, 6
        txt.draw(img, title, P_W // 2, 4, size=16, alt=alt, anchor="ct")
        txt.draw(img, sub, P_W // 2, 24, size=12, color=(90, 90, 90), alt=sub_alt, anchor="ct")

    H, W = frame.shape[:2]
    aw, ah = P_W - cbw - 2 * pad, P_H - top - pad
    if aw < 8 or ah < 16:
        return img                      # холст слишком мал -- только заголовок
    s = min(aw / W, ah / H)
    pw, ph = int(W * s), int(H * s)
    if pw < 8 or ph < 16:
        return img
    x0, y0 = pad + (aw - pw) // 2, top + (ah - ph) // 2
    plot = np.full((ph, pw, 3), 255, np.uint8)

    if edges:
        g = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (pw, ph), interpolation=cv2.INTER_AREA)
        plot[cv2.Canny(g, 60, 150) > 0] = (205, 205, 205)

    cell = dc.cell
    if val is not None and scale > 0:
        gh, gw = val.shape
        vis = np.clip(dc.cover / 0.05, 0.0, 1.0)      # вне движения -- белый
        t = np.clip(sign * val / scale, -1.0, 1.0) * vis
        rw, rh = max(1, int(round(gw * cell * s))), max(1, int(round(gh * cell * s)))
        up = cv2.resize(t.astype(np.float32), (rw, rh), interpolation=cv2.INTER_CUBIC)
        rw, rh = min(rw, pw), min(rh, ph)
        up = np.clip(up[:rh, :rw], -1, 1)
        col = colormap(up)
        a = np.clip(np.abs(up) * 4.0, 0.0, 1.0)[..., None]   # почти ноль -- видно контуры
        reg = plot[:rh, :rw].astype(np.float32)
        plot[:rh, :rw] = (reg * (1 - a) + col.astype(np.float32) * a).astype(np.uint8)

    if st is not None and st.valid and st.mask.any():
        m = cv2.resize(st.mask.astype(np.uint8), (pw, ph), interpolation=cv2.INTER_NEAREST) > 0
        plot[m] = (0.45 * plot[m] + 0.55 * np.array([80, 80, 80])).astype(np.uint8)
        if not compact:
            cv2.rectangle(plot, (6, ph - 20), (18, ph - 8), (80, 80, 80), -1)
            txt.draw(plot, "стоящие машины", 22, ph - 22, size=11, color=(60, 60, 60), alt="stopped")

    # стрелки -- поле скоростей; на мини-холстах угловой панели их не рисуем:
    # там они короче 5 px и сливаются в штриховку
    if not compact and dc.grid is not None and dc.scale_arrow > 0:
        gh, gw = dc.grid.shape[:2]
        for jy in range(gh):
            for ix in range(gw):
                if not dc.active[jy, ix]:
                    continue
                L, du, dv = arrow_len(*dc.grid[jy, ix], dc)
                L *= s
                if L < 4:
                    continue
                cx, cy = (ix * cell + cell / 2) * s, (jy * cell + cell / 2) * s
                cv2.arrowedLine(plot, (int(cx), int(cy)), (int(cx + du * L), int(cy + dv * L)),
                                (0, 0, 0), 1, cv2.LINE_AA, tipLength=0.35)

    img[y0:y0 + ph, x0:x0 + pw] = plot
    cv2.rectangle(img, (x0 - 1, y0 - 1), (x0 + pw, y0 + ph), (150, 150, 150), 1)

    # шкала справа: сверху +scale, снизу -scale; в полном холсте шкала короче
    # картинки на 18 px -- под ней подпись множителя и единиц
    bx, bw = (x0 + pw + 4, 6) if compact else (x0 + pw + 12, 12)
    by0, by1 = (y0 + 4, y0 + ph - 4) if compact else (y0 + 4, y0 + ph - 18)
    if by1 - by0 < 4:
        return img
    vals = np.linspace(1, -1, by1 - by0)
    bar = colormap(sign * vals)[:, None, :].repeat(bw, axis=1)
    img[by0:by1, bx:bx + bw] = bar
    cv2.rectangle(img, (bx - 1, by0 - 1), (bx + bw, by1), (120, 120, 120), 1)
    if compact:
        num(img, "+", bx + bw + 1, by0 + 3, size=0.3)
        num(img, "-", bx + bw + 1, by1 - 3, size=0.3)
    elif scale > 0:
        e = math.floor(math.log10(scale))
        for v in (1, 0.5, 0, -0.5, -1):
            yy = int(by0 + (1 - v) / 2 * (by1 - by0 - 1))
            cv2.line(img, (bx + bw, yy), (bx + bw + 3, yy), (120, 120, 120), 1)
            num(img, f"{v * scale / 10 ** e:+.1f}" if v else "0", bx + bw + 5, yy)
        unit = "1/с" if getattr(dc, "fps", 1.0) != 1.0 else "1/кадр"
        mult = "" if e == 0 else f"×1e{e} "
        txt.draw(img, mult + unit, bx - 2, by1 + 3, size=11, color=(60, 60, 60),
                 alt=mult.replace("×", "x") + ("1/s" if unit == "1/с" else "1/fr"))
    return img


# ---------------------------------------------------------------------------
#  Графики по времени
# ---------------------------------------------------------------------------

def _segments(xs, ys):
    """Куски ряда без nan -- nan рисуется разрывом линии."""
    ok = np.isfinite(ys)
    segs, cur = [], []
    for x, y, o in zip(xs, ys, ok):
        if o:
            cur.append((x, y))
        elif cur:
            segs.append(cur)
            cur = []
    if cur:
        segs.append(cur)
    return segs


def line_plot(img, rect, t, series, title, alt, txt, y_sym=False, t_min=10.0, xlabels=True, y_min_top=0.0):
    x0, y0, x1, y1 = rect
    title_end = x0 + txt.draw(img, title, x0, y0 - 18, size=13, alt=alt)
    cv2.rectangle(img, (x0, y0), (x1, y1), (150, 150, 150), 1)
    if not t:
        return
    t_end = max(t[-1], t_min)
    ys = np.concatenate([np.asarray(v, np.float64) for v, _, _ in series])
    hi = max(robust_top(np.abs(ys) if y_sym else ys), y_min_top)
    hi = hi * 1.2 if hi > 0 else 1.0
    lo = -hi if y_sym else 0.0

    def X(tt):
        return x0 + (np.asarray(tt, np.float64) / t_end) * (x1 - x0)

    def Y(vv):
        v = np.clip(np.asarray(vv, np.float64), lo, hi)   # выше оси -- прижать к краю
        return y1 - (v - lo) / (hi - lo) * (y1 - y0)

    for tv in (sym_ticks(hi / 1.2) if y_sym else nice_ticks(lo, hi, 3)):
        yy = int(Y(tv))
        cv2.line(img, (x0, yy), (x1, yy), (230, 230, 230), 1)
        num(img, fmt(tv), x0 - 4, yy, anchor="r")
    for tt in nice_ticks(0, t_end, 5):
        xx = int(X(tt))
        cv2.line(img, (xx, y0), (xx, y1), (235, 235, 235), 1)
        if xlabels:
            num(img, f"{tt:g}", xx, y1 + 9, anchor="c")
    if xlabels:
        num(img, "t, s", x1, y1 + 20, anchor="r")
    cv2.rectangle(img, (x0, y0), (x1, y1), (150, 150, 150), 1)

    # сначала все линии, потом легенда -- в строке заголовка справа, вне графика
    tx = X(t)
    for vals, color, label in series:
        yv = np.asarray(vals, np.float64)
        for seg in _segments(tx, yv):
            pts = np.array([(x, Y(y)) for x, y in seg], np.int32)
            cv2.polylines(img, [pts], False, color, 2, cv2.LINE_AA)
        if np.isfinite(yv[-1]):
            cv2.circle(img, (int(tx[-1]), int(Y(yv[-1]))), 3, color, -1, cv2.LINE_AA)
    items = []
    for vals, color, label in series:
        s_ = f"{label} = {fmt(vals[-1])}"
        items.append((s_, color, cv2.getTextSize(s_, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0][0]))
    total = sum(w + 26 for _, _, w in items)
    # легенда в строке заголовка; если не помещается -- в правый верхний угол
    # графика на белой подложке
    inside = x1 - total < title_end + 8
    ly = y0 + 10 if inside else y0 - 10
    if inside:
        cv2.rectangle(img, (x1 - total - 4, y0 + 2), (x1 - 2, y0 + 18), (255, 255, 255), -1)
    xr = x1 - (6 if inside else 0)
    for s_, color, w in reversed(items):
        num(img, s_, xr - w, ly, size=0.4, color=color)
        cv2.line(img, (xr - w - 16, ly), (xr - w - 4, ly), color, 2)
        xr -= w + 26


def stats_canvas(P_W, P_H, sm: SpeedMeter, hdiv, hrot, hstop, txt: Text, inc=None):
    """Три графика по времени: скорость потока, стоящие машины, div и |rot|."""
    img = np.full((P_H, P_W, 3), 255, np.uint8)
    left, right = 52, P_W - 12
    free = P_H - 128                       # высота под три графика
    h1, h2 = int(free * 0.38), int(free * 0.22)
    r1 = (left, 26, right, 26 + h1)
    ys = r1[3] + 14                        # строка состояния
    r2 = (left, ys + 34, right, ys + 34 + h2)
    r3 = (left, r2[3] + 26, right, P_H - 26)
    line_plot(img, r1, sm.t, [(sm.hv, (180, 90, 20), "V, px/s")],
              "Средняя скорость потока V (за 1 с)", "Mean flow speed V (1 s)", txt, xlabels=False)
    # падение скорости -- косвенный признак ДТП
    if math.isfinite(sm.drop):
        color = (0, 0, 200) if sm.drop >= 0.5 else (60, 60, 60)
        drop_s = f"{sm.drop * 100:.0f}%"
    else:
        color, drop_s = (120, 120, 120), "-"
    x = left
    x += txt.draw(img, "падение скорости:", x, ys - 7, size=12, color=color, alt="drop:") + 6
    num(img, drop_s, x, ys, size=0.42, color=color)
    x += 60
    x += txt.draw(img, "в движении дороги:", x, ys - 7, size=12, color=(60, 60, 60), alt="moving:") + 6
    num(img, f"{sm.area * 100:.1f}%", x, ys, size=0.42)
    if inc is not None and inc.active:
        txt.draw(img, "ДТП?", right, ys - 9, size=15, color=(0, 0, 230), alt="ACCIDENT?", anchor="rt")
    line_plot(img, r2, sm.t, [(hstop, (40, 110, 230), "%")],
              "Стоящие машины, % дороги (застой)", "Stopped vehicles, % of road", txt,
              xlabels=False, y_min_top=1.0)
    line_plot(img, r3, sm.t, [(hdiv, (40, 40, 200), "div"), (hrot, (150, 60, 140), "|rot|")],
              "Дивергенция и |ротор| по дороге, 1/с", "div and |rot| on road, 1/s", txt, y_sym=True)
    return img


def arrow_len(u, v, dc: DivCurl, base: float = 1.1):
    """Длина стрелки ~ корень из скорости. При линейной длине ползущая полоса
    затора (в 20-30 раз медленнее соседней) давала стрелки короче 5 px, и их
    не было видно совсем. Направление стрелки -- точное."""
    m = math.hypot(u, v)
    if m <= 0 or dc.scale_arrow <= 0:
        return 0.0, 0.0, 0.0
    L = min(base * dc.cell * math.sqrt(m / dc.scale_arrow), 1.8 * dc.cell)
    return L, u / m, v / m


def draw_stopped(out, st, inc, txt: Text):
    """Стоящие машины -- оранжевая рамка и сколько секунд стоят; если по правилу
    это "ДТП?" -- красная толстая рамка с подписью."""
    if st is None or not st.valid:
        return
    red = set(map(tuple, inc.boxes)) if inc is not None else set()
    for b in st.blobs:
        x, y, w, h, _, age = b
        if tuple(b) in red:
            cv2.rectangle(out, (x, y), (x + w, y + h), (0, 0, 0), 5)
            cv2.rectangle(out, (x, y), (x + w, y + h), (0, 0, 230), 3)
            lab = f"ДТП? стоят {age:.0f} с"
            txt.draw(out, lab, x + 2, max(0, y - 20), size=14, color=(0, 0, 230), alt=f"ACCIDENT? {age:.0f}s")
        else:
            cv2.rectangle(out, (x, y), (x + w, y + h), (0, 0, 0), 3)
            cv2.rectangle(out, (x, y), (x + w, y + h), (0, 140, 255), 1)
            num(out, f"{age:.0f}s", x + 2, max(8, y - 6), size=0.38, color=(0, 140, 255))


def video_panel(frame, dc: DivCurl, txt: Text, st=None, inc=None):
    out = frame.copy()
    cell = dc.cell
    if dc.grid is not None and dc.scale_arrow > 0:
        for jy in range(dc.grid.shape[0]):
            for ix in range(dc.grid.shape[1]):
                if not dc.active[jy, ix]:
                    continue
                L, du, dv = arrow_len(*dc.grid[jy, ix], dc)
                if L < 5:
                    continue
                p0 = (int(ix * cell + cell / 2), int(jy * cell + cell / 2))
                p1 = (int(p0[0] + du * L), int(p0[1] + dv * L))
                # тёмная обводка -- стрелку видно и на светлом асфальте
                cv2.arrowedLine(out, p0, p1, (0, 0, 0), 3, cv2.LINE_AA, tipLength=0.35)
                cv2.arrowedLine(out, p0, p1, (0, 255, 0), 1, cv2.LINE_AA, tipLength=0.35)
    draw_stopped(out, st, inc, txt)       # рамки поверх стрелок
    return out


def make_corner(frame, dc: DivCurl, sm: SpeedMeter, txt: Text, frac: float = 0.42, st=None, inc=None):
    """Панель для правого верхнего угла: мини-холсты дивергенции и ротора,
    под ними -- средняя скорость потока и её падение. Возвращает (панель, x0, y0)
    или None, если кадр слишком мал для панели.
    Считается один раз на кадр и накладывается на сколько угодно выходных видео."""
    H, W = frame.shape[:2]
    cw = int(W * frac / 2)
    if cw < 60:
        return None
    ch = int(cw * H / W) + max(10, cw // 13) + 8
    line_h = max(16, cw // 9)
    if ch + line_h + 4 + 12 > H:
        return None
    div_c = field_canvas(cw, ch, frame, dc, "div", txt, compact=True, st=st)
    rot_c = field_canvas(cw, ch, frame, dc, "rot", txt, compact=True, st=st)
    panel = np.full((ch + line_h + 4, 2 * cw, 3), 255, np.uint8)
    panel[:ch, :cw] = div_c
    panel[:ch, cw:] = rot_c
    if math.isfinite(sm.drop):
        color = (0, 0, 200) if sm.drop >= 0.5 else (40, 40, 40)
        drop_s = f"падение {sm.drop * 100:.0f}%"
    else:
        color, drop_s = (110, 110, 110), "падение -"
    v_s = f"{sm.v:.0f}" if math.isfinite(sm.v) else "-"
    stop_s = f"   стоят {st.frac * 100:.1f}%" if st is not None and st.valid else ""
    txt.draw(panel, f"V = {v_s} px/s   {drop_s}{stop_s}", 4, ch + 2, size=max(10, line_h - 5), color=color,
             alt=f"V={v_s} px/s")
    cv2.line(panel, (cw, 0), (cw, ch), (180, 180, 180), 1)
    if inc is not None and inc.active:
        txt.draw(panel, "ДТП?", panel.shape[1] - 4, ch + 1, size=max(11, line_h - 3), color=(0, 0, 230),
                 alt="ACCIDENT?", anchor="rt")
        cv2.rectangle(panel, (0, 0), (panel.shape[1] - 1, panel.shape[0] - 1), (0, 0, 230), 3)
    else:
        cv2.rectangle(panel, (0, 0), (panel.shape[1] - 1, panel.shape[0] - 1), (120, 120, 120), 1)
    return panel, W - panel.shape[1] - 6, 6


def paste_corner(img, corner):
    if corner is None:
        return img
    panel, x0, y0 = corner
    out = img.copy()
    ph, pw = panel.shape[:2]
    out[y0:y0 + ph, x0:x0 + pw] = panel     # непрозрачно: надписи кадра не проступают
    return out


def corner_panel(frame, dc: DivCurl, sm: SpeedMeter, txt: Text, frac: float = 0.42, st=None, inc=None):
    """Кадр + панель с ротором, дивергенцией и скоростью в правом верхнем углу;
    рамки стоящих машин (красные -- "ДТП?") рисуются и здесь."""
    out = frame.copy()
    draw_stopped(out, st, inc, txt)
    return paste_corner(out, make_corner(frame, dc, sm, txt, frac, st=st, inc=inc))


# ---------------------------------------------------------------------------

OUTPUTS = ("combined", "video_arrows", "divergence", "curl", "graphs", "corner")


def process(video, out_root, rel, args, txt):
    stem = os.path.splitext(os.path.basename(video))[0]
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

    cleaner = FlowCleaner(egomotion=bool(args.egomotion))
    dc = DivCurl(cell=default_cell(W, args.cell), memory=args.memory, fps=fps)
    sm = SpeedMeter(fps, scale=scale)
    guard = SceneGuard(fps)
    guard.update(prev_gray)
    sd = StoppedDetector(fps)
    inc = IncidentRule(fps)
    hdiv, hrot, hstop = [], [], []
    n_cuts = 0

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
    table.writerow(["frame", "time_s", "valid", "cut", "speed_px_s", "speed_frame_px_s", "speed_base_px_s",
                    "drop", "moving_area"] + [k + "_per_s" for k in DivCurl.STAT_KEYS] + ["active_cells",
                    "stopped_frac", "occupied", "stopped_max_age_s", "incident"])

    limit = int(args.seconds * fps) if args.seconds > 0 else 0
    idx = 0
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
            # монтажная склейка: всё, что накоплено о прежней сцене, сбрасываем
            cleaner = FlowCleaner(egomotion=bool(args.egomotion))
            dc.reset()
            sm.reset()
            inc.reset()
            n_cuts += 1
        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        valid = valid and not cut
        c = cleaner.update(flow, gray, prev_gray)
        if valid:
            dc.update(c)
        # стоящие машины (застой): поток их не видит, детектор -- по отличию от фона
        st = sd.update(frame, c.moving, c.road_w, cut=cut, valid=valid)
        sm.update(c, t, valid=valid, occupied=st.occupied)
        inc.update(sm, st, t)               # правило "ДТП?"
        hdiv.append(dc.stats["div_mean"] if valid else float("nan"))
        hrot.append(dc.stats["rot_mean_abs"] if valid else float("nan"))
        hstop.append(st.frac * 100 if valid else float("nan"))
        table.writerow([idx, f"{t:.3f}", int(valid), int(cut), f"{sm.v:.3f}", f"{sm.v_frame:.3f}",
                        f"{sm.base:.3f}", f"{sm.drop:.4f}", f"{sm.area:.5f}"]
                       + [f"{dc.stats[k]:.6f}" for k in DivCurl.STAT_KEYS] + [dc.stats["active_cells"],
                          f"{st.frac:.5f}", int(st.occupied), f"{st.max_age:.1f}", int(inc.active)])

        prev_gray = gray
        if not writers:
            continue                    # --outputs none: только CSV
        parts = {
            "video_arrows": video_panel(frame, dc, txt, st=st, inc=inc),
            "graphs": stats_canvas(W, H, sm, hdiv, hrot, hstop, txt, inc=inc),
            "divergence": field_canvas(W, H, frame, dc, "div", txt, edges=bool(args.edges), st=st),
            "curl": field_canvas(W, H, frame, dc, "rot", txt, edges=bool(args.edges), st=st),
            "corner": corner_panel(frame, dc, sm, txt, st=st, inc=inc),
        }
        out = np.vstack([np.hstack([parts["video_arrows"], parts["graphs"]]),
                         np.hstack([parts["divergence"], parts["curl"]])])
        cv2.line(out, (W, 0), (W, 2 * H), (180, 180, 180), 1)
        cv2.line(out, (0, H), (2 * W, H), (180, 180, 180), 1)
        parts["combined"] = out
        for name, w in writers.items():
            w.write(parts[name])

    cap.release()
    for w in writers.values():
        w.release()
    f_csv.close()
    first = "" if math.isnan(inc.t_first) else f", ДТП? впервые на {inc.t_first:.1f} с"
    print(f"готово: {os.path.join(rel, stem)}  ({idx} кадров, склеек: {n_cuts}{first})")


def main():
    ap = argparse.ArgumentParser(description="Видео + учебные графики дивергенции и ротора, синхронно.")
    ap.add_argument("--video", required=True, help="Файл или папка (обработаются все .mp4 в ней и подпапках)")
    ap.add_argument("--out_dir", default="canvas_out")
    ap.add_argument("--max_width", type=int, default=640, help="Ширина обработки и одной части видео, px")
    ap.add_argument("--cell", type=int, default=24, help="Размер клетки поля при ширине 640 px (для других -- пропорционально)")
    ap.add_argument("--memory", type=float, default=0.9, help="Сглаживание поля во времени, 0..1")
    ap.add_argument("--egomotion", type=int, default=1)
    ap.add_argument("--edges", type=int, default=1, help="1 = серые контуры кадра на холстах")
    ap.add_argument("--seconds", type=float, default=0, help="Обработать только первые N секунд")
    ap.add_argument("--outputs", default="all",
                    help="Какие видео писать через запятую: " + ",".join(OUTPUTS) +
                         "; all -- все, none -- только CSV")
    ap.add_argument("--rel", default=".",
                    help="Для одного файла: подпапка внутри каждой папки результата (например класс)")
    args = ap.parse_args()
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
            process(v, args.out_dir, os.path.relpath(os.path.dirname(v), args.video), args, txt)
    else:
        process(args.video, args.out_dir, args.rel, args, txt)


if __name__ == "__main__":
    main()
