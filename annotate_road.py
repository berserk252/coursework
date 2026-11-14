"""
Ручная разметка частей дороги (полигонов) мышкой.

    python annotate_road.py --video ../traffic_flow_dataset/5_avariya/view3.mp4

Открывается кадр видео. По умолчанию это «пустая дорога»: медиана кадров по
всему ролику, машины на ней почти не видны, поэтому видна разметка полос.
Если разметка этого видео уже есть, она загружается, и её можно поправить.

Мышь:
    левая кнопка          -- добавить вершину в текущую часть
                             (рядом с готовой вершиной -- взять её и тащить)
    правая кнопка         -- удалить ближайшую вершину текущей части
Клавиши (работают и в русской раскладке):
    N / Enter / Пробел    -- закончить текущую часть, начать следующую
    Tab                   -- выбрать следующую часть для правки
    Z / Backspace         -- убрать последнюю вершину текущей части
    X                     -- удалить текущую часть целиком
    R                     -- стереть все части (разметить заново)
    B                     -- переключить фон: пустая дорога / кадр видео
    T                     -- фон: следы машин за весь ролик (видно, где ездят)
    A / D                 -- кадр видео на 2 с назад / вперёд
    S                     -- сохранить
    Q / Esc               -- выйти (если есть несохранённые правки -- нажать дважды)

Сохраняется в <parts_dir>/<имя видео>.json (по умолчанию ../разметка_дорог).
Разметка сразу для всех видео, по камерам, -- annotate_cameras.py (то же окно).

Проверить разметку без окна -- картинка с полигонами поверх кадра:
    python annotate_road.py --video ... --preview view3_разметка.png
"""

import argparse
import os

if os.environ.get("WAYLAND_DISPLAY"):
    os.environ["QT_QPA_PLATFORM"] = "xcb"        # у Qt из OpenCV есть только плагин xcb

import cv2
import numpy as np

from flow_canvas import Text
from road_parts import RoadParts, part_color

WIN = "Razmetka dorogi"
HELP = ["ЛКМ: вершина (у вершины — тащить)   ПКМ: удалить вершину",
        "N/Enter: следующая часть   Tab: выбрать часть   Z: отменить точку   X: удалить часть   R: стереть всё",
        "B: пустая дорога / кадр   T: следы машин   A/D: кадр -2/+2 с   S: сохранить   Q: выйти   H: скрыть подсказку"]

# клавиши в русской раскладке -> латинская буква на той же клавише
RU_LAT = dict(zip("йцукенгшщзхъфывапролджэячсмитьбю", "qwertyuiop[]asdfghjkl;'zxcvbnm,."))
KOI = "юабцдефгхийклмнопярстужвьызшэщчъ"              # порядок кодов кириллицы в X11 (0x6c0..)
SPECIAL = {27: "esc", 9: "tab", 13: "enter", 10: "enter", 8: "bksp", 32: "space",
           0xff1b: "esc", 0xff09: "tab", 0xff0d: "enter", 0xff8d: "enter", 0xff08: "bksp",
           0xff51: "left", 0xff53: "right", 0xff55: "pgup", 0xff56: "pgdn"}


def norm_key(k: int):
    """Код из cv2.waitKey -> латинская буква или имя клавиши ("enter", "left", ...);
    None -- ничего не нажато или клавиша не нужна."""
    if k < 0:
        return None
    if k in SPECIAL:
        return SPECIAL[k]
    if 0x6c0 <= k <= 0x6ff:                      # кириллица (код клавиши X11)
        ch = KOI[(k - 0x6c0) % 32]
    elif k <= 0x7e or 0x400 <= k <= 0x4ff:      # латиница или кириллица (юникод)
        ch = chr(k).lower()
    else:
        return None                              # Shift, F1 и прочие
    ch = RU_LAT.get(ch, ch)
    return ch if len(ch) == 1 and ch.isprintable() else None


def check_parts(rp: RoadParts, W: int, H: int, min_frac: float = 0.003):
    """Причина, по которой разметку нельзя сохранить, или None. Части не должны
    перекрываться: общий пиксель достаётся части с меньшим номером, и часть внутри
    другой в анализе оказывается пустой."""
    h = int(round(400 * H / W))
    lab = rp.label_map(400, h)
    for k in range(len(rp)):
        if (lab == k).mean() >= min_frac:
            continue
        own = np.zeros((h, 400), np.uint8)
        cv2.fillPoly(own, [rp.pixels(k, 400, h)], 1)
        if own.mean() < min_frac:
            return f"{rp.parts[k]['name']} слишком маленькая -- растяните её или удалите (Tab до неё, X)"
        j = int(np.bincount(lab[own > 0][lab[own > 0] >= 0]).argmax())
        return (f"{rp.parts[k]['name']} лежит внутри {rp.parts[j]['name'].replace('Часть', 'Части')} -- "
                f"части не должны перекрываться, обведите их отдельно")
    return None


def read_key(delay: int = 30):
    return norm_key(cv2.waitKey(delay))


def sample_frames(cap, t0: float, n: int, width: int = 0):
    """n кадров равномерно от t0 до конца ролика (уменьшенных до width, если задано)."""
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    for i in np.linspace(int(t0 * fps), max(int(t0 * fps), total - 2), n).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, f = cap.read()
        if ok:
            if width and f.shape[1] > width:
                f = cv2.resize(f, (width, int(round(f.shape[0] * width / f.shape[1]))), interpolation=cv2.INTER_AREA)
            frames.append(f)
    return frames


def read_frame(cap, t: float):
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, min(int(t * fps), n - 1)))
    ok, f = cap.read()
    return f if ok else None


def median_background(cap, t0: float, n: int = 40, width: int = 0):
    """Медиана n кадров от t0 до конца: движущиеся машины исчезают, остаётся дорога."""
    frames = sample_frames(cap, t0, n, width)
    return np.median(np.stack(frames), axis=0).astype(np.uint8) if frames else None


def car_trails(cap, t0: float, n: int = 60, width: int = 0):
    """Следы машин: каждый пиксель берётся из кадра, где он сильнее всего отличается
    от пустой дороги (медианы). Видно, где за ролик проезжали машины."""
    frames = sample_frames(cap, t0, n, width)
    if not frames:
        return None
    st = np.stack(frames)
    med = np.median(st, axis=0).astype(np.uint8)
    d = np.abs(st.astype(np.int16) - med.astype(np.int16)).sum(axis=-1)
    k = d.argmax(axis=0)
    comp = np.take_along_axis(st, k[None, ..., None].repeat(3, axis=-1), axis=0)[0]
    return np.where((d.max(axis=0) > 60)[..., None], comp, med)


class Source:
    """Фоны для разметки одного видео: кадр по времени, пустая дорога, следы машин.
    Пустая дорога и следы считаются при первом запросе."""

    def __init__(self, path: str, t0: float = 0.0, width: int = 1200):
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise SystemExit(f"не открывается: {path}")
        self.W = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.H = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.t0, self.width = t0, width
        self._median = self._trails = None

    def frame(self, t: float):
        return read_frame(self.cap, t)

    def median(self):
        if self._median is None:
            self._median = median_background(self.cap, self.t0, width=self.width)
        return self._median

    def trails(self):
        if self._trails is None:
            self._trails = car_trails(self.cap, self.t0, width=self.width)
        return self._trails


def render(base, rp: RoadParts, cur: int, txt: Text, lines=()):
    """Кадр с частями: заливка, контур, вершины; текущая часть -- толще; сверху строки текста."""
    img = base.copy()
    H, W = img.shape[:2]
    over = img.copy()
    for k, p in enumerate(rp.parts):
        if len(p["points"]) >= 3:
            cv2.fillPoly(over, [rp.pixels(k, W, H)], part_color(k))
    cv2.addWeighted(over, 0.25, img, 0.75, 0, dst=img)
    for k, p in enumerate(rp.parts):
        if not p["points"]:
            continue
        pts = rp.pixels(k, W, H)
        color = part_color(k)
        cv2.polylines(img, [pts], len(pts) >= 3, (255, 255, 255), 4 if k == cur else 3, cv2.LINE_AA)
        cv2.polylines(img, [pts], len(pts) >= 3, color, 2 if k == cur else 1, cv2.LINE_AA)
        for i, (x, y) in enumerate(pts):
            cv2.circle(img, (int(x), int(y)), 5 if k == cur else 3, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(img, (int(x), int(y)), 4 if k == cur else 2, color, -1, cv2.LINE_AA)
        c = pts.mean(axis=0).astype(int)
        name = p["name"] + ("  (текущая)" if k == cur else "")
        txt.draw(img, name, int(c[0]) + 1, int(c[1]) + 1, size=16, color=(255, 255, 255))
        txt.draw(img, name, int(c[0]), int(c[1]), size=16, color=color)
    y = 6
    for line in lines:
        if not line:
            continue
        w = txt.draw(np.zeros((1, 1, 3), np.uint8), line, 0, 0, size=14)
        cv2.rectangle(img, (4, y - 2), (12 + w, y + 18), (255, 255, 255), -1)
        txt.draw(img, line, 8, y, size=14, color=(30, 30, 30))
        y += 22
    return img


class Editor:
    """Окно правки частей дороги (одно на всю программу)."""

    def __init__(self, txt: Text, width: int = 1200):
        self.txt, self.width = txt, width
        self.read_key = read_key                 # подменяется в тестах
        self.rp, self.cur = None, 0
        self.DW = self.DH = 1
        self.drag, self.dirty = None, False
        cv2.namedWindow(WIN, cv2.WINDOW_AUTOSIZE | cv2.WINDOW_GUI_NORMAL)
        cv2.setMouseCallback(WIN, self.mouse)

    # --- мышь ----------------------------------------------------------------

    def _to_norm(self, x, y):
        return [min(max(x / self.DW, 0.0), 1.0), min(max(y / self.DH, 0.0), 1.0)]

    def _nearest(self, k, x, y, r=10):
        best, bi = r * r, None
        for i, (px, py) in enumerate(self.rp.parts[k]["points"]):
            d = (px * self.DW - x) ** 2 + (py * self.DH - y) ** 2
            if d <= best:
                best, bi = d, i
        return bi

    def mouse(self, ev, x, y, flags=0, param=None):
        if self.rp is None:
            return
        parts = self.rp.parts
        if ev == cv2.EVENT_LBUTTONDOWN:
            # вершина любой части рядом -- тащим её (и делаем эту часть текущей)
            for k in [self.cur] + [j for j in range(len(parts)) if j != self.cur]:
                i = self._nearest(k, x, y)
                if i is not None:
                    self.cur = k
                    self.drag = (k, i)
                    return
            parts[self.cur]["points"].append(self._to_norm(x, y))
            self.dirty = True
        elif ev == cv2.EVENT_MOUSEMOVE and self.drag is not None:
            k, i = self.drag
            parts[k]["points"][i] = self._to_norm(x, y)
            self.dirty = True
        elif ev == cv2.EVENT_LBUTTONUP:
            self.drag = None
        elif ev == cv2.EVENT_RBUTTONDOWN:
            i = self._nearest(self.cur, x, y, r=14)
            if i is not None:
                parts[self.cur]["points"].pop(i)
                self.dirty = True

    # --- окно ----------------------------------------------------------------

    def _show(self, base, lines):
        base = cv2.resize(base, (self.DW, self.DH), interpolation=cv2.INTER_AREA
                          if base.shape[1] > self.DW else cv2.INTER_CUBIC)
        cv2.imshow(WIN, render(base, self.rp, self.cur, self.txt, lines))

    def edit(self, rp: RoadParts, src: Source, on_save, header=(), help_lines=HELP,
             show: str = "frame", camera: bool = False, on_view=None) -> str:
        """Правка rp поверх кадров src.
        on_save(rp) -> текст статуса; не сохранилось -- ValueError с причиной.
        show -- начальный фон: "frame", "median" или "trails".
        camera -- режим по камерам: S сохраняет и возвращает "saved", ./]/-> -- "next",
            ,/[/<- -- "prev"; on_view(rp) -- клавиша V.
        Возвращает "saved", "next", "prev" или "quit"."""
        self.rp = rp
        if not rp.parts:
            rp.parts.append({"name": "Часть 1", "points": []})
        self.cur = len(rp.parts) - 1
        self.DW = self.width
        self.DH = int(round(src.H * self.width / src.W))
        self.drag, self.dirty = None, False
        armed = None                                  # клавиша, которую надо нажать ещё раз
        t = src.t0
        frame = src.frame(t)
        mode = show
        help_on = True
        status = ""
        names = {"frame": lambda: f"кадр {t:.0f} с", "median": lambda: "пустая дорога (медиана кадров)",
                 "trails": lambda: "следы машин за ролик"}
        while True:
            bases = {"median": src._median, "trails": src._trails, "frame": frame}
            base = bases.get(mode)
            if base is None and mode in ("median", "trails"):
                # считается несколько секунд -- сначала покажем, что идёт расчёт
                if frame is not None:
                    self._show(frame, list(header) + ["считаю фон: " + names[mode]() + " ..."])
                    cv2.waitKey(1)
                base = src.median() if mode == "median" else src.trails()
            if base is None:
                base = frame if frame is not None else np.zeros((src.H, src.W, 3), np.uint8)
            st = f"фон: {names[mode]()}   частей: {sum(len(p['points']) >= 3 for p in rp.parts)}   " + \
                 (status or ("есть несохранённые правки" if self.dirty else ""))
            self._show(base, list(header) + (list(help_lines) if help_on else []) + [st])
            key = self.read_key(30)
            if key is None:
                if cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1:
                    return "quit"
                continue
            status = ""
            if armed is not None and key != armed:
                armed = None
            go = None
            if key in ("q", "esc"):
                go = "quit"
            elif camera and key in (".", "]", "right", "pgdn"):
                go = "next"
            elif camera and key in (",", "[", "left", "pgup"):
                go = "prev"
            if go is not None:
                if self.dirty and armed != key:
                    armed = key
                    status = "не сохранено! S -- сохранить, ещё раз ту же клавишу -- без сохранения"
                    continue
                return go
            if key in ("n", "enter", "space"):
                if len(rp.parts[self.cur]["points"]) < 3:
                    status = "в части меньше 3 вершин -- сначала поставьте вершины"
                elif rp.parts[-1]["points"]:
                    rp.parts.append({"name": f"Часть {len(rp.parts) + 1}", "points": []})
                    self.cur = len(rp.parts) - 1
                else:
                    self.cur = len(rp.parts) - 1
            elif key == "tab":
                self.cur = (self.cur + 1) % len(rp.parts)
            elif key in ("z", "bksp"):
                if rp.parts[self.cur]["points"]:
                    rp.parts[self.cur]["points"].pop()
                    self.dirty = True
            elif key == "x":
                rp.parts.pop(self.cur)
                if not rp.parts:
                    rp.parts.append({"name": "Часть 1", "points": []})
                for k, p in enumerate(rp.parts):          # номера частей -- по порядку
                    p["name"] = f"Часть {k + 1}"
                self.cur = min(self.cur, len(rp.parts) - 1)
                self.dirty = True
            elif key == "r":
                rp.parts[:] = [{"name": "Часть 1", "points": []}]
                self.cur = 0
                self.dirty = True
            elif key == "b":
                mode = "frame" if mode == "median" else "median"
            elif key == "t":
                mode = "frame" if mode == "trails" else "trails"
            elif key in ("a", "d"):
                t = max(0.0, t + (2.0 if key == "d" else -2.0))
                frame = src.frame(t)
                mode = "frame"
            elif key == "h":
                help_on = not help_on
            elif key == "v" and on_view is not None:
                status = on_view(rp) or ""
            elif key == "s":
                good = [p for p in rp.parts if len(p["points"]) >= 3]
                if len(good) < 2:
                    status = "нужно хотя бы две части по 3+ вершины"
                    continue
                bad = check_parts(RoadParts(good), src.W, src.H)
                if bad:
                    status = "не сохранено: " + bad
                    continue
                try:
                    status = on_save(rp)
                except ValueError as e:
                    status = str(e)
                    continue
                self.dirty = False
                print(status)
                if camera:
                    return "saved"

    def close(self):
        cv2.destroyWindow(WIN)


def main():
    ap = argparse.ArgumentParser(description="Ручная разметка частей дороги полигонами.")
    ap.add_argument("--video", required=True)
    ap.add_argument("--parts_dir", default=os.path.join("..", "разметка_дорог"))
    ap.add_argument("--t0", type=float, default=0.0,
                    help="Начало сцены, с (для пустой дороги; например, после заставки)")
    ap.add_argument("--width", type=int, default=1200, help="Ширина окна, px")
    ap.add_argument("--preview", default="", help="Не открывать окно: сохранить картинку с разметкой и выйти")
    args = ap.parse_args()

    src = Source(args.video, args.t0, args.width)
    path = RoadParts.path_for(args.video, args.parts_dir)
    if os.path.exists(path):
        rp = RoadParts.load(path)
        print(f"загружена разметка: {path} ({len(rp)} частей)")
    else:
        rp = RoadParts([], os.path.basename(args.video), [src.W, src.H])
    rp.video, rp.frame_size = os.path.basename(args.video), [src.W, src.H]

    txt = Text()
    if args.preview:
        img = cv2.resize(src.median(), (args.width, int(round(src.H * args.width / src.W))),
                         interpolation=cv2.INTER_CUBIC)
        rp.draw(img, txt, fill_alpha=0.25, size=16)
        cv2.imwrite(args.preview, img)
        print(f"сохранено: {args.preview}")
        return

    def on_save(rp):
        rp.save(path)
        return f"сохранено: {path}"

    ed = Editor(txt, args.width)
    ed.edit(rp, src, on_save, show="median")
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
