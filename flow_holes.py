"""
Дырки в векторном поле: место внутри части дороги, где обычно ездят, а сейчас
давно нет движения, хотя вокруг в той же части машины едут как обычно. Это
предположительно ДТП или что-то, что мешает движению. Анализ только по полю
(оптический поток), без детекции машин.

У стоящего объекта скорость 0, поэтому в поле на его месте пусто -- как на
пустом асфальте. Одного нуля мало: полоса без машин тоже ноль. Поэтому для
каждой клетки считаем, как часто через неё проезжают машины (проезд -- начало
движения в клетке после паузы):

    gap   -- обычный интервал между машинами: (последнее движение - начало сцены) / проездов.
            Текущий простой в него не входит, иначе за минуту простоя дырка
            "привыкла" бы к нулю; время до первого проезда входит -- иначе
            редкая полоса с тремя проездами подряд выглядела бы плотной;
    r_now -- частота проездов сейчас (бегущее среднее, память now_s ~ 10 с).

Клетка -- кандидат в дырку, если
    * это полоса: проездов было не меньше min_passes;
    * давно не ездят: движения нет не меньше idle_min_s секунд и не меньше
      gap_k обычных интервалов между машинами (gap) -- на редкой полосе
      пауза в 10 с нормальна, на плотной нет.
Пустая полоса тоже подходит под это. Отличие препятствия: машины к нему
ПОДЪЕЗЖАЮТ, но через него не проезжают (объезжают или встают). Поэтому
кандидат -- зерно дырки, если с его последнего движения в клетках выше по
течению (против обычного направления потока, на 1..up_cells клеток, та же
часть дороги) начались проезды -- не меньше arr_min (счётчик забывает старые
подъезды с постоянной arr_tau_s). Дырка -- зёрна и соседние кандидаты не дальше
grow_px от зерна: так в неё попадает весь стоящий объект, а не только край,
к которому подъезжают.
Новая дырка принимается, если
    * вокруг ездят как обычно: в кольце шириной ring_px вокруг неё (та же часть)
      сумма r_now / сумма 1/gap >= ring_ratio. Если поток поредел везде или
      встал весь -- это не дырка;
    * она не больше max_part_frac части: встала почти вся часть -- это пробка
      или опустевшая полоса, а не место на дороге;
    * она заходит внутрь части хотя бы на edge_px от её границы (край кадра
      границей не считается): узкая полоска вдоль линии между частями -- это
      разметка, по которой не ездят, а край соседней машины иногда задевает.
Принятая дырка держится, пока в её клетках нет движения (машины вокруг могли
встать в пробку, а препятствие стоит). Показывается дырка, только когда она
продержалась confirm_s секунд: редкая полоса, к которой подъехала машина,
выглядит дыркой, пока машина не проедет её насквозь, -- это 1-2 секунды.

Перекрытие (по C. Cuevas, R. Martínez, D. Berjón, N. García, "Detection of
Stationary Foreground Objects Using Multiple Nonparametric Background-Foreground
Models on a Finite State Machine", UPM, 2017, https://oa.upm.es/50852/ --
состояние "occluded stationary foreground object").
Стоящую машину на снимке сверху то и дело закрывают машины соседних полос,
рядом ходят люди, открываются двери -- в клетке появляется движение. Раньше
любое движение выкидывало клетку из дырки и обнуляло её "нет движения".
Теперь у клетки дырки три состояния, как в конечном автомате статьи:
    дырка -> (в клетке движение) -> дырка, перекрыта -> (движение кончилось) -> дырка
    дырка, перекрыта -> (движение идёт непрерывно дольше occl_max_s) -> свободна
Пока клетка перекрыта, она остаётся в дырке и её время без движения не
сбрасывается; отпускает её только непрерывное движение -- через неё снова едут.
Перекрытие допускается только у подтверждённой дырки (продержалась confirm_s
без единого движения): иначе редкая полоса, через которую машины проезжают
короткими эпизодами, держалась бы дыркой как "перекрытая".

Контур (по истории движения пикселей -- Motion History Image, Bobick & Davis,
"The Recognition of Human Movement Using Temporal Templates", IEEE PAMI 2001;
в OpenCV -- cv::motempl::updateMotionHistory). Клетки 12 px дают ступенчатый
контур. Для каждого блока block_px (4 px при 640) хранится время последнего
движения; контур дырки -- клетки дырки плюс соседние блоки, где движения не было
не меньше idle_min_s, затем морфологическое закрытие и отбрасывание мелочи.
Так граница проходит по месту, где реально ездили, с точностью до блока.

Когда в дырке нет движения не меньше incident_s секунд -- "ДТП?".

Клетки -- квадраты cell_px при ширине кадра 640 (для других -- пропорционально).
Дырка выдаётся полигоном (контур области), а не прямоугольником.

    hd = HoleDetector(fps, labels, scale)      # labels -- RoadParts.label_map(W, H)
    st = hd.update(c, t, valid=valid, cut=cut) # c -- CleanFlow (flow_filters)
    for h in st.holes:
        h.polygons       -- контуры дырки в пикселях кадра обработки
        h.idle_s         -- сколько секунд в дырке нет движения (медиана по клеткам;
                            короткие перекрытия его не сбрасывают)
        h.incident       -- нет движения >= incident_s -> "ДТП?"
        h.v_in, h.v_ring -- скорость поля в дырке и вокруг неё, px/с исходного видео
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np

NAN = float("nan")


def pool_mean(a: np.ndarray, cell: int) -> np.ndarray:
    """Среднее по клеткам cell x cell (остаток справа и снизу отбрасывается)."""
    h, w = a.shape[:2]
    gh, gw = h // cell, w // cell
    return a[: gh * cell, : gw * cell].reshape(gh, cell, gw, cell).mean(axis=(1, 3))


@dataclass
class Hole:
    id: int
    part: int                         # номер части дороги
    cells: np.ndarray                 # (gh, gw) bool
    polygons: list                    # [np.ndarray (N, 2) int32] в пикселях кадра обработки
    area_px: int = 0
    idle_s: float = 0.0               # сколько нет движения, медиана по клеткам дырки
    age_s: float = 0.0                # сколько существует дырка
    ring_ratio: float = 0.0           # вокруг: частота проездов сейчас / обычная
    v_in: float = NAN                 # средняя скорость поля в дырке, px/с исходного видео
    v_ring: float = NAN               # скорость движения вокруг (медиана), px/с
    incident: bool = False
    bg_diff: float = NAN              # доля точек дырки, отличных от долгого фона (bg_model)
    bg_match: bool = False            # кадр на месте дырки совпадает с фоном -- пустой асфальт
    bg_valid: bool | None = None      # None — фон отключён, False — сравнение пока недостоверно
    bg_coverage: float = NAN          # доля ядра с устойчивым эталоном
    core: np.ndarray = None           # (gh, gw) bool: клетки дырки до разрастания на тихую область (по ним сверяем с фоном)
    mask: np.ndarray = None           # точная пиксельная маска, включая пустые промежутки внутри контура
    kind: str = "flow_gap"            # flow_gap / stationary: стоящий объект сам по себе не доказывает ДТП
    bg_removed_frac: float = 0.0      # какая доля исходной области исключена как фон/неизвестность


@dataclass
class HoleState:
    holes: list = field(default_factory=list)
    incident: bool = False


class HoleDetector:
    def __init__(self, fps: float, labels: np.ndarray, scale: float = 1.0, cell_px: int = 12,
                 move_frac: float = 0.2, min_passes: int = 3, pass_gap_s: float = 1.0,
                 now_s: float = 10.0, idle_min_s: float = 4.0, gap_k: float = 3.0,
                 up_cells: int = 2, arr_min: float = 2.0, arr_tau_s: float = 15.0, grow_px: int = 48,
                 ring_px: int = 36, ring_ratio: float = 0.4, grace_s: float = 3.0, min_cells: int = 2,
                 max_part_frac: float = 0.15, edge_px: int = 24, confirm_s: float = 3.0, incident_s: float = 20.0,
                 id_keep_s: float = 2.0, occl_max_s: float = 2.0, block_px: int = 4,
                 pause_usual: bool = True, soft_grow: bool = True,
                 fill_px: int = 84, act_s: float = 8.0, act_max: float = 0.25,
                 div_accum: float = 0.0, div_up=(2, 6), div_base_s: float = 30.0):
        """fps -- частота кадров; labels -- (H, W) номер части дороги (-1 -- не дорога),
        в разрешении обработки; scale -- во сколько раз кадр уменьшен при обработке
        (для скорости в px/с исходного видео);
        cell_px, ring_px -- клетка и ширина кольца, px при ширине кадра 640;
        move_frac -- клетка "движется", если движется не меньше этой доли её пикселей;
        min_passes, pass_gap_s -- клетка -- полоса, если через неё проехали min_passes раз
            (новый проезд -- после паузы не короче pass_gap_s);
        now_s -- память текущей частоты проездов, с;
        idle_min_s, gap_k -- "давно не ездят" (см. описание модуля);
        up_cells, arr_min, arr_tau_s -- "к месту подъезжают": сколько клеток выше по
            течению смотреть, сколько подъездов нужно и как быстро старые забываются;
        grow_px -- насколько дырка растёт от зерна по кандидатам;
        ring_px, ring_ratio -- "вокруг ездят как обычно"; grace_s -- если кольцо
            затихло меньше чем на столько секунд, дырка не гаснет;
        min_cells, max_part_frac -- наименьшая дырка в клетках и наибольшая (доля части);
        edge_px -- насколько дырка должна заходить внутрь части от её границы;
        confirm_s -- сколько секунд дырка должна продержаться, чтобы её показать;
        incident_s -- сколько секунд нет движения, чтобы дырка стала "ДТП?";
        id_keep_s -- номер дырки сохраняется, если она пропала не дольше чем на столько секунд;
        occl_max_s -- движение в клетке дырки короче этого -- перекрытие, длиннее -- через
            клетку снова едут, она выходит из дырки;
        block_px -- блок истории движения для контура, px при ширине 640;
        pause_usual -- "давно не ездят" сравнивать с обычной длиной пауз в клетке
            (иначе -- с интервалом между проездами, как в первых версиях);
        soft_grow -- дырка растёт и в клетки части, где ездили редко (меньше min_passes);
        fill_px, act_s, act_max -- вся область аварии: подтверждённая дырка разрастается на
            соседние "тихие" клетки части (движение в них было меньше act_max доли времени за
            последние act_s секунд) не дальше fill_px от клеток полос дырки. Стоящие машины,
            куда раньше не ездили или рядом ползёт пробка, попадают в дырку целиком, а не
            только зёрнами, к которым подъезжали; 0 -- не разрастать.
        div_accum -- "к месту подъезжают" по дивергенции: перед клеткой (на div_up клеток
            против потока) дивергенция ниже своего обычного значения (среднее за div_base_s)
            хотя бы на столько, 1/с; 0 -- не использовать. По умолчанию выключено:
            проверка на видео с авариями (20.10) показала, что дивергенция колеблется и на
            обычных полосах, и при 0.01 ложных дырок становится в разы больше (view3:
            около 30 вместо 3), а настоящие дырки от ложных она не отличает."""
        self.fps = float(fps)
        self.scale = float(scale)
        H, W = labels.shape
        k = W / 640.0
        self.cell = c = max(6, int(round(cell_px * k)))
        self.ring = max(1, int(round(ring_px * k / c)))
        self.grow = max(1, int(round(grow_px * k / c)))
        self.up_cells = up_cells
        self.arr_min = arr_min
        self.arr_decay = math.exp(-1.0 / max(1.0, fps * arr_tau_s))
        self.move_frac = move_frac
        self.min_passes = min_passes
        self.pass_gap_s = pass_gap_s
        self.a_now = math.exp(-1.0 / max(1.0, fps * now_s))
        self.idle_min_s = idle_min_s
        self.gap_k = gap_k
        self.ring_ratio = ring_ratio
        self.grace_s = grace_s
        self.min_cells = min_cells
        self.max_part_frac = max_part_frac
        self.incident_s = incident_s
        self.id_keep_s = id_keep_s
        self.confirm_s = confirm_s
        self.occl_max_s = occl_max_s
        self.pause_usual = pause_usual
        self.soft_grow = soft_grow
        self.fill = int(round(fill_px * k / c)) if fill_px > 0 else 0
        self.a_act = math.exp(-1.0 / max(1.0, fps * act_s))
        self.act_s = act_s
        self.act_max = act_max
        self.div_accum = div_accum
        self.div_up = div_up
        self.a_div = math.exp(-1.0 / max(1.0, fps * div_base_s))
        self.block = b = max(2, int(round(block_px * k)))
        self.bh, self.bw = H // b, W // b

        self.H, self.W = H, W
        self.n_parts = max(1, int(labels.max()) + 1)
        self.gh, self.gw = H // c, W // c
        self.part_px = [labels == p for p in range(self.n_parts)]
        self.cell_part_px = labels
        frac = np.stack([pool_mean(m.astype(np.float32), c) for m in self.part_px])   # (P, gh, gw)
        self.cell_part = np.where(frac.max(axis=0) >= 0.5, frac.argmax(axis=0), -1).astype(np.int16)
        self.part_frac = frac
        self.k_grow = np.ones((3, 3), np.uint8)
        self.k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        # внутренние клетки частей: дальше edge_px от границы части; за краем кадра
        # часть продолжается (край кадра -- не граница полосы)
        e = max(1, int(round(edge_px * k / c)))
        pad = np.pad(self.cell_part, e, mode="edge")
        k_e = np.ones((2 * e + 1, 2 * e + 1), np.uint8)
        self.interior = np.zeros(self.cell_part.shape, bool)
        for p in range(self.n_parts):
            inner = cv2.erode((pad == p).astype(np.uint8), k_e)[e:-e, e:-e] > 0
            self.interior |= inner
        self.k_ring = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * self.ring + 1, 2 * self.ring + 1))
        self.next_id = 1
        self.reset()

    def reset(self) -> None:
        """Сброс после монтажной склейки: история движения прежней сцены не годится."""
        g = (self.gh, self.gw)
        self.n = 0                               # годных кадров с начала сцены
        self.last_move = np.full(g, -np.inf)     # когда в клетке последний раз было движение
        self.passes = np.zeros(g, np.int32)      # сколько раз через клетку проезжали
        self.pause_sum = np.zeros(g, np.float64) # сумма пауз перед проездами, с
        self.prev_move = np.zeros(g, bool)
        self.act = np.ones(g, np.float32)        # доля времени с движением за последние act_s секунд
        self.prev_raw = np.zeros(g, bool)
        self.t0 = None                           # начало сцены
        self.r_now = np.zeros(g, np.float32)     # частота проездов сейчас, 1/с
        self.w_now = 0.0
        self.vsum = np.zeros(g + (2,), np.float32)   # сумма векторов поля -- обычное направление
        self.up_idx = None                       # (gh*gw, up_cells) клетки выше по течению
        self.up_div_idx = None                   # то же дальше: для дивергенции перед клеткой
        self.div_base = None                     # обычная дивергенция клетки (медленное среднее)
        self.arr = np.zeros(g, np.float32)       # подъезды с последнего движения клетки (с забыванием)
        self.ring_ok = np.full(g, -np.inf)       # когда для клетки последний раз подтвердилось "вокруг ездят"
        self.hole_since = np.full(g, np.nan)     # с какого момента клетка в дырке
        self.occl = np.zeros(g, np.float32)      # сколько секунд подряд в клетке движение
        self.still_since = np.full(g, -np.inf)   # с какого момента в клетке нет движения
                                                 # (у дырки короткие перекрытия его не сдвигают)
        self.b_last = np.full((self.bh, self.bw), -np.inf)   # история движения блоков (MHI)
        self.b_prev = np.zeros((self.bh, self.bw), bool)
        self.recent: dict = {}                   # номер дырки -> (клетки, когда видели)
        self.state = HoleState()

    # ------------------------------------------------------------------

    def _upstream(self, ks=None) -> np.ndarray:
        """Для каждой клетки -- клетки выше по течению: центр - k * клетка * направление,
        k = 1..up_cells (или ks), в той же части дороги. Нет направления или клетки --
        индекс N (фиктивная клетка без проездов)."""
        ks = list(ks) if ks is not None else list(range(1, self.up_cells + 1))
        gh, gw, N = self.gh, self.gw, self.gh * self.gw
        m = np.linalg.norm(self.vsum, axis=2)
        ok = m > 1e-6
        dx = np.where(ok, self.vsum[..., 0] / np.maximum(m, 1e-6), 0.0)
        dy = np.where(ok, self.vsum[..., 1] / np.maximum(m, 1e-6), 0.0)
        jj, ii = np.mgrid[0:gh, 0:gw]
        idx = np.full((gh, gw, len(ks)), N, np.int64)
        for n, k in enumerate(ks):
            uj = np.round(jj - k * dy).astype(int)
            ui = np.round(ii - k * dx).astype(int)
            inside = ok & (uj >= 0) & (uj < gh) & (ui >= 0) & (ui < gw)
            uj, ui = np.clip(uj, 0, gh - 1), np.clip(ui, 0, gw - 1)
            same = inside & (self.cell_part[uj, ui] == self.cell_part) & (self.cell_part >= 0)
            same &= (uj != jj) | (ui != ii)
            idx[..., n] = np.where(same, uj * gw + ui, N)
        return idx.reshape(N, len(ks))

    def update(self, c, t: float, valid: bool = True, cut: bool = False, div=None, keep=None) -> HoleState:
        """div -- дивергенция поля (DivCurl.div, 1/с) на своей сетке или None;
        keep -- функция без аргументов -> bool (gh, gw): клетки, где по картинке пустой асфальт
        (см. bg_model); при разрастании дырки на область аварии они не берутся."""
        if cut:
            self.reset()
        if not valid:
            return self.state                  # негодный кадр: решение не меняем
        cell = self.cell
        self.n += 1
        if self.t0 is None:
            self.t0 = t
        # доля движущихся пикселей своей части в каждой клетке. Одиночные точки
        # убираем открытием: на сжатом видео бывают кадры, где шум сжатия даёт
        # россыпь "движения" по всему кадру, в том числе на стоящих машинах
        px = cv2.morphologyEx(c.moving.astype(np.uint8), cv2.MORPH_OPEN, self.k_open) > 0
        mv = np.zeros((self.gh, self.gw), np.float32)
        for p in range(self.n_parts):
            mp = pool_mean((px & self.part_px[p]).astype(np.float32), cell)
            sel = self.cell_part == p
            mv[sel] = mp[sel] / np.maximum(self.part_frac[p][sel], 1e-6)
        road = self.cell_part >= 0
        raw = (mv >= self.move_frac) & road
        # движение -- только если оно есть два кадра подряд: машина проезжает клетку
        # за несколько кадров, а вспышка шума -- один кадр
        moving = raw & self.prev_raw
        self.prev_raw = raw
        self.act = (self.a_act * self.act + (1 - self.a_act) * moving).astype(np.float32)

        # история движения блоков (MHI): время последнего движения, тоже два кадра подряд
        b_raw = pool_mean((px & (self.cell_part_px >= 0)).astype(np.float32), self.block) >= 0.3
        self.b_last[b_raw & self.b_prev] = t
        self.b_prev = b_raw

        # перекрытие клеток дырки (конечный автомат): движение короче occl_max_s
        # время без движения у клетки дырки не сдвигает
        dt = 1.0 / self.fps
        was = np.isfinite(self.hole_since)
        tol = was & ((t - np.where(was, self.hole_since, t)) >= self.confirm_s)   # подтверждённая дырка
        self.occl = np.where(moving, self.occl + dt, 0.0).astype(np.float32)
        released = moving & (~tol | (self.occl > self.occl_max_s))
        self.still_since[released] = t

        # проезды: движение началось после паузы >= pass_gap_s
        start = moving & ~self.prev_move & ((t - self.last_move) >= self.pass_gap_s)
        self.passes[start] += 1
        # пауза перед этим проездом (до первого проезда -- от начала сцены)
        pz = np.where(np.isfinite(self.last_move), t - self.last_move, t - self.t0)
        self.pause_sum[start] += pz[start]
        self.last_move[moving] = t
        self.prev_move = moving

        # обычное направление потока в клетке -- сумма векторов очищенного поля
        f = c.field()
        self.vsum[..., 0] += pool_mean(f[..., 0], cell)
        self.vsum[..., 1] += pool_mean(f[..., 1], cell)
        if self.up_idx is None or self.n % max(1, int(round(self.fps))) == 0:
            self.up_idx = self._upstream()
            self.up_div_idx = self._upstream(range(self.div_up[0], self.div_up[1] + 1))
        # подъезды: проезд начался выше по течению (среднее по клеткам выше по
        # течению -- одна машина проходит их все и должна считаться один раз)
        flat = np.append(start.ravel(), False).astype(np.float32)
        valid_up = (self.up_idx < self.gh * self.gw).sum(axis=1)
        up = flat[self.up_idx].sum(axis=1) / np.maximum(valid_up, 1)
        self.arr = self.arr * self.arr_decay + up.reshape(self.gh, self.gw)
        self.arr[moving] = 0.0

        # накопление перед клеткой по дивергенции. Дивергенция плотности потока < 0 --
        # в зону въезжает больше машин, чем выезжает (уравнение неразрывности), то есть
        # машины перед местом копятся. Из-за перспективы знак сам по себе ничего не
        # значит, поэтому сравниваем с обычной дивергенцией той же клетки на этом видео
        accum = np.zeros((self.gh, self.gw), bool)
        if self.div_accum > 0 and div is not None:
            d = cv2.resize(div.astype(np.float32), (self.gw, self.gh), interpolation=cv2.INTER_LINEAR)
            if self.div_base is None:
                self.div_base = d.copy()
            anom = np.append((d - self.div_base).ravel(), np.nan)
            ups = anom[self.up_div_idx]
            ok_up = np.isfinite(ups).sum(axis=1) > 0
            mean_up = np.where(ok_up, np.nansum(ups, axis=1) / np.maximum(np.isfinite(ups).sum(axis=1), 1), 0.0)
            accum = (mean_up < -self.div_accum).reshape(self.gh, self.gw)
            keep = np.isfinite(self.hole_since)            # в дырках "обычное" не обновляем
            self.div_base = np.where(keep, self.div_base, self.a_div * self.div_base + (1 - self.a_div) * d)

        # частота проездов сейчас, 1/с: бегущее среднее с поправкой на короткую историю
        ev = start.astype(np.float32) * self.fps
        self.r_now = self.a_now * self.r_now + (1 - self.a_now) * ev
        self.w_now = self.a_now * self.w_now + (1 - self.a_now)
        now = self.r_now / max(self.w_now, 1e-6)

        idle = t - self.last_move                                   # inf -- не ездили ни разу
        still = t - self.still_since                                # то же, но с учётом перекрытий
        lane = road & (self.passes >= self.min_passes)              # здесь ездят
        gap = np.where(lane, (self.last_move - self.t0) / np.maximum(self.passes, 1), np.inf)
        usual = np.where(lane, 1.0 / np.maximum(gap, 1e-3), 0.0)   # обычная частота проездов, 1/с
        # обычная пауза: сколько в клетке обычно тихо между проездами. Интервал между
        # проездами для этого не годится: в плотном потоке машины идут без пауз, это
        # один долгий "проезд", и интервал выходит в разы больше настоящих пауз
        pause = np.where(lane, self.pause_sum / np.maximum(self.passes, 1), np.inf)
        wait = pause if self.pause_usual else gap
        cand = lane & (idle >= np.maximum(self.idle_min_s, self.gap_k * wait))
        # клетки части, где ездили редко (меньше min_passes проездов): своей статистики
        # нет -- берём обычную паузу части. Начать дырку они не могут, но в них растёт
        # уже подтверждённая дырка: стоящий объект часто стоит там, где до аварии
        # ездили мало (у бордюра, на месте, куда машину отбросило)
        soft = np.zeros_like(cand)
        if self.soft_grow:
            for p in range(self.n_parts):
                lp = lane & (self.cell_part == p)
                if lp.sum() >= 3:
                    w_p = float(np.median(wait[lp]))
                    soft |= (self.cell_part == p) & ~lane & (idle >= max(self.idle_min_s, self.gap_k * w_p))

        # зёрна: кандидаты, к которым подъезжают (или уже дырка -- она держится,
        # пока клетка кандидат: машины могли перестать подъезжать, а объект стоит)
        seeds = cand & ((self.arr >= self.arr_min) | accum | was)

        # принятые дырки держатся, пока в клетке нет движения; подтверждённые --
        # и пока клетка лишь перекрыта
        holes_mask = (was & cand) | (tol & lane & (still >= self.idle_min_s))
        ratio_cells = np.zeros(cand.shape, np.float32)
        for p in range(self.n_parts):
            cand_p = (cand & (self.cell_part == p)).astype(np.uint8)
            cp = (seeds & (self.cell_part == p)).astype(np.uint8)
            if not cp.any():
                continue
            for _ in range(self.grow):              # рост от зёрен по кандидатам
                cp = cv2.dilate(cp, self.k_grow) & cand_p
            lane_p = lane & (self.cell_part == p)
            n_lane = max(int(lane_p.sum()), 1)
            n_lab, lab = cv2.connectedComponents(cp, connectivity=8)
            for i in range(1, n_lab):
                comp = lab == i
                core = comp & cand                  # размер и "внутри части" -- по клеткам полос
                if core.sum() > self.max_part_frac * n_lane or not (core & self.interior).any():
                    self.ring_ok[comp] = -np.inf
                    continue                    # встала почти вся часть или полоска вдоль границы части
                ring = (cv2.dilate(comp.astype(np.uint8), self.k_ring) > 0) & ~comp & lane_p
                if ring.sum() < 3:
                    continue
                ratio = float(now[ring].sum() / max(usual[ring].sum(), 1e-6))
                ratio_cells[comp] = ratio
                if ratio >= self.ring_ratio:
                    self.ring_ok[comp] = t
                holes_mask |= comp & ((t - self.ring_ok) <= self.grace_s)

        self._core = holes_mask.copy()

        # подтверждённые дырки растут в редкие клетки своей части -- не дальше grow_px
        # от клеток полос, которые сейчас в дырке; пересчитывается каждый кадр, так что
        # без клеток полос дырка не держится
        if self.soft_grow and soft.any():
            for p in range(self.n_parts):
                in_p = self.cell_part == p
                ext = (holes_mask & tol & lane & in_p).astype(np.uint8)
                if not ext.any():
                    continue
                allow = ((soft | holes_mask) & in_p).astype(np.uint8)
                for _ in range(self.grow):
                    ext = cv2.dilate(ext, self.k_grow) & allow
                holes_mask |= ext > 0

        # вся область аварии: рост по тихим клеткам части (стоящие машины, куда не ездили)
        if self.fill > 0 and (t - self.t0) >= self.act_s:
            quiet = road & (self.act < self.act_max)
            empty = keep() if (keep is not None and holes_mask.any()) else None
            if empty is not None:
                holes_mask &= ~empty        # пустая полоса не должна сливаться с аварией в одну дырку
            for p in range(self.n_parts):
                in_p = self.cell_part == p
                core_p = (holes_mask & tol & lane & in_p).astype(np.uint8)
                if not core_p.any():
                    continue
                allow = ((quiet | holes_mask) & in_p).astype(np.uint8)
                n_lab, lab = cv2.connectedComponents(core_p, connectivity=8)
                if empty is None and keep is not None:
                    empty = keep()
                for i in range(1, n_lab):
                    comp = lab == i
                    if empty is not None and empty[comp].mean() > 0.6:
                        continue                # зерно на пустом асфальте: разрастаться нечему
                    ext = comp.astype(np.uint8)
                    for _ in range(self.fill):
                        ext = cv2.dilate(ext, self.k_grow) & allow
                    grown = ext > 0
                    if empty is not None:
                        grown &= ~empty | comp  # клетки, где кадр совпал с пустым фоном, не берём
                    holes_mask |= grown

        # клетки, где движение возобновилось, из дырки выходят
        self.hole_since[~holes_mask] = np.nan
        self.hole_since[holes_mask & ~was] = t
        self.ring_ok[~cand] = -np.inf

        # сколько нет движения: у клеток, где не ездили ни разу, -- с начала сцены
        idle_h = np.minimum(np.where(holes_mask, still, idle), t - self.t0)
        holes = self._extract(holes_mask, idle_h, ratio_cells, c, t, lane, self._core)
        self.state = HoleState(holes=holes, incident=any(h.incident for h in holes))
        return self.state

    # ------------------------------------------------------------------

    def _extract(self, holes_mask, idle, ratio_cells, c, t, lane, core=None) -> list:
        cell = self.cell
        out = []
        for p in range(self.n_parts):
            hp = (holes_mask & (self.cell_part == p)).astype(np.uint8)
            if not hp.any():
                continue
            n_lab, lab = cv2.connectedComponents(hp, connectivity=8)
            for i in range(1, n_lab):
                comp = lab == i
                if comp.sum() < self.min_cells:
                    continue
                m = self._outline(comp, p, t)
                cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                eps = max(1.0, 1.5 * self.W / 640)
                polys = [cv2.approxPolyDP(cn, eps, True).reshape(-1, 2) for cn in cnts
                         if cv2.contourArea(cn) >= 4]
                if not polys:
                    continue
                inside = m > 0
                v_in = float(c.mag_clean[inside].mean()) * self.fps / self.scale
                ring = (cv2.dilate(comp.astype(np.uint8), self.k_ring) > 0) & ~comp & (self.cell_part == p)
                ring_up = cv2.resize(ring.astype(np.uint8), (self.gw * cell, self.gh * cell),
                                     interpolation=cv2.INTER_NEAREST) > 0
                rp = np.zeros((self.H, self.W), bool)
                rp[: ring_up.shape[0], : ring_up.shape[1]] = ring_up
                rp &= c.moving & self.part_px[p]
                v_ring = float(np.median(c.mag_clean[rp])) * self.fps / self.scale if rp.sum() >= 20 else NAN
                since = self.hole_since[comp]
                age = float(t - np.nanmin(since)) if np.isfinite(since).any() else 0.0
                hid = self._match_id(comp, t)
                if age < self.confirm_s:
                    continue                    # ещё не подтверждена
                # время без движения -- по клеткам полос (редкие клетки стоят тихо всегда)
                idle_s = float(np.median(idle[comp & lane] if (comp & lane).any() else idle[comp]))
                out.append(Hole(id=hid, part=p, cells=comp, polygons=polys,
                                core=(comp & core) if core is not None and (comp & core).any() else comp,
                                area_px=int(inside.sum()), idle_s=idle_s, age_s=age,
                                ring_ratio=float(ratio_cells[comp].max()),
                                v_in=v_in, v_ring=v_ring,
                                incident=idle_s >= self.incident_s))
        return out

    def _outline(self, comp: np.ndarray, p: int, t: float) -> np.ndarray:
        """Маска дырки в пикселях: клетки дырки (с перекрытыми) плюс блоки в одной
        клетке вокруг, где по истории движения (MHI) не ездили idle_min_s секунд;
        затем закрытие (сглаживает ступеньки) и только внутри своей части."""
        cell, b = self.cell, self.block
        core = np.zeros((self.H, self.W), bool)
        up = cv2.resize(comp.astype(np.uint8), (self.gw * cell, self.gh * cell), interpolation=cv2.INTER_NEAREST)
        core[: up.shape[0], : up.shape[1]] = up > 0
        near = cv2.dilate(core.astype(np.uint8), np.ones((2 * cell + 1, 2 * cell + 1), np.uint8)) > 0
        quiet = (t - self.b_last) >= self.idle_min_s
        qb = cv2.resize(quiet.astype(np.uint8), (self.bw * b, self.bh * b), interpolation=cv2.INTER_NEAREST)
        q = np.zeros((self.H, self.W), bool)
        q[: qb.shape[0], : qb.shape[1]] = qb > 0
        # внутри клеток дырки перекрытия разрешены; по истории движения подрезаем
        # только край клеток (полклетки) и добавляем тихие блоки в клетке вокруг
        inner = cv2.erode(core.astype(np.uint8), np.ones((cell + 1, cell + 1), np.uint8)) > 0
        m = inner | (core & q) | (near & q)
        m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_CLOSE,
                             cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (cell | 1, cell | 1)))
        m &= self.part_px[p].astype(np.uint8)
        # компоненты, не касающиеся клеток дырки, отбрасываем (это просто тихий асфальт рядом)
        n, lab = cv2.connectedComponents(m, connectivity=8)
        if n > 2:
            keep = np.unique(lab[core & (m > 0)])
            m = np.isin(lab, keep[keep > 0]).astype(np.uint8)
        return m

    def _match_id(self, comp: np.ndarray, t: float) -> int:
        """Номер дырки сохраняется, пока она пересекается с дыркой, которую видели
        не раньше id_keep_s секунд назад."""
        self.recent = {k: v for k, v in self.recent.items() if t - v[1] <= self.id_keep_s}
        best, best_ov = None, 0
        for k, (cells, _) in self.recent.items():
            ov = int((cells & comp).sum())
            if ov > best_ov:
                best, best_ov = k, ov
        if best is None:
            best = self.next_id
            self.next_id += 1
        self.recent[best] = (comp, t)
        return best
