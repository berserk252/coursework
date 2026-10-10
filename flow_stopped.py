"""
Детектор стоящих объектов на дороге (то, чего оптический поток не видит в принципе:
у стоящей машины поток = 0, как у асфальта).

    стоящее = |кадр - фон| > thresh            "здесь есть то, чего нет в фоне"
              и не движется (FlowCleaner.moving, расширенное)
              и в зоне дороги (где по дороге уже ездили машины)
              и держится >= confirm_s секунд
              и добавляет в картинку края (у машины краёв больше, чем у асфальта под ней)

Фон -- своё бегущее среднее с ВЫБОРОЧНЫМ обновлением (у MOG2 одна скорость
обучения на весь кадр: он либо быстро впитывает вставшую машину, либо не
успевает за светом):
    * где движение              -- фон не трогаем;
    * где свободный асфальт     -- быстро (tau_free_s ~ 10 с): свет, тени облаков;
    * где стоит объект          -- очень медленно (tau_stop_s ~ 600 с): отличие
      вставшей машины от фона тает как exp(-t/tau_stop), т.е. объект с контрастом
      1.2*thresh держится ~110 с, 2*thresh -- ~7 мин.

Устойчивость:
    * дрожание камеры   -- сравнение с допуском tol_px (окно (2*tol+1)^2);
    * медленный дрейф   -- раз в reg_every_s фон совмещается с кадром (фазовая
      корреляция, только сдвиг 0.3..4 px рабочего кадра);
    * экспозиция        -- глобальный коэффициент яркости по свободным пикселям;
    * блики, смена цвета светофора на мокром асфальте, дрейф края столба --
      отличие от фона без новых краёв: объект принимается, только если краёв в
      КАДРЕ хотя бы в edge_gain раз больше, чем в фоне на том же месте;
    * "призраки"        -- машина стояла в момент старта фона и уехала: краёв в
      кадре МЕНЬШЕ, чем в фоне (< ghost_ratio) -- вписываем это место в фон;
    * зона дороги       -- только где движение по маске дороги (road_w >= zone_road_w)
      набралось >= zone_min_s секунд: снег, дождь, листва зону не строят;
    * склейка / чёрный кадр -- reset() (cut=True из SceneGuard) / кадр пропускается.

Ограничения: машина, стоящая с ПЕРВОГО кадра сцены (очередь в заторе с самого
начала ролика), попадает в фон и не находится, пока не уедет; если по дороге
ещё никто не проехал, зоны нет и стоящих нет.

Использование (после FlowCleaner, на тех же кадрах ширины 640):

    sd = StoppedDetector(fps)
    ...
    cut, valid = guard.update(gray)                  # SceneGuard
    c = cleaner.update(flow, gray, prev_gray)        # FlowCleaner
    st = sd.update(frame, c.moving, c.road_w, cut=cut, valid=valid)
    sm.update(c, t, valid=valid, occupied=st.occupied)   # SpeedMeter: drop=1 только если дорога занята
    st.frac      -- доля зоны дороги, занятая стоящими объектами
    st.mask      -- маска стоящих (рабочее разрешение; st.scale -> пиксели кадра)
    st.blobs     -- [(x, y, w, h, площадь, возраст_с)] в пикселях кадра; пересекающиеся
                    рамки слиты в одну по IoU (merge_boxes)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import cv2
import numpy as np


def iou(a, b) -> float:
    """Intersection over Union двух рамок (x, y, w, h): площадь пересечения / площадь объединения."""
    ax1, ay1, ax2, ay2 = a[0], a[1], a[0] + a[2], a[1] + a[3]
    bx1, by1, bx2, by2 = b[0], b[1], b[0] + b[2], b[1] + b[3]
    iw, ih = min(ax2, bx2) - max(ax1, bx1), min(ay2, by2) - max(ay1, by1)
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    return inter / float(a[2] * a[3] + b[2] * b[3] - inter)


def merge_boxes(blobs, iou_thresh: float = 0.0, pad: int = 4):
    """Сливает пересекающиеся рамки стоящих объектов в одну.

    Одна авария -- это несколько машин вплотную; детектор видит их отдельными
    пятнами, и рамки накладываются друг на друга. Две рамки объединяются,
    если IoU рамок, расширенных на pad пикселей, больше iou_thresh (0 --
    любое пересечение или касание). Объединение повторяется, пока есть что
    сливать (A касается B, B касается C -> одна рамка A+B+C). Новая рамка --
    общий охватывающий прямоугольник, площадь -- сумма, возраст -- наибольший.
    blobs: [(x, y, w, h, площадь, возраст_с)].
    """
    boxes = [list(b) for b in blobs]
    merged = True
    while merged and len(boxes) > 1:
        merged = False
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                a, b = boxes[i], boxes[j]
                ea = (a[0] - pad, a[1] - pad, a[2] + 2 * pad, a[3] + 2 * pad)
                eb = (b[0] - pad, b[1] - pad, b[2] + 2 * pad, b[3] + 2 * pad)
                if iou(ea, eb) > iou_thresh:
                    x1, y1 = min(a[0], b[0]), min(a[1], b[1])
                    x2, y2 = max(a[0] + a[2], b[0] + b[2]), max(a[1] + a[3], b[1] + b[3])
                    boxes[i] = [x1, y1, x2 - x1, y2 - y1, a[4] + b[4], max(a[5], b[5])]
                    del boxes[j]
                    merged = True
                    break
            if merged:
                break
    return [tuple(b) for b in boxes]


@dataclass
class StoppedState:
    mask: np.ndarray                 # (h, w) bool, рабочее разрешение
    zone: np.ndarray                 # (h, w) bool, зона дороги
    frac: float = 0.0                # доля зоны, занятая стоящими
    moving_frac: float = 0.0         # доля зоны, где сейчас движение
    occupied: bool = False           # есть стоящие объекты (для SpeedMeter.drop)
    blobs: list = field(default_factory=list)
    max_age: float = 0.0             # сколько секунд стоит самый старый объект
    scale: float = 1.0               # пиксели кадра / рабочие пиксели
    valid: bool = True


class StoppedDetector:
    def __init__(self, fps: float, work_w: int = 320, thresh: float = 24.0,
                 tau_free_s: float = 10.0, tau_stop_s: float = 600.0,
                 confirm_s: float = 5.0, boot_s: float = 2.0,
                 move_dilate: int = 7, tol_px: int = 1,
                 zone_dilate: int = 9, zone_min_s: float = 0.5, zone_road_w: float = 0.3,
                 min_blob: int = 40, edge_gain: float = 1.2, occupied_frac: float = 0.005,
                 ghost_ratio: float = 0.75, ghost_every_s: float = 0.5, reg_every_s: float = 0.5,
                 merge_iou: float = 0.0, merge_pad: float = 0.016):
        """fps -- частота кадров; work_w -- ширина рабочей копии кадра, px;
        thresh -- отличие от фона (макс. по каналам BGR, 0..255);
        tau_free_s / tau_stop_s -- память фона на свободном асфальте / под стоящим объектом;
        confirm_s -- сколько стоять, чтобы засчитать; boot_s -- быстрый старт фона;
        move_dilate -- расширение маски движения (рабочие px); tol_px -- допуск на дрожание;
        zone_min_s, zone_road_w -- зона дороги: где движение при road_w >= zone_road_w
            набралось >= zone_min_s секунд (+ текущая road_w >= 0.5), расширенная на zone_dilate;
        min_blob -- минимальная площадь объекта (рабочие px; 40 при 320 = 160 px кадра 640);
        edge_gain -- во сколько раз краёв в кадре должно быть больше, чем в фоне (объект, а не блик);
        occupied_frac -- доля зоны, начиная с которой дорога считается занятой;
        ghost_ratio -- краёв в кадре меньше этой доли от фона -> призрак уехавшей машины;
        reg_every_s -- как часто совмещать фон с кадром (0 -- не совмещать);
        merge_iou, merge_pad -- слияние пересекающихся рамок: IoU > merge_iou после
            расширения рамок на merge_pad * ширина кадра (0.016 -> 10 px при 640, около
            половины машины, чтобы сливались и машины вплотную); None -- не сливать."""
        self.fps = float(fps)
        self.work_w = int(work_w)
        self.thresh = float(thresh)
        self.a_free = 1.0 / max(1.0, fps * tau_free_s)
        self.a_stop = 1.0 / max(1.0, fps * tau_stop_s)
        self.confirm_s = float(confirm_s)
        self.boot_n = int(round(boot_s * fps))
        self.k_move = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (move_dilate, move_dilate))
        self.k_tol = np.ones((2 * tol_px + 1, 2 * tol_px + 1), np.uint8) if tol_px > 0 else None
        self.k_zone = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (zone_dilate, zone_dilate))
        self.zone_min_n = max(1.0, zone_min_s * fps)
        self.zone_road_w = zone_road_w
        self.min_blob = min_blob
        self.edge_gain = edge_gain
        self.occupied_frac = occupied_frac
        self.ghost_ratio = ghost_ratio
        self.ghost_every = max(1, int(round(ghost_every_s * fps)))
        self.reg_every = int(round(reg_every_s * fps)) if reg_every_s > 0 else 0
        self.merge_iou = merge_iou
        self.merge_pad = merge_pad
        self.ms: list[float] = []          # время update, мс (для отчёта)
        self.reset()

    def reset(self) -> None:
        """Сброс после склейки: фон прежней сцены к новой не относится."""
        self.B = None            # фон, float32 BGR, рабочее разрешение
        self.age = None          # сколько секунд подряд пиксель "отличается от фона и не движется"
        self.zone_hist = None    # сколько кадров в пикселе было движение по дороге
        self.win = None          # окно Ханнинга для фазовой корреляции
        self.n = 0
        self.gain = np.ones(3, np.float32)

    def update(self, frame: np.ndarray, moving: np.ndarray, road_w: np.ndarray | None = None,
               cut: bool = False, valid: bool = True) -> StoppedState:
        t0 = time.perf_counter()
        if cut:
            self.reset()
        H0, W0 = frame.shape[:2]
        W = self.work_w
        H = int(round(H0 * W / W0))
        if not valid:
            z = np.zeros((H, W), bool)
            return StoppedState(z, z, scale=W0 / W, valid=False)
        small = cv2.resize(frame, (W, H), interpolation=cv2.INTER_AREA).astype(np.float32)
        small = cv2.GaussianBlur(small, (0, 0), 0.8)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        mov = cv2.resize(moving.astype(np.uint8), (W, H), interpolation=cv2.INTER_AREA) > 0
        mov_d = cv2.dilate(mov.astype(np.uint8), self.k_move) > 0
        rw = None if road_w is None else cv2.resize(road_w.astype(np.float32), (W, H), interpolation=cv2.INTER_AREA)

        if self.B is None:
            self.B = small.copy()
            self.age = np.zeros((H, W), np.float32)
            self.zone_hist = np.zeros((H, W), np.float32)
            self.win = cv2.createHanningWindow((W, H), cv2.CV_32F)
        self.n += 1

        # медленный дрейф камеры: совмещаем фон с кадром (только сдвиг)
        if self.reg_every and self.n > self.boot_n and self.n % self.reg_every == 0:
            (dx, dy), resp = cv2.phaseCorrelate(cv2.cvtColor(self.B, cv2.COLOR_BGR2GRAY), gray, self.win)
            if resp > 0.1 and 0.3 <= np.hypot(dx, dy) <= 4.0:
                M = np.float32([[1, 0, dx], [0, 1, dy]])
                self.B = cv2.warpAffine(self.B, M, (W, H), borderMode=cv2.BORDER_REPLICATE)

        # зона дороги: где по дороге (road_w) двигалось >= zone_min_s секунд
        self.zone_hist += mov if rw is None else (mov & (rw >= self.zone_road_w))
        zone = (self.zone_hist >= self.zone_min_n).astype(np.uint8)
        if rw is not None:
            zone |= (rw >= 0.5).astype(np.uint8) & (self.zone_hist > 0).astype(np.uint8)
        zone = cv2.morphologyEx(cv2.dilate(zone, self.k_zone), cv2.MORPH_CLOSE, self.k_zone) > 0

        # экспозиция: отношение медиан по свободным пикселям
        free = ~mov_d
        if self.n > 1 and free.sum() > 500:
            idx = np.flatnonzero(free.ravel())[::7]
            g = np.median(small.reshape(-1, 3)[idx], axis=0) / np.maximum(np.median(self.B.reshape(-1, 3)[idx], axis=0), 1.0)
            self.gain = np.clip(g, 0.8, 1.25).astype(np.float32)
        Bg = self.B * self.gain
        # отличие от фона с допуском на дрожание камеры
        if self.k_tol is not None:
            D = np.maximum(small - cv2.dilate(Bg, self.k_tol), cv2.erode(Bg, self.k_tol) - small).max(axis=2)
            D = np.maximum(D, 0.0)
        else:
            D = np.abs(small - Bg).max(axis=2)
        fg = cv2.morphologyEx((D > self.thresh).astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)) > 0

        # возраст стояния: растёт, где отличие есть и движения нет; движение перед
        # объектом (обгон, заслонение) возраст не сбрасывает
        self.age[fg & ~mov_d] += 1.0 / self.fps
        self.age[~fg] = 0.0
        stopped = (self.age >= self.confirm_s) & zone

        n_lab, lab, st, _ = cv2.connectedComponentsWithStats(stopped.astype(np.uint8), connectivity=8)
        keep = np.zeros(n_lab, bool)
        keep[1:] = st[1:, cv2.CC_STAT_AREA] >= self.min_blob
        if keep.any():
            gc, gb = self._edges(gray), self._edges(cv2.cvtColor(self.B, cv2.COLOR_BGR2GRAY))
            ghost_check = self.n % self.ghost_every == 0
            for i in np.flatnonzero(keep):
                x, y, w, h = st[i, :4]
                sl = (slice(max(0, y - 2), y + h + 2), slice(max(0, x - 2), x + w + 2))
                reg = cv2.dilate((lab[sl] == i).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
                ec, eb = float(gc[sl][reg].sum()), float(gb[sl][reg].sum())
                if ghost_check and ec < self.ghost_ratio * eb:     # призрак: в фон его, возраст в ноль
                    m = lab[sl] == i
                    self.B[sl][m] = small[sl][m] / self.gain
                    self.age[sl][m] = 0.0
                    keep[i] = False
                elif ec < self.edge_gain * eb:                     # отличие без новых краёв: блик, дрейф
                    keep[i] = False
        stopped = keep[lab]

        # обновление фона: движение -- никак, стоящее/отличающееся -- медленно, асфальт -- быстро
        if self.n <= self.boot_n:
            rate = np.where(mov_d, 0.0, 0.25).astype(np.float32)
        else:
            near = (D > 0.5 * self.thresh) | stopped
            rate = np.where(mov_d, 0.0, np.where(near, self.a_stop, self.a_free)).astype(np.float32)
        self.B += rate[..., None] * (small / self.gain - self.B)

        za = float(zone.sum())
        s = W0 / W
        blobs = []
        for i in np.flatnonzero(keep):
            x, y, w, h, a = st[i]
            blobs.append((int(x * s), int(y * s), int(w * s), int(h * s), int(a * s * s),
                          float(self.age[lab == i].max())))
        if self.merge_iou is not None:
            blobs = merge_boxes(blobs, self.merge_iou, max(4, int(round(self.merge_pad * W0))))
        frac = float((stopped & zone).sum()) / za if za > 0 else 0.0
        out = StoppedState(mask=stopped, zone=zone, frac=frac,
                           moving_frac=float((mov & zone).sum()) / za if za > 0 else 0.0,
                           occupied=bool(blobs) and frac >= self.occupied_frac,
                           blobs=blobs, max_age=max((b[5] for b in blobs), default=0.0), scale=s)
        self.ms.append(1000.0 * (time.perf_counter() - t0))
        return out

    @staticmethod
    def _edges(g: np.ndarray) -> np.ndarray:
        return np.abs(cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)) + np.abs(cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3))
