"""
Средняя скорость транспортного потока по времени -- косвенный признак ДТП.

Скорость в кадре = медиана модуля очищенного потока (FlowCleaner) по точкам
дороги, где сейчас есть движение. Переводим в пиксели ИСХОДНОГО видео в секунду,
чтобы цифры не зависели от того, во сколько раз кадр уменьшили при обработке:

    V_кадр = median(|v|) / resize * fps          [px/с]

"В единицу времени" -- медиана V_кадр за последнюю секунду (V). Медиана, а
не среднее: один испорченный кадр (склейка, блик) не должен давать
секундное плато ложной скорости.

Почему это признак аварии: после ДТП машины тормозят, встают и объезжают
место аварии -- средняя скорость потока резко падает относительно той,
что была до этого. Поэтому храним базовую скорость V_base и считаем падение:

    drop = 1 - V / V_base       0 -- едут как обычно, 1 -- встали полностью

V_base: первые warmup_s секунд -- медиана V (drop пока не выдаётся, база
ещё не известна), дальше -- бегущее среднее с памятью ~base_s секунд.

Пустая дорога и полная остановка для оптического потока одинаковы: движения
нет. Поэтому если движения нет, drop выдаётся только когда известно, что
дорога ЗАНЯТА (occupied=True -- например, детектор стоящих машин); иначе
drop = nan ("нет движения", а не "падение 100%").

Ограничение: из-за перспективы дальние машины в пикселях идут медленнее
ближних, поэтому V сравнима только во времени в пределах одного видео,
а не между разными камерами.
"""

from __future__ import annotations

import math
from collections import deque

import numpy as np

NAN = float("nan")


class SpeedMeter:
    def __init__(self, fps: float, scale: float = 1.0, window_s: float = 1.0,
                 base_s: float = 10.0, warmup_s: float = 3.0, min_area: float = 0.001):
        self.fps = float(fps)
        self.scale = float(scale)
        self.window = max(1, round(fps * window_s))
        self.base_a = math.exp(-1.0 / max(fps * base_s, 1.0))
        self.warmup = max(1, round(fps * warmup_s))
        self.min_area = min_area
        # история для графиков (через склейки не сбрасывается)
        self.t, self.hv, self.harea, self.hdrop = [], [], [], []
        self.reset()

    def reset(self) -> None:
        """Сброс после монтажной склейки: скорость новой сцены с прежней не сравнима."""
        self.win = deque(maxlen=self.window)
        self.warm = []        # значения V для начальной базы
        self.v_frame = 0.0    # скорость в текущем кадре, px/с (nan -- движения нет)
        self.v = 0.0          # медиана за последнюю секунду, px/с
        self.base = NAN       # базовая (обычная) скорость, px/с; nan -- ещё не известна
        self.drop = NAN       # падение скорости относительно базовой, 0..1; nan -- не определено
        self.area = 0.0       # доля дороги, где сейчас есть движение

    def update(self, c, t: float, valid: bool = True, occupied: bool | None = None) -> "SpeedMeter":
        """c -- CleanFlow; valid=False -- кадр негоден (склейка, чёрный кадр):
        ничего не измеряем; occupied -- дорога занята (есть стоящие машины)."""
        if not valid:
            self.v_frame, self.v, self.drop, self.area = NAN, NAN, NAN, 0.0
            self._log(t)
            return self
        road = c.road_w >= 0.5
        # пока маска дороги не накопила статистику (первые ~0.5 с) -- весь кадр
        if road.sum() < 0.02 * road.size:
            road = np.ones_like(road)
        mv = c.moving & road
        self.area = float(mv.sum()) / float(road.sum())
        if self.area >= self.min_area:
            self.v_frame = float(np.median(c.mag_clean[mv])) * self.fps / self.scale
        else:
            self.v_frame = NAN
        self.win.append(self.v_frame)
        vals = [x for x in self.win if not math.isnan(x)]
        moving_now = bool(vals)
        self.v = float(np.median(vals)) if moving_now else 0.0

        if moving_now:
            if math.isnan(self.base):
                self.warm.append(self.v)
                if len(self.warm) >= self.warmup:
                    self.base = float(np.median(self.warm))
            else:
                self.base = self.base_a * self.base + (1 - self.base_a) * self.v

        if math.isnan(self.base):
            self.drop = NAN
        elif moving_now:
            self.drop = float(np.clip(1.0 - self.v / self.base, 0.0, 1.0))
        else:
            self.drop = 1.0 if occupied else NAN
        self._log(t)
        return self

    def _log(self, t: float) -> None:
        self.t.append(t)
        self.hv.append(self.v)
        self.harea.append(self.area)
        self.hdrop.append(self.drop)


class IncidentRule:
    """Простое правило "возможное ДТП" -- первый вариант подхода по оптическому потоку.

        ДТП? = на дороге что-то стоит не меньше stop_s секунд (StoppedDetector)
               И за последние lookback_s секунд средняя скорость потока падала
               не меньше чем на drop_min (SpeedMeter.drop)

    Почему так: после ДТП разбитые машины стоят на полосе, а поток вокруг них
    резко замедляется. Одна остановка без падения скорости -- это, например,
    припаркованная машина; одно падение скорости без стоящих -- обычное
    торможение потока. Падение берётся за окно, а не в текущий момент:
    базовая скорость через ~10 с подстраивается под новую норму, и drop
    уменьшается, хотя машины всё ещё стоят.

    Ограничение: длинная очередь на красный свет (стоит > stop_s и поток
    замедлился) по этому правилу тоже даст "ДТП?". Отличить можно по тому,
    что на светофоре полоса потом уезжает целиком -- это следующий шаг.
    """

    def __init__(self, fps: float, stop_s: float = 20.0, drop_min: float = 0.35,
                 lookback_s: float = 40.0):
        """drop_min 0.35, а не 0.5: на view4 после удара фургон уходит на обочину,
        остальные полосы едут, и средняя скорость падает только на 36-42%;
        на b2D9 -- на 68-87%. Ложных срабатываний на нормальных видео и пустых
        дорогах нет и при 0.35: там нигде ничего не стоит дольше 20 с."""
        self.fps = float(fps)
        self.stop_s = float(stop_s)
        self.drop_min = float(drop_min)
        self.lookback = max(1, round(fps * lookback_s))
        self.t_first = NAN        # когда правило сработало впервые (через склейки не сбрасывается)
        self.reset()

    def reset(self) -> None:
        self.drops = deque(maxlen=self.lookback)
        self.active = False       # сейчас есть "ДТП?"
        self.boxes = []           # рамки стоящих, которые признаны "ДТП?"

    def update(self, sm: SpeedMeter, st, t: float) -> "IncidentRule":
        """sm -- SpeedMeter после update; st -- StoppedState после update."""
        if not st.valid:
            return self       # негодный кадр (вспышка, чёрный кадр): решение не меняем
        self.drops.append(sm.drop)
        vals = [d for d in self.drops if not math.isnan(d)]
        max_drop = max(vals) if vals else 0.0
        long_stop = [b for b in st.blobs if b[5] >= self.stop_s]
        self.active = bool(long_stop) and max_drop >= self.drop_min
        self.boxes = long_stop if self.active else []
        if self.active and math.isnan(self.t_first):
            self.t_first = t
        return self
