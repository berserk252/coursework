"""
Долгий фон дороги и сравнение с ним текущего кадра.

Зачем. Ротор и дивергенция "скачут", и в поле появляется дырка (нет движения там,
где обычно ездят) -- но пустая полоса, на которой просто нет машин, тоже даёт
дырку. Отличить её от стоящего объекта можно по картинке: если на месте дырки
кадр совпадает с фоном (пустой асфальт), объекта нет и тревогу давать не надо.

Фон собирается с задержкой из неподвижных точек. При запуске в точке
различаются два уровня яркости: проезжающая машина не должна портить среднее
асфальта и его дисперсию. После накопления устойчивого эталона в него попадают
только похожие наблюдения; новый стоящий объект не становится фоном.
Общий сдвиг освещения вычитается перед обучением. Внутри дырок устойчивый фон
заморожен, а ещё неизвестный фон может доучиваться на задержанных кадрах.

    bg = BackgroundModel(fps, road_mask)
    bg.update(gray, moving, frozen)         # каждый кадр
    frac, ok = bg.compare(gray, hole_mask)  # доля точек дырки, отличных от фона
"""

from __future__ import annotations

from collections import deque
import warnings

import cv2
import numpy as np

from mask_regions import motion_footprint, thin_color_lines


class BackgroundModel:
    def __init__(self, fps: float, road: np.ndarray, delay_s: float = 12.0, cap_s: float = 30.0,
                 rate_hz: float = 5.0, min_cnt: int = 10, diff_thr: float = 28.0, std_max: float = 14.0):
        """road -- (H, W) bool, дорога; delay_s -- задержка пополнения фона, с;
        cap_s -- память фона (наибольший вес накопленного среднего), с;
        rate_hz -- сколько раз в секунду фон пополняется; min_cnt -- сколько пополнений
        нужно точке, чтобы фону в ней можно было верить; diff_thr -- отличие яркости
        (после поправки на общее освещение), с которого точка считается отличной от фона."""
        self.road = road
        self.step = max(1, int(round(fps / rate_hz)))
        actual_rate = fps / self.step
        self.delay_n = max(1, int(round(delay_s * actual_rate)))
        self.bootstrap_n = max(1,min(self.delay_n,round(8 * actual_rate)))
        self.cap = max(2.0, cap_s * actual_rate)
        self.recovery_count = max(min_cnt,round(max(10.,delay_s)*actual_rate))
        self.min_cnt = min_cnt
        self.diff_thr = diff_thr
        self.std_max = std_max
        self.k_dil = np.ones((7, 7), np.uint8)
        self.reset()

    def reset(self) -> None:
        H, W = self.road.shape
        self.bg = np.zeros((H, W), np.float32)
        self.cnt = np.zeros((H, W), np.float32)
        self.var = np.zeros((H, W), np.float32)      # разброс яркости точки во времени
        self._other = np.zeros((H, W), np.float32)
        self._other_cnt = np.zeros((H, W), np.float32)
        self._other_var = np.zeros((H, W), np.float32)
        self._clear_mean=np.zeros((H,W),np.float32)
        self._clear_count=np.zeros((H,W),np.float32)
        self._clear_var=np.zeros((H,W),np.float32)
        self.recovered=np.zeros((H,W),bool)
        self.newly_recovered=np.zeros((H,W),bool)
        self.ring: deque = deque()
        self.i = 0
        self._bootstrapped = False

    def _bootstrap(self):
        """Робастный ранний эталон вместо первых десяти неподвижных образцов.

        Первые образцы могут принадлежать проезжающей машине. Если сразу
        заморозить её как известный фон, асфальт затем уже не восстановится.
        Медиана окна выбирает преобладающий уровень; дисперсия считается только
        внутри этого уровня, без смешивания асфальта и кузовов.
        """
        # Эталон запуска берём из ранней части окна. Иначе раннее ДТП
        # успевает занять большую часть неподвижных наблюдений за 12 с.
        # Проезжающие машины убирает медиана всех ранних кадров; маска
        # движения проверяет, что выбранный уровень хотя бы раз был тихим.
        initial=list(self.ring)[:self.bootstrap_n]
        samples = np.stack([g for g, _ in initial])
        still = np.stack([s for _, s in initial]) & self.road[None]
        samples[:,~self.road] = np.nan
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            center = np.nanmedian(samples, axis=0)
        selected = self.road[None] & (np.abs(samples - center[None]) <= self.std_max)
        selected &= (selected & still).any(axis=0)[None]
        count = selected.sum(axis=0).astype(np.float32)
        mean = np.where(selected, samples, 0).sum(axis=0) / np.maximum(count, 1)
        variance = np.where(selected, (samples - mean[None]) ** 2, 0).sum(axis=0) / np.maximum(count, 1)
        self.bg[:] = mean
        self.cnt[:] = count
        self.var[:] = variance

    @staticmethod
    def smooth(gray: np.ndarray) -> np.ndarray:
        return cv2.GaussianBlur(gray, (5, 5), 0).astype(np.float32)

    def trusted(self) -> np.ndarray:
        return (self.cnt >= self.min_cnt) & (self.var <= self.std_max ** 2)

    def coverage(self, region: np.ndarray) -> float:
        """Доля области, для которой накоплен устойчивый эталон."""
        return float(self.trusted()[region].mean()) if region.any() else 0.0

    def foreground(self, gray: np.ndarray, still_ref: np.ndarray | None = None):
        """Пиксельный передний план и известный фон; неизвестность не равна объекту.

        Порог учитывает шум конкретного пикселя. Локальный допуск 1 px подавляет
        дрожание неизменных линий разметки, но не размывает маску на пустой асфальт.
        """
        good = self.trusted() & self.road
        reference = good if still_ref is None else good & still_ref
        g = self.smooth(gray)
        g = g - self._offset(g, reference)
        low = cv2.erode(self.bg, np.ones((3, 3), np.uint8))
        high = cv2.dilate(self.bg, np.ones((3, 3), np.uint8))
        distance = np.maximum(np.maximum(low - g, g - high), 0)
        threshold = np.clip(8 + 2.5 * np.sqrt(np.maximum(self.var, 0)), 12, self.diff_thr)
        return (distance > threshold) & good, good

    def _offset(self, gray: np.ndarray, reference: np.ndarray) -> float:
        values = (gray - self.bg)[reference]
        if values.size < 20:
            return 0.0
        return float(np.median(values[::max(1, values.size // 4000)]))

    def _learn(self, gray, mask, mean, count, variance) -> None:
        weight = 1.0 / np.minimum(count[mask] + 1.0, self.cap)
        delta = gray[mask] - mean[mask]
        mean[mask] += weight * delta
        variance[mask] += weight * ((1 - weight) * delta * delta - variance[mask])
        count[mask] += 1.0

    def update(self, gray: np.ndarray, moving: np.ndarray, frozen: np.ndarray, frame=None) -> None:
        """gray -- кадр; moving -- где сейчас движение; frozen -- защита устойчивого
        фона в дырках. Неизвестный фон может доучиваться. Кадр попадает в фон
        из очереди через delay_s."""
        self.i += 1
        self.newly_recovered.fill(False)
        if self.i % self.step:
            return
        protected = motion_footprint(moving)
        still = cv2.dilate(protected.astype(np.uint8), self.k_dil) == 0
        self.ring.append((self.smooth(gray), still))
        if self._bootstrapped:
            self._recover_empty(gray,still,frame)
        if len(self.ring) <= self.delay_n:
            return
        if not self._bootstrapped:
            self._bootstrap()
            # Эти исходные наблюдения уже учтены. Повторно из очереди их не
            # используем; следующие обновления вновь проходят полную задержку.
            self.ring.clear()
            self._bootstrapped = True
            return
        g, st = self.ring.popleft()
        good = self.trusted()
        # Нельзя навсегда заморозить ещё не накопленный эталон: иначе молодая
        # дырка блокирует обучение, compare() остаётся NaN и пустая полоса
        # больше никогда не проверяется по фону.
        mask = st & self.road & (~frozen | ~good)
        g = g - self._offset(g, st & self.road & good & ~frozen)
        # У известного фона новый непохожий объект — передний план. Отсутствия
        # оптического потока внутри однотонной машины недостаточно для обучения.
        threshold = np.where(good, self.diff_thr, min(self.diff_thr, self.std_max))
        same = mask & ((self.cnt == 0) | (np.abs(g - self.bg) <= threshold))
        self._learn(g, same, self.bg, self.cnt, self.var)

        # Пока эталон неустойчив, отдельная гипотеза хранит другой уровень
        # яркости. Выбросы от машин не входят в дисперсию основной гипотезы.
        alternative = mask & ~same & ~good
        fresh = alternative & ((self._other_cnt == 0)
                               | (np.abs(g - self._other) > min(self.diff_thr, self.std_max)))
        self._other[fresh] = g[fresh]
        self._other_cnt[fresh] = 0
        self._other_var[fresh] = 0
        self._learn(g, alternative, self._other, self._other_cnt, self._other_var)
        promote = alternative & (self._other_cnt > self.cnt)
        for primary, secondary in ((self.bg, self._other), (self.cnt, self._other_cnt),
                                   (self.var, self._other_var)):
            previous = primary[promote].copy()
            primary[promote] = secondary[promote]
            secondary[promote] = previous

    def _recover_empty(self,gray,still,frame):
        """Исправить начальный след машины по длительно наблюдаемому асфальту.

        Недостаточно неподвижности: прежние детали должны исчезнуть, новых
        деталей мало, а яркость участка должна совпасть с соседним известным
        асфальтом. Учится отдельная гипотеза, не меньше 10–12 с наблюдений.
        """
        foreground,known=self.foreground(gray,still_ref=still)
        if not foreground.any():return
        ignored=np.zeros_like(known) if frame is None else thin_color_lines(frame)
        edges=(cv2.Canny(self.smooth(gray).astype(np.uint8),50,120)>0)&~ignored
        old=cv2.Canny(np.clip(self.bg,0,255).astype(np.uint8),50,120)>0
        near_old=cv2.dilate(old.astype(np.uint8),np.ones((5,5),np.uint8))>0
        novel=edges&~near_old
        g=self.smooth(gray)
        g-=self._offset(g,known&still&~foreground)
        candidate=np.zeros_like(known)
        # Разница разных знаков на кузове и окне разъединяет один старый
        # объект. Соседние фрагменты проверяем совместно, а переобучаем
        # только действительно отличающиеся и похожие на асфальт точки.
        grouping=max(9,round(17*gray.shape[1]/640)|1)
        region=cv2.dilate((foreground&known&still).astype(np.uint8),
                         cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(grouping,grouping)))>0
        region &= known&still
        n,lab,stats,_=cv2.connectedComponentsWithStats(region.astype(np.uint8),8)
        size=max(5,round(17*gray.shape[1]/640)|1)
        kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(size,size))
        for i in range(1,n):
            if stats[i,cv2.CC_STAT_AREA]<max(20,round(60*(gray.shape[1]/640)**2)):continue
            mask=lab==i;near=cv2.dilate(mask.astype(np.uint8),np.ones((3,3),np.uint8))>0
            if (mask&foreground).sum()>.2*gray.size:continue
            old_count=int((old&near).sum());current_count=int((edges&near).sum())
            if old_count<10 or current_count>.7*old_count or (novel&near).sum()>.5*old_count:continue
            ring=(cv2.dilate(mask.astype(np.uint8),kernel)>0)&~mask&known&still&~foreground
            if ring.sum()<20:continue
            level=float(np.median(self.bg[ring]))
            if abs(float(np.median(g[mask]))-level)>self.std_max:continue
            # Даже при соприкосновении с новым транспортом учится только
            # поверхность, похожая на наблюдаемый соседний асфальт.
            candidate|=mask&foreground&~ignored&(np.abs(g-level)<=self.std_max)
        if not candidate.any():return
        changed=candidate&(self._clear_count>0)&(np.abs(g-self._clear_mean)>self.std_max)
        self._clear_count[changed]=0;self._clear_var[changed]=0
        self._learn(g,candidate,self._clear_mean,self._clear_count,self._clear_var)
        ready=candidate&(self._clear_count>=self.recovery_count)&(self._clear_var<=self.std_max**2)
        if ready.any():
            self.bg[ready]=self._clear_mean[ready];self.var[ready]=self._clear_var[ready]
            self.cnt[ready]=self._clear_count[ready]
            self._other_cnt[ready]=0;self._clear_count[ready]=0
            self.recovered|=ready;self.newly_recovered|=ready

    def compare(self, gray: np.ndarray, region: np.ndarray, still_ref: np.ndarray | None = None):
        """Доля точек region, где кадр отличается от фона больше diff_thr, и достаточно ли
        в region устойчивого фона, чтобы судить (>= 50 % точек с min_cnt пополнениями и малым разбросом).
        still_ref -- точки, по которым оценивается общий сдвиг яркости (по умолчанию дорога)."""
        # Нужен устойчивый эталон на достаточной части области; неизвестные
        # точки исключены из оценки различия, а не считаются совпавшими.
        good = self.trusted()
        reg = region & good
        if region.sum() == 0 or reg.sum() < 0.5 * region.sum() or reg.sum() < 20:
            return float("nan"), False
        foreground, _ = self.foreground(gray, still_ref)
        return float(foreground[reg].mean()), True

    def empty_cells(self, gray: np.ndarray, cell: int, gh: int, gw: int, frac_thr: float = 0.25,
                    still_ref: np.ndarray | None = None) -> np.ndarray:
        """(gh, gw) bool: клетка -- пустой асфальт: фону в ней верим (>= 50 % точек устойчивы)
        и кадр с ним совпадает (отличных точек меньше frac_thr)."""
        diff, good = self.foreground(gray, still_ref)
        h, w = gh * cell, gw * cell
        gd = diff[:h, :w].reshape(gh, cell, gw, cell).mean(axis=(1, 3))
        gg = good[:h, :w].reshape(gh, cell, gw, cell).mean(axis=(1, 3))
        return (gg >= 0.5) & (gd / np.maximum(gg, 1e-6) < frac_thr)
