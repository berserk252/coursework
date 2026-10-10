"""
Очистка оптического потока -- общий модуль для всех скриптов курсовой:
Heatmap.py, opticalflowwithdecay.py, flow_heatmap_with_decay.py, divergence_curl.py.

Порядок обработки одного кадра (FlowCleaner.update):

    1. компенсация движения камеры   -- дрон, панорама, тряска штатива
    2. бегущее среднее по u и v      -- F_t = a*F_new + (1-a)*F_{t-1} (опционально)
    3. мёртвая зона                  -- порог по шумовому полу и по размытию Farneback
    4. постоянство направления       -- дрожание меняет знак, машина едет в одну сторону
    5. изменение яркости             -- где картинка не изменилась, там движения нет
    6. маска дороги                  -- деревья и блики качаются, векторы сокращаются

Шаги 3-6 перенесены сюда из flow_heatmap_with_decay.py без изменений, чтобы
тот же самый, уже проверенный, конвейер работал во всех скриптах.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


# ---------------------------------------------------------------------------
#  1. Компенсация движения камеры
# ---------------------------------------------------------------------------

class CameraMotion:
    """Оценивает движение камеры между кадрами и вычитает его из потока.

    Зачем: если камера движется (дрон) или дрожит (штатив на ветру), к
    скорости машин в каждом пикселе прибавляется скорость самой камеры, и
    "движется" весь кадр, включая асфальт.

    Как: фон занимает большую часть кадра, поэтому движение камеры -- это
    аффинное преобразование (сдвиг + поворот + масштаб + наклон), которое
    лучше всего объясняет поток в БОЛЬШИНСТВЕ точек. Его ищем RANSAC по
    самому плотному потоку на сетке точек: машины для RANSAC -- выбросы.
    Затем из каждого пикселя вычитаем скорость, которую создаёт камера:

        V_камеры(x, y) = M @ (x, y, 1) - (x, y)
        V_машин        = V_измеренное - V_камеры

    Почему по плотному потоку, а не по отдельным углам (goodFeaturesToTrack
    + Lucas-Kanade, как в traffic_field/egomotion.py): проверено на трёх
    видео, медиана |потока| по кадру (это фон) после компенсации:

        видео                      без     углы+LK    плотный поток
        1301121611 (дрон)          0.69    0.70       0.38
        1900 (камера на столбе)    0.077   0.32       0.061
        128195 (деревья)           0.077   --         0.048

    Углы+LK на статичной камере делали фон в 4 раза ХУЖЕ: порог RANSAC 2 px
    огромен по сравнению с настоящим дрожанием 0.07 px, и в модель уходила
    ошибка. Плотный поток -- та же величина, которую мы потом чистим, поэтому
    модель подгоняется ровно под неё, и порог можно взять 0.3 px.

    Предохранитель: компенсация применяется, только если она действительно
    уменьшила фон (медиану |потока|) хотя бы на 10%. Иначе поток остаётся
    как есть -- хуже, чем без компенсации, не станет ни на одном видео.
    """

    def __init__(self, step: int = 8, ransac_px: float = 0.3, min_gain: float = 0.9):
        self.step = step
        self.ransac_px = ransac_px
        self.min_gain = min_gain
        self.matrix: np.ndarray | None = None   # 2x3, prev -> curr; None = не применяли
        self.magnitude = 0.0                    # средняя скорость камеры, px/кадр
        self._shape = None

    def _prepare(self, h: int, w: int) -> None:
        if self._shape == (h, w):
            return
        self._shape = (h, w)
        ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
        self._xs, self._ys = xs, ys
        m = 2 * self.step                       # края кадра у Farneback ненадёжны
        gy, gx = np.mgrid[m:max(m + 1, h - m):self.step, m:max(m + 1, w - m):self.step]
        self._gy, self._gx = gy.ravel(), gx.ravel()
        self._src = np.stack([self._gx, self._gy], axis=1).astype(np.float32)

    def camera_field(self, M: np.ndarray) -> np.ndarray:
        """Поле скоростей, которое создаёт движение камеры, (H, W, 2)."""
        xs, ys = self._xs, self._ys
        return np.stack([M[0, 0] * xs + M[0, 1] * ys + M[0, 2] - xs,
                         M[1, 0] * xs + M[1, 1] * ys + M[1, 2] - ys], axis=2)

    def compensate(self, flow: np.ndarray) -> np.ndarray:
        h, w = flow.shape[:2]
        self._prepare(h, w)
        self.matrix = None
        self.magnitude = 0.0
        if len(self._src) < 12:
            return flow
        dst = self._src + flow[self._gy, self._gx]
        M, _ = cv2.estimateAffine2D(self._src, dst, method=cv2.RANSAC,
                                    ransacReprojThreshold=self.ransac_px,
                                    maxIters=2000, confidence=0.99)
        if M is None:
            return flow
        cam = self.camera_field(M)
        comp = flow - cam
        before = float(np.median(np.linalg.norm(flow[::2, ::2], axis=2)))
        after = float(np.median(np.linalg.norm(comp[::2, ::2], axis=2)))
        if after >= self.min_gain * before:
            return flow
        self.matrix = M.astype(np.float32)
        self.magnitude = float(np.mean(np.linalg.norm(cam[::4, ::4], axis=2)))
        return comp.astype(np.float32)

    def align_prev(self, prev_gray: np.ndarray) -> np.ndarray:
        """Предыдущий кадр, сдвинутый так же, как сдвинулась камера.

        Нужен фильтру по изменению яркости: при движущейся камере меняется
        яркость ВСЕГО кадра, и фильтр перестаёт что-либо отсекать.
        """
        if self.matrix is None:
            return prev_gray
        h, w = prev_gray.shape[:2]
        return cv2.warpAffine(prev_gray, self.matrix, (w, h), borderMode=cv2.BORDER_REPLICATE)


# ---------------------------------------------------------------------------
#  3. Подавление шума и размытия оптического потока
# ---------------------------------------------------------------------------

def noise_floor(mag: np.ndarray, prev: float, decay: float = 0.98) -> float:
    """Оценка шумового пола: робастная статистика по самым медленным пикселям.

    Нижний квартиль кадра -- это заведомо неподвижный фон (асфальт, разметка,
    здания). Его скорость обязана быть нулевой, всё что там осталось -- шум
    алгоритма. 1.4826*MAD ~ СКО для нормального распределения.
    """
    m = mag.reshape(-1)
    if m.size > 100_000:
        m = m[:: m.size // 50_000]
    low = m[m <= np.percentile(m, 25.0)]
    if low.size == 0:
        est = 0.0
    else:
        med = float(np.median(low))
        est = float(1.4826 * np.median(np.abs(low - med)) + med)
    return est if prev <= 0 else decay * prev + (1.0 - decay) * est


def apply_deadband(flow: np.ndarray, mag: np.ndarray, floor: float,
                   k: float = 3.0, rel: float = 0.08):
    """Обнуляет всё ниже порога. Порог -- максимум из двух:

        k * шумовой_пол      -- собственный шум оптического потока
        rel * p95(скорость)  -- размытие Farneback вокруг машины

    Второе слагаемое принципиально: Farneback считает поток по окну 15x15 и
    размазывает движение далеко за габариты автомобиля, на обочину и газон.
    Величина размытия пропорциональна скорости самой машины, а не шумовому
    полу, поэтому абсолютный порог (как было: mag < 1.0) её не убирает.
    Измерено: при пороге по одному лишь шумовому полу поток проходит на 65%
    площади кадра, в котором едут четыре машины; с относительным порогом --
    на 39%, при этом скорость самих машин не меняется вовсе.

    Порог жёсткий, а не мягкий: мягкий вычитает величину порога из модуля и
    занижает скорость, а она дальше идёт в цвет карты.
    """
    tau = max(k * floor, rel * float(np.percentile(mag, 95.0)))
    # предохранитель: порог не имеет права стереть больше 98% поля
    cap = float(np.percentile(mag, 98.0))
    if cap > 0:
        tau = min(tau, cap)
    keep = mag >= max(tau, 1e-6)
    return np.where(keep, mag, 0.0).astype(np.float32), keep, tau


# ---------------------------------------------------------------------------
#  6. Маска дороги
# ---------------------------------------------------------------------------

class RoadMask:
    """Маска дороги по согласованности движения во времени.

    Деревья, кусты и блики РЕАЛЬНО движутся, поэтому никакой порог по
    скорости их не уберёт. Но их движение колебательное: за N кадров
    векторная сумма близка к нулю, тогда как у полосы движения векторы
    складываются. Отношение

        coherence = |сумма векторов| / сумма модулей

    у дороги близко к 1 (измерено 0.93), у растительности -- к нулю (0.09).

    decay < 1 -- забывать старую статистику. FlowCleaner включает это, когда
    камера движется: дорога тогда ползёт по кадру, и маска должна ползти за ней.
    """

    def __init__(self, cell: int = 16, coherence_min: float = 0.45,
                 activity_min: float = 0.01, warmup: int = 45):
        self.cell = cell
        self.coherence_min = coherence_min
        self.activity_min = activity_min
        self.warmup = warmup
        self.vec_sum = None
        self.mag_sum = None
        self.n = 0
        self.min_frames = 8      # ~0.3 с: меньше статистика ещё бессмысленна
        self.smooth = None
        self.decay = 1.0

    def update(self, flow: np.ndarray) -> None:
        h, w = flow.shape[:2]
        gh, gw = max(1, h // self.cell), max(1, w // self.cell)
        fx = cv2.resize(flow[..., 0], (gw, gh), interpolation=cv2.INTER_AREA)
        fy = cv2.resize(flow[..., 1], (gw, gh), interpolation=cv2.INTER_AREA)
        g = np.stack([fx, fy], axis=-1)
        if self.vec_sum is None:
            self.vec_sum = np.zeros_like(g)
            self.mag_sum = np.zeros((gh, gw), np.float32)
        if self.decay < 1.0:
            self.vec_sum *= self.decay
            self.mag_sum *= self.decay
        self.vec_sum += g
        self.mag_sum += np.linalg.norm(g, axis=2)
        self.n += 1

    def mask(self, shape) -> np.ndarray:
        """Маска в разрешении кадра, 0..1.

        Раньше до warmup-кадра маска была единицей (пропускала всё), а на
        warmup-кадре резко включалась -- на видео это был скачок: на
        heatmap_99 окраска падала с 53% до 23% за один кадр (кадр 44).

        Теперь маска считается с самого начала -- колебание деревьев имеет
        период меньше секунды, и согласованность различает их уже через
        несколько кадров, -- а сама маска сглажена во времени, поэтому
        меняться скачком она не может в принципе.
        """
        if self.vec_sum is None or self.n < self.min_frames:
            self.smooth = None
            return np.zeros(shape, np.float32)
        coh = np.linalg.norm(self.vec_sum, axis=2) / np.maximum(self.mag_sum, 1e-6)
        act = self.mag_sum / max(self.n, 1)
        act = act / max(float(act.max()), 1e-6)
        # Порог активности -- доля от САМОЙ быстрой клетки кадра. Было 0.05:
        # ползущая полоса затора рядом с быстрой полосой не дотягивала до 5% и
        # выпадала из маски целиком -- ни стрелок, ни скорости, ни div/rot.
        # Измерено на 3_/471 (затор справа): активность правых полос 3.9% от
        # максимума, в маску попадало 27% их площади; с 0.01 -- 94%, движение там
        # 19.6% -> 36.6%; на 4_/440 road&moving 9.7% -> 14.8%. Ложное движение на
        # 10 пустых дорогах не изменилось (<=0.03 п.п.), деревья на 128195 по-прежнему
        # отсекает согласованность (8.70% -> 8.85%).
        m = ((coh >= self.coherence_min) & (act >= self.activity_min)).astype(np.float32)
        m = cv2.GaussianBlur(m, (0, 0), 1.0)
        m = cv2.resize(m, (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)
        # стартуем с нуля, а не с m: иначе в момент появления маски карта
        # вспыхивает на полную силу за один кадр (был скачок 14.8 на кадре 7)
        if self.smooth is None or self.smooth.shape != m.shape:
            self.smooth = np.zeros_like(m)
        self.smooth = 0.9 * self.smooth + 0.1 * m
        return self.smooth


# ---------------------------------------------------------------------------
#  Весь конвейер
# ---------------------------------------------------------------------------

@dataclass
class CleanFlow:
    flow: np.ndarray       # (H, W, 2) поток без движения камеры (и сглаженный, если ema_alpha < 1)
    mag: np.ndarray        # (H, W) его модуль, до порогов
    mag_clean: np.ndarray  # (H, W) модуль после всех фильтров, фон = 0
    moving: np.ndarray     # (H, W) bool, где есть настоящее движение
    road_w: np.ndarray     # (H, W) 0..1, маска дороги (только для показа)

    def field(self) -> np.ndarray:
        """Очищенное векторное поле: фон, дрожание и шум обнулены."""
        return np.where(self.moving[..., None], self.flow, 0.0).astype(np.float32)


class FlowCleaner:
    """Один вызов update() на кадр: сырой поток Farneback -> очищенный.

        cleaner = FlowCleaner()
        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, ...)
        c = cleaner.update(flow, gray, prev_gray)
        c.mag_clean, c.moving, c.road_w, c.field()
    """

    def __init__(self, egomotion: bool = True, ema_alpha: float = 1.0,
                 deadband_k: float = 2.0, deadband_rel: float = 0.02,
                 coh_beta: float = 0.7, coh_min: float = 0.7,
                 change_thresh: float = 6.0, road_mask: bool = True,
                 road_decay_moving: float = 0.98, camera_moving_px: float = 0.3):
        self.camera = CameraMotion() if egomotion else None
        self.ema_alpha = float(ema_alpha)
        self.deadband_k = float(deadband_k)
        self.deadband_rel = float(deadband_rel)
        self.coh_beta = float(coh_beta)
        self.coh_min = float(coh_min)
        self.change_thresh = float(change_thresh)
        self.road = RoadMask(cell=16, warmup=45) if road_mask else None
        self.road_decay_moving = road_decay_moving
        self.camera_moving_px = camera_moving_px
        self.noise = 0.0          # оценка шумового пола, px/кадр
        self.tau = 0.0            # текущий порог мёртвой зоны
        self._ema = None
        self._coh_vec = None      # короткое среднее ВЕКТОРА потока в каждой точке
        self._coh_mag = None      # короткое среднее МОДУЛЯ потока в каждой точке

    def update(self, flow: np.ndarray, gray: np.ndarray, prev_gray: np.ndarray) -> CleanFlow:
        h, w = flow.shape[:2]

        # 1. движение камеры
        if self.camera is not None:
            flow = self.camera.compensate(flow)
            prev_gray = self.camera.align_prev(prev_gray)

        # 2. бегущее среднее по ЗНАКОВЫМ компонентам u и v. По модулю |v|
        # усреднять нельзя: модуль всегда >= 0, шум не сокращается, а только
        # становится гладким. У знаковых компонент плюсы и минусы дрожания
        # взаимно гасятся.
        if self.ema_alpha < 1.0:
            a = self.ema_alpha
            self._ema = flow.copy() if self._ema is None else a * flow + (1.0 - a) * self._ema
            flow = self._ema.astype(np.float32)

        mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)

        # 3. Порог адаптивный: по шумовому полу И по доле от скорости в кадре.
        # Абсолютный порог (mag < 1.0 в старой версии) не годится из-за
        # перспективы: дальние машины идут медленнее порога и пропадают,
        # а размытие вокруг ближних порог проходит.
        # Значения по умолчанию k=2, rel=0.02 (было 3 и 0.08). С rel=0.08 порог
        # задавала самая быстрая ближняя машина: один грузовик поднимал его до
        # 0.9 px/кадр и стирал все дальние и ползущие машины -- "застой" не был
        # виден. Измерено (медленные машины 0.05-0.2 px/кадр): у места ДТП на
        # b2D9 до удара 14% -> 61%, затор встречки 45% -> 72%, пробки 3_/4_
        # 41% -> 61%; ложное движение на 10 пустых дорогах не изменилось
        # (0.53% -> 0.54% в среднем), ореол вокруг машин 11.9% -> 12.3%.
        # coh_min и change_thresh снижать нельзя: ложное движение растёт в 5-20 раз.
        self.noise = noise_floor(mag, self.noise)
        mag_clean, moving, self.tau = apply_deadband(
            flow, mag, self.noise, k=self.deadband_k, rel=self.deadband_rel)

        # 4. ФИЛЬТР ПОСТОЯНСТВА НАПРАВЛЕНИЯ.
        # На неподвижной разметке Farneback видит 0.163 px/кадр -- в 2.7
        # раза больше, чем на асфальте рядом (0.061). Причина: камера
        # слегка дрожит (~0.04 px/кадр, знак меняется), а на резкой
        # границе белой полосы даже такой сдвиг и шум сжатия дают заметный
        # поток; на ровном асфальте его почти не измерить. Порог по
        # величине пропускал 42% пикселей неподвижной разметки.
        # Отличие от машины -- во времени: дрожание меняет направление,
        # машина едет в одну сторону. Отношение
        #     |среднее вектора| / среднее модуля
        # за последние ~3 кадра у дрожания близко к нулю, у машины -- к 1.
        # Для нового движения оно равно 1 сразу (вектор и модуль растут
        # одинаково), поэтому машины не запаздывают.
        # Измерено на 1900: разметка 42% -> 11%, машины 100% -> 99.3%;
        # дальше 40 px от машин проходит 0.8% неподвижной дороги.
        if self.coh_min > 0:
            b = self.coh_beta
            if self._coh_vec is None:
                self._coh_vec = np.zeros_like(flow)
                self._coh_mag = np.zeros_like(mag)
            self._coh_vec = b * self._coh_vec + (1.0 - b) * flow
            self._coh_mag = b * self._coh_mag + (1.0 - b) * mag
            coh = np.linalg.norm(self._coh_vec, axis=2) / np.maximum(self._coh_mag, 1e-6)
            moving = moving & (coh >= self.coh_min)
            mag_clean = np.where(moving, mag_clean, 0.0).astype(np.float32)

        # 5. ФИЛЬТР ПО ИЗМЕНЕНИЮ ЯРКОСТИ.
        # Farneback считает поток по окну 15x15 и размазывает движение
        # машины на ~15 px в стороны. Это размытие движется ВМЕСТЕ с
        # машиной, направление у него постоянное, и фильтр выше его не
        # ловит -- поэтому пунктир рядом с полосой продолжал краситься.
        # Но у по-настоящему неподвижного пикселя яркость между кадрами
        # не меняется, сколько бы потока ни насчитал рядом Farneback.
        # Маску изменения расширяем, иначе теряются ровные однотонные
        # участки кузова, где яркость тоже почти не меняется.
        # Измерено на 1900: размытие у машин 21.6% -> 11.2%,
        # машины 99.3% -> 94.0%.
        if self.change_thresh > 0:
            diff = cv2.GaussianBlur(
                cv2.absdiff(gray, prev_gray).astype(np.float32), (0, 0), 1.0)
            changed = cv2.dilate((diff > self.change_thresh).astype(np.uint8),
                                 np.ones((11, 11), np.uint8)) > 0
            moving = moving & changed
            mag_clean = np.where(moving, mag_clean, 0.0).astype(np.float32)

        # 6. Маска влияет ТОЛЬКО на прозрачность при показе, а не на
        # статистику. Если отсекать ею сами измерения (как было:
        # moving & (road_w > 0.5)), то в момент, когда плавно растущая
        # маска пересекает порог, у целой области одновременно
        # начинается учёт -- на видео это скачок (был 9.2 на кадре 13
        # при фоне ~3). Так статистика копится с первого кадра везде,
        # а маска лишь плавно гасит то, что не дорога.
        if self.road is not None:
            moving_cam = self.camera is not None and self.camera.magnitude > self.camera_moving_px
            self.road.decay = self.road_decay_moving if moving_cam else 1.0
            self.road.update(flow)
            road_w = self.road.mask((h, w))
        else:
            road_w = np.ones((h, w), np.float32)

        return CleanFlow(flow, mag, mag_clean, moving, road_w)


# ---------------------------------------------------------------------------
#  Монтажные склейки и негодные кадры
# ---------------------------------------------------------------------------

class SceneGuard:
    """Находит монтажные склейки, заставки и чёрные кадры.

    Зачем: видео из интернета часто смонтированы -- новостная заставка,
    склейка двух ракурсов, чёрная концовка. Через склейку Farneback видит
    "движение" всего кадра: на заставке b2D9m3upSZQ поток доходил до
    100-550 px/кадр, средняя скорость -- до 2000 px/с. Состояние всех
    фильтров (маска дороги, бегущие средние, базовая скорость) после этого
    испорчено надолго. Поэтому на склейке состояние сбрасывается.

    Склейка -- резкий скачок картинки на миниатюре 160x90:
        d = mean|кадр_t - кадр_{t-1}|  >  max(20, 8 * медиана d за ~2 с)
        или расстояние гистограмм яркости (32 бина, 0.5*L1) > 0.5
    Перед сравнением яркость кадра приводится к яркости предыдущего
    (кадр * mean_{t-1} / mean_t): вспышка, мерцание, автоэкспозиция меняют
    яркость всего кадра, но картинка та же -- это не склейка.
    Негодный кадр -- почти однотонный или почти чёрный (СКО яркости < 4 или
    средняя < 5), или вспышка: средняя яркость скачком изменилась больше чем
    на max(8, 12%), а картинка та же. По негодному кадру ничего не измеряем,
    но и не сбрасываем.

    На b2D9m3upSZQ находит заставку -> камера (5.92 с), смену ракурса
    (11.17 с) и чёрную концовку (~52 с). На view3 кадр 459 (36.72 с) целиком
    темнее на 20% (137 -> 110 -> 135): без приведения яркости это была
    "склейка" (d = 27), сброс стирал стоящий после ДТП фургон. С приведением
    d = 2.9 -- обычный кадр; он и следующий помечаются вспышкой.
    """

    def __init__(self, fps: float = 25.0):
        self.hist_len = max(5, int(round(2 * fps)))
        self.prev = None
        self.prev_hist = None
        self.prev_mean = 0.0
        self.prev_valid = False
        self.ds = []

    @staticmethod
    def _hist(small: np.ndarray) -> np.ndarray:
        hist = cv2.calcHist([np.clip(small, 0, 255).astype(np.uint8)], [0], None, [32], [0, 256]).ravel()
        return hist / max(float(hist.sum()), 1.0)

    def update(self, gray: np.ndarray):
        """Возвращает (cut, valid): cut -- на этом кадре склейка, valid -- кадр годен."""
        small = cv2.resize(gray, (160, 90), interpolation=cv2.INTER_AREA).astype(np.float32)
        mean = float(small.mean())
        # негоден -- почти однотонный или почти чёрный; не "просто тёмный": у ночной
        # трассы средняя яркость ~10 при СКО ~16, и это рабочая картинка
        valid = float(small.std()) >= 4.0 and mean >= 5.0
        hist = self._hist(small)
        cut = False
        if self.prev is not None:
            gain = self.prev_mean / max(mean, 1.0)
            d = float(np.mean(np.abs(small - self.prev)))
            hd = 0.5 * float(np.abs(hist - self.prev_hist).sum())
            # яркость приводим к прежней, только если оба кадра рабочие: чёрный
            # кадр -> картинка (и обратно) остаётся склейкой. Берём меньшее из
            # "как есть" и "после приведения": приведение может только убрать
            # ложную склейку, но не добавить новую (у порога при gain ~1.02
            # приведённая разница бывает чуть больше исходной)
            if self.prev_valid and valid and 0.5 <= gain <= 2.0:
                norm = small * gain
                d = min(d, float(np.mean(np.abs(norm - self.prev))))
                hd = min(hd, 0.5 * float(np.abs(self._hist(norm) - self.prev_hist).sum()))
            med = float(np.median(self.ds)) if self.ds else d
            cut = d > max(20.0, 8.0 * med) or hd > 0.5
            self.ds.append(d)
            if len(self.ds) > self.hist_len:
                self.ds.pop(0)
            if cut:
                self.ds = []
            elif abs(mean - self.prev_mean) > max(8.0, 0.12 * self.prev_mean):
                valid = False           # вспышка: кадр пропускаем, состояние не трогаем
        self.prev, self.prev_hist, self.prev_mean = small, hist, mean
        self.prev_valid = float(small.std()) >= 4.0 and mean >= 5.0
        return cut, valid
