import cv2
import numpy as np

from flow_filters import FlowCleaner

VIDEO_PATH = "video_small.mp4"

DECAYS = [None, 0.90, 0.95, 0.98]  # None = без decay
NAMES = [
    "heatmap_nodecay.mp4",
    "heatmap_90.mp4",
    "heatmap_95.mp4",
    "heatmap_98.mp4"
]

EGOMOTION = True   # вычитать движение камеры (дрон, тряска)
ROAD_MASK = True   # гасить деревья, кусты, блики

cap = cv2.VideoCapture(VIDEO_PATH)
FPS = cap.get(cv2.CAP_PROP_FPS) or 30

ret, frame = cap.read()
if not ret:
    raise ValueError("Cannot read video")

H, W = frame.shape[:2]
prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

cleaner = FlowCleaner(egomotion=EGOMOTION, road_mask=ROAD_MASK)

# 🔥 отдельный аккумулятор под каждый decay
accums = [np.zeros((H, W), dtype=np.float32) for _ in DECAYS]

# 🔥 writers
writers = []
for name in NAMES:
    out = cv2.VideoWriter(
        name,
        cv2.VideoWriter_fourcc(*"mp4v"),
        FPS,
        (W, H)
    )
    if not out.isOpened():
        raise ValueError(f"Failed writer: {name}")
    writers.append(out)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # ✅ считаем flow 1 раз
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, gray,
        None,
        0.5, 3, 15, 3, 5, 1.2, 0
    )

    # Очищенный модуль скорости: без движения камеры, шума, дрожания и
    # размытия Farneback вокруг машин. Фон в нём ровно 0.
    #
    # Раньше был фиксированный порог mag < 1.0. Шум ниже порога он убирал, но
    # (1) дальние машины из-за перспективы идут медленнее 1 px/кадр и
    # пропадали, (2) размытие вокруг ближних машин и качающиеся деревья
    # порог проходили. Накопление H = d*H + mag раздувает любой остаток шума
    # в 1/(1-d) раз (при d=0.98 -- в 50 раз), и карта заливалась целиком.
    # Ноль же не накапливается ни при каком d.
    c = cleaner.update(flow, gray, prev_gray)
    mag = c.mag_clean
    # маска дороги -- только при показе, накопление идёт везде с первого кадра
    road_w = c.road_w

    # 🔥 обновляем все варианты сразу
    for i, decay in enumerate(DECAYS):
        if decay is None:
            # ❗ без decay → просто накапливаем
            accums[i] += mag
        else:
            accums[i] = accums[i] * decay + mag

        current = accums[i] * road_w

        # percentiles (робастная нормализация)
        # Важно: цветовая шкала должна быть "больше = горячее".
        # Перцентиль -- только по точкам, где движение было: после очистки
        # фон равен нулю, и p99 по всему кадру был бы нулём, если машины
        # занимают меньше 1% кадра.
        low = 0.0
        live = current[current > 0]
        high = float(np.percentile(live, 99)) if live.size else 0.0

        if high - low < 1e-6:
            norm = np.zeros_like(current)
        else:
            clipped = np.clip(current, low, high)
            norm = (clipped - low) / (high - low)

        # heatmap: КАК У ТЕБЯ ПО СМЫСЛУ
        # красный = медленно/мало движения, зелёный = быстро/много движения
        # делаем градиент в HSV: H=0 (red) -> H=60deg (green)
        h = (norm * 30.0).astype(np.uint8)  # 0..30 (OpenCV hue 0..179)
        s = np.full((H, W), 255, dtype=np.uint8)
        v = np.full((H, W), 255, dtype=np.uint8)
        hsv = cv2.merge([h, s, v])
        heatmap = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        # Прозрачность пропорциональна значению. Раньше любая точка с
        # norm > 1/255 красилась с одинаковой силой 0.6: след, от которого
        # осталось 0.4%, выглядел так же, как свежий. При decay 0.98 след
        # гаснет до 1/255 за ~270 кадров, и вся дорога была сплошь красной.
        # Теперь старый след плавно исчезает, а пустой кадр не затемняется.
        alpha = (np.sqrt(norm) * 0.6)[..., None]
        vis = (frame * (1.0 - alpha) + heatmap * alpha).astype(np.uint8)

        writers[i].write(vis)

    prev_gray = gray

cap.release()
for w in writers:
    w.release()
