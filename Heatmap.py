import cv2
import numpy as np

VIDEO_PATH = "video_small.mp4"

DECAYS = [None, 0.90, 0.95, 0.98]  # None = без decay
NAMES = [
    "heatmap_nodecay.mp4",
    "heatmap_90.mp4",
    "heatmap_95.mp4",
    "heatmap_98.mp4"
]

cap = cv2.VideoCapture(VIDEO_PATH)

ret, frame = cap.read()
if not ret:
    raise ValueError("Cannot read video")

H, W = frame.shape[:2]
prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# 🔥 отдельный аккумулятор под каждый decay
accums = [np.zeros((H, W), dtype=np.float32) for _ in DECAYS]

# 🔥 writers
writers = []
for name in NAMES:
    out = cv2.VideoWriter(
        name,
        cv2.VideoWriter_fourcc(*"mp4v"),
        30,
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

    # magnitude
    mag = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)

    # фильтр шума
    mag[mag < 1.0] = 0

    # 🔥 обновляем все варианты сразу
    for i, decay in enumerate(DECAYS):
        if decay is None:
            # ❗ без decay → просто накапливаем
            accums[i] += mag
        else:
            accums[i] = accums[i] * decay + mag

        current = accums[i]

        # percentiles (робастная нормализация)
        # Важно: цветовая шкала должна быть "больше = горячее".
        low = 0.0
        high = float(np.percentile(current, 99))

        if high - low < 1e-6:
            norm = np.zeros_like(current)
        else:
            clipped = np.clip(current, low, high)
            norm = (clipped - low) / (high - low)

        # heatmap: КАК У ТЕБЯ ПО СМЫСЛУ
        # красный = медленно/мало движения, зелёный = быстро/много движения
        # делаем градиент в HSV: H=0 (red) -> H=60deg (green)
        norm8 = (norm * 255.0).astype(np.uint8)
        h = (norm * 30.0).astype(np.uint8)  # 0..30 (OpenCV hue 0..179)
        s = np.full((H, W), 255, dtype=np.uint8)
        v = np.full((H, W), 255, dtype=np.uint8)
        hsv = cv2.merge([h, s, v])
        heatmap = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        heatmap[norm8 == 0] = (0, 0, 0)

        vis = cv2.addWeighted(frame, 0.7, heatmap, 0.6, 0)

        writers[i].write(vis)

    prev_gray = gray

cap.release()
for w in writers:
    w.release()