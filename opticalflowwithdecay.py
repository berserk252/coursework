import cv2
import numpy as np

from flow_filters import FlowCleaner

VIDEO_PATH = "video_small.mp4"

DECAYS = [0.90, 0.95, 0.98]
NAMES = ["opticalflow90.mp4", "opticalflow95.mp4", "opticalflow98.mp4"]

STEP = 16
SCALE = 3
# Порог в px/кадр поверх адаптивного порога FlowCleaner. 0 = только адаптивный.
# Раньше было 0.8: из-за перспективы дальние машины идут медленнее 0.8 px/кадр
# и стрелок не получали вовсе, а шум вокруг ближних порог проходил.
ARROW_MIN_MAG = 0.0
ARROW_COLOR = (0, 255, 255)
ARROW_THICKNESS = 1
ARROW_TIP = 0.35

EGOMOTION = True   # вычитать движение камеры (дрон, тряска)
ROAD_MASK = True   # не рисовать стрелки на деревьях, кустах, бликах
EMA_ALPHA = 0.5    # бегущее среднее по u, v: F = a*F_new + (1-a)*F_old; 1.0 = выкл

cap = cv2.VideoCapture(VIDEO_PATH)
FPS = cap.get(cv2.CAP_PROP_FPS) or 30

ret, frame = cap.read()
if not ret:
    raise ValueError("Video error")

H, W = frame.shape[:2]
prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# Точки -- НЕПОДВИЖНАЯ сетка. Раньше каждая точка сдвигалась на flow * SCALE и
# оставалась там, а координата округлялась через int(), то есть к нулю:
# шум -0.3 px сдвигал точку на целый пиксель влево/вверх, шум +0.3 -- никуда.
# Измерено на 1900: через 600 кадров 67% точек прилипли к левому и верхнему
# краю кадра, сетка покрывала 8% кадра -- на дороге стрелок не оставалось.
points = []
for y in range(STEP // 2, H, STEP):
    for x in range(STEP // 2, W, STEP):
        points.append([x, y])
points = np.array(points, dtype=np.int32)

cleaner = FlowCleaner(egomotion=EGOMOTION, road_mask=ROAD_MASK, ema_alpha=EMA_ALPHA)

# trail
trails = [np.zeros((H, W, 3), dtype=np.float32) for _ in DECAYS]

# ✅ writers СОЗДАЁМ ОДИН РАЗ
writers = []
for name in NAMES:
    out = cv2.VideoWriter(
        name,
        cv2.VideoWriter_fourcc(*"mp4v"),
        FPS,
        (W, H)
    )
    if not out.isOpened():
        raise ValueError(f"Failed to open writer: {name}")
    writers.append(out)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, gray,
        None,
        0.5, 3, 15, 3, 5, 1.2, 0
    )

    # без движения камеры, шума, дрожания и размытия; фон = 0
    c = cleaner.update(flow, gray, prev_gray)
    field = c.field()
    if ROAD_MASK:
        field[c.road_w < 0.5] = 0.0

    # стрелки рисуем на отдельном оверлее, чтобы они были видны поверх trail
    arrows = np.zeros_like(frame)

    for (x, y) in points:
        fx, fy = field[y, x]
        mag = float((fx * fx + fy * fy) ** 0.5)
        if mag <= ARROW_MIN_MAG or mag == 0.0:
            # неподвижная точка: раньше и для неё рисовалась линия нулевой
            # длины, то есть точка, и trail копил по белой точке в каждом узле
            continue

        nx = int(round(x + fx * SCALE))
        ny = int(round(y + fy * SCALE))

        if 0 <= nx < W and 0 <= ny < H:
            cv2.arrowedLine(
                arrows,
                (int(x), int(y)),
                (nx, ny),
                ARROW_COLOR,
                ARROW_THICKNESS,
                tipLength=ARROW_TIP,
            )
            for trail in trails:
                cv2.line(trail, (int(x), int(y)), (nx, ny), (255.0, 255.0, 255.0), 2)

    # применяем decay + запись
    for i, decay in enumerate(DECAYS):
        trails[i] *= decay
        trail_u8 = np.clip(trails[i], 0, 255).astype(np.uint8)

        vis = cv2.addWeighted(frame, 1.0, trail_u8, 0.85, 0)
        # стрелки поверх
        vis = cv2.addWeighted(vis, 1.0, arrows, 1.0, 0)
        writers[i].write(vis)

    prev_gray = gray

cap.release()
for w in writers:
    w.release()
