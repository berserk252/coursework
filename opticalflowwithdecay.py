import cv2
import numpy as np

VIDEO_PATH = "video_small.mp4"

DECAYS = [0.90, 0.95, 0.98]
NAMES = ["opticalflow90.mp4", "opticalflow95.mp4", "opticalflow98.mp4"]

STEP = 16
SCALE = 3
ARROW_MIN_MAG = 0.8
ARROW_COLOR = (0, 255, 255)
ARROW_THICKNESS = 1
ARROW_TIP = 0.35

cap = cv2.VideoCapture(VIDEO_PATH)

ret, frame = cap.read()
if not ret:
    raise ValueError("Video error")

H, W = frame.shape[:2]
prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# точки
points = []
for y in range(0, H, STEP):
    for x in range(0, W, STEP):
        points.append([x, y])
points = np.array(points, dtype=np.float32)

# trail
trails = [np.zeros((H, W, 3), dtype=np.float32) for _ in DECAYS]

# ✅ writers СОЗДАЁМ ОДИН РАЗ
writers = []
for name in NAMES:
    out = cv2.VideoWriter(
        name,
        cv2.VideoWriter_fourcc(*"mp4v"),
        30,
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

    new_points = []
    # стрелки рисуем на отдельном оверлее, чтобы они были видны поверх trail
    arrows = np.zeros_like(frame)

    for (x, y) in points:
        fx, fy = flow[int(y), int(x)]

        nx = int(x + fx * SCALE)
        ny = int(y + fy * SCALE)

        if 0 <= nx < W and 0 <= ny < H:
            mag = float((fx * fx + fy * fy) ** 0.5)
            if mag >= ARROW_MIN_MAG:
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
            new_points.append([nx, ny])
        else:
            new_points.append([x, y])

    points = np.array(new_points, dtype=np.float32)

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