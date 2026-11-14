"""Связные области для защиты фона и восстановления кузова, без детектора объектов."""
import cv2
import numpy as np


def enclosed_holes(mask, min_area=1, max_area=None):
    """Замкнутые просветы заданной площади, без внешнего фона.

    Открытые промежутки между объектами и внешний фон не заполняются.
    Размеры задаются в пикселях изображения обработки, а не исходного видео.
    """
    count, regions, stats, _ = cv2.connectedComponentsWithStats((~mask).astype(np.uint8), 8)
    accepted = np.zeros(count, bool)
    height, width = mask.shape
    for ident in range(1, count):
        x, y, w, h, area = stats[ident]
        accepted[ident] = (area >= min_area and (max_area is None or area <= max_area) and x > 0 and y > 0
                           and x + w < width and y + h < height)
    return accepted[regions]


def fill_small_holes(mask, max_area):
    return mask | enclosed_holes(mask, max_area=max_area)


def motion_footprint(moving):
    """Защитить от обучения весь движущийся объект, включая однотонную середину.

    Внутри ровного кузова Farneback может дать мало движения. Одних движущихся
    краёв недостаточно: иначе кузов попадает в фон. Здесь допустимо исключить
    из обучения немного больше точек; это не маска обнаруженного препятствия.
    """
    scale = moving.shape[1] / 640
    size = max(3, round(7 * scale) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    closed = cv2.morphologyEx(moving.astype(np.uint8), cv2.MORPH_CLOSE, kernel) > 0
    return fill_small_holes(closed, max(64, round(4096 * scale ** 2))) | moving


def color_interior(seed, frame, domain, scale):
    """Достроить похожие по цвету пиксели внутри оболочки исходного компонента.

    Оболочка лишь ограничивает поиск: целиком она не закрашивается. Новый
    пиксель должен быть не дальше 8 px от исходной маски (при ширине 640)
    и отличаться от ближайшего исходного пикселя не больше 18 в Lab.
    За внешнюю оболочку отдельного компонента рост запрещён.
    """
    count, regions, stats, _ = cv2.connectedComponentsWithStats(seed.astype(np.uint8), 8)
    accepted = np.zeros(count, bool)
    accepted[1:] = stats[1:, cv2.CC_STAT_AREA] >= max(10, round(24 * scale ** 2))
    support = accepted[regions]
    if not support.any():
        return seed
    envelope = np.zeros(seed.shape, np.uint8)
    contours, _ = cv2.findContours(support.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        if cv2.contourArea(contour) <= .15 * seed.size:
            cv2.fillConvexPoly(envelope, cv2.convexHull(contour), 1)
    distance, nearest = cv2.distanceTransformWithLabels((~support).astype(np.uint8), cv2.DIST_L2, 5,
                                                       labelType=cv2.DIST_LABEL_PIXEL)
    lab = cv2.cvtColor(cv2.GaussianBlur(frame, (3, 3), 0), cv2.COLOR_BGR2LAB).astype(np.int16)
    colors = np.zeros((int(nearest.max()) + 1, 3), np.int16)
    colors[nearest[support]] = lab[support]
    delta = (lab - colors[nearest]).astype(np.int32)
    same_color = np.sum(delta * delta, axis=2) <= 18 ** 2
    added = domain & (envelope > 0) & (distance <= max(2, 8 * scale)) & same_color
    return seed | added


def complete_foreground(foreground, known, labels, frame=None):
    """Восстановить узкие разрывы и небольшие внутренние участки кузова.

    Разница с фоном даёт начальные пиксели, а не готовую сегментацию машины.
    Внутри компонента похожие по цвету точки достраиваются на расстоянии до
    8 px от исходной маски. Замыкание 5 px и заполнение просветов до 400 px
    восстанавливают окна и мелкие разрывы. Размеры указаны при ширине 640.
    Большие исходные просветы, неизвестный фон и соседняя часть дороги
    в маску не включаются.
    """
    scale = foreground.shape[1] / 640
    size = max(3, round(5 * scale) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    max_area = max(16, round(400 * scale ** 2))
    result = np.zeros_like(foreground, dtype=bool)
    for part in np.unique(labels[labels >= 0]):
        domain = known & (labels == part)
        seed = foreground & domain
        if not seed.any():
            continue
        # Большой известный просвет не должен уменьшаться даже по углам
        # вследствие замыкания: сохраняем его исходную границу.
        asphalt = enclosed_holes(seed, min_area=max_area + 1)
        if frame is not None:
            seed = color_interior(seed, frame, domain, scale)
        closed = cv2.morphologyEx(seed.astype(np.uint8), cv2.MORPH_CLOSE, kernel) > 0
        result |= (fill_small_holes(closed, max_area) | seed) & domain & ~asphalt
    return result
