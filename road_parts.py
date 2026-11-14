"""
Ручная разметка дороги: части дороги (полигоны), где разрешено движение.

Разметка одного видео -- JSON в папке разметки (по умолчанию ../разметка_дорог),
имя файла = имя видео без расширения:

    {
      "video": "view3.mp4",
      "frame_size": [478, 300],
      "parts": [
        {"name": "Часть 1", "points": [[0.00, 0.21], [0.46, 0.01], ...]},
        {"name": "Часть 2", "points": [...]}
      ]
    }

Координаты вершин -- доли ширины и высоты кадра (0..1), поэтому разметка
не зависит от того, в каком разрешении видео обрабатывается.

Размечает человек (annotate_road.py), а не алгоритм: анализ векторного поля
идёт только внутри этих полигонов, и каждая часть сравнивается сама с собой
(дырка -- место внутри части, где движения нет, а вокруг в той же части есть).

    rp = RoadParts.load("../разметка_дорог/view3.json")
    labels = rp.label_map(W, H)        # (H, W) int: -1 -- не дорога, k -- часть k
    rp.draw(img)                       # контуры частей с подписями
"""

from __future__ import annotations

import json
import os

import cv2
import numpy as np

# цвета частей (BGR): не красный и не оранжевый -- ими рисуются дырки,
# не зелёный -- им рисуются стрелки поля
PART_COLORS = [(230, 130, 20), (190, 60, 190), (150, 150, 0), (40, 90, 160), (120, 120, 120)]


def part_color(k: int):
    return PART_COLORS[k % len(PART_COLORS)]


class RoadParts:
    def __init__(self, parts=None, video: str = "", frame_size=None):
        self.parts = parts or []          # [{"name": str, "points": [[x, y], ...]}], доли кадра
        self.video = video
        self.frame_size = frame_size
        self._label_xy = {}               # место подписи части (кэш: считается эрозией маски)

    # --- файл -------------------------------------------------------------

    @staticmethod
    def path_for(video: str, parts_dir: str) -> str:
        stem = os.path.splitext(os.path.basename(video))[0]
        return os.path.join(parts_dir, stem + ".json")

    @classmethod
    def load(cls, path: str) -> "RoadParts":
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        parts = [p for p in d.get("parts", []) if len(p.get("points", [])) >= 3]
        return cls(parts, d.get("video", ""), d.get("frame_size"))

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        d = {"video": self.video, "frame_size": self.frame_size,
             "parts": [{"name": p["name"], "points": [[round(x, 4), round(y, 4)] for x, y in p["points"]]}
                       for p in self.parts if len(p["points"]) >= 3]}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(d, f, ensure_ascii=False, indent=1)

    # --- геометрия --------------------------------------------------------

    def __len__(self):
        return len(self.parts)

    def names(self):
        return [p["name"] for p in self.parts]

    def pixels(self, k: int, W: int, H: int) -> np.ndarray:
        """Вершины части k в пикселях кадра W x H, (N, 2) int32."""
        p = np.asarray(self.parts[k]["points"], np.float64)
        return np.round(p * [W, H]).astype(np.int32)

    def label_map(self, W: int, H: int) -> np.ndarray:
        """(H, W) int8: -1 -- вне разметки, k -- часть k. Если полигоны
        перекрываются, пиксель достаётся части с меньшим номером."""
        lab = np.full((H, W), -1, np.int8)
        for k in reversed(range(len(self.parts))):
            m = np.zeros((H, W), np.uint8)
            cv2.fillPoly(m, [self.pixels(k, W, H)], 1)
            lab[m > 0] = k
        return lab

    # --- рисование --------------------------------------------------------

    def draw(self, img, txt=None, thickness: int = 2, fill_alpha: float = 0.0,
             labels: bool = True, size: int = 13):
        """Контуры частей поверх img. img -- кадр или его уменьшенная копия
        (холсты div/rot): координаты -- доли, поэтому размер не важен."""
        H, W = img.shape[:2]
        for k in range(len(self.parts)):
            color = part_color(k)
            pts = self.pixels(k, W, H)
            if fill_alpha > 0:
                over = img.copy()
                cv2.fillPoly(over, [pts], color)
                cv2.addWeighted(over, fill_alpha, img, 1 - fill_alpha, 0, dst=img)
            dashed_polyline(img, pts, color, thickness)
            if labels and txt is not None:
                key = (k, W, H)
                if key not in self._label_xy:
                    self._label_xy[key] = label_point(pts, W, H)
                x, y = self._label_xy[key]
                txt.draw(img, self.parts[k]["name"], x + 1, y + 1, size=size, color=(255, 255, 255),
                         alt=f"part {k + 1}")
                txt.draw(img, self.parts[k]["name"], x, y, size=size, color=color, alt=f"part {k + 1}")
        return img


def label_point(pts: np.ndarray, W: int, H: int):
    """Место подписи части: внутри полигона, у правого края его средней по высоте строки
    (слева у дорог на снимках чаще всего обочина, где встают машины)."""
    m = np.zeros((H, W), np.uint8)
    cv2.fillPoly(m, [pts], 1)
    m = cv2.erode(m, np.ones((15, 15), np.uint8))
    ys, xs = np.nonzero(m)
    if len(xs) == 0:
        x, y = pts.mean(axis=0)
        return int(x), int(y)
    y = int(np.percentile(ys, 60))
    row = xs[ys == y]
    return int(np.percentile(row, 80)) - 40, y - 8


def dashed_polyline(img, pts: np.ndarray, color, thickness: int = 2, dash: int = 10, gap: int = 6):
    """Замкнутый пунктир: часть дороги -- это область анализа, а не объект."""
    n = len(pts)
    for i in range(n):
        p0, p1 = pts[i].astype(np.float64), pts[(i + 1) % n].astype(np.float64)
        L = float(np.hypot(*(p1 - p0)))
        if L < 1:
            continue
        d = (p1 - p0) / L
        s = 0.0
        while s < L:
            e = min(s + dash, L)
            a, b = p0 + d * s, p0 + d * e
            cv2.line(img, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])), (255, 255, 255), thickness + 2, cv2.LINE_AA)
            cv2.line(img, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])), color, thickness, cv2.LINE_AA)
            s += dash + gap
