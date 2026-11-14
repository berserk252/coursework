"""Передний план целиком: ограничение контуров и стоящий транспорт без истории проездов."""
from dataclasses import replace

import cv2
import numpy as np

from flow_holes import Hole


def contours(mask):
    # Внутренние контуры сохраняем: RETR_EXTERNAL заливал бы асфальт между машинами.
    found, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return [c.reshape(-1, 2) for c in found if len(c) >= 3]


def mask_of(hole, shape):
    if hole.mask is not None:
        return hole.mask.astype(bool)
    result = np.zeros(shape, np.uint8)
    cv2.fillPoly(result, hole.polygons, 1)
    return result > 0


def clip_holes(holes, foreground, known, labels, min_area=24):
    """Весь полигон ограничен наблюдаемым передним планом, а не только его ядро."""
    result = []
    for hole in holes:
        original = mask_of(hole, labels.shape) & (labels == hole.part)
        mask = original & foreground & known
        n, lab, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
        allowed = [i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= min_area]
        mask = np.isin(lab, allowed) if allowed else np.zeros_like(mask)
        if not mask.any():
            continue
        result.append(replace(hole, mask=mask, polygons=contours(mask), area_px=int(mask.sum()),
                              bg_valid=True, bg_coverage=1.0, bg_diff=1.0, bg_match=False,
                              bg_removed_frac=1-int(mask.sum())/max(int(original.sum()),1)))
    return result


class ForegroundObjects:
    """Показывает неподвижный передний план, даже если в нём не было трёх проездов.

    Новые края отсекают плоские изменения освещения и следы уехавших машин.
    Метка stationary означает занятую область/очередь, а не отдельное ДТП.
    """
    def __init__(self, shape, fps, min_area=60, confirm_s=2.0):
        self.shape, self.fps = shape, float(fps)
        self.min_area, self.confirm_s = min_area, confirm_s
        self.reset()

    def reset(self):
        self.age = np.zeros(self.shape, np.float32)
        self.previous = {}
        self.next_id = 1000000

    def update(self, gray, background, foreground, moving, labels, t, cell, gh, gw, cut=False, valid=True):
        if cut:
            self.reset()
        if not valid:
            return []
        moving = cv2.dilate(moving.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        quiet = foreground & ~moving & (labels >= 0)
        self.age = np.where(quiet, self.age + 1 / self.fps, 0).astype(np.float32)
        edges = cv2.Canny(gray, 50, 120) > 0
        old_edges = cv2.Canny(np.clip(background, 0, 255).astype(np.uint8), 50, 120)
        novel = edges & (cv2.dilate(old_edges, np.ones((5, 5), np.uint8)) == 0)
        output, current = [], {}
        for part in range(int(labels.max()) + 1):
            n, lab, stats, _ = cv2.connectedComponentsWithStats((foreground & (labels == part)).astype(np.uint8), 8)
            for i in range(1, n):
                if stats[i, cv2.CC_STAT_AREA] < self.min_area:
                    continue
                mask = lab == i
                stable = self.age[mask]
                if (stable >= self.confirm_s).mean() < .15:
                    continue
                near = cv2.dilate(mask.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
                if (novel & near).sum() < max(5, .003 * mask.sum()):
                    continue
                if edges[mask].mean() < .8 * (old_edges[mask] > 0).mean():
                    continue                         # опустевшая область прежней машины
                matches = [(int((mask & old).sum()) / max(int((mask | old).sum()), 1), ident)
                           for ident, (old, p) in self.previous.items() if p == part and ident not in current]
                score, ident = max(matches, default=(0, 0))
                if score < .2:
                    ident = self.next_id
                    self.next_id += 1
                current[ident] = (mask, part)
                grid = cv2.resize(mask[:gh*cell, :gw*cell].astype(np.float32), (gw, gh), interpolation=cv2.INTER_AREA) > .15
                output.append(Hole(id=ident, part=part, cells=grid, core=grid, polygons=contours(mask),
                                   mask=mask, area_px=int(mask.sum()), idle_s=float(np.quantile(stable, .75)),
                                   age_s=float(np.quantile(stable, .75)), v_in=0., bg_diff=1., bg_valid=True,
                                   bg_coverage=1., incident=False, kind="stationary"))
        self.previous = current
        return output


def combine_objects(holes, objects):
    """Не накладываем два контура на один объект; очередь расширяет только подтверждённую маску."""
    for obj in objects:
        matches = [h for h in holes if h.part == obj.part and h.mask is not None
                   and (h.mask & obj.mask).sum() > .15 * obj.mask.sum()]
        if matches:
            hole = matches[0]
            hole.mask |= obj.mask
            hole.polygons = contours(hole.mask)
            hole.area_px = int(hole.mask.sum())
        else:
            holes.append(obj)
    return holes
