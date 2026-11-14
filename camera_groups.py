"""
Видео по камерам. Ролики одной камеры (тот же вид; камера могла немного сдвинуться
или изменить зум) размечаются один раз, разметка переносится на остальные.

Ролик попадает к камере, если его кадр совмещается с кадром представителя камеры
гомографией по особым точкам ORB: не меньше min_inliers согласованных точек,
масштаб 0.5..2, без сильной перспективы. Представитель -- самый длинный ролик
камеры (по нему лучше видно, где ездят). Берётся первый кадр; если он тёмный
(заставка) -- кадр из середины ролика.

Группы хранятся в <parts_dir>/_камеры.json, пути видео -- относительно parts_dir:
    {"cameras": [{"rep": "../traffic_flow_dataset/5_avariya/view3.mp4", "done": false,
                  "members": [{"video": "...", "size": [W, H], "H": [[..], [..], [..]]}]}]}
H переводит координаты кадра представителя (доли 0..1) в координаты ролика.
done -- камеру разметил человек (annotate_cameras.py).

    cams = CameraGroups.load_or_build(videos, parts_dir)
    for k, cam in enumerate(cams.cameras): ...
    cams.apply(k, rp)            # разметку представителя -- на все ролики камеры
"""

from __future__ import annotations

import json
import os

import cv2
import numpy as np

from road_parts import RoadParts

VIDEO_EXT = (".mp4", ".avi", ".mov", ".mkv")
W = 640                                          # ширина кадра для поиска особых точек


def list_videos(root: str) -> list:
    """Файл или все видео в папке и подпапках, по порядку."""
    if os.path.isfile(root):
        return [root]
    out = []
    for d, _, files in os.walk(root, followlinks=True):
        out += [os.path.join(d, f) for f in files if f.lower().endswith(VIDEO_EXT)]
    return sorted(out)


def probe(path: str):
    """(кадр шириной W, длительность с, исходный размер) или None."""
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ok, f = cap.read()
    if not ok:
        return None
    if f.std() < 12 and n > 2:                   # тёмный кадр или заставка
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(n * 0.6))
        ok2, g = cap.read()
        if ok2:
            f = g
    h0, w0 = f.shape[:2]
    small = cv2.resize(f, (W, int(round(h0 * W / w0))), interpolation=cv2.INTER_AREA)
    return small, n / fps, (w0, h0)


class Features:
    def __init__(self):
        self.orb = cv2.ORB_create(3000)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING)

    def __call__(self, img):
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kp, des = self.orb.detectAndCompute(g, None)
        return np.float32([k.pt for k in kp]), des, g.shape

    def match(self, a, b):
        """Гомография кадр a -> кадр b (в долях кадра) и число согласованных точек."""
        pa, da, (ha, wa) = a
        pb, db, (hb, wb) = b
        if da is None or db is None or len(da) < 20 or len(db) < 20:
            return None, 0
        m = self.bf.knnMatch(da, db, k=2)
        good = [x[0] for x in m if len(x) == 2 and x[0].distance < 0.75 * x[1].distance]
        if len(good) < 25:
            return None, len(good)
        src = pa[[g.queryIdx for g in good]]
        dst = pb[[g.trainIdx for g in good]]
        H, inl = cv2.findHomography(src, dst, cv2.RANSAC, 4.0)
        if H is None:
            return None, 0
        n = int(inl.sum())
        s = np.sqrt(abs(np.linalg.det(H[:2, :2])))
        if not (0.5 < s < 2.0) or np.abs(H[2, :2]).max() > 2e-3:
            return None, n                       # не тот же вид: сильный зум или перспектива
        corners = cv2.perspectiveTransform(np.float32([[0, 0], [wa, 0], [wa, ha], [0, ha]])[None], H)[0]
        if np.abs(corners).max() > 3 * W:
            return None, n
        Hn = np.diag([1.0 / wb, 1.0 / hb, 1.0]) @ H @ np.diag([wa, ha, 1.0])
        return Hn / Hn[2, 2], n


def warp_parts(parts: list, Hn) -> list:
    """Части (доли кадра представителя) -> доли кадра ролика."""
    out = []
    for p in parts:
        if len(p["points"]) < 3:
            continue
        q = cv2.perspectiveTransform(np.float32(p["points"])[None], np.asarray(Hn, np.float64))[0]
        q = np.clip(q, 0.0, 1.0)
        out.append({"name": p["name"], "points": [[round(float(x), 4), round(float(y), 4)] for x, y in q]})
    return out


class CameraGroups:
    NAME = "_камеры.json"

    def __init__(self, parts_dir: str, cameras: list):
        self.parts_dir = parts_dir
        self.cameras = cameras

    # --- пути ----------------------------------------------------------------

    def path(self, rel: str) -> str:
        return os.path.normpath(os.path.join(self.parts_dir, rel))

    def rel(self, path: str) -> str:
        return os.path.relpath(path, self.parts_dir)

    def save(self) -> None:
        os.makedirs(self.parts_dir, exist_ok=True)
        with open(os.path.join(self.parts_dir, self.NAME), "w", encoding="utf-8") as f:
            json.dump({"cameras": self.cameras}, f, ensure_ascii=False, indent=1)

    # --- группировка ---------------------------------------------------------

    @classmethod
    def load_or_build(cls, videos: list, parts_dir: str, min_inliers: int = 60, rebuild: bool = False,
                      log=print) -> "CameraGroups":
        """Группы из <parts_dir>/_камеры.json; новые видео добавляются к своим камерам
        (или становятся новыми камерами), пропавшие с диска -- убираются."""
        p = os.path.join(parts_dir, cls.NAME)
        cams = []
        if os.path.exists(p) and not rebuild:
            with open(p, encoding="utf-8") as f:
                cams = json.load(f)["cameras"]
        self = cls(parts_dir, cams)
        want = {os.path.normpath(os.path.abspath(v)) for v in videos}
        # пропавшие видео
        for cam in self.cameras:
            cam["members"] = [m for m in cam["members"]
                              if os.path.normpath(os.path.abspath(self.path(m["video"]))) in want]
        self.cameras = [c for c in self.cameras if c["members"]]
        for cam in self.cameras:
            if not any(m["video"] == cam["rep"] for m in cam["members"]):
                # представителя нет -- им становится первый ролик, гомографии пересчитываем от него
                inv = np.linalg.inv(np.asarray(cam["members"][0]["H"], np.float64))
                for m in cam["members"]:
                    h = np.asarray(m["H"], np.float64) @ inv
                    m["H"] = np.round(h / h[2, 2], 8).tolist()
                cam["rep"] = cam["members"][0]["video"]
        have = {os.path.normpath(os.path.abspath(self.path(m["video"]))) for c in self.cameras for m in c["members"]}
        new = [v for v in videos if os.path.normpath(os.path.abspath(v)) not in have]
        if not new:
            return self
        log(f"ищу камеры для {len(new)} видео ...")
        threads = cv2.getNumThreads()
        cv2.setNumThreads(2)                     # не грузить весь процессор
        feat = Features()
        info = {}
        for i, v in enumerate(new):
            r = probe(v)
            if r is not None:
                info[v] = (feat(r[0]), r[1], r[2])
            if (i + 1) % 20 == 0:
                log(f"  прочитано {i + 1}/{len(new)}")
        rep_feat = []                            # (номер камеры, особые точки представителя)
        for k, cam in enumerate(self.cameras):
            r = probe(self.path(cam["rep"]))
            if r is not None:
                rep_feat.append((k, feat(r[0])))
        # длинные ролики -- первыми: они становятся представителями новых камер
        for v in sorted(info, key=lambda v: -info[v][1]):
            fv, dur, size = info[v]
            best = (None, 0, None)
            for k, fr in rep_feat:
                Hn, n = feat.match(fr, fv)
                if Hn is not None and n > best[1]:
                    best = (k, n, Hn)
            if best[0] is not None and best[1] >= min_inliers:
                self.cameras[best[0]]["members"].append(
                    {"video": self.rel(v), "size": list(size), "H": np.round(best[2], 8).tolist()})
            else:
                self.cameras.append({"rep": self.rel(v), "done": False,
                                     "members": [{"video": self.rel(v), "size": list(size),
                                                  "H": np.eye(3).tolist()}]})
                rep_feat.append((len(self.cameras) - 1, fv))
        for cam in self.cameras:
            cam["members"].sort(key=lambda m: m["video"])
        self.cameras.sort(key=lambda c: c["rep"])
        cv2.setNumThreads(threads)
        self.save()
        log(f"камер: {len(self.cameras)}, видео: {sum(len(c['members']) for c in self.cameras)}")
        return self

    # --- разметка --------------------------------------------------------------

    def parts_path(self, video_rel: str) -> str:
        return RoadParts.path_for(video_rel, self.parts_dir)

    def load_rep(self, k: int) -> RoadParts:
        """Разметка представителя камеры k (пустая, если её нет)."""
        cam = self.cameras[k]
        rep = next(m for m in cam["members"] if m["video"] == cam["rep"])
        p = self.parts_path(cam["rep"])
        rp = RoadParts.load(p) if os.path.exists(p) else RoadParts([])
        rp.video, rp.frame_size = os.path.basename(cam["rep"]), rep["size"]
        return rp

    def apply(self, k: int, rp: RoadParts) -> int:
        """Разметку rp (в кадре представителя) -- на все ролики камеры k. Возвращает число файлов."""
        cam = self.cameras[k]
        done = set()
        for m in cam["members"]:
            p = self.parts_path(m["video"])
            if p in done:
                continue                             # тот же ролик в двух папках классов
            parts = warp_parts(rp.parts, m["H"])
            RoadParts(parts, os.path.basename(m["video"]), m["size"]).save(p)
            done.add(p)
        cam["done"] = True
        self.save()
        return len(done)
