"""
Разметка частей дороги по камерам: окно разметки открывается по очереди для каждой
камеры (повторяющиеся ролики одной камеры -- один раз), и разметка сразу
записывается на все видео этой камеры.

    python annotate_cameras.py                        # все видео ../traffic_flow_dataset
    python annotate_cameras.py --video ../newVideos
    python annotate_cameras.py --start 5              # начать с камеры 5
    python annotate_cameras.py --list                 # только список камер, без окна

Первый запуск делит видео на камеры (около минуты), дальше группы берутся из
<parts_dir>/_камеры.json (camera_groups.py). Открывается первая камера, которую
ещё не размечали. Если у неё уже есть разметка (черновик), она показана: её
можно поправить или стереть клавишей R. Сверху написано, на какие видео пойдёт
разметка.

Мышь и клавиши -- как в annotate_road.py, плюс:
    S                 -- сохранить на все видео камеры и открыть следующую
    .  ]  или  ->     -- следующая камера без сохранения
    ,  [  или  <-     -- предыдущая камера
    V                 -- отдельное окно: первые кадры всех видео камеры с этой разметкой
    Q / Esc           -- выйти (сохранённое остаётся; в следующий раз продолжится
                         с первой неразмеченной камеры)
Клавиши работают и в русской раскладке.
"""

import argparse
import math
import os

import cv2
import numpy as np

from annotate_road import HELP, Editor, Source
from camera_groups import CameraGroups, list_videos, probe, warp_parts
from flow_canvas import Text
from road_parts import RoadParts

WIN_ALL = "Videos of this camera"
HELP_CAM = ["S: сохранить на все видео камеры и дальше   . или ->: следующая камера   , или <-: предыдущая   "
            "V: все видео камеры"] + HELP


def short(video: str) -> str:
    return os.path.splitext(os.path.basename(video))[0].replace("negative_samples__negative_samples_", "ns")


def members_sheet(cams: CameraGroups, k: int, rp: RoadParts, txt: Text, cache: dict, width: int = 1500):
    """Первые кадры всех видео камеры с разметкой rp, перенесённой на каждое."""
    cam = cams.cameras[k]
    ms = cam["members"]
    cols = max(1, min(6, math.ceil(math.sqrt(len(ms) * 1.6))))
    tw = width // cols
    tiles = []
    for m in ms:
        if m["video"] not in cache:
            r = probe(cams.path(m["video"]))
            cache[m["video"]] = r[0] if r is not None else None
        f = cache[m["video"]]
        if f is None:
            continue
        img = cv2.resize(f, (tw - 4, int(round(f.shape[0] * (tw - 4) / f.shape[1]))), interpolation=cv2.INTER_AREA)
        RoadParts(warp_parts(rp.parts, m["H"])).draw(img, None, thickness=1, fill_alpha=0.3, labels=False)
        w = txt.draw(np.zeros((1, 1, 3), np.uint8), short(m["video"]), 0, 0, size=13)
        cv2.rectangle(img, (0, 0), (w + 8, 19), (255, 255, 255), -1)
        txt.draw(img, short(m["video"]), 4, 2, size=13, color=(20, 20, 20))
        tiles.append(cv2.copyMakeBorder(img, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=(255, 255, 255)))
    if not tiles:
        return None
    h = max(t.shape[0] for t in tiles)
    tiles = [cv2.copyMakeBorder(t, 0, h - t.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
             for t in tiles]
    while len(tiles) % cols:
        tiles.append(np.full_like(tiles[0], 255))
    return np.vstack([np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)])


def main():
    ap = argparse.ArgumentParser(description="Разметка частей дороги по камерам, по очереди.")
    ap.add_argument("--video", default=os.path.join("..", "traffic_flow_dataset"),
                    help="Папка с видео (все видео в ней и подпапках) или файл")
    ap.add_argument("--parts_dir", default=os.path.join("..", "разметка_дорог"))
    ap.add_argument("--width", type=int, default=1200, help="Ширина окна, px")
    ap.add_argument("--start", type=int, default=0, help="Номер камеры, с которой начать (1..)")
    ap.add_argument("--regroup", action="store_true", help="Заново разделить видео на камеры")
    ap.add_argument("--list", action="store_true", help="Только напечатать камеры и выйти")
    args = ap.parse_args()

    videos = list_videos(args.video)
    if not videos:
        raise SystemExit(f"нет видео: {args.video}")
    cams = CameraGroups.load_or_build(videos, args.parts_dir, rebuild=args.regroup)
    N = len(cams.cameras)
    print(f"видео: {len(videos)}, камер: {N}, уже размечено: {sum(c['done'] for c in cams.cameras)}")
    if args.list:
        for k, cam in enumerate(cams.cameras):
            print(f"{k + 1:3d} {'+' if cam['done'] else ' '} {short(cam['rep'])}  ({len(cam['members'])} видео)")
        return

    if args.start:
        k = min(max(args.start, 1), N) - 1
    else:
        k = next((i for i, c in enumerate(cams.cameras) if not c["done"]), 0)
    txt = Text()
    ed = Editor(txt, args.width)
    thumbs = {}
    while True:
        cam = cams.cameras[k]
        ms = cam["members"]
        names = [short(m["video"]) for m in ms]
        src = Source(cams.path(cam["rep"]), width=args.width)
        rp = cams.load_rep(k)
        n_done = sum(c["done"] for c in cams.cameras)
        tail = ", ".join(names[:10]) + (f" … (+{len(names) - 10})" if len(names) > 10 else "")
        header = [f"Камера {k + 1} из {N}   (размечено {n_done})   "
                  f"{'РАЗМЕЧЕНА' if cam['done'] else 'не размечена'}   показан: {short(cam['rep'])}",
                  f"Разметка пойдёт на {len(ms)} видео: {tail}"]
        print(f"\nкамера {k + 1}/{N}: {short(cam['rep'])}; разметка пойдёт на {len(ms)} видео: {', '.join(names)}")

        def on_save(rp, k=k):
            n = cams.apply(k, rp)
            return f"камера {k + 1}: сохранено на {n} видео"

        def on_view(rp, k=k):
            sheet = members_sheet(cams, k, rp, txt, thumbs)
            if sheet is not None:
                cv2.imshow(WIN_ALL, sheet)
            return f"открыто окно со всеми видео камеры ({len(ms)})"

        res = ed.edit(rp, src, on_save, header=header, help_lines=HELP_CAM, show="frame",
                      camera=True, on_view=on_view)
        try:
            cv2.destroyWindow(WIN_ALL)
        except cv2.error:
            pass
        if res == "quit":
            break
        if res == "prev":
            k = (k - 1) % N
        elif res == "next":
            k = (k + 1) % N
        else:                                   # сохранено -- к следующей неразмеченной
            rest = [i for i in list(range(k + 1, N)) + list(range(0, k)) if not cams.cameras[i]["done"]]
            if rest:
                k = rest[0]
            elif k + 1 < N:                     # все размечены -- правка по порядку дальше
                k += 1
            else:
                print("все камеры размечены")
                break
    cv2.destroyAllWindows()
    print(f"размечено камер: {sum(c['done'] for c in cams.cameras)} из {N}")


if __name__ == "__main__":
    main()
