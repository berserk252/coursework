"""
Пакетный прогон flow_canvas.py и flow_heatmap_with_decay.py по папкам с видео.

    python run_dataset.py --inputs ../traffic_flow_dataset ../newVideos --out_root "../../Результаты прогона"

Структура результата (имя папки датасета и подпапки классов сохраняются):
    <out_root>/canvas/<тип>/<датасет>/<класс>/<видео>.mp4     -- flow_canvas.py
    <out_root>/canvas/stats/<датасет>/<класс>/<видео>.csv
    <out_root>/flow_heatmap/<датасет>/<класс>/<видео>/*.mp4   -- flow_heatmap_with_decay.py
    <out_root>/run_log.csv                                    -- журнал: что, сколько секунд, код возврата

Щадит машину: --jobs процессов параллельно, у каждого пониженный приоритет
(nice) и 1 поток OpenCV. Уже обработанные видео пропускаются, поэтому
прерванный прогон можно просто запустить ещё раз той же командой.
"""

import argparse
import csv
import glob
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable


def duration(path):
    c = cv2.VideoCapture(path)
    n, fps = c.get(cv2.CAP_PROP_FRAME_COUNT), c.get(cv2.CAP_PROP_FPS) or 25
    c.release()
    return n / fps if n > 0 else 0.0


def main():
    ap = argparse.ArgumentParser(description="Пакетный прогон по папкам с видео.")
    ap.add_argument("--inputs", nargs="+", required=True, help="Папки с видео (рекурсивно, *.mp4)")
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--scripts", default="canvas,heatmap", help="canvas, heatmap или оба через запятую")
    ap.add_argument("--jobs", type=int, default=4, help="Сколько процессов одновременно")
    ap.add_argument("--max_width", type=int, default=640)
    ap.add_argument("--canvas_outputs", default="all", help="--outputs для flow_canvas.py (all/none/список)")
    ap.add_argument("--heatmap_outputs", default="heat,flow,combined", help="--outputs для flow_heatmap")
    ap.add_argument("--nice", type=int, default=10)
    ap.add_argument("--force", action="store_true", help="Пересчитать даже уже готовые")
    ap.add_argument("--log", default="run_log.csv", help="Имя журнала внутри out_root")
    args = ap.parse_args()

    scripts = {x.strip() for x in args.scripts.split(",") if x.strip()}
    os.makedirs(args.out_root, exist_ok=True)
    jobs = []
    for inp in args.inputs:
        inp = os.path.abspath(inp)
        ds = os.path.basename(inp.rstrip("/"))
        for v in sorted(glob.glob(os.path.join(inp, "**", "*.mp4"), recursive=True)):
            rel = os.path.join(ds, os.path.relpath(os.path.dirname(v), inp))
            stem = os.path.splitext(os.path.basename(v))[0]
            dur = duration(v)
            if "canvas" in scripts:
                done = os.path.join(args.out_root, "canvas", "stats", rel, stem + ".csv")
                cmd = [PY, os.path.join(HERE, "flow_canvas.py"), "--video", v,
                       "--out_dir", os.path.join(args.out_root, "canvas"), "--rel", rel,
                       "--max_width", str(args.max_width), "--outputs", args.canvas_outputs]
                jobs.append(("canvas", v, dur, done, cmd))
            if "heatmap" in scripts:
                od = os.path.join(args.out_root, "flow_heatmap", rel, stem)
                done = os.path.join(od, "flow_stats.csv")
                cmd = [PY, os.path.join(HERE, "flow_heatmap_with_decay.py"), "--video", v,
                       "--out_dir", od, "--max_width", str(args.max_width),
                       "--outputs", args.heatmap_outputs]
                jobs.append(("heatmap", v, dur, done, cmd, od))

    # длинные вперёд -- процессы заканчивают примерно одновременно
    jobs.sort(key=lambda j: -j[2])
    todo = [j for j in jobs if args.force or not os.path.exists(j[3] + ".done")]
    total_s = sum(j[2] for j in todo)
    print(f"заданий: {len(jobs)}, к выполнению: {len(todo)}, видео суммарно {total_s / 60:.1f} мин", flush=True)

    log_path = os.path.join(args.out_root, args.log)
    new_log = not os.path.exists(log_path)
    lock = threading.Lock()
    log_f = open(log_path, "a", newline="")
    log = csv.writer(log_f)
    if new_log:
        log.writerow(["script", "video", "video_s", "wall_s", "returncode", "finished_at"])
    env = dict(os.environ, CV_THREADS="1", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    state = {"done": 0, "fail": 0, "t0": time.time()}

    def run(job):
        script, video, dur, done = job[:4]
        cmd = job[4]
        if script == "heatmap":
            os.makedirs(job[5], exist_ok=True)
        t0 = time.time()
        r = subprocess.run(["nice", "-n", str(args.nice)] + cmd, env=env,
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        wall = time.time() - t0
        ok = r.returncode == 0 and os.path.exists(done)
        if ok:
            open(done + ".done", "w").close()
        with lock:
            state["done" if ok else "fail"] += 1
            log.writerow([script, video, f"{dur:.1f}", f"{wall:.1f}", r.returncode,
                          time.strftime("%Y-%m-%d %H:%M:%S")])
            log_f.flush()
            n = state["done"] + state["fail"]
            el = (time.time() - state["t0"]) / 60
            print(f"[{n}/{len(todo)}] {'ok ' if ok else 'ОШИБКА'} {script:7s} {os.path.basename(video)} "
                  f"({dur:.0f} с видео за {wall:.0f} с), прошло {el:.0f} мин", flush=True)
            if not ok:
                print("   " + "\n   ".join(r.stdout.strip().splitlines()[-6:]), flush=True)

    with ThreadPoolExecutor(max_workers=args.jobs) as ex:
        list(ex.map(run, todo))
    log_f.close()
    print(f"готово: успешно {state['done']}, ошибок {state['fail']}, "
          f"за {(time.time() - state['t0']) / 60:.0f} мин", flush=True)


if __name__ == "__main__":
    main()
