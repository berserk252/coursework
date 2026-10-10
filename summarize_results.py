"""
Сводная таблица по результатам пакетного прогона (run_dataset.py).

    python summarize_results.py --root "../../Результаты прогона"

Читает <root>/canvas/stats/**/*.csv (по кадрам: скорость, падение скорости,
доля движения, дивергенция, ротор, ...) и пишет:
    <root>/summary_videos.csv   -- одна строка на видео
    <root>/summary_classes.csv  -- средние по классам (папкам)
    <root>/summary.md           -- то же таблицами для чтения

Признаки на видео (только по годным кадрам, valid=1):
    speed_med      медиана средней скорости потока V, px/с
    drop_max_2s    максимум падения скорости, удержанного 2 с (медиана за 2 с)
    drop_t         когда это было, с
    moving_med     медиана доли дороги в движении
    div_mean_med   медиана средней дивергенции, 1/с
    div_min_p5     5-й перцентиль минимальной дивергенции (сильное схождение), 1/с
    rot_mean_med   медиана среднего |ротора|, 1/с
    rot_max_p95    95-й перцентиль максимального |ротора|, 1/с
    + любые другие числовые колонки CSV (например stopped_area): медиана и максимум
"""

import argparse
import csv
import glob
import math
import os
from collections import defaultdict

import numpy as np

BASE_COLS = {"frame", "time_s", "valid", "cut"}


def rolling_median(x, n):
    out = np.full(len(x), np.nan)
    for i in range(len(x)):
        w = x[max(0, i - n + 1):i + 1]
        w = w[np.isfinite(w)]
        if len(w) >= max(1, n // 2):
            out[i] = np.median(w)
    return out


def load(path):
    with open(path) as f:
        r = csv.DictReader(f)
        rows = list(r)
    if not rows:
        return None
    cols = {}
    for k in rows[0]:
        try:
            cols[k] = np.array([float(row[k]) if row[k] not in ("", "nan") else np.nan for row in rows])
        except ValueError:
            pass
    return cols


def nanstat(f, x):
    x = x[np.isfinite(x)]
    return float(f(x)) if x.size else float("nan")


def video_features(cols):
    t = cols["time_s"]
    valid = cols.get("valid", np.ones_like(t)) > 0
    fps = 1.0 / np.median(np.diff(t)) if len(t) > 2 else 25.0
    feat = {"duration_s": float(t[-1]) if len(t) else 0.0,
            "valid_share": float(valid.mean()) if len(t) else 0.0,
            "cuts": int(np.nansum(cols.get("cut", np.zeros_like(t))))}

    def col(name):
        x = cols.get(name)
        if x is None:
            return np.full_like(t, np.nan)
        return np.where(valid, x, np.nan)

    feat["speed_med"] = nanstat(np.median, col("speed_px_s"))
    drop2 = rolling_median(col("drop"), max(1, int(round(2 * fps))))
    if np.isfinite(drop2).any():
        i = int(np.nanargmax(drop2))
        feat["drop_max_2s"], feat["drop_t"] = float(drop2[i]), float(t[i])
    else:
        feat["drop_max_2s"], feat["drop_t"] = float("nan"), float("nan")
    feat["moving_med"] = nanstat(np.median, col("moving_area"))
    feat["div_mean_med"] = nanstat(np.median, col("div_mean_per_s"))
    feat["div_min_p5"] = nanstat(lambda a: np.percentile(a, 5), col("div_min_per_s"))
    feat["rot_mean_med"] = nanstat(np.median, col("rot_mean_abs_per_s"))
    feat["rot_max_p95"] = nanstat(lambda a: np.percentile(a, 95), col("rot_max_abs_per_s"))
    known = BASE_COLS | {"speed_px_s", "speed_frame_px_s", "speed_base_px_s", "drop", "moving_area",
                         "div_mean_per_s", "div_min_per_s", "div_max_per_s", "rot_mean_abs_per_s",
                         "rot_max_abs_per_s", "active_cells"}
    for k in cols:
        if k not in known:
            feat[k + "_med"] = nanstat(np.median, col(k))
            feat[k + "_max"] = nanstat(np.max, col(k))
    return feat


def fmt(v):
    if isinstance(v, str):
        return v
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return "-"
    if isinstance(v, int):
        return str(v)
    a = abs(v)
    return f"{v:.0f}" if a >= 100 else f"{v:.3g}"


def main():
    ap = argparse.ArgumentParser(description="Сводная таблица по результатам прогона.")
    ap.add_argument("--root", required=True)
    args = ap.parse_args()
    stats_root = os.path.join(args.root, "canvas", "stats")
    files = sorted(glob.glob(os.path.join(stats_root, "**", "*.csv"), recursive=True))
    rows = []
    for p in files:
        cols = load(p)
        if cols is None or "time_s" not in cols:
            continue
        rel = os.path.relpath(p, stats_root)
        parts = rel.split(os.sep)
        rows.append({"dataset": parts[0] if len(parts) > 2 else "", "class": parts[-2] if len(parts) > 1 else "",
                     "video": os.path.splitext(parts[-1])[0], **video_features(cols)})
    if not rows:
        print("нет CSV в", stats_root)
        return
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(os.path.join(args.root, "summary_videos.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)

    groups = defaultdict(list)
    for r in rows:
        groups[(r["dataset"], r["class"])].append(r)
    num_keys = [k for k in keys if k not in ("dataset", "class", "video")]
    crow = []
    for (ds, cl), rs in sorted(groups.items()):
        c = {"dataset": ds, "class": cl, "videos": len(rs)}
        for k in num_keys:
            vals = np.array([r[k] for r in rs if isinstance(r.get(k), (int, float))], float)
            c[k] = nanstat(np.median, vals)
        crow.append(c)
    with open(os.path.join(args.root, "summary_classes.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(crow[0].keys()))
        w.writeheader()
        w.writerows(crow)

    show = ["speed_med", "drop_max_2s", "moving_med", "div_mean_med", "div_min_p5", "rot_mean_med",
            "rot_max_p95"] + [k for k in num_keys if k.endswith("_max") and k.startswith("stopped")]
    with open(os.path.join(args.root, "summary.md"), "w") as f:
        f.write("# Сводка по классам (медианы по видео)\n\n")
        hdr = ["датасет", "класс", "видео"] + show
        f.write("| " + " | ".join(hdr) + " |\n|" + "---|" * len(hdr) + "\n")
        for c in crow:
            f.write("| " + " | ".join([c["dataset"], c["class"], str(c["videos"])] +
                                      [fmt(c.get(k)) for k in show]) + " |\n")
        f.write("\n# По видео\n\n")
        hdr = ["класс", "видео", "длит, с", "склеек"] + show + ["drop_t"]
        f.write("| " + " | ".join(hdr) + " |\n|" + "---|" * len(hdr) + "\n")
        for r in rows:
            f.write("| " + " | ".join([r["class"], r["video"], fmt(r["duration_s"]), fmt(r["cuts"])] +
                                      [fmt(r.get(k)) for k in show] + [fmt(r.get("drop_t"))]) + " |\n")
    print(f"видео: {len(rows)}, классов: {len(crow)} -> summary_videos.csv, summary_classes.csv, summary.md")


if __name__ == "__main__":
    main()
