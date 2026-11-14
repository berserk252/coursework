"""Возобновляемый полный прогон flow_polygons.py, строго без нейросетей.

Первый вход сохраняет подпапки классов, дополнительные — также имя датасета.
Результаты каждого ролика публикуются только после полной проверки пяти MP4.
"""
import argparse
import csv
import fcntl
import hashlib
import html
import importlib.abc
import json
import os
from pathlib import Path
import re
import runpy
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from datetime import datetime, timezone

import cv2

KINDS = ('combined', 'video_arrows', 'divergence', 'curl', 'graphs')
TABLES = ('stats', 'holes', 'field_stats')
ALGORITHM = ('flow_polygons.py', 'flow_canvas.py', 'flow_filters.py', 'divergence_curl.py',
             'flow_stats.py', 'bg_model.py', 'flow_holes.py', 'foreground_objects.py',
             'auto_road.py', 'road_parts.py', 'mask_regions.py')
FORBIDDEN = {'torch', 'torchvision', 'ultralytics', 'vehicle_tracker'}
SETTINGS = ['--road_mode', 'auto', '--road_seconds', '10', '--max_width', '640',
            '--cell', '24', '--memory', '0.9', '--egomotion', '1', '--edges', '1',
            '--incident_s', '20', '--bg', '1', '--bg_diff_frac', '0.25', '--outputs', 'all']


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def meta(path):
    cap = cv2.VideoCapture(str(path))
    ok, first = cap.read()
    result = {'frames': int(cap.get(7)), 'fps': cap.get(5),
              'width': int(cap.get(3)), 'height': int(cap.get(4))}
    cap.release()
    if not ok or result['frames'] < 2 or result['fps'] <= 0:
        raise ValueError(f'Нечитаемый или слишком короткий исходник: {path}')
    result['width'], result['height'] = first.shape[1], first.shape[0]
    result['pairs'] = result['frames'] - 1
    result['duration_s'] = result['frames'] / result['fps']
    return result


def prepare(args, out, audit):
    if not args.inputs:
        raise ValueError('Для первого запуска нужны --inputs')
    here = Path(__file__).resolve().parent
    roots = [Path(p).resolve() for p in args.inputs]
    if any(out.is_relative_to(p) for p in roots):
        raise ValueError('Папка результатов должна находиться вне входных папок')
    files, keys = [], set()
    for index, root in enumerate(roots):
        paths = sorted(p for p in root.rglob('*') if p.is_file() and p.suffix.lower() == '.mp4')
        if not paths:
            raise ValueError(f'Нет MP4: {root}')
        for source in paths:
            rel = source.parent.relative_to(root)
            if index:
                rel = Path(root.name) / rel
            key = (rel / source.stem).as_posix()
            if key in keys:
                raise ValueError(f'Совпадение выходных имён: {key}')
            keys.add(key)
            item = dict(meta(source), key=key, rel=rel.as_posix(), stem=source.stem,
                        source=str(source), sha256=sha(source), dataset=root.name)
            item['snapshots'] = sorted({min(item['pairs'], max(1, round(t * item['fps'])))
                                        for t in (20, 30, 40, item['duration_s'] / 2)})
            files.append(item)
    snapshot = audit / 'code_snapshot'
    snapshot.mkdir(parents=True, exist_ok=False)
    for source in here.glob('*.py'):
        shutil.copy2(source, snapshot / source.name)
    for name in ('requirements-flow.txt', 'README_скрипты.md'):
        shutil.copy2(here / name, snapshot / name)
    protected = {}
    if args.protect:
        for root in map(Path, args.protect):
            if not root.exists():
                raise ValueError(f'Нет защищаемой папки: {root}')
            for p in sorted(root.rglob('*')):
                if p.is_file():
                    protected[str(p.resolve())] = sha(p)
    manifest = {'created_utc': now(), 'inputs': list(map(str, roots)), 'out_root': str(out),
                'settings': SETTINGS, 'videos': files, 'total': len(files),
                'pairs': sum(v['pairs'] for v in files),
                'duration_s': sum(v['duration_s'] for v in files),
                'algorithm_sha256': {name: sha(snapshot / name) for name in ALGORITHM},
                'runner_sha256': sha(snapshot / Path(__file__).name),
                'protected_sha256': protected, 'python': sys.version,
                'opencv': cv2.__version__, 'reuse': str(Path(args.reuse).resolve()) if args.reuse else None,
                'baseline': str(Path(args.baseline).resolve()) if args.baseline else None,
                'report_copy': str(Path(args.report_copy).resolve()) if args.report_copy else None}
    save(audit / 'manifest.json', manifest)
    save(out / 'status.json', {'run_status': 'prepared', 'started_utc': now(),
         'videos': {v['key']: {'status': 'pending'} for v in files}, 'total': len(files),
         'total_pairs': manifest['pairs'], 'verified_pairs': 0, 'counts': {'pending': len(files)}})
    return manifest


def check_snapshot(manifest, audit):
    for name, digest in manifest['algorithm_sha256'].items():
        if sha(audit / 'code_snapshot' / name) != digest:
            raise ValueError(f'Изменён зафиксированный алгоритм: {name}')
    if sha(audit / 'code_snapshot' / Path(__file__).name) != manifest['runner_sha256']:
        raise ValueError('Изменён зафиксированный пакетный скрипт')


def paths(root, item):
    return {kind: root / kind / item['rel'] / (item['stem'] + ('.mp4' if kind in KINDS else '.csv'))
            for kind in KINDS + TABLES}


def verify(root, item):
    artifacts = paths(root, item)
    with artifacts['stats'].open() as stream:
        rows = list(csv.DictReader(stream))
    with artifacts['field_stats'].open() as stream:
        fields = list(csv.DictReader(stream))
    with artifacts['holes'].open() as stream:
        holes = list(csv.DictReader(stream))
    expected = list(range(1, item['pairs'] + 1))
    for table in (rows, fields):
        if [int(r['frame']) for r in table] != expected:
            raise ValueError(f"Неполная таблица: {item['key']}")
    if any(r['road_mode'] != 'auto' for r in rows):
        raise ValueError('Использована ручная разметка')
    if FORBIDDEN.intersection(rows[0]) or {'vehicles', 'stationary_vehicles', 'queue_parts'}.intersection(rows[0]):
        raise ValueError('Нейросетевые поля в таблице потока')
    for h in holes:
        if h['incident'] == '1' and (h['bg_valid'] != '1' or h['bg_match'] != '0' or h['kind'] == 'stationary'):
            raise ValueError('Красная метка без подтверждения фоном')
    calibration = root / 'auto_road' / item['rel'] / item['stem'] / 'calibration.json'
    plans = json.loads(calibration.read_text())
    count = max(1, max(p['parts'] for p in plans))
    scopes = ['hole', 'rest'] + [s for i in range(1, count + 1) for s in (f'part{i}', f'part{i}_hole')]
    for scope in scopes:
        for name in ('div_per_s', 'rot_per_s', 'rot_abs_per_s', 'cells'):
            if f'{scope}_{name}' not in fields[0]:
                raise ValueError(f'Нет раздельной статистики {scope}_{name}')
    proof = json.loads((root / 'flow_only_proof.json').read_text())
    if not proof.get('flow_only'):
        raise ValueError('Нет доказательства запуска без нейросети')
    scale = min(1., 640 / item['width'])
    width, height = round(item['width'] * scale), round(item['height'] * scale)
    width -= width % 2
    height -= height % 2
    videos = {}
    for kind in KINDS:
        cap = cv2.VideoCapture(str(artifacts[kind]))
        fps, n = cap.get(5), 0
        size = (width * 2, height * 2) if kind == 'combined' else (width, height)
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if (frame.shape[1], frame.shape[0]) != size:
                raise ValueError(f'Неверная геометрия {kind}: {frame.shape}')
            n += 1
        cap.release()
        if n != item['pairs'] or abs(fps - item['fps']) > .0005:
            raise ValueError(f"Неполное видео {kind}: {n}, ожидалось {item['pairs']}; FPS {fps}")
        videos[kind] = {'frames': n, 'fps': fps, 'bytes': artifacts[kind].stat().st_size}
    return {'frames': item['pairs'], 'scenes': len(plans), 'parts_by_scene': [p['parts'] for p in plans],
            'cuts': sum(int(r['cut']) for r in rows), 'hole_rows': len(holes),
            'frames_with_regions': sum(int(r['holes']) > 0 for r in rows),
            'incident_frames': sum(int(r['incident']) for r in rows),
            'first_incident_s': next((float(r['time_s']) for r in rows if r['incident'] == '1'), None),
            'videos': videos, 'artifacts_sha256': {k: sha(p) for k, p in artifacts.items()}}


def worker(item, manifest, audit):
    check_snapshot(manifest, audit)
    blocked = []
    class NoNeural(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname.split('.')[0] in FORBIDDEN:
                blocked.append(fullname)
                raise ImportError('Нейросети запрещены в прогоне потока: ' + fullname)
    sys.meta_path.insert(0, NoNeural())
    work = audit / 'work' / item['key']
    work.mkdir(parents=True, exist_ok=True)
    snapshot = audit / 'code_snapshot'
    sys.path.insert(0, str(snapshot))
    sys.argv = [str(snapshot / 'flow_polygons.py'), '--video', item['source'], '--out_dir', str(work),
                '--rel', item['rel'], '--parts_dir', str(audit / 'no_manual_annotations'),
                '--snapshot_times', ','.join(str(n / item['fps']) for n in item['snapshots']), *manifest['settings']]
    status = 'failed'
    try:
        print('COMMAND', json.dumps(sys.argv, ensure_ascii=False), flush=True)
        runpy.run_path(str(snapshot / 'flow_polygons.py'), run_name='__main__')
        status = 'complete'
    finally:
        loaded = sorted(m for m in sys.modules if m.split('.')[0] in FORBIDDEN)
        save(work / 'flow_only_proof.json', {'status': status, 'blocked_imports': blocked,
             'neural_modules_loaded': loaded, 'flow_only': status == 'complete' and not blocked and not loaded})


def reuse_preview(item, manifest, work):
    if not manifest['reuse']:
        return False
    previous = Path(manifest['reuse'])
    old_audit = previous / 'служебные_данные'
    old_manifest = json.loads((old_audit / 'manifest.json').read_text())
    if old_manifest.get('source') != item['source']:
        return False
    if any(sha(old_audit / 'code_final' / name) != digest for name, digest in manifest['algorithm_sha256'].items()):
        raise ValueError('Проба для повторного использования получена другой версией алгоритма')
    proof = json.loads((old_audit / 'no_yolo_proof.json').read_text())
    if not proof.get('flow_only'):
        raise ValueError('Проба использовала нейросеть')
    for name, source in paths(previous, item).items():
        target = paths(work, item)[name]
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    for kind in ('auto_road', 'snapshots'):
        source = previous / kind / item['rel']
        target = work / kind / item['rel']
        if kind == 'auto_road':
            shutil.copytree(source / item['stem'], target / item['stem'])
        elif source.exists():
            target.mkdir(parents=True, exist_ok=True)
            for p in source.glob(item['stem'] + '_*.png'):
                shutil.copy2(p, target / p.name)
    save(work / 'flow_only_proof.json', dict(proof, reused_from=str(previous)))
    return True


def publish(work, out, item, audit):
    for name, source in paths(work, item).items():
        target = paths(out, item)[name]
        target.parent.mkdir(parents=True, exist_ok=True)
        source.replace(target)
    for kind in ('auto_road', 'snapshots'):
        source = work / kind / item['rel']
        target = out / kind / item['rel']
        target.mkdir(parents=True, exist_ok=True)
        if kind == 'auto_road':
            destination = target / item['stem']
            if destination.exists():
                shutil.rmtree(destination)
            (source / item['stem']).replace(destination)
        elif source.exists():
            for p in source.iterdir():
                p.replace(target / p.name)
    destination = audit / 'proofs' / (item['key'] + '.json')
    destination.parent.mkdir(parents=True, exist_ok=True)
    (work / 'flow_only_proof.json').replace(destination)
    shutil.rmtree(work)


def summary(out, manifest, state):
    columns = ['video', 'dataset', 'status', 'frames', 'frames_with_regions', 'incident_frames', 'first_incident_s', 'wall_s', 'origin']
    with (out / 'summary.csv.tmp').open('w', newline='') as stream:
        table = csv.DictWriter(stream, columns)
        table.writeheader()
        for item in manifest['videos']:
            row = state['videos'][item['key']]
            table.writerow(dict(video=item['key'], dataset=item['dataset'], **{k: row.get(k, '') for k in columns[2:]}))
    (out / 'summary.csv.tmp').replace(out / 'summary.csv')


def report(out, manifest, state):
    figures = out / 'comparisons'
    figures.mkdir(exist_ok=True)
    from PIL import Image, ImageDraw, ImageFont
    font = ImageFont.truetype('/usr/share/fonts/noto/NotoSans-Regular.ttf', 18)
    pictures = []
    baseline = Path(manifest['baseline']) if manifest['baseline'] else None
    for item in manifest['videos']:
        if '5_avariya' not in Path(item['rel']).parts or baseline is None:
            continue
        for source in sorted((baseline / item['stem']).glob('video_arrows_*.jpg')):
            index = int(source.stem.rsplit('_', 1)[1])
            old = cv2.imread(str(source))
            cap = cv2.VideoCapture(str(paths(out, item)['video_arrows']))
            cap.set(cv2.CAP_PROP_POS_FRAMES, index - 1)
            ok, new = cap.read()
            cap.release()
            if old is None or not ok:
                raise ValueError(f'Нет парного кадра: {source}')
            old = cv2.resize(old, (640, 360))
            new = cv2.resize(new, (640, 360))
            image = Image.new('RGB', (1280, 406), 'white')
            for i, (label, frame) in enumerate((('Раньше: поток, ручные зоны', old), ('Сейчас: поток и фон, без YOLO', new))):
                ImageDraw.Draw(image).text((i * 640 + 6, 6), label, font=font, fill='black')
                image.paste(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)), (i * 640, 40))
            name = f"{item['stem']}_{index}.jpg"
            image.save(figures / name, quality=93)
            pictures.append((item['stem'], index / item['fps'], name))
    escape = html.escape
    rows = []
    for item in manifest['videos']:
        s = state['videos'][item['key']]
        rows.append('<tr>' + ''.join(f'<td>{escape(str(x))}</td>' for x in
                    (item['key'], s['frames'], s['frames_with_regions'], s['incident_frames'], s['first_incident_s'])) + '</tr>')
    document = '''<!doctype html><meta charset="utf-8"><style>
    @page {size:A4 landscape;margin:15mm} body{font:12px sans-serif;line-height:1.4}
    table{border-collapse:collapse;width:100%;font-size:10px}td,th{border:1px solid #ccc;padding:4px;text-align:left}
    thead{display:table-header-group}figure{break-inside:avoid; margin:18px 0}img{width:100%}h2{break-before:page}
    </style><h1>Полный прогон оптического потока без YOLO — 14.11.2026</h1>'''
    document += f"<p>Полностью обработано {len(manifest['videos'])} роликов, {manifest['pairs']} межкадровых пар. Пять MP4 на ролик. Все MP4 декодированы целиком; CSV проверены на непрерывность кадров. Ручные зоны и нейросетевые модули не использовались.</p>"
    document += '<p>Настройки: ширина до 640 px, автоматическая дорога по 10 с движения каждой сцены, накопленный фон, восстановление связанных участков кузова по цвету. Совпадающие маленькие участки внутри кузова восстанавливаются; большие замкнутые фоновые просветы и неизвестные точки исключаются.</p>'
    document += '<p>Ограничения: сходные цвета, тени, перекрытия и неизвестный начальный фон оставляют ошибки. Области не гарантируют разделения каждого автомобиля. Числа красных меток — срабатывания правила, а не точность распознавания ДТП. Парные кадры сравнивают версии целиком: ручные зоны старой версии и автоматические зоны новой отличаются.</p>'
    document += '<table><thead><tr><th>Видео</th><th>Пар</th><th>Кадров с областями</th><th>Кадров ДТП?</th><th>Первое ДТП?, с</th></tr></thead><tbody>' + ''.join(rows) + '</tbody></table><h2>Парные кадры аварий</h2>'
    for stem, second, name in pictures:
        document += f'<figure><figcaption>{escape(stem)}, {second:.2f} с</figcaption><img src="comparisons/{escape(name)}"></figure>'
    (out / 'report.html').write_text(document)
    browser = shutil.which('chromium')
    if browser:
        (out / 'служебные_данные').mkdir(parents=True, exist_ok=True)
        command = [browser, '--headless', '--disable-gpu', '--disable-background-networking',
                   '--no-pdf-header-footer', f'--user-data-dir={out / "служебные_данные/browser_profile"}',
                   f'--print-to-pdf={out / "Отчёт_полный_прогон_без_YOLO_14_11_2026.pdf"}', (out / 'report.html').as_uri()]
        with (out / 'служебные_данные/report.log').open('w') as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=180)
        pdf = out / 'Отчёт_полный_прогон_без_YOLO_14_11_2026.pdf'
        if not pdf.is_file() or not pdf.read_bytes().startswith(b'%PDF'):
            raise ValueError('Chromium не создал PDF')
        if shutil.which('qpdf'):
            subprocess.run(['qpdf', '--check', str(pdf)], check=True, stdout=subprocess.DEVNULL)
        if manifest.get('report_copy'):
            destination = Path(manifest['report_copy'])
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(pdf, destination)
    return {'html': str(out / 'report.html'), 'comparison_pairs': len(pictures), 'pdf_created': browser is not None}


def run(args, out, audit, manifest):
    state_path = out / 'status.json'
    state = json.loads(state_path.read_text()) if state_path.exists() else {
        'started_utc': now(), 'videos': {v['key']: {'status': 'pending'} for v in manifest['videos']}}
    lock = threading.Lock()
    def persist():
        counts = {}
        for row in state['videos'].values():
            counts[row['status']] = counts.get(row['status'], 0) + 1
        state.update(updated_utc=now(), counts=counts, total=manifest['total'], total_pairs=manifest['pairs'],
                     verified_pairs=sum(r.get('frames', 0) for r in state['videos'].values() if r['status'] == 'verified'))
        save(state_path, state)
    for item in manifest['videos']:
        row = state['videos'][item['key']]
        if row.get('status') == 'verified':
            if any(sha(paths(out, item)[k]) != digest for k, digest in row['artifacts_sha256'].items()):
                raise ValueError(f"Изменён готовый результат {item['key']}")
        else:
            row['status'] = 'pending'
    state['run_status'] = 'running'
    persist()
    def one(item):
        key = item['key']
        if state['videos'][key]['status'] == 'verified':
            return
        for attempt in (1, 2):
            start = time.monotonic()
            work = audit / 'work' / key
            if work.exists():
                shutil.rmtree(work)
            work.mkdir(parents=True)
            with lock:
                state['videos'][key] = {'status': 'running', 'started_utc': now(), 'attempt': attempt}
                persist()
            try:
                if sha(item['source']) != item['sha256']:
                    raise ValueError('Изменён исходник')
                origin = 'reused_preview' if reuse_preview(item, manifest, work) else 'computed'
                if origin == 'computed':
                    command = [sys.executable, '-u', str(audit / 'code_snapshot/run_polygons_dataset.py'),
                               '--out_root', str(out), '--worker', key]
                    log_path = audit / 'logs' / (key + '.log')
                    log_path.parent.mkdir(parents=True, exist_ok=True)
                    save(log_path.with_suffix('.command.json'), command)
                    env = dict(os.environ, CV_THREADS='1', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', OPENCV_FFMPEG_THREADS='1')
                    with log_path.open('w') as log:
                        subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
                with lock:
                    state['videos'][key]['status'] = 'verifying'
                    persist()
                result = verify(work, item)
                if sha(item['source']) != item['sha256']:
                    raise ValueError('Исходник изменился во время обработки')
                publish(work, out, item, audit)
                result.update(status='verified', origin=origin, completed_utc=now(), wall_s=round(time.monotonic() - start, 2))
                with lock:
                    state['videos'][key].update(result)
                    persist()
                    summary(out, manifest, state)
                print('VERIFIED', key, flush=True)
                return
            except Exception as error:
                with lock:
                    state['videos'][key].update(status='failed', error=str(error), completed_utc=now())
                    persist()
                print('FAILED', key, repr(error), flush=True)
    # Короткий контроль и view4 завершаются рано; аварии идут перед остальными классами.
    ordered = sorted(manifest['videos'], key=lambda v: (0 if manifest['reuse'] and v['stem'] == 'b2D9m3upSZQ' else
                     1 if v['stem'] == 'negative_samples__negative_samples_803' else 2 if v['stem'] == 'view4' else
                     3 if '5_avariya' in Path(v['rel']).parts else 4, -v['duration_s']))
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        pending = {pool.submit(one, item) for item in ordered}
        while pending:
            done, pending = wait(pending, timeout=10, return_when=FIRST_COMPLETED)
            for future in done:
                future.result()
            with lock:
                for key, row in state['videos'].items():
                    if row['status'] == 'running':
                        log_path = audit / 'logs' / (key + '.log')
                        if log_path.exists():
                            with log_path.open('rb') as stream:
                                stream.seek(max(0, log_path.stat().st_size - 8192))
                                matches = re.findall(rb'frame=(\d+) time_s=([\d.]+)', stream.read())
                            if matches:
                                row.update(last_frame=int(matches[-1][0]), source_time_s=float(matches[-1][1]))
                persist()
    if any(r['status'] != 'verified' for r in state['videos'].values()):
        state['run_status'] = 'failed'
        persist()
        raise RuntimeError('Есть не завершившиеся ролики; подробности в status.json')
    for p, digest in manifest['protected_sha256'].items():
        if sha(p) != digest:
            raise ValueError(f'Изменён защищённый файл: {p}')
    for item in manifest['videos']:
        if sha(item['source']) != item['sha256']:
            raise ValueError(f"Изменён исходник: {item['source']}")
    state['run_status'] = 'building_report'
    persist()
    summary(out, manifest, state)
    save(out / 'processing_complete.json', {'videos': manifest['total'], 'output_videos': 5 * manifest['total'],
         'pairs': manifest['pairs'], 'completed_utc': now(), 'protected_files_unchanged': len(manifest['protected_sha256'])})
    report_info = report(out, manifest, state)
    state['run_status'] = 'complete'
    state['completed_utc'] = now()
    persist()
    save(out / 'completion.json', {'status': 'complete', 'videos': manifest['total'],
         'output_videos': 5 * manifest['total'], 'pairs': manifest['pairs'], 'completed_utc': now(),
         'flow_only': True, 'protected_files_unchanged': len(manifest['protected_sha256']), **report_info})
    print('COMPLETE', manifest['total'], 'videos', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', nargs='+')
    parser.add_argument('--out_root', required=True)
    parser.add_argument('--jobs', type=int, default=3)
    parser.add_argument('--protect', nargs='*')
    parser.add_argument('--reuse')
    parser.add_argument('--baseline')
    parser.add_argument('--report_copy', help='Копия итогового PDF по указанному пути')
    parser.add_argument('--prepare_only', action='store_true')
    parser.add_argument('--worker')
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error('--jobs должен быть положительным')
    cv2.setNumThreads(1)
    out = Path(args.out_root).resolve()
    audit = out / 'служебные_данные'
    audit.mkdir(parents=True, exist_ok=True)
    if args.worker:
        manifest = json.loads((audit / 'manifest.json').read_text())
        return worker(next(v for v in manifest['videos'] if v['key'] == args.worker), manifest, audit)
    with (audit / 'runner.lock').open('w') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest_path = audit / 'manifest.json'
        manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else prepare(args, out, audit)
        check_snapshot(manifest, audit)
        print(f"INPUTS {manifest['total']} videos; {manifest['pairs']} pairs; {manifest['duration_s']/60:.1f} source minutes", flush=True)
        if not args.prepare_only:
            try:
                run(args, out, audit, manifest)
            except Exception as error:
                state_path = out / 'status.json'
                state = json.loads(state_path.read_text())
                state.update(run_status='failed', error=str(error), updated_utc=now())
                save(state_path, state)
                raise


if __name__ == '__main__':
    main()
