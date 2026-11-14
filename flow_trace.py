"""Read-only, full-precision frame traces for repeated polygon experiments."""
from dataclasses import replace
import gzip
import hashlib
import json
from pathlib import Path
import pickle
import sys

import cv2
import numpy as np

PREFIX_FILES = ('flow_polygons.py', 'flow_canvas.py', 'flow_filters.py', 'divergence_curl.py',
                'flow_stats.py', 'bg_model.py', 'flow_holes.py', 'foreground_objects.py',
                'auto_road.py', 'road_parts.py', 'mask_regions.py', 'flow_trace.py')


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


class TraceWriter:
    """Cache numerical analysis, never modify live detector objects or arrays."""
    def __init__(self, directory, source, fps, shape, n_parts, plans, scale, chunk_frames=1200):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.chunk_frames = chunk_frames
        self.header = {'version': 1, 'source': str(Path(source).resolve()), 'source_sha256': sha(source),
                       'fps': fps, 'shape': tuple(shape), 'n_parts': n_parts, 'plans': plans, 'scale': scale,
                       'prefix_sha256': {name: sha(Path(__file__).with_name(name)) for name in PREFIX_FILES},
                       'spatial_sha256': sha(Path(__file__).with_name('spatial_support.py')),
                       'python': sys.version, 'opencv': cv2.__version__}
        self.stream = None
        self.chunks = []
        self.count = 0
        self.total = 0

    def _open(self, idx, sm, ps, fh):
        self.path = self.directory / f'chunk_{len(self.chunks):04d}.pkl.gz'
        self.stream = gzip.open(self.path, 'wb', compresslevel=1)
        header = dict(self.header, first_frame=idx, times=list(sm.t[:-1]),
                      part_speeds=[list(values[:-1]) for values in ps.hist], field_history=list(fh[:-1]))
        pickle.dump(header, self.stream, protocol=5)
        self.first = idx
        self.count = 0

    def add(self, idx, t, dc, base_holes, holes, mag_clean, known, foreground, pixel_age,
            background, scene_id, sm, ps, fh, basic_stats, gray=None):
        if self.stream is None:
            self._open(idx, sm, ps, fh)
        base = [replace(h, v_in=float(mag_clean[h.mask].mean()) * self.header['fps'] / self.header['scale'])
                for h in base_holes]
        record = {'frame': idx, 'time_s': t, 'scene_id': scene_id, 'dc': dc, 'base_holes': base,
                  'holes': holes, 'background_u8': np.clip(background, 0, 255).astype(np.uint8),
                  'known': known, 'foreground': foreground, 'pixel_age': pixel_age,
                  'part_speeds': [values[-1] for values in ps.hist], 'field_stats': fh[-1],
                  'basic_stats': basic_stats,
                  'gray_sha256': hashlib.sha256(gray.tobytes()).hexdigest() if gray is not None else None}
        pickle.dump(record, self.stream, protocol=5)
        self.count += 1
        self.total += 1
        self.last = idx
        if self.count >= self.chunk_frames:
            self._close_chunk()

    def _close_chunk(self):
        if self.stream is None:
            return
        self.stream.close()
        self.stream = None
        self.chunks.append({'file': self.path.name, 'first_frame': self.first, 'last_frame': self.last,
                            'frames': self.count, 'sha256': sha(self.path)})
        value = {'source': self.header['source'], 'source_sha256': self.header['source_sha256'],
                 'frames': self.total, 'chunks': self.chunks, 'complete': False}
        self._save(value)

    def _save(self, value):
        target = self.directory / 'manifest.json'
        temporary = target.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
        temporary.replace(target)

    def close(self):
        self._close_chunk()
        self._save({'source': self.header['source'], 'source_sha256': self.header['source_sha256'],
                    'frames': self.total, 'chunks': self.chunks, 'complete': True})
