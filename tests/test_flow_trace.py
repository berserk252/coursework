"""The optional replay cache preserves precision and never mutates live holes."""
import gzip
import pickle
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import sys
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flow_holes import Hole
from flow_trace import TraceWriter


class TraceTests(unittest.TestCase):
    def test_chunk_prefix_precision_and_no_live_mutation(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'source.mp4'
            source.write_bytes(b'fixture')
            shape = (8, 16)
            mask = np.ones(shape, bool)
            hole = Hole(1, 0, mask, [], mask=mask, v_in=19.123456789)
            writer = TraceWriter(root / 'trace', source, 24., shape, 1, [], .5, chunk_frames=1)
            sm = SimpleNamespace(t=[0., 1. / 24])
            ps = SimpleNamespace(hist=[[2.123456789, 3.123456789]])
            fh = [[float('nan')], [1.123456789]]
            grid = np.full((2, 3), .123456789, np.float64)
            writer.add(1, 1. / 24, SimpleNamespace(div=grid), [hole], [hole],
                       np.ones(shape), mask, mask, np.zeros(shape), np.zeros(shape), 0, sm, ps, fh, {},
                       gray=np.zeros(shape, np.uint8))
            writer.close()
            with gzip.open(root / 'trace/chunk_0000.pkl.gz', 'rb') as stream:
                header, record = pickle.load(stream), pickle.load(stream)
            self.assertEqual(header['times'], [0.])
            self.assertEqual(header['part_speeds'], [[2.123456789]])
            self.assertEqual(record['holes'][0].v_in, 19.123456789)
            self.assertEqual(record['base_holes'][0].v_in, 48.)
            self.assertEqual(hole.v_in, 19.123456789)
            np.testing.assert_array_equal(record['dc'].div, grid)
            self.assertEqual(record['field_stats'], fh[-1])


if __name__ == '__main__':
    unittest.main()
