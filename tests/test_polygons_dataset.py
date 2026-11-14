"""Повторное использование пробы требует того же содержимого, кода и всех выходов."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from run_polygons_dataset import reuse_preview, sha


class ReuseTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.audit = self.root / 'служебные_данные'
        self.audit.mkdir()
        self.item = {'source': '/source/video.mp4', 'sha256': 'current-content',
                     'rel': '5_avariya', 'stem': 'video'}
        self.manifest = {'reuse': str(self.root), 'algorithm_sha256': {}}

    def source_record(self, digest):
        (self.audit / 'manifest.json').write_text(json.dumps({
            'source': self.item['source'],
            'protected_files': {self.item['source']: {'sha256': digest}}}))

    def test_same_path_with_changed_content_is_recomputed(self):
        self.source_record('old-content')
        self.assertFalse(reuse_preview(self.item, self.manifest, self.root / 'work'))

    def test_different_processing_code_is_recomputed(self):
        self.source_record('current-content')
        code = self.audit / 'code_final'
        code.mkdir()
        (code / 'flow_polygons.py').write_text('old algorithm')
        self.manifest['algorithm_sha256'] = {'flow_polygons.py': 'new-algorithm-hash'}
        self.assertFalse(reuse_preview(self.item, self.manifest, self.root / 'work'))

    def test_missing_output_is_recomputed(self):
        self.source_record('current-content')
        self.assertFalse(reuse_preview(self.item, self.manifest, self.root / 'work'))


if __name__ == '__main__':
    unittest.main()
