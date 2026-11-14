"""Регрессии: асфальт внутри контура, белый кузов и объект без истории проездов."""
import sys
import unittest
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bg_model import BackgroundModel
from flow_holes import Hole
from foreground_objects import ForegroundObjects, clip_holes
from flow_polygons import draw_holes


class ForegroundTests(unittest.TestCase):
    def setUp(self):
        self.shape = (96, 144)
        self.labels = np.zeros(self.shape, np.int8)
        self.region = np.zeros(self.shape, bool)
        self.region[20:76, 28:116] = True
        self.hole = Hole(1, 0, np.ones((8, 12), bool),
                         [np.array([[5,5],[138,5],[138,90],[5,90]], np.int32)], incident=True)

    def test_entire_polygon_is_trimmed_and_asphalt_cavity_stays_empty(self):
        foreground = self.region.copy()
        foreground[34:62, 56:88] = False
        result = clip_holes([self.hole], foreground, np.ones(self.shape,bool), self.labels)
        self.assertEqual(len(result), 1)
        self.assertFalse((result[0].mask & ~foreground).any())
        self.assertFalse(result[0].mask[48,72])
        image = np.full((*self.shape,3), 180, np.uint8)
        draw_holes(image, result, None, labels=False)
        self.assertTrue(np.all(image[48,72] == 180))
        self.assertGreater(result[0].bg_removed_frac, .5)

    def test_unknown_background_never_becomes_confirmed_object(self):
        self.assertEqual(clip_holes([self.hole], self.region, np.zeros(self.shape,bool), self.labels), [])

    def test_overlapping_regions_do_not_hide_cars_with_repeated_fill(self):
        first = replace(self.hole, mask=self.region.copy(), incident=False)
        second_mask = np.zeros(self.shape, bool)
        second_mask[30:86, 50:126] = True
        second = replace(first, id=2, mask=second_mask)
        image = np.full((*self.shape, 3), 180, np.uint8)
        draw_holes(image, [first, second], None, labels=False)
        # Внутри одного и двух полигонов одна и та же слабая заливка.
        np.testing.assert_array_equal(image[40, 40], image[45, 80])
        self.assertGreater(int(image[45, 80, 0]), 150)

    def test_asphalt_cavity_has_no_thick_internal_outline(self):
        mask = self.region.copy()
        mask[34:62, 56:88] = False
        hole = replace(self.hole, mask=mask)
        image = np.full((*self.shape, 3), 180, np.uint8)
        draw_holes(image, [hole], None, labels=False)
        np.testing.assert_array_equal(image[34:62, 56:88], 180)

    def test_white_body_is_found_with_low_noise_reference(self):
        road = self.labels >= 0
        model = BackgroundModel(10, road, delay_s=1)
        clear = np.zeros(self.shape,bool)
        asphalt = np.full(self.shape,180,np.uint8)
        for _ in range(60):model.update(asphalt,clear,clear)
        frame = asphalt.copy();frame[self.region] = 205
        cv2.rectangle(frame,(35,25),(105,65),65,2)
        fg, known = model.foreground(frame)
        self.assertGreater(fg[40:55,40:52].mean(), .9)
        detector = ForegroundObjects(self.shape,10,confirm_s=2)
        found = []
        for i in range(30):
            found = detector.update(frame,model.bg,fg,clear,self.labels,i/10,12,8,12)
        self.assertTrue(found)
        self.assertTrue(all(h.kind == "stationary" and not h.incident for h in found))


if __name__ == "__main__":
    unittest.main()
