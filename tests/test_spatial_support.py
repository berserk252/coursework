"""A local vehicle detail must not validate a large unrelated asphalt region."""
import sys
import unittest
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flow_holes import Hole
from foreground_objects import contours
from spatial_support import SpatialSupport, appearance_bounded


class SpatialSupportTests(unittest.TestCase):
    def setUp(self):
        self.shape = (360, 640)
        self.background = np.full(self.shape, 150, np.float32)
        self.frame = self.background.astype(np.uint8)
        self.body = np.zeros(self.shape, bool)
        self.body[120:150, 240:300] = True
        self.frame[self.body] = 225
        self.frame[126:138, 248:291] = 45
        self.hole = Hole(20, 0, np.ones((30, 53), bool), contours(self.body),
                         mask=self.body, area_px=int(self.body.sum()), kind='stationary',
                         observed_s=5., bg_valid=True, bg_coverage=1.)
        self.gate = SpatialSupport(self.shape)

    def test_local_detail_does_not_validate_large_background(self):
        broad = np.zeros(self.shape, bool)
        broad[60:260, 120:450] = True
        hole = replace(self.hole, mask=broad, polygons=contours(broad), area_px=int(broad.sum()), observed_s=90.)
        self.assertEqual(self.gate.update([hole], self.frame, self.background, 100.), [])

    def test_vehicle_body_survives_loss_of_new_edges_in_reference(self):
        result = self.gate.update([self.hole], self.frame, self.background, 5.)
        self.assertEqual(len(result), 1)
        self.assertTrue(np.array_equal(result[0].mask, self.body))
        for second in np.arange(5.1, 30.1, .1):
            hole = replace(self.hole, observed_s=second)
            result = self.gate.update([hole], self.frame, self.frame.astype(np.float32), second)
            self.assertEqual(len(result), 1)
            self.assertTrue(np.array_equal(result[0].mask, self.body))

    def test_new_unsupported_growth_cannot_inherit_vehicle_footprint(self):
        self.gate.update([self.hole], self.frame, self.background, 5.)
        broad = np.zeros(self.shape, bool)
        broad[60:260, 120:450] = True
        hole = replace(self.hole, mask=broad, polygons=contours(broad), area_px=int(broad.sum()), observed_s=6.)
        result = self.gate.update([hole], self.frame, self.background, 6., known=np.ones(self.shape, bool),
                                  foreground=broad, pixel_age=np.full(self.shape, 30.))
        self.assertEqual(len(result), 1)
        bounded = cv2.dilate(self.body.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))) > 0
        self.assertFalse((result[0].mask & ~bounded).any())
        self.assertTrue(result[0].mask[self.body].all())
        self.assertEqual(result[0].area_px, int(result[0].mask.sum()))
        self.assertGreater(result[0].bg_removed_frac, .9)
        self.assertEqual(hole.area_px, int(broad.sum()))

    def test_red_anchor_and_bounded_coarse_body_are_preserved(self):
        red = replace(self.hole, incident=True)
        result = self.gate.update([red], self.background.astype(np.uint8), self.background, 5.)
        self.assertEqual(len(result), 1)
        self.assertIs(result[0], red)
        coarse = replace(self.hole, id=40, kind='flow_gap')
        result = self.gate.update([coarse], self.background.astype(np.uint8), self.background, 5.)
        self.assertIs(result[0], coarse)

    def test_reused_id_cannot_inherit_previous_vehicle(self):
        self.gate.update([self.hole], self.frame, self.background, 5.)
        fresh = replace(self.hole, observed_s=2.)
        result = self.gate.update([fresh], self.frame, self.frame.astype(np.float32), 20.)
        self.assertEqual(result, [])

    def test_unsupported_long_roadside_branch_is_removed_without_losing_body(self):
        mask=self.body.copy()
        mask[132:144,20:250]=True
        result=self.gate.update([replace(self.hole,kind='flow_gap',mask=mask,
                                 polygons=contours(mask),area_px=int(mask.sum()))],
                                self.frame,self.background,5.)
        self.assertEqual(len(result),1)
        self.assertFalse(result[0].mask[:,20:200].any())
        self.assertTrue(result[0].mask[self.body].all())

    def test_new_details_preserve_a_thin_distant_vehicle_next_to_large_body(self):
        mask=self.body.copy();mask[80:92,400:430]=True
        novel=np.zeros(self.shape,bool);novel[80:92,400:430]=True
        result=appearance_bounded(mask,novel,1.)
        self.assertTrue(result[80:92,400:430].all())
        self.assertTrue(result[self.body].all())


if __name__ == '__main__':
    unittest.main()
