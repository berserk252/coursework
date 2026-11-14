"""Регрессии фона: пустой асфальт, загрязнение машинами и стоящий объект."""
import math
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bg_model import BackgroundModel


class BackgroundTests(unittest.TestCase):
    def setUp(self):
        self.road = np.ones((72, 96), bool)
        self.clear = np.zeros_like(self.road)
        self.region = self.clear.copy()
        self.region[22:50, 32:64] = True
        self.asphalt = np.full(self.road.shape, 180, np.uint8)
        self.model = BackgroundModel(5, self.road, delay_s=1)

    def feed(self, frame, count=40, frozen=None):
        for _ in range(count):
            self.model.update(frame, self.clear, self.clear if frozen is None else frozen)

    def test_vehicle_outliers_do_not_destroy_asphalt_reference(self):
        self.feed(self.asphalt)
        for index in range(250):
            frame = self.asphalt.copy()
            if index % 4 == 0:
                frame[self.region] = 25
            self.model.update(frame, self.clear, self.clear)
        fraction, valid = self.model.compare(self.asphalt, self.region)
        self.assertTrue(valid)
        self.assertLess(fraction, .05)

    def test_bootstrap_recovers_from_first_vehicle_sample(self):
        for index in range(120):
            frame = self.asphalt.copy()
            if index % 3 == 0:
                frame[self.region] = 25
            self.model.update(frame, self.clear, self.clear)
        fraction, valid = self.model.compare(self.asphalt, self.region)
        self.assertTrue(valid)
        self.assertLess(fraction, .05)

    def test_new_stationary_object_is_not_absorbed(self):
        self.feed(self.asphalt)
        occupied = self.asphalt.copy()
        occupied[self.region] = 40
        self.feed(occupied, count=350)
        fraction, valid = self.model.compare(occupied, self.region)
        self.assertTrue(valid)
        self.assertGreater(fraction, .85)

    def test_freeze_does_not_prevent_unknown_asphalt_bootstrap(self):
        self.feed(self.asphalt, frozen=self.road)
        fraction, valid = self.model.compare(self.asphalt, self.region)
        self.assertTrue(valid)
        self.assertLess(fraction, .05)
        occupied = self.asphalt.copy()
        occupied[self.region] = 40
        self.feed(occupied, count=100, frozen=self.region)
        fraction, valid = self.model.compare(occupied, self.region)
        self.assertTrue(valid)
        self.assertGreater(fraction, .85)

    def test_global_lighting_change_is_not_foreground(self):
        self.feed(self.asphalt)
        changed = np.full(self.road.shape, 230, np.uint8)
        self.feed(changed, count=150)
        fraction, valid = self.model.compare(changed, self.region)
        self.assertTrue(valid)
        self.assertLess(fraction, .05)

    def test_untrained_and_reset_background_remains_unknown(self):
        fraction, valid = self.model.compare(self.asphalt, self.region)
        self.assertFalse(valid)
        self.assertTrue(math.isnan(fraction))
        self.feed(self.asphalt)
        self.model.reset()
        self.assertFalse(self.model.compare(self.asphalt, self.region)[1])


if __name__ == "__main__":
    unittest.main()
