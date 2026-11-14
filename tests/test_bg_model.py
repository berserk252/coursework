"""Регрессии фона: пустой асфальт, загрязнение машинами и стоящий объект."""
import math
import sys
import unittest
from pathlib import Path

import numpy as np
import cv2

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

    def test_first_vehicle_run_does_not_lock_the_initial_background(self):
        model = BackgroundModel(5, self.road, delay_s=8)
        vehicle = self.asphalt.copy()
        vehicle[self.region] = 45
        # Ранний однотонный кузов занимает первые 12 наблюдений. Дальше
        # открывается асфальт, который преобладает в полном окне прогрева.
        for i in range(100):
            model.update(vehicle if i < 12 else self.asphalt, self.clear, self.clear)
        fraction, valid = model.compare(self.asphalt, self.region)
        self.assertTrue(valid)
        self.assertLess(fraction, .05)
        self.assertGreater(float(model.bg[self.region].mean()), 175)

    def test_new_stationary_object_is_not_absorbed(self):
        self.feed(self.asphalt)
        occupied = self.asphalt.copy()
        occupied[self.region] = 40
        self.feed(occupied, count=350)
        fraction, valid = self.model.compare(occupied, self.region)
        self.assertTrue(valid)
        self.assertGreater(fraction, .85)

    def test_departed_initial_vehicle_can_be_replaced_by_verified_clear_asphalt(self):
        model=BackgroundModel(5,self.road,delay_s=1)
        occupied=self.asphalt.copy();occupied[self.region]=230
        cv2.rectangle(occupied,(38,27),(57,43),55,-1)
        for _ in range(60):model.update(occupied,self.clear,self.clear)
        self.assertTrue(model.trusted()[35,46])
        for _ in range(130):model.update(self.asphalt,self.clear,self.road)
        fraction,valid=model.compare(self.asphalt,self.region)
        self.assertTrue(valid);self.assertLess(fraction,.05)
        self.assertTrue(model.recovered[self.region].any())

    def test_background_recovery_does_not_absorb_a_new_vehicle_touching_the_ghost(self):
        model=BackgroundModel(5,self.road,delay_s=1)
        initial=self.asphalt.copy();initial[self.region]=230
        cv2.rectangle(initial,(38,27),(57,43),55,-1)
        for _ in range(60):model.update(initial,self.clear,self.clear)
        current=self.asphalt.copy();vehicle=self.clear.copy();vehicle[25:43,64:80]=True
        current[vehicle]=220;cv2.rectangle(current,(68,28),(74,33),65,-1)
        for _ in range(150):model.update(current,self.clear,self.road)
        difference,valid=model.compare(current,vehicle)
        self.assertTrue(valid);self.assertGreater(difference,.8)


    def test_early_collision_does_not_become_the_initial_road(self):
        model=BackgroundModel(5,self.road,delay_s=12)
        occupied=self.asphalt.copy();occupied[self.region]=40
        for i in range(65):
            model.update(self.asphalt if i<42 else occupied,self.clear,self.clear)
        fraction,valid=model.compare(occupied,self.region)
        self.assertTrue(valid);self.assertGreater(fraction,.85)

    def test_sampling_rounding_does_not_shorten_delay_at_12_5_fps(self):
        model=BackgroundModel(12.5,self.road,delay_s=12)
        for _ in range(149):model.update(self.asphalt,self.clear,self.clear)
        self.assertFalse(model.trusted().any())
        for _ in range(10):model.update(self.asphalt,self.clear,self.clear)
        self.assertTrue(model.trusted().all())

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
