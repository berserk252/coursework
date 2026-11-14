"""Кузов со слабым потоком и окна, похожие на асфальт; дорога между машинами."""
import sys
import unittest
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bg_model import BackgroundModel
from mask_regions import complete_foreground


class MaskRegionTests(unittest.TestCase):
    def setUp(self):
        self.shape = (160, 640)
        self.labels = np.zeros(self.shape, np.int8)
        self.known = np.ones(self.shape, bool)

    def test_window_inside_body_is_part_of_the_same_object(self):
        pixels = np.zeros(self.shape, bool)
        pixels[40:80, 100:180] = True
        pixels[48:59, 120:145] = False  # Окно сравнялось по яркости с асфальтом.
        result = complete_foreground(pixels, self.known, self.labels)
        self.assertTrue(result[40:80, 100:180].all())
        self.assertFalse(result[90:, :].any())

    def test_asphalt_corridor_and_large_cavity_stay_background(self):
        pixels = np.zeros(self.shape, bool)
        pixels[30:100, 80:160] = True
        pixels[30:100, 178:240] = True
        pixels[45:85, 95:135] = False  # Большой просвет — не окно кузова.
        result = complete_foreground(pixels, self.known, self.labels)
        self.assertFalse(result[:, 162:176].any())
        self.assertFalse(result[45:85, 95:135].any())

    def test_unknown_pixels_and_other_road_parts_are_not_completed(self):
        pixels = np.zeros(self.shape, bool)
        pixels[40:80, 100:180] = True
        self.known[50:60, 120:135] = False
        self.labels[:, 150:154] = -1
        result = complete_foreground(pixels, self.known, self.labels)
        self.assertFalse(result[50:60, 120:135].any())
        self.assertFalse(result[:, 150:154].any())

    def test_empty_asphalt_does_not_create_an_object(self):
        result = complete_foreground(np.zeros(self.shape, bool), self.known, self.labels)
        self.assertFalse(result.any())

    def test_missing_white_hood_is_restored_without_outward_growth(self):
        body = np.zeros(self.shape, bool)
        body[40:100, 100:180] = True
        pixels = body.copy()
        pixels[60:74, 100:125] = False  # Пропуск открыт наружу, заполнения дыр мало.
        frame = np.full((*self.shape, 3), 110, np.uint8)
        frame[body] = 225
        result = complete_foreground(pixels, self.known, self.labels, frame)
        self.assertTrue(result[60:74, 100:125].all())
        self.assertFalse(result[~body].any())

    def test_same_color_road_between_separate_objects_is_not_filled(self):
        pixels = np.zeros(self.shape, bool)
        pixels[40:100, 100:150] = True
        pixels[40:100, 160:200] = True
        frame = np.full((*self.shape, 3), 180, np.uint8)
        result = complete_foreground(pixels, self.known, self.labels, frame)
        self.assertFalse(result[:, 152:158].any())

    def test_textureless_moving_body_does_not_become_background(self):
        shape = (96, 160)
        road = np.ones(shape, bool)
        clear = np.zeros(shape, bool)
        moving_edges = np.zeros(shape, np.uint8)
        cv2.rectangle(moving_edges, (52, 32), (71, 49), 1, 2)
        vehicle = np.full(shape, 120, np.uint8)
        vehicle[32:50, 52:72] = 210
        model = BackgroundModel(5, road, delay_s=1)
        # Движение определилось по краям, внутри однородного кузова его нет.
        for _ in range(40):
            model.update(vehicle, moving_edges > 0, clear)
        asphalt = np.full(shape, 120, np.uint8)
        for _ in range(80):
            model.update(asphalt, clear, clear)
        self.assertTrue(model.trusted()[40, 61])
        self.assertLess(abs(float(model.bg[40, 61]) - 120), 2)
        foreground, _ = model.foreground(asphalt)
        self.assertFalse(foreground[36:46, 56:67].any())


if __name__ == '__main__':
    unittest.main()
