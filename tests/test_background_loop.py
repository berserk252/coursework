import sys
import unittest
from pathlib import Path
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from background_loop import ColorSamples


class RepeatedVideoTests(unittest.TestCase):
    def test_repeated_frame_does_not_invent_unique_evidence(self):
        samples=ColorSamples((8,12));mask=np.ones((8,12),bool)
        for _ in range(30):samples.add(np.full((8,12,3),160,np.uint8),mask,7)
        self.assertTrue((samples.unique==1).all())
        self.assertTrue((samples.count==30).all())

    def test_protected_vehicle_never_enters_color_background(self):
        samples=ColorSamples((8,12));road=np.ones((8,12),bool);road[2:6,3:9]=False
        frame=np.full((8,12,3),170,np.uint8);frame[~road]=250
        for i in range(20):samples.add(frame,road,i)
        self.assertTrue((samples.unique[~road]==0).all())
        self.assertTrue((samples.mean[road]==170).all())


if __name__=='__main__':unittest.main()
