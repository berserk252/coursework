import sys
import unittest
from pathlib import Path
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from auto_road import MotionRoadEstimator


class AutoRoadTests(unittest.TestCase):
    def test_opposite_flows_are_distinct_and_median_stays_outside(self):
        e=MotionRoadEstimator((120,180))
        flow=np.zeros((120,180,2),np.float32)
        move=np.zeros((120,180),bool)
        move[15:115,20:58]=True;move[15:115,120:158]=True
        flow[15:115,20:58,1]=2;flow[15:115,120:158,1]=-2
        for _ in range(30):e.update(flow,move)
        parts,meta=e.finish()
        self.assertEqual(len(parts),2)
        labels=parts.label_map(180,120)
        self.assertLess(labels[60,90],0)
        self.assertNotEqual(labels[60,40],labels[60,140])
        self.assertLess(meta['directions'][0]['unit_vector'][1]*meta['directions'][1]['unit_vector'][1],0)

    def test_still_frame_does_not_invent_road(self):
        e=MotionRoadEstimator((100,150))
        for _ in range(30):e.update(np.zeros((100,150,2),np.float32),np.zeros((100,150),bool))
        self.assertEqual(len(e.finish()[0]),0)

    def test_global_camera_flow_is_rejected(self):
        e=MotionRoadEstimator((100,150))
        for _ in range(20):e.update(np.ones((100,150,2),np.float32),np.ones((100,150),bool))
        self.assertEqual(len(e.finish()[0]),0)


if __name__=='__main__':unittest.main()
