import sys
import unittest
import importlib.util
from types import SimpleNamespace
from pathlib import Path

import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from vehicle_tracker import MotionState


class VehicleMotionTests(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec('ultralytics'), 'YOLO environment required')
    def test_bytetrack_keeps_ids_across_short_occlusion(self):
        from ultralytics.trackers.byte_tracker import BYTETracker
        from ultralytics.engine.results import Boxes
        tracker=BYTETracker(SimpleNamespace(track_high_thresh=.25,track_low_thresh=.1,
                                            new_track_thresh=.25,track_buffer=6,match_thresh=.8,fuse_score=True))
        original=None
        for step in range(7):
            rows=[[10+step,20,30+step,50,.95,0],[90-step,20,110-step,50,.95,0]]
            if step==3:rows=rows[1:]
            found=tracker.update(Boxes(np.array(rows,np.float32),(100,160)))
            ids={int(row[4]) for row in found}
            if original is None:original=ids
            if step!=3:self.assertEqual(ids,original)
    def test_speed_uses_time_and_is_independent_of_sample_rate(self):
        speeds=[]
        for hz in (2,10):
            state=MotionState()
            for t in np.arange(0,5,1/hz):
                state.update(np.array([10+10*t,20,40+10*t,50]),float(t))
            speeds.append(state.speed)
            self.assertEqual(state.state,'moving')
        self.assertAlmostEqual(speeds[0],10,places=4)
        self.assertAlmostEqual(speeds[1],10,places=4)

    def test_stopped_car_survives_detection_jitter(self):
        state=MotionState()
        for i,t in enumerate(np.arange(0,6,.2)):
            jitter=(-1)**i*.4
            state.update(np.array([20+jitter,30,60+jitter,80]),float(t))
        self.assertEqual(state.state,'stationary')
        self.assertGreater(state.stop_s,4)

    def test_hidden_interval_is_not_counted_as_observed_stop(self):
        state=MotionState()
        for t in np.arange(0,5,.2):state.update(np.array([20,30,60,80]),float(t))
        state.update(np.array([20,30,60,80]),10.)
        self.assertEqual(state.state,'warming')
        self.assertEqual(state.stop_s,0)


if __name__=='__main__':unittest.main()
