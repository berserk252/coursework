import sys
import unittest
import tempfile
from pathlib import Path
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from auto_road import MotionRoadEstimator
from road_parts import RoadParts


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

    def test_temporal_road_appearance_preserves_corridor_without_filling_grass(self):
        shape=(120,180);e=MotionRoadEstimator(shape)
        frame=np.full((*shape,3),(40,130,40),np.uint8)
        frame[10:118,20:80]=(150,150,150)
        flow=np.zeros((*shape,2),np.float32);flow[15:115,36:54,1]=2
        moving=np.zeros(shape,bool);moving[15:115,36:54]=True
        for _ in range(35):e.update(flow,moving,frame)
        parts,_=e.finish();labels=parts.label_map(180,120)
        self.assertGreaterEqual(labels[60,40],0)
        self.assertGreaterEqual(labels[60,52],0)
        self.assertLess(labels[60,20],0)
        self.assertLess(labels[60,100],0)

    def test_short_timestamp_bursts_do_not_define_a_road(self):
        shape=(120,180);e=MotionRoadEstimator(shape)
        flow=np.zeros((*shape,2),np.float32);flow[100:119,10:160,0]=2
        moving=np.zeros(shape,bool);moving[100:119,10:160]=True
        for i in range(60):e.update(flow,moving if i%10==0 else np.zeros(shape,bool))
        self.assertEqual(len(e.finish()[0]),0)

    def test_duplicate_cctv_frames_do_not_erase_the_road(self):
        shape=(120,180);e=MotionRoadEstimator(shape,fps=25)
        flow=np.zeros((*shape,2),np.float32);flow[15:115,36:64,1]=2
        moving=np.zeros(shape,bool);moving[15:115,36:64]=True
        for i in range(100):e.update(flow,moving if i%2==0 else np.zeros(shape,bool))
        self.assertTrue(e.finish()[0].label_map(180,120)[60,50]>=0)

    def test_border_exit_is_extended_along_flow_without_filling_grass(self):
        shape=(120,180);e=MotionRoadEstimator(shape)
        frame=np.full((*shape,3),(40,130,40),np.uint8);frame[:,0:80]=150
        ys,xs=np.mgrid[:shape[0],:shape[1]]
        speed=(1+(ys-2*xs+130)/100).astype(np.float32)
        flow=np.zeros((*shape,2),np.float32)
        flow[20:86,:65,0]=-2*speed[20:86,:65];flow[20:86,:65,1]=speed[20:86,:65]
        moving=np.zeros(shape,bool);moving[20:86,:65]=True
        for _ in range(35):e.update(flow,moving,frame)
        parts,_=e.finish();labels=parts.label_map(180,120)
        self.assertGreaterEqual(labels[94,5],0)
        self.assertLess(labels[94,110],0)

    def test_border_extension_keeps_a_white_vehicle_on_brighter_near_asphalt(self):
        shape=(120,180);e=MotionRoadEstimator(shape)
        ys,xs=np.mgrid[:shape[0],:shape[1]]
        frame=np.full((*shape,3),(40,130,40),np.uint8)
        road=(110+.7*ys).astype(np.uint8)
        frame[:,:80]=road[:,:80,None]
        frame[90:104,:30]=220
        speed=(1+(ys-2*xs+130)/100).astype(np.float32)
        flow=np.zeros((*shape,2),np.float32)
        flow[20:86,:65,0]=-2*speed[20:86,:65];flow[20:86,:65,1]=speed[20:86,:65]
        moving=np.zeros(shape,bool);moving[20:86,:65]=True
        for _ in range(35):e.update(flow,moving,frame)
        labels=e.finish()[0].label_map(180,120)
        self.assertGreaterEqual(labels[96,5],0)
        self.assertLess(labels[96,110],0)

    def test_road_exclusion_survives_save_and_load(self):
        parts=RoadParts([{'name':'road','points':[[.1,.1],[.9,.1],[.9,.9],[.1,.9]],
                          'holes':[[[.4,.4],[.6,.4],[.6,.6],[.4,.6]]]}])
        with tempfile.TemporaryDirectory() as directory:
            path=str(Path(directory)/'road.json');parts.save(path);restored=RoadParts.load(path)
        self.assertLess(restored.label_map(100,100)[50,50],0)
        self.assertGreaterEqual(restored.label_map(100,100)[25,25],0)


if __name__=='__main__':unittest.main()
