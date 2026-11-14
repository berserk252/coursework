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
from foreground_objects import ForegroundObjects, FlowRegionConfirmation, clip_holes, combine_objects, link_flow_regions, stable_output_regions
from flow_polygons import draw_holes
from flow_filters import CleanFlow
from flow_stats import SpeedMeter


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

    def test_old_pixel_age_does_not_publish_a_fresh_region(self):
        detector=ForegroundObjects(self.shape,10,confirm_s=.5)
        detector.age[self.region]=90
        detector.probability[self.region]=1
        detector.active[self.region]=True
        background=np.full(self.shape,180,np.float32)
        frame=background.astype(np.uint8);frame[self.region]=50
        clear=np.zeros(self.shape,bool)
        objects=detector.update(frame,background,self.region,clear,self.labels,0,12,8,12)
        self.assertTrue(objects)
        self.assertEqual(stable_output_regions(objects,.5),[])
        for i in range(1,20):
            objects=detector.update(frame,background,self.region,clear,self.labels,i/10,12,8,12)
        self.assertTrue(stable_output_regions(objects,.5))

    def test_old_weak_road_lines_do_not_become_a_stationary_object_after_contrast_change(self):
        stripe = (np.indices(self.shape)[1] // 18 % 2).astype(np.uint8)
        background = cv2.GaussianBlur(100 + 30 * stripe, (5, 5), 0).astype(np.float32)
        current = 100 + 70 * stripe
        old_strong = cv2.Canny(background.astype(np.uint8), 50, 120)
        new_strong = cv2.Canny(cv2.GaussianBlur(current, (5, 5), 0), 50, 120) > 0
        previous_novel = new_strong & (cv2.dilate(old_strong, np.ones((5, 5), np.uint8)) == 0)
        self.assertGreater(int((previous_novel & self.region).sum()), 30)
        hole = replace(self.hole, mask=self.region, incident=False, kind='stationary', observed_s=10)
        self.assertEqual(stable_output_regions([hole], gray=current, background=background), [])

    def test_new_vehicle_details_preserve_a_stationary_candidate(self):
        background = np.full(self.shape, 170, np.float32)
        current = background.astype(np.uint8)
        current[self.region] = 225
        current[30:50, 45:95] = 45
        hole = replace(self.hole, mask=self.region, incident=False, kind='stationary', observed_s=10)
        self.assertEqual(len(stable_output_regions([hole], gray=current, background=background)), 1)

    def test_additional_stationary_check_keeps_an_existing_flow_gap(self):
        background = np.full(self.shape, 170, np.float32)
        hole = replace(self.hole, mask=self.region, incident=False, kind='flow_gap', observed_s=10)
        result = stable_output_regions([hole], gray=background.astype(np.uint8), background=background)
        self.assertEqual(len(result), 1)
        self.assertIs(result[0], hole)

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

    def test_local_light_change_on_fine_static_texture_is_not_an_object(self):
        y, x = np.indices(self.shape)
        asphalt = (110 + np.where((x // 2 + y // 2) % 2, 35, -35)).astype(np.uint8)
        clear = np.zeros(self.shape, bool)
        model = BackgroundModel(10, ~clear, delay_s=1)
        for _ in range(60):
            model.update(asphalt, clear, clear)
        current = asphalt.copy()
        current[self.region] += 20
        foreground, known = model.foreground(current)
        self.assertGreater(foreground[self.region].mean(), .9)
        detector = ForegroundObjects(self.shape, 10)
        for i in range(40):
            found = detector.update(current, model.bg, foreground, clear, self.labels,
                                    i / 10, 12, 8, 12, known=known)
        # Fine background texture must not count as new vehicle details merely
        # because the stored reference was smoothed and the current image was not.
        self.assertEqual(found, [])

    def test_old_flow_hole_does_not_make_a_fresh_fragment_red(self):
        parent = replace(self.hole, idle_s=90, incident=True, core=np.ones((8,12), bool))
        obj = replace(self.hole, id=1000001, mask=self.region.copy(), incident=False,
                      kind='stationary', idle_s=2, bg_valid=True, bg_diff=.8, area_px=int(self.region.sum()))
        result = link_flow_regions([parent], [obj], self.labels, 12)
        self.assertEqual(len(result), 1)
        self.assertFalse(result[0].incident)
        self.assertEqual(result[0].idle_s, 2)
        result = link_flow_regions([replace(parent, incident=False)], [replace(obj, idle_s=25)], self.labels, 12)
        self.assertTrue(result[0].incident)

    def test_queue_merge_never_expands_a_red_mask(self):
        red_mask = np.zeros(self.shape, bool);red_mask[30:60,40:60] = True
        red = replace(self.hole, mask=red_mask.copy(), incident=True, area_px=int(red_mask.sum()))
        queue = replace(self.hole, id=99, mask=self.region.copy(), incident=False, kind='stationary')
        result = combine_objects([red],[queue])
        np.testing.assert_array_equal(result[0].mask,red_mask)
        self.assertTrue(all(not h.incident for h in result[1:]))

    def test_new_pixels_in_connected_body_do_not_inherit_red_age(self):
        parent=replace(self.hole,core=np.ones((8,12),bool),idle_s=90,incident=True)
        obj=replace(self.hole,id=1000000,mask=self.region,area_px=int(self.region.sum()),
                    bg_valid=True,bg_diff=.9,idle_s=25,incident=False)
        ages=np.where(self.region,25.,0.);ages[:,80:]=2.
        result=link_flow_regions([parent],[obj],self.labels,12,pixel_age=ages,
                                 foreground=self.region,known=np.ones(self.shape,bool))
        red=np.logical_or.reduce([h.mask for h in result if h.incident])
        self.assertTrue(red[40,45]);self.assertFalse(red[:,80:].any())
        self.assertTrue(any(not h.incident and h.mask[40,100] for h in result))

    def test_no_observed_road_does_not_enable_full_frame_speed(self):
        flow=np.ones((*self.shape,2),np.float32);magnitude=np.ones(self.shape,np.float32)
        current=CleanFlow(flow,magnitude,magnitude,np.ones(self.shape,bool),np.zeros(self.shape,np.float32))
        meter=SpeedMeter(10,strict_road=True);meter.update(current,0.)
        self.assertTrue(np.isnan(meter.v));self.assertEqual(meter.area,0.)

    def test_region_confirmation_preserves_body_and_caps_new_queue_growth(self):
        rule=FlowRegionConfirmation(20)
        parent=replace(self.hole,core=np.ones((8,12),bool),idle_s=90,incident=True)
        obj=replace(self.hole,id=1000000,mask=self.region.copy(),area_px=int(self.region.sum()),
                    age_s=2,idle_s=30,bg_valid=True,bg_diff=.9,incident=False)
        known=np.ones(self.shape,bool)
        fresh=rule.update([parent],[obj],self.labels,12,30,self.region,known)
        self.assertFalse(any(h.incident for h in fresh))
        confirmed=rule.update([parent],[replace(obj,age_s=21)],self.labels,12,31,self.region,known)
        red=next(h for h in confirmed if h.incident)
        np.testing.assert_array_equal(red.mask,self.region)
        grown=self.region.copy();grown[20:76,116:136]=True
        # Та же область присоединила новую часть очереди; её собственный
        # таймер сбросился, но уже подтверждённая исходная машина остаётся.
        changed=replace(obj,mask=grown,area_px=int(grown.sum()),age_s=0)
        output=rule.update([parent],[changed],self.labels,12,31.1,grown,known)
        red=np.logical_or.reduce([h.mask for h in output if h.incident])
        self.assertFalse(red[:,121:].any())
        self.assertTrue(any(not h.incident and h.mask[45,130] for h in output))
        published=stable_output_regions(output)
        self.assertTrue(any(h.incident for h in published))
        self.assertFalse(any(not h.incident for h in published))

    def test_region_confirmation_rechecks_empty_background_and_reset(self):
        rule=FlowRegionConfirmation(20);known=np.ones(self.shape,bool);clear=np.zeros(self.shape,bool)
        parent=replace(self.hole,core=np.ones((8,12),bool),idle_s=90,incident=True)
        obj=replace(self.hole,id=1000000,mask=self.region.copy(),area_px=int(self.region.sum()),
                    age_s=25,bg_valid=True,bg_diff=.9,incident=False)
        output=rule.update([parent],[obj],self.labels,12,30,clear,known)
        self.assertFalse(any(h.incident for h in output));self.assertFalse(rule.anchors)
        rule.update([parent],[obj],self.labels,12,31,self.region,known)
        self.assertTrue(rule.anchors)
        rule.reset();self.assertFalse(rule.anchors)

    def test_background_repair_removes_empty_pixels_without_resetting_the_remaining_car(self):
        detector=ForegroundObjects(self.shape,10,confirm_s=.5)
        background=np.full(self.shape,180,np.float32);frame=background.astype(np.uint8)
        frame[self.region]=50;clear=np.zeros(self.shape,bool);known=np.ones(self.shape,bool)
        for i in range(25):objects=detector.update(frame,background,self.region,clear,self.labels,i/10,12,8,12)
        self.assertTrue(objects);before=objects[0]
        rule=FlowRegionConfirmation(1)
        parent=replace(self.hole,core=np.ones((8,12),bool))
        self.assertTrue(any(h.incident for h in rule.update([parent],objects,self.labels,12,2.4,self.region,known)))
        cleared=self.region.copy();cleared[:,55:]=False
        detector.forget_background(cleared);rule.forget_background(cleared)
        foreground=self.region&~cleared;frame[cleared]=180
        objects=detector.update(frame,background,foreground,clear,self.labels,2.5,12,8,12)
        self.assertTrue(objects);self.assertEqual(objects[0].id,before.id)
        self.assertGreaterEqual(objects[0].age_s,before.age_s)
        result=rule.update([parent],objects,self.labels,12,2.5,foreground,known)
        self.assertTrue(any(h.incident for h in result))
        self.assertFalse(any((h.mask&cleared).any() for h in result))

    def test_undetected_slow_translation_does_not_accumulate_stationary_track_age(self):
        shape=(100,400);labels=np.zeros(shape,np.int8);clear=np.zeros(shape,bool)
        background=np.full(shape,180,np.float32);detector=ForegroundObjects(shape,10,confirm_s=.5)
        for i in range(230):
            frame=background.astype(np.uint8);fg=clear.copy();fg[30:65,20+i:70+i]=True;frame[fg]=50
            objects=detector.update(frame,background,fg,clear,labels,i/10,8,12,50)
            self.assertTrue(all(h.age_s<3 for h in objects))

    def test_single_frame_change_and_moving_body_are_not_stationary_objects(self):
        model = ForegroundObjects(self.shape, 10, confirm_s=.5)
        background = np.full(self.shape,180,np.float32)
        frame = background.astype(np.uint8);frame[self.region]=50
        clear = np.zeros(self.shape,bool)
        self.assertEqual(model.update(frame,background,self.region,clear,self.labels,0,12,8,12),[])
        for i in range(1,30):
            found=model.update(background.astype(np.uint8),background,clear,clear,self.labels,i/10,12,8,12)
        self.assertFalse(found)
        for i in range(30,70):
            found=model.update(frame,background,self.region,self.region,self.labels,i/10,12,8,12)
        self.assertFalse(found)

    def test_real_background_evidence_is_not_forced_to_one(self):
        model=ForegroundObjects(self.shape,10,confirm_s=.5)
        background=np.full(self.shape,180,np.float32);frame=background.astype(np.uint8)
        frame[self.region]=50
        fg=self.region.copy();fg[40:43,50:54]=False
        clear=np.zeros(self.shape,bool)
        found=[]
        for i in range(40):found=model.update(frame,background,fg,clear,self.labels,i/10,12,8,12)
        self.assertTrue(found)
        self.assertTrue(any(h.bg_diff < .9999 for h in found))

    def test_repeated_frames_do_not_turn_moving_body_into_stationary_object(self):
        model=ForegroundObjects(self.shape,10,confirm_s=.5)
        background=np.full(self.shape,180,np.float32);frame=background.astype(np.uint8)
        frame[self.region]=50;clear=np.zeros(self.shape,bool)
        for i in range(50):
            found=model.update(frame,background,self.region,self.region if i%2==0 else clear,
                               self.labels,i/10,12,8,12)
            self.assertFalse(found)

    def test_stopped_body_touching_moving_queue_is_not_discarded(self):
        model=ForegroundObjects(self.shape,10,confirm_s=.5)
        background=np.full(self.shape,180,np.float32);frame=background.astype(np.uint8)
        frame[self.region]=50
        moving=np.zeros(self.shape,bool);moving[20:76,80:116]=True
        found=[]
        for i in range(40):
            found=model.update(frame,background,self.region,moving,self.labels,i/10,12,8,12)
        self.assertTrue(found)
        mask=np.logical_or.reduce([h.mask for h in found])
        self.assertTrue(mask[45,45]);self.assertFalse(mask[45,105])

    def test_short_occlusion_preserves_object_but_resumed_driving_releases_it(self):
        model=ForegroundObjects(self.shape,10,confirm_s=.5)
        background=np.full(self.shape,180,np.float32);frame=background.astype(np.uint8)
        frame[self.region]=50;clear=np.zeros(self.shape,bool)
        for i in range(20):found=model.update(frame,background,self.region,clear,self.labels,i/10,12,8,12)
        self.assertTrue(found)
        for i in range(20,26):
            found=model.update(frame,background,self.region,self.region,self.labels,i/10,12,8,12)
            self.assertTrue(found)
        for i in range(26,55):found=model.update(frame,background,self.region,self.region,self.labels,i/10,12,8,12)
        self.assertFalse(found)


if __name__ == "__main__":
    unittest.main()
