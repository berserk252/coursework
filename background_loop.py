"""Последний эксперимент: повторять view4.mp4 и накапливать отдельный JPG-фон.

Использует сохранённые маски YOLO из завершённого прогона. Повторение не создаёт
новых наблюдений: доверие требует разных исходных кадров. Невидимый асфальт под
постоянно стоящей машиной остаётся неизвестным и отмечается отдельной маской.
"""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from auto_road import dimensions
from bg_model import BackgroundModel
from flow_filters import FlowCleaner


class ColorSamples:
    def __init__(self,shape):
        self.mean=np.zeros((*shape,3),np.float32)
        self.count=np.zeros(shape,np.uint16)
        self.unique=np.zeros(shape,np.uint16)
        self.seen={}

    def add(self,frame,mask,source_index,offset=0.):
        shape=mask.shape
        old=np.unpackbits(self.seen.get(source_index,np.zeros((mask.size+7)//8,np.uint8)),count=mask.size).reshape(shape)>0
        self.unique[mask & ~old]+=1
        self.seen[source_index]=np.packbits(old|mask)
        weight=1/np.minimum(self.count[mask].astype(np.float32)+1,600)
        values=frame[mask].astype(np.float32)-offset
        self.mean[mask]+=weight[:,None]*(values-self.mean[mask])
        self.count[mask]=np.minimum(self.count[mask].astype(np.uint32)+1,65000)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--video',required=True);ap.add_argument('--semantic_cache',required=True)
    ap.add_argument('--out_dir',required=True);ap.add_argument('--cycles',type=int,default=3)
    ap.add_argument('--memory_s',type=float,default=120)
    args=ap.parse_args()
    started=datetime.now(timezone.utc).isoformat()
    if Path(args.video).suffix.lower()!='.mp4':raise SystemExit('Этот эксперимент предназначен для MP4')
    if args.cycles<1:raise SystemExit('cycles должно быть >= 1')
    cv2.setNumThreads(1);out=Path(args.out_dir);out.mkdir(parents=True,exist_ok=True)
    cache=np.load(args.semantic_cache);shape=tuple(int(x) for x in cache['shape'])
    sample_frames=cache['frames'];packed=cache['masks']
    if len(sample_frames)==0:raise SystemExit('Нет проверенных масок транспорта')
    cap=cv2.VideoCapture(args.video);fps=cap.get(5);source_frames=int(cap.get(7));cap.release()
    road=np.ones(shape,bool)
    # При 12.5 FPS округление шага даёт 6.25 обновлений/с вместо заданных 5.
    # Передаём фактическую частоту, чтобы память 120 с не сокращалась до 96 с.
    sampling_hz=fps/max(1,round(fps/5))
    model=BackgroundModel(fps,road,delay_s=3,cap_s=args.memory_s,rate_hz=sampling_hz)
    colors=ColorSamples(shape);empty=np.zeros(shape,bool);records=[];previous_background=None;previous_valid=None
    for cycle in range(args.cycles):
        cap=cv2.VideoCapture(args.video);previous=None;cleaner=FlowCleaner(egomotion=True,road_mask=False);index=0
        while True:
            ok,frame=cap.read()
            if not ok:break
            frame=cv2.resize(frame,(shape[1],shape[0]),interpolation=cv2.INTER_AREA)
            gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            # Небольшое окно вокруг момента закрывает пропуски отдельных детекций.
            nearby=np.flatnonzero(np.abs(sample_frames-index)<=max(2,round(.4*fps)))
            protected=empty.copy()
            for j in nearby:protected|=np.unpackbits(packed[j],count=gray.size).reshape(shape)>0
            protected=cv2.dilate(protected.astype(np.uint8),np.ones((3,3),np.uint8))>0
            if previous is None:
                moving=empty
            else:
                flow=cv2.calcOpticalFlowFarneback(previous,gray,None,.5,3,15,3,5,1.2,0)
                moving=cleaner.update(flow,gray,previous).moving>0
            excluded=moving|protected
            model.update(gray,excluded,empty,frame=frame)
            if model.i%model.step==0:
                fg,known=model.foreground(gray,still_ref=~excluded)
                accepted=known & ~fg & ~excluded
                smooth=model.smooth(gray)
                offset=model._offset(smooth,accepted)
                colors.add(frame,accepted,index,offset)
            previous=gray;index+=1
        cap.release()
        valid=model.trusted() & (colors.unique>=10)
        image=np.clip(colors.mean,0,255).astype(np.uint8);image[~valid]=0
        cv2.imwrite(str(out/f'view4_background_cycle_{cycle+1}.jpg'),image,[cv2.IMWRITE_JPEG_QUALITY,95])
        cv2.imwrite(str(out/f'view4_background_valid_cycle_{cycle+1}.png'),valid.astype(np.uint8)*255)
        common=valid & previous_valid if previous_valid is not None else np.zeros(shape,bool)
        delta=float(np.abs(image.astype(float)-previous_background.astype(float))[common].mean()) if common.any() else None
        record={'cycle':cycle+1,'frames':index,'valid_fraction':float(valid.mean()),
                'unknown_fraction':float((~valid).mean()),'mean_change_from_previous_cycle':delta,
                'comparison_fraction':float(common.mean()),
                'max_unique_source_observations':int(colors.unique.max())}
        records.append(record);previous_background=image.copy();previous_valid=valid.copy();print(json.dumps(record),flush=True)
    cv2.imwrite(str(out/'view4_background.jpg'),image,[cv2.IMWRITE_JPEG_QUALITY,95])
    cv2.imwrite(str(out/'view4_background_valid.png'),valid.astype(np.uint8)*255)
    np.savez_compressed(out/'background_state.npz',gray=model.bg,variance=model.var,count=model.cnt,
                        rgb=colors.mean,valid=valid,unique_observations=colors.unique)
    result={'video':str(Path(args.video).resolve()),'source_frames':source_frames,'fps':fps,
            'started_utc':started,'completed_utc':datetime.now(timezone.utc).isoformat(),
            'cycles':args.cycles,'memory_s':args.memory_s,'unique_observations_required':10,
            'sampling_hz':fps/model.step,'effective_memory_s':model.cap*model.step/fps,
            'vehicle_mask_cache':str(Path(args.semantic_cache).resolve()),'records':records,
            'unknown_pixels':'black in JPG; excluded from background comparisons',
            'no_loop_boundary_optical_flow':True,'manual_annotations_used':False}
    (out/'experiment.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')


if __name__=='__main__':main()
