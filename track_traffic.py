"""Отдельный запуск YOLO+ByteTrack: движения, остановки и очередь; без ручной разметки."""
import argparse
import csv
import json
from pathlib import Path

import cv2

from auto_road import infer_road,dimensions,scene_plans
from flow_canvas import Text
from flow_filters import SceneGuard
from vehicle_tracker import VehicleTracker,TRACK_FIELDS,track_row


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    root=Path(__file__).resolve().parent.parent
    ap.add_argument('--video',required=True);ap.add_argument('--out_dir',required=True)
    ap.add_argument('--start',type=float,default=0);ap.add_argument('--seconds',type=float,default=0)
    ap.add_argument('--model',default=str(root/'Training Results/weights/best.pt'))
    ap.add_argument('--seg_model',default=str(root/'models/yolo11s-seg.pt'))
    ap.add_argument('--hz',type=float,default=4);ap.add_argument('--imgsz',type=int,default=1280)
    args=ap.parse_args();cv2.setNumThreads(1)
    out=Path(args.out_dir);out.mkdir(parents=True,exist_ok=True)
    # Фрагмент из середины затора использует начало своей сцены: внутри самой
    # очереди уже нет движения, по которому можно восстановить проезжую часть.
    plans=scene_plans(args.video,max_width=640,limit_s=args.start) if args.start>0 else None
    if plans:
        probe=cv2.VideoCapture(args.video);source_fps=probe.get(5);probe.release()
        _,parts,calibration=max((p for p in plans if p[0]<=round(args.start*source_fps)),key=lambda p:p[0])
    else:
        parts,calibration=infer_road(args.video,max_width=640)
    parts.save(str(out/'auto_road.json'));(out/'calibration.json').write_text(json.dumps(calibration,ensure_ascii=False,indent=2)+'\n')
    tracker=VehicleTracker(args.model or None,args.seg_model or None,hz=args.hz,imgsz=args.imgsz,config_dir=out/'yolo_config')
    (out/'models.json').write_text(json.dumps(tracker.metadata(),ensure_ascii=False,indent=2)+'\n')
    cap=cv2.VideoCapture(args.video);fps=cap.get(5);size=dimensions(cap.get(3),cap.get(4),640)
    start=round(args.start*fps);cap.set(cv2.CAP_PROP_POS_FRAMES,start)
    writer=cv2.VideoWriter(str(out/'tracking.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),fps,size)
    if not writer.isOpened():raise RuntimeError('Не открылся VideoWriter')
    labels=parts.label_map(*size);text=Text();guard=SceneGuard(fps)
    index=0;seen=set();stopped=set();max_queue=0;scene=0
    stream=(out/'tracks.csv').open('w',newline='');table=csv.writer(stream);table.writerow(TRACK_FIELDS)
    snapshots={round(x*fps) for x in (2.,5.,10.,15.,19.)}
    while True:
        ok,original=cap.read()
        if not ok or (args.seconds and index/fps>=args.seconds):break
        frame=cv2.resize(original,size,interpolation=cv2.INTER_AREA);t=(start+index)/fps
        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY);cut,valid=guard.update(gray)
        if cut:
            scene+=1;parts,calibration=infer_road(args.video,t0=t,max_width=640);labels=parts.label_map(*size)
        vehicles,observed=tracker.update(frame,t,labels,cut=cut,inference_frame=original) if valid else ([],False)
        if observed:
            for v in vehicles:
                table.writerow(track_row(v,start+index,t,scene,scale=size[0]/original.shape[1]));seen.add((scene,v.id))
                if v.motion.state=='stationary':stopped.add((scene,v.id))
        queued=sum(sum(v.part==p and v.motion.state=='stationary' for v in vehicles)>=3 for p in range(len(parts)))
        max_queue=max(max_queue,queued)
        tracker.draw(frame,vehicles,text,t);parts.draw(frame,text,size=10)
        text.draw(frame,f'Транспорт: {len(vehicles)}; стоят: {sum(v.motion.state=="stationary" for v in vehicles)}; зон очереди: {queued}',4,4,size=12,color=(255,255,255))
        writer.write(frame)
        if index in snapshots:cv2.imwrite(str(out/f'frame_{t:.1f}s.png'),frame)
        index+=1
    cap.release();writer.release();stream.close()
    result={'frames':index,'fps':fps,'start_s':args.start,'unique_track_ids':len(seen),'stationary_ids':len(stopped),
            'max_queue_zones':max_queue,'manual_annotations_used':False,'speed_units':'source image px/s'}
    (out/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':main()
