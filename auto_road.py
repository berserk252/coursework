"""Автоматические зоны движения: первые 10 с каждой сцены, без ручных полигонов.

Это калибровка записанного видео с просмотром окна вперёд, не потоковый детектор.
Зона объединяет наблюдаемый поток одного направления; число физических полос
из одного оптического потока не предполагается известным.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from flow_canvas import Text
from flow_filters import FlowCleaner, SceneGuard
from road_parts import RoadParts


class MotionRoadEstimator:
    def __init__(self, shape, cell=8):
        self.shape, self.cell = shape, cell
        self.votes = np.zeros(shape, np.float32)
        self.direction = np.zeros((*shape, 2), np.float32)
        self.magnitude = np.zeros(shape, np.float32)
        self.frames = 0

    def update(self, flow, moving):
        if moving.mean() > .65:
            return                         # целый движущийся кадр не является доказательством дороги
        magnitude = np.linalg.norm(flow, axis=2)
        active = moving & (magnitude > .2 * self.shape[1] / 640)
        self.votes += active
        self.magnitude += magnitude * active
        self.direction += flow / np.maximum(magnitude[..., None], 1e-6) * active[..., None]
        self.frames += 1

    def finish(self, video="", max_parts=6):
        h, w = self.shape
        if self.frames < 4:
            return RoadParts([], video, [w,h]), {"reason": "недостаточно кадров движения"}
        coherence = np.linalg.norm(self.direction, axis=2) / np.maximum(self.votes, 1)
        support = (self.votes >= max(4, self.frames * .07)) & (coherence >= .60)
        vectors = self.direction[support]
        if len(vectors) < 40:
            return RoadParts([], video, [w,h]), {"reason": "нет надёжных наблюдений потока"}
        unit = vectors / np.maximum(np.linalg.norm(vectors,axis=1,keepdims=True),1e-6)
        full_unit = self.direction / np.maximum(np.linalg.norm(self.direction,axis=2,keepdims=True),1e-6)
        angles = np.arctan2(unit[:,1],unit[:,0])
        hist,bins = np.histogram(angles,bins=24,range=(-np.pi,np.pi))
        density = hist+np.roll(hist,1)+np.roll(hist,-1)
        first = int(density.argmax());centers=[(bins[first]+bins[first+1])/2]
        alternatives=[i for i in range(24) if abs(np.angle(np.exp(1j*((bins[i]+bins[i+1])/2-centers[0]))))>1.3]
        if alternatives:
            second=max(alternatives,key=lambda i:density[i])
            if density[second]>.2*density[first]:centers.append((bins[second]+bins[second+1])/2)
        directions=np.array([[np.cos(a),np.sin(a)] for a in centers])
        for _ in range(12):
            which=(unit@directions.T).argmax(axis=1)
            for j in range(len(directions)):
                v=unit[which==j].sum(axis=0)
                if np.linalg.norm(v)>0:directions[j]=v/np.linalg.norm(v)
        assignment=(full_unit@directions.T).argmax(axis=2)
        groups=range(len(directions))
        candidates = []
        k = max(3, round(w/640*13) | 1)
        close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(k,k))
        expand = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(max(3,k//2|1),)*2)
        for j in groups:
            m = (support & (assignment==j)).astype(np.uint8)
            m = cv2.morphologyEx(m,cv2.MORPH_CLOSE,close)
            m = cv2.dilate(m,expand)
            count, lab, stats, _ = cv2.connectedComponentsWithStats(m,8)
            biggest = stats[1:,cv2.CC_STAT_AREA].max(initial=0)
            for component in range(1,count):
                mask = lab==component
                if stats[component,cv2.CC_STAT_AREA] < max(.005*h*w,.06*biggest) or (mask & support).sum() < 40:
                    continue
                contours, _ = cv2.findContours(mask.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
                contour = max(contours,key=cv2.contourArea)
                if 2*mask.sum()/max(cv2.arcLength(contour,True),1) < 12*w/640:
                    continue
                vote = self.direction[mask & support].sum(axis=0)
                direction = vote / max(float(np.linalg.norm(vote)),1e-6)
                candidates.append((int((mask & support).sum()),mask,direction))
        bridge=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(max(3,round(w*.05)|1),)*2)
        changed=True
        while changed:
            changed=False
            for i in range(len(candidates)):
                for j in range(i+1,len(candidates)):
                    a,b=candidates[i],candidates[j]
                    if a[2]@b[2] < .6 or not (cv2.dilate(a[1].astype(np.uint8),bridge).astype(bool)&b[1]).any():continue
                    mask=cv2.morphologyEx((a[1]|b[1]).astype(np.uint8),cv2.MORPH_CLOSE,bridge)>0
                    v=self.direction[mask&support].sum(axis=0);v/=max(float(np.linalg.norm(v)),1e-6)
                    candidates[i]=(int((mask&support).sum()),mask,v)
                    candidates.pop(j);changed=True;break
                if changed:break
        candidates = sorted(candidates,key=lambda x:-x[0])[:max_parts]
        candidates.sort(key=lambda x:float(np.nonzero(x[1])[1].mean()))
        parts, directions = [], []
        for i, (_,mask,direction) in enumerate(candidates):
            found,_=cv2.findContours(mask.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            points=cv2.approxPolyDP(max(found,key=cv2.contourArea),max(1.,w/640*2),True).reshape(-1,2)
            parts.append({"name":f"Автозона {i+1}","points":(points/[w,h]).tolist()})
            directions.append({"part":i+1,"unit_vector":direction.tolist(),
                               "angle_image_deg":float(np.degrees(np.arctan2(direction[1],direction[0])))})
        return RoadParts(parts,video,[w,h]), {"frames":self.frames,"support_pixels":int(support.sum()),
                                             "directions":directions,"method":"coherent compensated optical flow"}


def dimensions(w,h,max_width):
    scale=min(1.,max_width/w) if max_width>0 else 1.
    nw,nh=round(w*scale),round(h*scale)
    return nw-nw%2,nh-nh%2


def infer_road(video,t0=0.,seconds=10.,max_width=640):
    cap=cv2.VideoCapture(str(video));fps=cap.get(5) or 25
    size=dimensions(cap.get(3),cap.get(4),max_width)
    cap.set(cv2.CAP_PROP_POS_FRAMES,round(t0*fps));ok,frame=cap.read()
    if not ok:
        cap.release();return RoadParts([],Path(video).name,list(size)),{"reason":"нет кадра"}
    prep=lambda f:cv2.resize(f,size,interpolation=cv2.INTER_AREA)
    previous=cv2.cvtColor(prep(frame),cv2.COLOR_BGR2GRAY)
    estimator=MotionRoadEstimator(previous.shape)
    cleaner=FlowCleaner(egomotion=True,road_mask=False)
    guard=SceneGuard(fps);guard.update(previous)
    for _ in range(round(seconds*fps)):
        ok,frame=cap.read()
        if not ok:break
        gray=cv2.cvtColor(prep(frame),cv2.COLOR_BGR2GRAY)
        cut,valid=guard.update(gray)
        if cut:break
        flow=cv2.calcOpticalFlowFarneback(previous,gray,None,.5,3,15,3,5,1.2,0)
        c=cleaner.update(flow,gray,previous)
        if valid:estimator.update(c.field(),c.moving)
        previous=gray
    cap.release()
    parts,metadata=estimator.finish(Path(video).name)
    metadata.update(start_s=t0,learning_s=seconds,offline_lookahead=True)
    return parts,metadata


def scene_plans(video,seconds=10.,max_width=640,limit_s=0.):
    cap=cv2.VideoCapture(str(video));fps=cap.get(5) or 25
    size=dimensions(cap.get(3),cap.get(4),max_width)
    guard=SceneGuard(fps);starts=[0];index=0
    while True:
        ok,frame=cap.read()
        if not ok or (limit_s and index/fps>limit_s):break
        gray=cv2.cvtColor(cv2.resize(frame,size,interpolation=cv2.INTER_AREA),cv2.COLOR_BGR2GRAY)
        cut,_=guard.update(gray)
        if cut:starts.append(index)
        index+=1
    cap.release()
    result=[]
    for i,start in enumerate(starts):
        duration=min(seconds,(starts[i+1]-start)/fps) if i+1<len(starts) else seconds
        parts,meta=infer_road(video,start/fps,duration,max_width)
        result.append((start,parts,meta))
    return result


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--video',required=True);ap.add_argument('--out_dir',required=True)
    ap.add_argument('--seconds',type=float,default=10);ap.add_argument('--max_width',type=int,default=640)
    args=ap.parse_args();cv2.setNumThreads(1)
    out=Path(args.out_dir);out.mkdir(parents=True,exist_ok=True)
    parts,meta=infer_road(args.video,seconds=args.seconds,max_width=args.max_width)
    parts.save(str(out/(Path(args.video).stem+'.json')))
    (out/'calibration.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2)+'\n')
    cap=cv2.VideoCapture(args.video);ok,frame=cap.read();cap.release()
    if ok:
        frame=cv2.resize(frame,dimensions(frame.shape[1],frame.shape[0],args.max_width))
        parts.draw(frame,Text(),fill_alpha=.15)
        cv2.imwrite(str(out/'auto_road_preview.png'),frame)
    print(json.dumps(meta,ensure_ascii=False))


if __name__=='__main__':main()
