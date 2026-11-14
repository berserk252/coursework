"""YOLO + ByteTrack: ID, траектория, направление и длительность остановки.

Обученная best.pt используется вместе с COCO-сегментацией: в пользовательской
модели нет класса truck. Маска кузова берётся из сегментации, не из рамки.
Скорость — px/s и доля диагонали объекта/s; физическая скорость не предполагается.
"""
import hashlib
import os
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

LABELS = ('car','bus','truck','motorcycle')


@dataclass
class MotionState:
    history: deque = field(default_factory=lambda: deque(maxlen=120))
    stopped_since: float | None = None
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(2))
    speed: float = 0.
    stop_s: float = 0.
    state: str = 'warming'

    def update(self, box, t):
        point=np.array([(box[0]+box[2])/2,(box[1]+box[3])/2],float)
        if self.history and t-self.history[-1][0]>1.0:
            self.history.clear();self.stopped_since=None
        self.history.append((t,point))
        while len(self.history)>2 and t-self.history[0][0]>1.5:self.history.popleft()
        self.stop_s=0.
        if len(self.history)<3 or t-self.history[0][0]<.6:
            self.state='warming';return self
        times=np.array([r[0] for r in self.history]);points=np.array([r[1] for r in self.history])
        times-=times.mean()
        self.velocity=(times[:,None]*(points-points.mean(axis=0))).sum(axis=0)/max(float((times**2).sum()),1e-6)
        self.speed=float(np.linalg.norm(self.velocity))
        diagonal=float(np.hypot(box[2]-box[0],box[3]-box[1]))
        threshold=max(1.5,.025*diagonal)
        if self.speed <= threshold*(1.8 if self.state=='stationary' else 1):
            if self.stopped_since is None:self.stopped_since=t
            self.stop_s=t-self.stopped_since
            self.state='stationary' if self.stop_s>=2.0 else 'slow'
        else:
            self.stopped_since=None;self.state='moving'
        return self


@dataclass
class Vehicle:
    id: int
    box: np.ndarray
    label: str
    confidence: float
    mask: np.ndarray | None
    motion: MotionState
    time: float
    source: str
    part: int = -1


def overlap(a,b):
    lo=np.maximum(a[:2],b[:2]);hi=np.minimum(a[2:],b[2:]);area=float(np.maximum(hi-lo,0).prod())
    return area/max(float((a[2:]-a[:2]).prod()+(b[2:]-b[:2]).prod()-area),1e-6)


class VehicleTracker:
    def __init__(self, custom_model, seg_model=None, hz=4., imgsz=1280, conf=.15, config_dir=None):
        if config_dir:
            path=Path(config_dir);path.mkdir(parents=True,exist_ok=True)
            os.environ['YOLO_CONFIG_DIR']=str(path)
        os.environ.setdefault('YOLO_AUTOINSTALL','false')
        os.environ.setdefault('MPLCONFIGDIR','/tmp/traffic-matplotlib')
        import torch
        from ultralytics import YOLO
        from ultralytics.trackers.byte_tracker import BYTETracker
        torch.set_num_threads(2)
        self._torch = torch
        self.models=[]
        for kind,path in (('seg',seg_model),('custom',custom_model)):
            if path:
                path=Path(path)
                if not path.is_file():raise FileNotFoundError(path)
                self.models.append((kind,YOLO(str(path)),path))
        if not self.models:raise ValueError('Нужна хотя бы одна модель YOLO')
        self.hz,self.imgsz,self.conf=float(hz),imgsz,conf
        args=SimpleNamespace(track_high_thresh=.22,track_low_thresh=.10,new_track_thresh=.22,
                             track_buffer=max(2,round(2*hz)),match_thresh=.8,fuse_score=True)
        self.tracker=BYTETracker(args)
        self.reset()

    def reset(self):
        self.tracker.reset()
        self.states={};self.current=[];self.last_inference=-np.inf;self.next_inference=-np.inf;self.rows=[]

    def metadata(self):
        return {'tracker':'Ultralytics ByteTrack','inference_hz':self.hz,'imgsz':self.imgsz,
                'speed_units':'image px/s; not km/h','models':[
                    {'kind':kind,'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                     'names':model.names} for kind,model,path in self.models]}

    def detect(self,frame,target_shape=None):
        detections=[]
        target_shape=target_shape or frame.shape[:2]
        sy,sx=target_shape[0]/frame.shape[0],target_shape[1]/frame.shape[1]
        for kind,model,_ in self.models:
            ids=[i for i,name in model.names.items() if name in (*LABELS,'motorbike')]
            size=self.imgsz if kind=='seg' and frame.shape[1]>=900 else 640
            result=model.predict(frame,imgsz=size,conf=.12 if kind=='custom' else self.conf,
                                 classes=ids,device='cpu',retina_masks=True,agnostic_nms=True,verbose=False)[0]
            # Инициализация предиктора Ultralytics переустанавливает число потоков.
            # Возвращаем лимит, чтобы параллельные видео не вытесняли друг друга.
            self._torch.set_num_threads(2)
            for i,box in enumerate(result.boxes):
                xyxy=box.xyxy[0].cpu().numpy().astype(np.float32)
                if float((xyxy[2:]-xyxy[:2]).prod()) > .20*frame.shape[0]*frame.shape[1]:continue
                xyxy*=np.array([sx,sy,sx,sy],np.float32)
                label=model.names[int(box.cls.item())].replace('motorbike','motorcycle')
                if kind=='custom' and any(overlap(xyxy,d['box'])>.4 for d in detections):continue
                mask=None
                if result.masks is not None:
                    mask=result.masks.data[i].cpu().numpy()>0.5
                    if mask.shape!=target_shape:mask=cv2.resize(mask.astype(np.uint8),(target_shape[1],target_shape[0]),interpolation=cv2.INTER_NEAREST)>0
                detections.append({'box':xyxy,'label':label,'conf':float(box.conf.item()),'mask':mask,'source':kind})
        return detections

    def update(self,frame,t,labels=None,cut=False,inference_frame=None):
        if cut:self.reset()
        if t+1e-6 < self.next_inference:return self.visible(t),False
        self.last_inference=t
        self.next_inference=(np.floor(t*self.hz)+1)/self.hz
        from ultralytics.engine.results import Boxes
        detections=self.detect(frame if inference_frame is None else inference_frame,frame.shape[:2])
        data=np.array([[*d['box'],d['conf'],LABELS.index(d['label'])] for d in detections],np.float32).reshape(-1,6)
        tracked=self.tracker.update(Boxes(data,frame.shape[:2]),frame)
        self.current=[]
        for row in tracked:
            ident,idx=int(row[4]),int(row[-1]);d=detections[idx]
            state=self.states.setdefault(ident,MotionState()).update(row[:4],t)
            vehicle=Vehicle(ident,row[:4].copy(),d['label'],float(row[5]),d['mask'],state,t,d['source'])
            if labels is not None:vehicle.part=self.assign_part(vehicle,labels)
            self.current.append(vehicle)
        return self.current,True

    def visible(self,t):
        # Не переносим потерянный контур на освободившийся асфальт.
        return [v for v in self.current if t-v.time <= min(.4,1.6/self.hz)]

    @staticmethod
    def assign_part(vehicle,labels):
        if vehicle.mask is not None:
            values=labels[vehicle.mask];values=values[values>=0]
            if len(values):return int(np.bincount(values).argmax())
        x=int(np.clip((vehicle.box[0]+vehicle.box[2])/2,0,labels.shape[1]-1))
        y=int(np.clip(vehicle.box[3]-1,0,labels.shape[0]-1))
        part=int(labels[y,x])
        if part>=0:return part
        best=(float('inf'),-1)
        for p in range(int(labels.max())+1):
            distance=cv2.distanceTransform((labels!=p).astype(np.uint8),cv2.DIST_L2,3)
            if distance[y,x]<best[0]:best=(float(distance[y,x]),p)
        return best[1] if best[0] <= max(12.,.4*(vehicle.box[2]-vehicle.box[0])) else -1

    @staticmethod
    def mask(vehicles,shape,stationary_only=False):
        result=np.zeros(shape,bool)
        for v in vehicles:
            if v.mask is not None and (not stationary_only or v.motion.state=='stationary'):
                result|=v.mask
        return result

    @staticmethod
    def draw(image,vehicles,text,t):
        for v in vehicles:
            color=(200,160,0) if v.motion.state=='stationary' else (40,180,40)
            if v.mask is not None:
                # Один объект используется на нескольких междетекторных кадрах.
                # Контур считаем один раз, смешиваем цвет только в его области.
                cached=getattr(v,'_drawing_cache',None)
                if cached is None or cached[0]!=image.shape[:2]:
                    mask=v.mask.astype(np.uint8)
                    if mask.shape!=image.shape[:2]:
                        mask=cv2.resize(mask,(image.shape[1],image.shape[0]),interpolation=cv2.INTER_NEAREST)
                    x,y,w,h=cv2.boundingRect(mask)
                    rings,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
                    cached=(image.shape[:2],(x,y,w,h),mask[y:y+h,x:x+w]>0,rings)
                    v._drawing_cache=cached
                _,(x,y,w,h),mask,rings=cached
                if w and h:
                    region=image[y:y+h,x:x+w]
                    over=region.copy();over[mask]=color
                    cv2.addWeighted(over,.25,region,.75,0,dst=region)
                cv2.drawContours(image,rings,-1,color,1,cv2.LINE_AA)
            x1,y1,x2,y2=np.round(v.box).astype(int)
            cv2.rectangle(image,(x1,y1),(x2,y2),color,1)
            detail=f' стоит {v.motion.stop_s:.0f}с' if v.motion.state=='stationary' else ''
            label=f'#{v.id}'+(f' {v.label}{detail}' if (x2-x1)*(y2-y1)>1300 else detail)
            text.draw(image,label,max(0,x1),max(0,y1-11),size=9,color=color)
            points=np.array([p for _,p in v.motion.history],np.int32)
            if len(points)>1:cv2.polylines(image,[points],False,color,1,cv2.LINE_AA)


TRACK_FIELDS=['frame','time_s','scene_id','track_id','class','confidence','source','x1','y1','x2','y2',
              'speed_px_s','direction_image_deg','stationary_s','motion_state','part','mask_area_px']


def track_row(vehicle,frame,t,scene_id=0,scale=1.):
    direction=float(np.degrees(np.arctan2(vehicle.motion.velocity[1],vehicle.motion.velocity[0]))) if vehicle.motion.speed>1e-6 else float('nan')
    return [frame,f'{t:.3f}',scene_id,vehicle.id,vehicle.label,f'{vehicle.confidence:.4f}',vehicle.source,
            *[f'{x/scale:.2f}' for x in vehicle.box],f'{vehicle.motion.speed/scale:.3f}',f'{direction:.2f}',
            f'{vehicle.motion.stop_s:.3f}',vehicle.motion.state,vehicle.part+1,
            int(vehicle.mask.sum()) if vehicle.mask is not None else 0]
