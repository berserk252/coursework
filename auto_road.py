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
    def __init__(self, shape, cell=8, fps=25):
        self.shape, self.cell = shape, cell
        self.votes = np.zeros(shape, np.float32)
        self.direction = np.zeros((*shape, 2), np.float32)
        self.magnitude = np.zeros(shape, np.float32)
        self.frames = 0
        self.fps = fps
        self.streak = np.zeros(shape,np.uint16)
        self.motion_gap = np.zeros(shape,np.uint16)
        self.samples = []

    def update(self, flow, moving, frame=None):
        if moving.mean() > .65:
            return                         # целый движущийся кадр не является доказательством дороги
        magnitude = np.linalg.norm(flow, axis=2)
        active = moving & (magnitude > .2 * self.shape[1] / 640)
        footprint=cv2.dilate(active.astype(np.uint8),np.ones((7,7),np.uint8))>0
        # CCTV часто содержит повторные кадры. Нулевой поток на таком кадре
        # не обрывает проезд, но секундная пауза между сменами часов — обрывает.
        self.motion_gap = np.where(footprint,0,np.minimum(self.motion_gap.astype(np.uint32)+1,65000)).astype(np.uint16)
        self.streak = np.where(footprint,np.minimum(self.streak.astype(np.uint32)+1,65000),
                               np.where(self.motion_gap<=max(1,round(self.fps*.15)),self.streak,0)).astype(np.uint16)
        active &= self.streak >= max(2,round(self.fps*.12))
        if frame is not None and self.frames % max(1,round(self.fps/3))==0:
            self.samples.append(frame.copy())
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
        vegetation = np.zeros(self.shape,bool)
        if len(self.samples)>=3:
            image = np.median(np.stack(self.samples),axis=0).astype(np.uint8)
            hsv = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
            green = ((hsv[...,0]>=30)&(hsv[...,0]<=95)&(hsv[...,1]>50)).astype(np.float32)
            dense_green = ((cv2.boxFilter(green,-1,(9,9))>.7) & (green>0)).astype(np.uint8)
            count, regions, stats, _ = cv2.connectedComponentsWithStats(dense_green,connectivity=8)
            keep=np.zeros(count,bool)
            for j in range(1,count):
                x,y,ww,hh,area=stats[j]
                keep[j]=(area>=max(40,120*w/640) and (x==0 or y==0 or x+ww==w or y+hh==h))
            vegetation=keep[regions]
            refined=[]
            for _,mask,direction in candidates:
                mask = self._appearance_region(image,mask,direction,vegetation)
                if (mask & support).sum()<40:
                    continue
                refined.append((int((mask&support).sum()),mask,direction))
            candidates=refined
            # Два куска одного направления после восстановления асфальта
            # не должны оставаться перекрывающимися автозонами.
            changed=True
            while changed:
                changed=False
                for i in range(len(candidates)):
                    for j in range(i+1,len(candidates)):
                        a,b=candidates[i],candidates[j]
                        if a[2]@b[2]>.9 and (a[1]&b[1]).sum()>.1*min(a[1].sum(),b[1].sum()):
                            candidates[i]=(a[0]+b[0],a[1]|b[1],a[2]);candidates.pop(j);changed=True;break
                    if changed:break
        candidates.sort(key=lambda x:float(np.nonzero(x[1])[1].mean()))
        parts, directions = [], []
        for i, (_,mask,direction) in enumerate(candidates):
            mask=cv2.morphologyEx(mask.astype(np.uint8),cv2.MORPH_OPEN,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)))
            mask=(cv2.GaussianBlur(mask.astype(np.float32),(9,9),0)>.4)&~vegetation
            found,hierarchy=cv2.findContours(mask.astype(np.uint8),cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
            if not found:continue
            outer=max((j for j in range(len(found)) if hierarchy[0,j,3]<0),key=lambda j:cv2.contourArea(found[j]))
            contour=found[outer];epsilon=max(1.,min(w/640*8,.0075*cv2.arcLength(contour,True)))
            ys,xs=np.nonzero(mask)
            extent=np.ptp(xs*direction[0]+ys*direction[1])
            thickness=2*mask.sum()/max(cv2.arcLength(contour,True),1)
            # Один дрожащий значок, пламя крупным планом и тонкая строка
            # часов не дают пространственно протяжённого коридора движения.
            if extent < .2*min(h,w) or thickness < max(4,.03*w):
                continue
            if len(self.samples)>=3:
                extended=self._extend_border_corridor(image,mask,direction,vegetation)
                if not np.array_equal(extended,mask):
                    mask=extended
                    found,hierarchy=cv2.findContours(mask.astype(np.uint8),cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
                    outer=max((j for j in range(len(found)) if hierarchy[0,j,3]<0),key=lambda j:cv2.contourArea(found[j]))
                    contour=found[outer];epsilon=max(1.,min(w/640*8,.0075*cv2.arcLength(contour,True)))
            points=cv2.approxPolyDP(contour,epsilon,True).reshape(-1,2)
            item={"name":f"Автозона {len(parts)+1}","points":(points/[w,h]).tolist()}
            holes=[]
            for j,cn in enumerate(found):
                if hierarchy[0,j,3]==outer and cv2.contourArea(cn)>=max(24,60*w/640):
                    ring=cv2.approxPolyDP(cn,max(1.,w/640*2),True).reshape(-1,2)
                    holes.append((ring/[w,h]).tolist())
            if holes:item['holes']=holes
            parts.append(item)
            directions.append({"part":len(parts),"unit_vector":direction.tolist(),
                               "angle_image_deg":float(np.degrees(np.arctan2(direction[1],direction[0])))})
        return RoadParts(parts,video,[w,h]), {"frames":self.frames,"support_pixels":int(support.sum()),
                                             "directions":directions,
                                             "appearance_refined":len(self.samples)>=3,
                                             "method":"persistent compensated flow and temporal road appearance"}

    def _extend_border_corridor(self,image,mask,direction,vegetation):
        """Продолжить обрезанный краем кадра наблюдаемый коридор по потоку.

        Конец десятисекундного следа — не физический край асфальта. Продление
        идёт только вперёд и только если поток выходит через границу кадра.
        Встречные потоки и внутренний дальний конец дороги не объединяются.
        """
        h,w=self.shape;dx,dy=direction
        ys,xs=np.mgrid[:h,:w];along=xs*dx+ys*dy
        reliable=mask&(self.votes>=max(4,self.frames*.07))
        if reliable.sum()<40:return mask
        positions=along[reliable];speed=self.magnitude[reliable]/np.maximum(self.votes[reliable],1)
        near=float(np.median(speed[positions>=np.percentile(positions,70)]))
        far=float(np.median(speed[positions<=np.percentile(positions,30)]))
        # Приближающийся поток ускоряется в пикселях по перспективе.
        # Удаляющийся поток сужается к горизонту: его продление средней
        # стрелкой пересекало бы разделительную полосу.
        if near<1.2*max(far,1e-3):return mask
        exits=((dx<-.25 and mask[:,:4].any()) or (dx>.25 and mask[:,-4:].any())
               or (dy<-.25 and mask[:4].any()) or (dy>.25 and mask[-4:].any()))
        if not exits:return mask
        tail=mask & (along>=np.percentile(along[mask],70))
        proposal=mask.copy();length=round(min(.25*w,.4*h))
        for distance in range(3,length+1,3):
            transform=np.array([[1,0,dx*distance],[0,1,dy*distance]],np.float32)
            proposal |= cv2.warpAffine(tail.astype(np.uint8),transform,(w,h),flags=cv2.INTER_NEAREST)>0
        padding=max(8,round(.1*w))
        border=np.zeros(self.shape,bool)
        if dx<-.25 and mask[:,:4].any():border|=xs<=1.5*padding
        if dx>.25 and mask[:,-4:].any():border|=w-1-xs<=1.5*padding
        if dy<-.25 and mask[:4].any():border|=ys<=1.5*padding
        if dy>.25 and mask[-4:].any():border|=h-1-ys<=1.5*padding
        proposal |= (cv2.dilate(tail.astype(np.uint8),cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                       (2*padding+1,)*2))>0)&border
        hsv=cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
        bright=((hsv[...,1]<40)&(hsv[...,2]>max(160,float(np.percentile(hsv[...,2][mask],90))+20))).astype(np.uint8)
        n,regions,stats,_=cv2.connectedComponentsWithStats(bright,8)
        large=np.zeros(n,bool)
        for j in range(1,n):
            x,y,bw,bh,area=stats[j]
            edges=int(x==0)+int(y==0)+int(x+bw==w)+int(y+bh==h)
            large[j]=area>.02*h*w and edges>=2
        snow=large[regions]
        visible=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)>=max(20,.25*float(np.percentile(hsv[...,2],90)))
        extension=proposal & ~vegetation & ~snow & visible
        return mask | extension

    def _appearance_region(self,image,mask,direction,vegetation):
        """Участки одного асфальта между следами движения, с ограниченным ростом.

        Медиана калибровочного окна уменьшает влияние проезжающих кузовов.
        Растительность — запрещённый фон. Соседний асфальт может входить в
        область даже без машины в каждом пикселе первых десяти секунд.
        """
        h,w=self.shape;ys,xs=np.nonzero(mask)
        if not len(xs):return mask
        margin=max(3,round(w/640*6))
        domain=cv2.dilate(mask.astype(np.uint8),cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*margin+1,)*2))>0
        if abs(direction[1])>=abs(direction[0]):
            domain[:max(0,int(ys.min())-4)]=False;domain[min(h,int(ys.max())+5):]=False
        else:
            domain[:,:max(0,int(xs.min())-4)]=False;domain[:,min(w,int(xs.max())+5):]=False
        domain &= ~vegetation
        seed=cv2.erode(mask.astype(np.uint8),np.ones((3,3),np.uint8))>0
        value=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        # Непрозрачный дым и чёрные поля не являются доказательством асфальта.
        visible = value>=max(20,.25*float(np.percentile(value,90)))
        saturation=cv2.cvtColor(image,cv2.COLOR_BGR2HSV)[...,1]
        plausible=visible & (saturation<=max(100,float(np.median(saturation))+25))
        seed &= domain & plausible
        if seed.sum()<40:return np.zeros_like(mask)
        gc=np.full(self.shape,cv2.GC_BGD,np.uint8)
        gc[domain]=cv2.GC_PR_BGD;gc[mask&domain]=cv2.GC_PR_FGD;gc[seed]=cv2.GC_FGD
        gc[~plausible]=cv2.GC_BGD
        bg_model=np.zeros((1,65),np.float64);fg_model=bg_model.copy()
        cv2.grabCut(image,gc,None,bg_model,fg_model,2,cv2.GC_INIT_WITH_MASK)
        result=((gc==cv2.GC_FGD)|(gc==cv2.GC_PR_FGD))&domain
        n,lab=cv2.connectedComponents(result.astype(np.uint8),connectivity=8)
        keep=np.unique(lab[seed]);result=np.isin(lab,keep[keep>0])
        # Окна и неподвижные кузова не вырезаются из самой дороги. При этом
        # зелёные разделительные островки остаются исключёнными.
        outer,_=cv2.findContours(result.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        filled=np.zeros(self.shape,np.uint8);cv2.drawContours(filled,outer,-1,1,cv2.FILLED)
        return (filled>0)&domain&~vegetation


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
    estimator=MotionRoadEstimator(previous.shape,fps=fps)
    cleaner=FlowCleaner(egomotion=True,road_mask=False)
    guard=SceneGuard(fps);guard.update(previous)
    for _ in range(round(seconds*fps)):
        ok,frame=cap.read()
        if not ok:break
        image=prep(frame);gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        cut,valid=guard.update(gray)
        if cut:break
        flow=cv2.calcOpticalFlowFarneback(previous,gray,None,.5,3,15,3,5,1.2,0)
        c=cleaner.update(flow,gray,previous)
        if valid:estimator.update(c.field(),c.moving,image)
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
