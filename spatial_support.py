"""Поддержка полигонов новыми деталями и сохранённым видом неподвижного объекта."""
from dataclasses import dataclass
import cv2,numpy as np

@dataclass
class Remembered:
 mask:np.ndarray
 features:np.ndarray
 created:float
 started:float
 last_match:float
 last_seen:float
 ready:bool=False

class _SupportMemory:
 def __init__(self,scale,confirm_s=0.,hold_s=2.,fraction=.5,radius=8.):
  self.scale=scale;self.confirm_s=confirm_s;self.hold_s=hold_s;self.fraction=fraction;self.radius=radius;self.states={}
 def update(self,actors,edges,novel,t):
  self.states={k:s for k,s in self.states.items() if t-s.last_seen<=self.hold_s+.5}
  if not actors:return []
  distance=cv2.distanceTransform((~novel).astype('uint8'),cv2.DIST_L2,5)
  near_edges=cv2.dilate(edges.astype('uint8'),np.ones((5,5),'uint8'))>0
  radius=max(1,round(2*self.scale));kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*radius+1,)*2)
  groups={}
  for a in actors:
   key=(a['scene'],a['part'],a['id']//2);groups.setdefault(key,[]).append(a)
  out=[]
  for key,group in groups.items():
   union=np.logical_or.reduce([a['mask'] for a in group]);red=np.logical_or.reduce([a['mask'] for a in group if a['red']]) if any(a['red'] for a in group) else np.zeros_like(union)
   created=t-max(a['age'] for a in group);state=self.states.get(key)
   if state and abs(state.created-created)>.2:self.states.pop(key);state=None
   area=int(union.sum());near=cv2.dilate(union.astype('uint8'),np.ones((3,3),'uint8'))>0
   count=int((novel&near).sum());support=float((distance[union]<=self.radius*self.scale).mean())
   strong=(area>=max(16,400*self.scale*self.scale) and count>=max(5,.005*area) and support>=self.fraction)
   if state is None and strong:
    state=Remembered(union.copy(),novel&near,created,t,t,t);self.states[key]=state
   elif state is None and red.any():
    state=Remembered(red.copy(),edges&red,created,t-self.confirm_s,t,t,True);self.states[key]=state
   if state:
    state.last_seen=t
    feature_count=int(state.features.sum());match=float((state.features&near_edges&near).sum()/max(feature_count,1))
    overlap=float((state.mask&union).sum()/max(state.mask.sum(),1))
    matched=feature_count>=5 and match>=.5 and overlap>=.5
    if matched:state.last_match=t
    tolerance=self.hold_s if state.ready else .25
    if t-state.last_match>tolerance:
     self.states.pop(key,None);state=None
     if strong:
      state=Remembered(union.copy(),novel&near,created,t,t,t);self.states[key]=state
    if state and matched and t-state.started>=self.confirm_s:state.ready=True
    if state and state.ready and strong and matched:
     added=union&~state.mask
     if added.sum()<=max(16,400*self.scale*self.scale) or float((distance[added]<=self.radius*self.scale).mean())>=self.fraction:
      state.mask=union.copy();state.features=novel&near
   for a in group:
    mask=a['mask'];size=int(mask.sum())
    if a['red']:out.append((a,mask,'confirmed'));continue
    if a['kind']=='flow_gap' and size<=.04*mask.size:out.append((a,mask,'bounded_coarse'));continue
    if size<=max(16,400*self.scale*self.scale):
     anear=cv2.dilate(mask.astype('uint8'),np.ones((3,3),'uint8'))>0
     if (novel&anear).sum()>=5 or (state and state.ready and (mask&state.mask).sum()>=.5*size):out.append((a,mask,'small'))
     continue
    if state and state.ready:
     allowed=cv2.dilate(state.mask.astype('uint8'),kernel)>0
     cropped=mask&allowed
     if cropped.sum()>=max(12,60*self.scale*self.scale):out.append((a,cropped,'tracked'))
  return out


def appearance_bounded(mask, novel, scale):
 """Убрать длинный узкий отросток, не поддержанный кузовом или новыми деталями.

 Крупный кузов содержит широкую внутреннюю область. Тонкая дальняя машина
 поддерживается новыми деталями. Так короткая мелочь внутри родительской
 области не сохраняет длинную полосу старого фона на обочине.
 """
 if mask.sum() <= max(16, 400*scale*scale):
  return mask
 radius=max(1,round(8*scale))
 solid=cv2.morphologyEx(mask.astype('uint8'),cv2.MORPH_OPEN,
       cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*radius+1,)*2))>0
 if not solid.any():
  return mask
 distance=cv2.distanceTransform((~(solid|novel)).astype('uint8'),cv2.DIST_L2,5)
 return mask & (distance <= max(1.,8*scale))


class SpatialSupport:
 """Не передавать уверенность небольшой детали всей большой фоновой области.

 Достоверная форма запоминается. При временном ослаблении разницы с эталоном
 сохраняется лишь поддерживаемое изображением прежнее место объекта. Большая
 новая площадь не наследует эту уверенность автоматически.
 """
 def __init__(self,shape):
  self.scale=shape[1]/640
  self.reset()
 def reset(self):
  self.memory=_SupportMemory(self.scale)
  self.last_edges=self.last_novel=None
 def update(self,holes,gray,background,t,frame=None,known=None,foreground=None,pixel_age=None,scene_id=0):
  from dataclasses import replace
  from foreground_objects import mask_of,contours
  from mask_regions import thin_color_lines
  edges=cv2.Canny(cv2.GaussianBlur(gray,(5,5),0),50,120)>0
  if frame is not None:edges &= ~thin_color_lines(frame)
  old=cv2.Canny(np.clip(background,0,255).astype('uint8'),30,70)
  novel=edges&(cv2.dilate(old,np.ones((5,5),'uint8'))==0)
  self.last_edges,self.last_novel=edges,novel
  actors=[{'mask':mask_of(h,gray.shape),'id':h.id,'part':h.part,'scene':scene_id,'age':h.observed_s,'red':h.incident,'kind':h.kind,'hole':h} for h in holes]
  result=[]
  for actor,mask,reason in self.memory.update(actors,edges,novel,t):
   hole=actor['hole']
   if not hole.incident:
    mask=appearance_bounded(mask,novel,self.scale)
   if np.array_equal(mask,actor['mask']):result.append(hole);continue
   observed=mask if known is None else mask&known
   coverage=float(observed.sum()/max(mask.sum(),1))
   difference=hole.bg_diff if foreground is None or not observed.any() else float(foreground[observed].mean())
   idle=hole.idle_s if pixel_age is None else float(np.quantile(pixel_age[mask],.4))
   result.append(replace(hole,mask=mask,polygons=contours(mask),area_px=int(mask.sum()),
       idle_s=idle,bg_diff=difference,bg_coverage=coverage,
       bg_valid=hole.bg_valid is True and coverage>=.5 and observed.sum()>=20,
       bg_inferred_frac=float((~known)[mask].mean()) if known is not None else hole.bg_inferred_frac,
       bg_removed_frac=1-(1-hole.bg_removed_frac)*mask.sum()/max(hole.area_px,1)))
  return result
