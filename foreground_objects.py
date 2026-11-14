"""Передний план целиком: ограничение контуров и стоящий транспорт без истории проездов."""
from dataclasses import replace
import math

import cv2
import numpy as np

from flow_holes import Hole
from mask_regions import complete_foreground, thin_color_lines


def contours(mask):
    # Внутренние контуры сохраняем: RETR_EXTERNAL заливал бы асфальт между машинами.
    found, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return [c.reshape(-1, 2) for c in found if len(c) >= 3]


def mask_of(hole, shape):
    if hole.mask is not None:
        return hole.mask.astype(bool)
    result = np.zeros(shape, np.uint8)
    cv2.fillPoly(result, hole.polygons, 1)
    return result > 0


def stable_output_regions(holes, minimum_s=2., gray=None, background=None, frame=None):
    """Оранжевый контур требует устойчивости самой области, а не старых пикселей.

    Внутреннее сопровождение продолжается до публикации контура. Подтверждённая
    тревога сохраняется по своей опорной маске даже при изменении соседней очереди.
    """
    stable = [h for h in holes if h.incident or h.observed_s >= minimum_s]
    if gray is None or background is None or not any(h.kind == 'stationary' for h in stable):
        return stable
    # Без дырки в потоке одной разницы яркости недостаточно: при изменении
    # контраста старые слабые линии разметки становятся сильными краями.
    # Для самостоятельного неподвижного кандидата исключаем и такие края
    # эталона. Уже связанная с дыркой область сохраняет основное подтверждение.
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 50, 120) > 0
    if frame is not None:
        edges &= ~thin_color_lines(frame)
    old = cv2.Canny(np.clip(background, 0, 255).astype(np.uint8), 30, 70)
    novel = edges & (cv2.dilate(old, np.ones((5, 5), np.uint8)) == 0)
    result = []
    for hole in stable:
        if hole.incident or hole.kind != 'stationary':
            result.append(hole)
            continue
        mask = mask_of(hole, gray.shape)
        near = cv2.dilate(mask.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        if (novel & near).sum() >= max(5, .005 * mask.sum()):
            result.append(hole)
    return result


def clip_holes(holes, foreground, known, labels, min_area=24, raw_foreground=None):
    """Весь полигон ограничен наблюдаемым передним планом, а не только его ядро."""
    result = []
    for hole in holes:
        original = mask_of(hole, labels.shape) & (labels == hole.part)
        mask = original & foreground & known
        n, lab, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
        allowed = [i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= min_area]
        mask = np.isin(lab, allowed) if allowed else np.zeros_like(mask)
        if not mask.any():
            continue
        coverage = float(known[original].mean()) if original.any() else 0.
        evidence = foreground if raw_foreground is None else raw_foreground
        difference = float(evidence[mask].mean())
        valid = coverage >= .5 and int((known & original).sum()) >= 20
        result.append(replace(hole, mask=mask, polygons=contours(mask), area_px=int(mask.sum()),
                              bg_valid=valid, bg_coverage=coverage, bg_diff=difference, bg_match=False,
                              incident=hole.incident and valid and difference >= .25,
                              bg_removed_frac=1-int(mask.sum())/max(int(original.sum()),1)))
    return result


class ForegroundObjects:
    """Показывает неподвижный передний план, даже если в нём не было трёх проездов.

    Новые края отсекают плоские изменения освещения и следы уехавших машин.
    Метка stationary означает занятую область/очередь, а не отдельное ДТП.
    """
    def __init__(self, shape, fps, min_area=60, confirm_s=2.0, smooth_s=.4):
        self.shape, self.fps = shape, float(fps)
        self.min_area, self.confirm_s = min_area, confirm_s
        self.smooth_s = smooth_s
        self.reset()

    def reset(self):
        self.age = np.zeros(self.shape, np.float32)
        self.gap = np.zeros(self.shape, np.float32)
        self.probability = np.zeros(self.shape, np.float32)
        self.active = np.zeros(self.shape, bool)
        self.motion_hold = np.zeros(self.shape, np.float32)
        self.last_t = None
        self.previous = {}
        self.next_id = 1000000

    def forget_background(self,cleared):
        if not cleared.any():return
        self.active[cleared]=False;self.probability[cleared]=0;self.age[cleared]=0
        for ident in list(self.previous):
            previous=self.previous[ident]
            previous['mask']=previous['mask']&~cleared
            previous['origin_mask']=previous['origin_mask']&~cleared
            if not previous['mask'].any() or not previous['origin_mask'].any():
                self.previous.pop(ident)
                continue
            x,y,w,h=cv2.boundingRect(previous['mask'].astype(np.uint8))
            previous['bbox']=(x,y,w,h);previous['area']=int(previous['mask'].sum())
            ox,oy,ow,oh=cv2.boundingRect(previous['origin_mask'].astype(np.uint8))
            previous['origin']=(ox+ow/2,oy+oh/2)

    def update(self, gray, background, foreground, moving, labels, t, cell, gh, gw, cut=False, valid=True,
               frame=None, known=None):
        if cut:
            self.reset()
        if not valid:
            self.last_t = t
            return []
        dt = 1 / self.fps if self.last_t is None else min(max(t - self.last_t, 0.), 3 / self.fps)
        self.last_t = t
        domain = labels >= 0
        known = domain if known is None else known & domain
        # Гистерезис убирает краткие вспышки разницы с фоном до восстановления
        # кузова: морфология не должна раздувать однокадровое пятно.
        alpha = 1 - math.exp(-dt / max(self.smooth_s, 1e-3))
        self.probability += alpha * (foreground.astype(np.float32) - self.probability)
        self.active = ((self.probability >= .65) | (self.active & (self.probability >= .35))) & known
        candidate = complete_foreground(self.active, known, labels, frame=frame, infer_unknown=True)
        candidate = (cv2.medianBlur(candidate.astype(np.uint8), 3) > 0) & domain
        moving = cv2.dilate(moving.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        # Повторный кадр CCTV не означает остановку. Короткая память держит
        # движение между такими кадрами и не даёт медленно едущему кузову
        # накопить возраст неподвижного объекта.
        self.motion_hold = np.where(moving, .3, np.maximum(self.motion_hold-dt, 0.)).astype(np.float32)
        moving = self.motion_hold > 0
        quiet = candidate & ~moving
        self.gap = np.where(quiet, 0., self.gap + dt).astype(np.float32)
        tolerance=np.where(self.age>=self.confirm_s,2.,.25)
        self.age = np.where(quiet, self.age + dt, np.where(self.gap <= tolerance, self.age, 0.)).astype(np.float32)
        # Сначала выделяем неподвижное ядро. Иначе однотонный стоящий фургон,
        # касающийся в маске едущей очереди, отбрасывался вместе со всей очередью.
        stable_mask = candidate & (self.age >= self.confirm_s)
        allow = candidate
        for _ in range(max(1, round(6 * labels.shape[1] / 640))):
            stable_mask = (cv2.dilate(stable_mask.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0) & allow
        size=max(3,round(5*labels.shape[1]/640)|1)
        stable_mask=cv2.morphologyEx(stable_mask.astype(np.uint8),cv2.MORPH_OPEN,
                                    cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(size,size)),
                                    borderType=cv2.BORDER_CONSTANT,borderValue=0)>0
        # Эталон уже сглажен BackgroundModel.smooth(). Одинаковый масштаб
        # деталей не даёт фактуре неподвижного фона стать «новыми» краями.
        edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 50, 120) > 0
        if frame is not None:
            edges &= ~thin_color_lines(frame)
        old_edges = cv2.Canny(np.clip(background, 0, 255).astype(np.uint8), 50, 120)
        novel = edges & (cv2.dilate(old_edges, np.ones((5, 5), np.uint8)) == 0)
        output, current, used = [], {}, set()
        minimum_area = max(12, round(self.min_area * (labels.shape[1] / 640) ** 2))
        for part in range(int(labels.max()) + 1):
            n, lab, stats, _ = cv2.connectedComponentsWithStats((stable_mask & (labels == part)).astype(np.uint8), 8)
            for i in sorted(range(1, n), key=lambda j: -stats[j, cv2.CC_STAT_AREA]):
                if stats[i, cv2.CC_STAT_AREA] < minimum_area:
                    continue
                mask = lab == i
                stable = self.age[mask]
                if (stable >= self.confirm_s).mean() < .5:
                    continue
                observed=mask & known
                if observed.sum()<20:
                    continue
                raw_fraction = float(foreground[observed].mean())
                if raw_fraction < .05:
                    continue  # исходник уже совпал с фоном: не держим «призрак»
                near = cv2.dilate(mask.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
                if (novel & near).sum() < max(5, .005 * mask.sum()):
                    continue
                if edges[mask].mean() < .8 * (old_edges[mask] > 0).mean():
                    continue                         # опустевшая область прежней машины
                x, y, w, h = stats[i, :4]
                area = int(stats[i, cv2.CC_STAT_AREA])
                matches = []
                for ident, previous in self.previous.items():
                    if previous['part'] != part or ident in used:
                        continue
                    ox, oy, ow, oh = previous['bbox']
                    x0, y0, x1, y1 = max(x, ox), max(y, oy), min(x+w, ox+ow), min(y+h, oy+oh)
                    if x0 >= x1 or y0 >= y1:
                        continue
                    # Пересечение только внутри общего bbox, без O(K²HW).
                    intersection = int((mask[y0:y1, x0:x1] & previous['mask'][y0:y1, x0:x1]).sum())
                    union = area + previous['area'] - intersection
                    matches.append((intersection / max(union, 1), ident))
                score, ident = max(matches, default=(0, 0))
                if score < .25:
                    ident = self.next_id
                    self.next_id += 1
                    created, difference, origin = t, raw_fraction, (x+w/2,y+h/2)
                    origin_mask=mask.copy()
                else:
                    previous = self.previous[ident]
                    created = previous['created']
                    origin=previous['origin']
                    origin_mask=previous['origin_mask']
                    retained=(mask&origin_mask).sum()/max(origin_mask.sum(),1)
                    if retained<.5 and math.hypot(x+w/2-origin[0],y+h/2-origin[1])>max(2*labels.shape[1]/640,.08*max(w,h)):
                        created,origin=t,(x+w/2,y+h/2)
                        origin_mask=mask.copy()
                    beta = 1 - math.exp(-dt)
                    difference = previous['bg_diff'] + beta * (raw_fraction - previous['bg_diff'])
                used.add(ident)
                current[ident] = {'mask': mask, 'part': part, 'bbox': (x,y,w,h), 'area': int(mask.sum()),
                                  'created': created, 'origin':origin, 'origin_mask':origin_mask,
                                  'last_seen': t, 'bg_diff': difference}
                if difference < .25:
                    continue
                grid = cv2.resize(mask[:gh*cell, :gw*cell].astype(np.float32), (gw, gh), interpolation=cv2.INTER_AREA) > .15
                core = np.zeros_like(mask)
                expanded = cv2.resize(grid.astype(np.uint8), (gw*cell, gh*cell), interpolation=cv2.INTER_NEAREST) > 0
                core[:gh*cell, :gw*cell] = expanded
                core &= labels == part
                coverage = min(float(known[core].mean()) if core.any() else 0., float(known[mask].mean()))
                bg_valid = coverage >= .5 and int((known & mask).sum()) >= 20
                output.append(Hole(id=ident, part=part, cells=grid, core=grid, polygons=contours(mask),
                                   mask=mask, area_px=int(mask.sum()), idle_s=float(np.quantile(stable, .4)),
                                   age_s=t-created, v_in=0., bg_diff=difference, bg_valid=bg_valid,
                                   bg_coverage=coverage, bg_inferred_frac=float((~known)[mask].mean()),
                                   observed_s=t-created,track_anchor=origin_mask,incident=False, kind="stationary"))
        self.previous = {ident: p for ident, p in self.previous.items() if t-p['last_seen'] <= .5 and ident not in used}
        self.previous.update(current)
        return output


def link_flow_regions(holes, objects, labels, cell, incident_s=20., pixel_age=None, foreground=None, known=None):
    """Подтвердить дырку устойчивым объектом, сохранив его собственный возраст.

    Расширенный старый контур не передаёт свой красный статус всем новым
    пикселям. Связь устанавливается по исходным клеткам ядра; остальные
    неподвижные области остаются кандидатами, включая стоящую очередь.
    """
    cores = []
    height, width = labels.shape
    for hole in holes:
        grid = hole.core if hole.core is not None else hole.cells
        up = cv2.resize(grid.astype(np.uint8), None, fx=cell, fy=cell, interpolation=cv2.INTER_NEAREST) > 0
        core = np.zeros(labels.shape, bool)
        core[:min(height,up.shape[0]), :min(width,up.shape[1])] = up[:height, :width]
        cores.append((hole, core))
    result = []
    for obj in objects:
        obj=replace(obj,id=2*obj.id+1)  # чётный ID — подтверждённая часть, нечётный — кандидат
        matches = [(overlap / max(obj.area_px, 1), h)
                   for h, core in cores if h.part == obj.part
                   if (overlap := int((obj.mask & core).sum())) >= 20]
        score, source = max(matches, key=lambda pair: pair[0], default=(0., None))
        if source is None or score < .05:
            result.append(obj)
            continue
        obj=replace(obj,kind='flow_gap',ring_ratio=source.ring_ratio,v_ring=source.v_ring)
        minimum=max(12,round(60*(width/640)**2),round(.75*cell**2))
        if pixel_age is None:
            result.append(replace(obj,incident=obj.idle_s>=incident_s and obj.bg_valid is True
                                  and obj.bg_diff>=.25 and obj.area_px>=minimum))
            continue
        red=obj.mask & (pixel_age>=incident_s)
        n,lab,stats,_=cv2.connectedComponentsWithStats(red.astype(np.uint8),8)
        keep=np.zeros(n,bool)
        for i in range(1,n):
            if stats[i,cv2.CC_STAT_AREA]<minimum:
                continue
            region=lab==i
            observed=region if known is None else region & known
            coverage=float(observed.sum()/max(region.sum(),1))
            difference=obj.bg_diff if foreground is None else float(foreground[observed].mean()) if observed.any() else 0.
            keep[i]=coverage>=.5 and observed.sum()>=20 and difference>=.25 and obj.bg_valid is True
        red=keep[lab]
        if not red.any():
            result.append(obj)
            continue
        observed=red if known is None else red & known
        difference=obj.bg_diff if foreground is None else float(foreground[observed].mean())
        result.append(replace(obj,id=obj.id-1,incident=True,mask=red,polygons=contours(red),
                              area_px=int(red.sum()),idle_s=float(np.quantile(pixel_age[red],.4)),
                              bg_diff=min(obj.bg_diff,difference),
                              bg_coverage=float(observed.sum()/red.sum()),
                              bg_inferred_frac=float(1-observed.sum()/red.sum())))
        rest=obj.mask & ~red
        if rest.sum()>=max(12,round(60*(width/640)**2)):
            result.append(replace(obj,mask=rest,polygons=contours(rest),area_px=int(rest.sum()),
                                  idle_s=float(np.quantile(pixel_age[rest],.4)),
                                  bg_inferred_frac=obj.bg_inferred_frac if known is None else float((~known)[rest].mean())))
    return result


class FlowRegionConfirmation:
    """Подтверждение устойчивой области, а не самых тёмных старых пикселей.

    После подтверждения сохраняется опорная маска именно этого объекта.
    Присоединившаяся очередь не расширяет её; непрерывное движение снимает
    объект в ForegroundObjects, а маленький новый фрагмент получает новый ID.
    """
    def __init__(self, incident_s=20.):
        self.incident_s=incident_s
        self.reset()

    def reset(self):
        self.anchors={}

    def forget_background(self,cleared):
        if not cleared.any():return
        for ident in list(self.anchors):
            self.anchors[ident]['mask'] &= ~cleared
            if not self.anchors[ident]['mask'].any():self.anchors.pop(ident)

    def update(self,holes,objects,labels,cell,t,foreground,known):
        self.anchors={key:value for key,value in self.anchors.items() if t-value['last_seen']<=.75}
        attached=link_flow_regions(holes,objects,labels,cell,incident_s=math.inf)
        output=[]
        minimum=max(12,round(60*(labels.shape[1]/640)**2),round(.75*cell**2))
        margin=max(1,round(3*labels.shape[1]/640))
        kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*margin+1,)*2)
        for obj in attached:
            anchor=self.anchors.get(obj.id)
            observed=obj.mask&known
            current_evidence=(observed.sum()>=20 and observed.sum()>=.5*obj.area_px
                              and foreground[observed].mean()>=.25)
            if (anchor is None and obj.kind=='flow_gap' and obj.age_s>=self.incident_s
                    and obj.bg_valid is True and obj.bg_diff>=.25 and current_evidence and obj.area_px>=minimum):
                anchor={'mask':(obj.track_anchor if obj.track_anchor is not None else obj.mask).copy(),
                        'confirmed_at':t,'observed_s':obj.age_s,
                        'last_seen':t,'ring_ratio':obj.ring_ratio,'v_ring':obj.v_ring}
                self.anchors[obj.id]=anchor
            if anchor is None:
                output.append(obj)
                continue
            overlap=(obj.mask&anchor['mask']).sum()/max(anchor['mask'].sum(),1)
            if overlap<.5:
                self.anchors.pop(obj.id,None)
                output.append(obj)
                continue
            anchor['last_seen']=t
            bounded=cv2.dilate(anchor['mask'].astype(np.uint8),kernel)>0
            red=obj.mask & bounded
            n,lab,stats,_=cv2.connectedComponentsWithStats(red.astype(np.uint8),8)
            keep=np.zeros(n,bool)
            for i in range(1,n):
                if stats[i,cv2.CC_STAT_AREA]<minimum:
                    continue
                region=lab==i;observed=region&known
                keep[i]=(observed.sum()>=20 and observed.sum()>=.5*region.sum()
                         and foreground[observed].mean()>=.25 and obj.bg_valid is True)
            red=keep[lab]
            if not red.any():
                output.append(obj)
                continue
            observed=red&known
            elapsed=anchor['observed_s']+t-anchor['confirmed_at']
            output.append(replace(obj,id=obj.id-1,kind='flow_gap',incident=True,mask=red,
                polygons=contours(red),area_px=int(red.sum()),idle_s=elapsed,
                confirmation_s=anchor['confirmed_at'],observed_s=elapsed,incident_anchor=anchor['mask'],
                ring_ratio=anchor['ring_ratio'],v_ring=anchor['v_ring'],
                bg_diff=min(obj.bg_diff,float(foreground[observed].mean())),
                bg_coverage=float(observed.sum()/red.sum()),bg_inferred_frac=float(1-observed.sum()/red.sum())))
            rest=obj.mask&~red
            if rest.sum()>=max(12,round(60*(labels.shape[1]/640)**2)):
                output.append(replace(obj,mask=rest,polygons=contours(rest),area_px=int(rest.sum()),
                                      bg_inferred_frac=float((~known)[rest].mean())))
        return output


def combine_objects(holes, objects):
    """Дополнить кандидатов без расширения красной области соседней очередью."""
    occupied = np.zeros(objects[0].mask.shape, bool) if objects else None
    if occupied is not None:
        for hole in holes:
            if hole.mask is not None:
                occupied |= hole.mask
    for obj in objects:
        mask = obj.mask & ~occupied
        if mask.any():
            holes.append(replace(obj, mask=mask, polygons=contours(mask), area_px=int(mask.sum()), incident=False))
            occupied |= mask
    return holes
