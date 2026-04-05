"""
simulation.py
-------------
First-person simulation with split-view display:
  LEFT  — first-person perspective (polished rendering)
  RIGHT — top-down 2D map showing robot path, objects, trees, FOV cone

Controls: SPACE = pause, Q/Esc = quit
"""
from __future__ import annotations
import copy, math, time
from dataclasses import dataclass
from typing import List, Optional
import cv2, numpy as np
from vision.attribute_extractor import AttributeExtractor
from vision.frame_processor import Detection, FrameData
from vision.sim_world import PerspectiveCamera, RobotState, SimDetector, TreeOccluder, WorldObject
from vision.visualizer import Visualizer
from scoring_interface import ScoringAlgorithm

# Color palette for top-down map (BGR)
_MAP_COLORS = {
    "red":(0,0,220),"orange":(0,120,230),"yellow":(0,220,220),
    "green":(0,180,40),"cyan":(200,200,0),"blue":(220,60,0),
    "purple":(180,40,160),"pink":(180,80,220),"white":(230,230,230),
}


@dataclass
class SimMetrics:
    algo_name:str=""; success:bool=False; frames_elapsed:int=0
    time_elapsed_s:float=0.0; path_length:float=0.0
    false_follows:int=0; fill_ratio_final:float=0.0

    def summary(self):
        s="SUCCESS" if self.success else "TIMEOUT"
        return (f"[{self.algo_name}] {s} | frames={self.frames_elapsed} | "
                f"time={self.time_elapsed_s:.2f}s | path={self.path_length:.2f} | "
                f"false_follows={self.false_follows} | final_fill={self.fill_ratio_final:.4f}")


class TopDownMap:
    """
    Renders a 2D top-down view of the world.
    Shows: trees, objects (color-coded), robot position + heading,
    robot path trail, FOV cone, target highlight ring.
    """
    def __init__(self, objects, trees, world_bounds=(-5,30,-10,15),
                 map_w=400, map_h=400, fov_degrees=60.0):
        self._objects = objects
        self._trees   = trees
        self._xmin, self._xmax, self._ymin, self._ymax = world_bounds
        self._w = map_w; self._h = map_h
        self._fov_half = math.radians(fov_degrees / 2)
        self._path: List[tuple] = []

    def _to_px(self, wx, wy):
        px = int((wx - self._xmin) / (self._xmax - self._xmin) * self._w)
        py = int((1 - (wy - self._ymin) / (self._ymax - self._ymin)) * self._h)
        return px, py

    def render(self, robot: RobotState, detections: List[Detection],
               metrics: SimMetrics, target: dict) -> np.ndarray:
        canvas = np.full((self._h, self._w, 3), (28, 35, 28), dtype=np.uint8)

        # Grid lines
        for gx in range(int(self._xmin), int(self._xmax)+1, 5):
            x, _ = self._to_px(gx, 0); cv2.line(canvas,(x,0),(x,self._h),(40,50,40),1)
        for gy in range(int(self._ymin), int(self._ymax)+1, 5):
            _, y = self._to_px(0, gy); cv2.line(canvas,(0,y),(self._w,y),(40,50,40),1)

        # Trees
        for tree in self._trees:
            px,py = self._to_px(tree.world_x, tree.world_y)
            r = max(2, int(tree.radius * self._w / (self._xmax - self._xmin)))
            cv2.circle(canvas,(px,py),r,(25,80,25),-1)
            cv2.circle(canvas,(px,py),r,(15,60,15),1)

        # Objects
        for obj in self._objects:
            px,py = self._to_px(obj.world_x, obj.world_y)
            bgr = _MAP_COLORS.get(obj.color,(180,180,180))
            if obj.is_target:
                cv2.circle(canvas,(px,py),9,(255,255,255),2)
                cv2.circle(canvas,(px,py),6,bgr,-1)
            else:
                cv2.circle(canvas,(px,py),5,bgr,-1)
            cv2.putText(canvas, obj.label[:3], (px+6,py+4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.28, (180,180,180), 1)

        # Robot path trail
        self._path.append((robot.x, robot.y))
        for i in range(1, len(self._path)):
            p1 = self._to_px(*self._path[i-1])
            p2 = self._to_px(*self._path[i])
            alpha = i / len(self._path)
            col = (int(40*alpha), int(180*alpha), int(80*alpha))
            cv2.line(canvas, p1, p2, col, 1)

        # FOV cone
        rx,ry = self._to_px(robot.x, robot.y)
        fov_len = int(0.25 * self._w)
        for angle_offset in np.linspace(-self._fov_half, self._fov_half, 12):
            angle = robot.heading + angle_offset
            ex = int(rx + fov_len * math.cos(angle))
            ey = int(ry - fov_len * math.sin(angle))
            cv2.line(canvas,(rx,ry),(ex,ey),(60,100,60),1)

        # FOV boundary lines
        for side in (-1, 1):
            angle = robot.heading + side * self._fov_half
            ex = int(rx + fov_len * math.cos(angle))
            ey = int(ry - fov_len * math.sin(angle))
            cv2.line(canvas,(rx,ry),(ex,ey),(80,140,80),1)

        # Robot dot + heading arrow
        cv2.circle(canvas,(rx,ry),6,(0,220,120),-1)
        cv2.circle(canvas,(rx,ry),6,(0,255,140),1)
        hdg_ex = int(rx + 14 * math.cos(robot.heading))
        hdg_ey = int(ry - 14 * math.sin(robot.heading))
        cv2.arrowedLine(canvas,(rx,ry),(hdg_ex,hdg_ey),(0,255,140),2,tipLength=0.4)

        # Legend / info block
        lines = [
            f"Algo: {metrics.algo_name}",
            f"Frames: {metrics.frames_elapsed}",
            f"Path: {metrics.path_length:.1f}",
            f"FF: {metrics.false_follows}",
            f"Fill: {metrics.fill_ratio_final:.4f}",
        ]
        for i, line in enumerate(lines):
            cv2.putText(canvas, line, (6, 14+i*14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (160,200,160), 1)

        # Map title
        cv2.putText(canvas, "TOP-DOWN VIEW", (self._w//2-55, self._h-6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (100,140,100), 1)

        return canvas


class Simulation:
    def __init__(self, objects, trees=None, camera_width=800, camera_height=600,
                 fov_degrees=60.0, success_fill=0.04, max_frames=6000,
                 robot_start=None, show_window=True, window_scale=1.0,
                 frame_delay_ms=16, pause_on_end=True):
        targets = [o for o in objects if o.is_target]
        assert len(targets)==1, "Exactly one WorldObject must have is_target=True"
        self._objects=objects; self._trees=trees or []
        self._cam_w=camera_width; self._cam_h=camera_height
        self._success_fill=success_fill; self._max_frames=max_frames
        self._robot_start=robot_start or RobotState()
        self._show_window=show_window; self._window_scale=window_scale
        self._frame_delay=max(1,frame_delay_ms); self._pause_on_end=pause_on_end
        self._fov=fov_degrees
        self._camera  = PerspectiveCamera(camera_width,camera_height,fov_degrees=fov_degrees)
        self._sim_det = SimDetector()
        self._extractor = AttributeExtractor()

    def run(self, algo, target, window_title=None, debug = False):
        robot   = copy.deepcopy(self._robot_start)
        metrics = SimMetrics(algo_name=algo.name)
        title   = window_title or f"Sim: {algo.name}"

        # Build top-down map
        xs=[o.world_x for o in self._objects]; ys=[o.world_y for o in self._objects]
        bounds=(min(xs)-3,max(xs)+3,min(ys)-3,max(ys)+3)
        tmap = TopDownMap(self._objects,self._trees,bounds,400,600,self._fov)

        # Combined window width: fp + map side by side
        win_w = self._cam_w + 400
        win_h = self._cam_h
        cv2.namedWindow(title, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(title, int(win_w*self._window_scale), int(win_h*self._window_scale))

        start_wall=time.time(); prev_x,prev_y=robot.x,robot.y
        paused=False; last_combined=None

        for frame_idx in range(self._max_frames):
            frame, projections = self._camera.render(robot,self._objects,self._trees)

            raw_dets = self._sim_det.detections_from_projections(projections,self._cam_w,self._cam_h)

            gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            detections=[]
            for rd in raw_dets:
                attrs=self._extractor.extract(frame,rd.bbox)
                attrs.motion=self._extractor.compute_motion(gray,rd.bbox)
                detections.append(Detection(label=rd.label,confidence=rd.confidence,
                                            bbox=rd.bbox,attributes=self._pack_attrs(attrs)))
            self._extractor.update_motion_buffer(frame)

            detections = algo.score(detections, target)
            fd=FrameData(frame_id=frame_idx,timestamp=round(frame_idx/30.0,3),detections=detections)
            best=self._best(detections)

            if best:
                proj=self._find_proj(best,projections)
                if proj is None or not proj["obj"].is_target:
                    metrics.false_follows+=1

            if best is None:
                robot.heading+=robot.search_turn
            else:
                self._steer(robot,best)
                robot.x+=robot.speed*math.cos(robot.heading)
                robot.y+=robot.speed*math.sin(robot.heading)

            metrics.path_length+=math.hypot(robot.x-prev_x,robot.y-prev_y)
            prev_x,prev_y=robot.x,robot.y

            target_proj=next((p for p in projections if p["obj"].is_target),None)
            fill=self._fill(target_proj); metrics.fill_ratio_final=fill
            metrics.frames_elapsed=frame_idx

            if frame_idx%30==0:
                bs=f"{best.label}({best.relevance_score:.3f})" if best else "searching..."
                ot=""
                if best:
                    p2=self._find_proj(best,projections)
                    ot=" [TARGET]" if(p2 and p2["obj"].is_target) else " [distractor]"
                if debug:
                    print(f"  f={frame_idx:04d} pos=({robot.x:.1f},{robot.y:.1f})"
                      f" hdg={math.degrees(robot.heading):.0f}deg"
                      f" following={bs}{ot} fill={fill:.4f}")

            if self._show_window:
                # Render first-person with overlay
                fp = frame.copy()
                self._draw_fp_overlay(fp,best,target_proj,fill,metrics,algo.name)

                # Render top-down map
                td = tmap.render(robot,detections,metrics,target)

                # Combine side by side
                combined = np.zeros((win_h,win_w,3),dtype=np.uint8)
                combined[:,:self._cam_w] = fp
                # Map is 400x600, same height as fp — fits directly
                combined[:,:]=np.concatenate([fp,td],axis=1)
                last_combined=combined.copy()

                if self._window_scale!=1.0:
                    dw=int(win_w*self._window_scale); dh=int(win_h*self._window_scale)
                    disp=cv2.resize(combined,(dw,dh))
                else:
                    disp=combined
                cv2.imshow(title,disp)

                key=cv2.waitKey(self._frame_delay)&0xFF
                if key in (ord('q'),ord('Q'),27): break
                if key==ord(' '):
                    paused=not paused
                    if debug:
                        print(f"  {'[PAUSED]' if paused else '[RESUMED]'}")
                while paused:
                    k2=cv2.waitKey(50)&0xFF
                    if k2==ord(' '): 
                        paused=False
                        if debug:
                            print("  [RESUMED]")
                    elif k2 in (ord('q'),ord('Q'),27): paused=False; break

            if fill>=self._success_fill:
                metrics.success=True; metrics.frames_elapsed=frame_idx+1; break
        else:
            metrics.frames_elapsed=self._max_frames

        metrics.time_elapsed_s=time.time()-start_wall

        if self._show_window and self._pause_on_end and last_combined is not None:
            banner=last_combined.copy()
            self._draw_result_banner(banner,metrics)
            if self._window_scale!=1.0:
                bw=int(win_w*self._window_scale); bh=int(win_h*self._window_scale)
                banner=cv2.resize(banner,(bw,bh))
            cv2.imshow(title,banner)
            if debug:
                print(f"\n  Run complete: {metrics.summary()}")
                print("  [Press SPACE or Q to continue]\n")
            while True:
                k=cv2.waitKey(50)&0xFF
                if k in (ord(' '),ord('q'),ord('Q'),27): break

        cv2.destroyWindow(title)
        return metrics

    def run_comparison(self, algorithms, seed, target, debug = False):
        all_metrics=[]
        for algo in algorithms:
            if(debug):
                print(f"\n{'='*60}\n  Running: {algo.name}\n{'='*60}")
            m=self.run(algo,target, debug)
            all_metrics.append(m)


        for m in all_metrics:
            s="SUCCESS" if m.success else "TIMEOUT"
            print(f"  {seed:<10}, {m.algo_name:<20}, {s:<10}, {m.frames_elapsed:>8}, "
                  f"{m.path_length:>8.2f}, {m.false_follows:>14}, {m.fill_ratio_final:>10.4f}")

        return all_metrics

    def _steer(self,robot,best):
        off=(best.bbox["x"]+best.bbox["width"]/2)-self._cam_w/2
        turn=max(-robot.turn_rate,min(robot.turn_rate,(off/(self._cam_w/2))*robot.turn_rate))
        robot.heading+=turn

    @staticmethod
    def _best(dets):
        s=[d for d in dets if d.relevance_score is not None]
        return max(s,key=lambda d:d.relevance_score) if s else None

    def _fill(self,tp):
        if tp is None: return 0.0
        return tp["screen_w"]*tp["screen_h"]/(self._cam_w*self._cam_h)

    @staticmethod
    def _find_proj(det,projs):
        for p in projs:
            px=p["screen_x"]-p["screen_w"]//2; py=p["screen_y"]-p["screen_h"]//2
            if abs(px-det.bbox["x"])<8 and abs(py-det.bbox["y"])<8: return p
        return None

    @staticmethod
    def _pack_attrs(attrs):
        r=[f"color:{attrs.color}"]
        s=attrs.size
        if   s<0.01: r.append("size:tiny")
        elif s<0.05: r.append("size:small")
        elif s<0.20: r.append("size:medium")
        else:        r.append("size:large")
        if attrs.position!="unknown": r.append(f"position:{attrs.position}")
        if attrs.motion  !="unknown": r.append(f"motion:{attrs.motion}")
        return r

    def _draw_fp_overlay(self,frame,best,target_proj,fill,metrics,algo_name):
        h,w=frame.shape[:2]
        lines=[
            (f"Algo: {algo_name}",                          (220,220,100)),
            (f"Fill: {fill:.4f} / {self._success_fill}",   (200,200,200)),
            (f"Frames: {metrics.frames_elapsed}",           (200,200,200)),
            (f"Path:   {metrics.path_length:.2f}",          (200,200,200)),
            (f"FalseFollows: {metrics.false_follows}",      (200,200,200)),
            (f"{'Tracking' if best else 'Searching...'}",
             (100,220,100) if best else (100,100,220)),
        ]
        for i,(text,color) in enumerate(lines):
            y=20+i*20
            (tw,th),_=cv2.getTextSize(text,cv2.FONT_HERSHEY_SIMPLEX,0.45,1)
            cv2.rectangle(frame,(w-tw-14,y-th-2),(w-6,y+4),(20,20,20),-1)
            cv2.putText(frame,text,(w-tw-10,y),cv2.FONT_HERSHEY_SIMPLEX,0.45,color,1,cv2.LINE_AA)

        # Fill bar
        bar_w=int(min(fill/self._success_fill,1.0)*(w-20))
        cv2.rectangle(frame,(10,h-37),(w-10,h-24),(40,40,40),-1)
        bc=(0,200,80) if fill>=self._success_fill else (0,140,220)
        if bar_w>0: cv2.rectangle(frame,(10,h-37),(10+bar_w,h-24),bc,-1)
        cv2.putText(frame,"target fill",(12,h-40),cv2.FONT_HERSHEY_SIMPLEX,0.35,(160,160,160),1)

        if best:
            bx,by,bw,bh=best.bbox["x"],best.bbox["y"],best.bbox["width"],best.bbox["height"]
            cv2.rectangle(frame,(bx,by),(bx+bw,by+bh),(0,255,80),2)
            cv2.putText(frame,f"{best.label} {best.relevance_score:.2f}",
                        (bx,max(0,by-6)),cv2.FONT_HERSHEY_SIMPLEX,0.4,(0,255,80),1,cv2.LINE_AA)

    def _draw_result_banner(self,frame,metrics):
        h,w=frame.shape[:2]
        ov=frame.copy()
        cv2.rectangle(ov,(w//4,h//4),(3*w//4,3*h//4),(15,15,15),-1)
        cv2.addWeighted(ov,0.82,frame,0.18,0,frame)
        s="SUCCESS" if metrics.success else "TIMEOUT"
        tc=(0,220,80) if metrics.success else (0,80,220)
        lines=[
            (s,tc,1.2,2),
            (f"Algorithm:     {metrics.algo_name}",(220,220,220),0.55,1),
            (f"Frames:        {metrics.frames_elapsed}",(200,200,200),0.50,1),
            (f"Path length:   {metrics.path_length:.2f}",(200,200,200),0.50,1),
            (f"False follows: {metrics.false_follows}",(200,200,200),0.50,1),
            (f"Final fill:    {metrics.fill_ratio_final:.4f}",(200,200,200),0.50,1),
            ("",(0,0,0),0.4,1),
            ("SPACE / Q  to continue",(140,140,140),0.45,1),
        ]
        total_h=sum(int(38*s) for _,_,s,_ in lines)
        y=h//2-total_h//2
        for text,color,scale,thickness in lines:
            if not text: y+=12; continue
            (tw,th),_=cv2.getTextSize(text,cv2.FONT_HERSHEY_SIMPLEX,scale,thickness)
            cv2.putText(frame,text,(w//2-tw//2,y+th),cv2.FONT_HERSHEY_SIMPLEX,
                        scale,color,thickness,cv2.LINE_AA)
            y+=th+int(10*scale)
