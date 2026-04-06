"""
sim_world.py — realistic first-person perspective world with tree occlusion.
"""
from __future__ import annotations
import math
import cv2
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class WorldObject:
    world_x: float; world_y: float; color: str; shape: str
    base_width: float; base_height: float; label: str; is_target: bool = False

@dataclass
class TreeOccluder:
    world_x: float; world_y: float; radius: float = 0.35; height: float = 4.0

@dataclass
class RobotState:
    x: float = 0.0; y: float = 0.0; heading: float = 0.0
    speed: float = 0.04; turn_rate: float = 0.05; search_turn: float = 0.015

class PerspectiveCamera:
    def __init__(self, width=800, height=600, fov_degrees=60.0,
                 camera_height=1.2, max_render_dist=40.0, min_render_dist=0.3, bg_seed=42,
                 headless=False):
        self.width = width; self.height = height
        self.max_dist = max_render_dist; self.min_dist = min_render_dist
        self.cam_h = camera_height
        self.focal = (width / 2) / math.tan(math.radians(fov_degrees / 2))
        self._bg = None if headless else self._make_background(width, height, bg_seed)

    def render(self, robot, objects, trees):
        frame     = self._bg.copy()
        tree_proj = self._project_trees(robot, trees)
        obj_proj  = self._project_objects(robot, objects, trees)
        for p in sorted(tree_proj + obj_proj, key=lambda p: p["distance"], reverse=True):
            if p.get("is_tree"):    self._draw_tree(frame, p)
            elif not p.get("occluded"): self._draw_object(frame, p)
        self._draw_hud(frame, robot)
        return frame, [p for p in obj_proj if not p.get("occluded")]

    def project_headless(self, robot, objects, trees):
        """
        Headless fast path — pure geometry, zero OpenCV drawing.
        Returns only the visible projection list. ~10x faster than render().
        Use this when SHOW_WINDOW=False.
        """
        return [p for p in self._project_objects(robot, objects, trees)
                if not p.get("occluded")]

    def _to_cam(self, robot, wx, wy):
        dx = wx-robot.x; dy = wy-robot.y; dist = math.hypot(dx, dy)
        ch = math.cos(-robot.heading); sh = math.sin(-robot.heading)
        return dx*ch - dy*sh, dx*sh + dy*ch, dist

    def _project_objects(self, robot, objects, trees):
        cx = self.width/2; cy = self.height/2; results = []
        for obj in objects:
            cam_x, cam_y, dist = self._to_cam(robot, obj.world_x, obj.world_y)
            if cam_x <= self.min_dist or dist > self.max_dist: continue
            sc = self.focal/cam_x
            sx = int(cx + cam_y*sc)
            sy_b = int(cy + self.cam_h*sc); sy_t = int(cy - (obj.base_height-self.cam_h)*sc)
            sw = max(2, int(obj.base_width*sc)); sh = max(2, sy_b-sy_t)
            if sx+sw//2 < 0 or sx-sw//2 > self.width: continue
            results.append({"obj":obj,"screen_x":sx,"screen_y":(sy_t+sy_b)//2,
                "screen_w":sw,"screen_h":sh,"sy_top":sy_t,"sy_bottom":sy_b,
                "distance":dist,"occluded":self._is_occluded(robot,obj.world_x,obj.world_y,dist,trees),"is_tree":False})
        return results

    def _project_trees(self, robot, trees):
        cx = self.width/2; cy = self.height/2; results = []
        for tree in trees:
            cam_x, cam_y, dist = self._to_cam(robot, tree.world_x, tree.world_y)
            if cam_x <= self.min_dist or dist > self.max_dist: continue
            sc = self.focal/cam_x; sx = int(cx+cam_y*sc)
            sy_b = int(cy+self.cam_h*sc); sy_t = int(cy-(tree.height-self.cam_h)*sc)
            sw = max(4, int(tree.radius*2*sc)); sh = max(4, sy_b-sy_t)
            if sx+sw//2 < 0 or sx-sw//2 > self.width: continue
            results.append({"tree":tree,"screen_x":sx,"screen_y":(sy_t+sy_b)//2,
                "screen_w":sw,"screen_h":sh,"sy_top":sy_t,"sy_bottom":sy_b,
                "distance":dist,"is_tree":True})
        return results

    def _is_occluded(self, robot, ox, oy, odist, trees):
        rx, ry = robot.x, robot.y; dx, dy = ox-rx, oy-ry
        for tree in trees:
            tx, ty = tree.world_x-rx, tree.world_y-ry
            t = max(0.0, min(1.0, (tx*dx+ty*dy)/(odist**2+1e-9)))
            if math.hypot(rx+t*dx-tree.world_x, ry+t*dy-tree.world_y) < tree.radius:
                if math.hypot(tree.world_x-rx, tree.world_y-ry) < odist:
                    return True
        return False

    def _draw_object(self, frame, v):
        from vision.test_frame_generator import COLOR_BGR, TestFrameGenerator
        bgr = COLOR_BGR.get(v["obj"].color, (200,200,200))
        TestFrameGenerator._draw_geometry(frame, v["obj"].shape,
            v["screen_x"], v["screen_y"], v["screen_w"], v["screen_h"], bgr)
        if v["obj"].is_target:
            cv2.ellipse(frame,(v["screen_x"],v["screen_y"]),
                (v["screen_w"]//2+3,v["screen_h"]//2+3),0,0,360,(255,255,255),1)

    def _draw_tree(self, frame, v):
        cx=v["screen_x"]; sw=v["screen_w"]; sh=v["screen_h"]
        sy_t=v["sy_top"]; sy_b=v["sy_bottom"]
        cv2.rectangle(frame,(cx-max(2,sw)//2,sy_t+sh//2),(cx+max(2,sw)//2,sy_b),(20,50,80),-1)
        cv2.ellipse(frame,(cx,sy_t+sh//3),(max(2,sw*2),max(2,sh//2)),0,0,360,(15,80,20),-1)
        cv2.ellipse(frame,(cx,sy_t+sh//3),(max(2,sw*2),max(2,sh//2)),0,0,360,(10,60,15),1)

    def _draw_hud(self, frame, robot):
        cv2.putText(frame,f"pos=({robot.x:.1f},{robot.y:.1f})  hdg={math.degrees(robot.heading):.0f}deg",
            (10,frame.shape[0]-10),cv2.FONT_HERSHEY_SIMPLEX,0.4,(180,180,180),1,cv2.LINE_AA)

    @staticmethod
    def _make_background(w, h, seed):
        rng = np.random.default_rng(seed); bg = np.zeros((h,w,3),dtype=np.uint8)
        horizon = int(h*0.45)
        for y in range(horizon):
            t=y/horizon; bg[y,:]=(int(60+t*20),int(45+t*15),int(30+t*10))
        for y in range(horizon,h):
            t=(y-horizon)/(h-horizon); bg[y,:]=(int(20+(1-t)*15),int(50+(1-t)*30),int(10+(1-t)*15))
        bg = np.clip(bg.astype(np.int16)+rng.integers(0,18,(h,w,3),dtype=np.uint8),0,255).astype(np.uint8)
        cv2.line(bg,(0,horizon),(w,horizon),(50,70,35),1)
        return bg


class SimDetector:
    def detections_from_projections(self, projections, frame_w, frame_h):
        from vision.detector import RawDetection
        frame_area=frame_w*frame_h; results=[]; label_ids={}; next_id=[0]
        def get_id(l):
            if l not in label_ids: label_ids[l]=next_id[0]; next_id[0]+=1
            return label_ids[l]
        for v in projections:
            obj=v["obj"]; sw,sh=v["screen_w"],v["screen_h"]; cx,cy=v["screen_x"],v["screen_y"]
            conf=float(np.clip((sw*sh)/frame_area*8.0,0.05,0.98))
            results.append(RawDetection(label=obj.label,confidence=round(conf,3),
                bbox={"x":max(0,cx-sw//2),"y":max(0,cy-sh//2),"width":sw,"height":sh},
                class_id=get_id(obj.label)))
        return results
