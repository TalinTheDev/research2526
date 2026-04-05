"""
semantic_scorer.py  (v3)
------------------------
Key fix from debug analysis:
  - Wrong-color children were scoring 0.85 because size_aligns gated them
    into the target band. Fix: color is STRICTLY required when the target
    specifies a color. No fallback to size.
  - Wrong-color same-label objects now score in clue band [0.45-0.64].
  - Only red child → target band [0.65-1.0].
"""
from __future__ import annotations
import math
from collections import deque
from typing import Deque, Dict, List, Optional
from vision.frame_processor import Detection
from scoring_interface import ScoringAlgorithm

KNOWN_COLORS = {
    "red","orange","yellow","green","cyan","blue","purple",
    "pink","brown","white","black","gray","grey",
}
LABEL_ASSOC: Dict[str, List[str]] = {
    "child":     ["person","kid","boy","girl","human"],
    "person":    ["human","child","adult","figure","man","woman"],
    "dog":       ["animal","pet"],
    "cat":       ["animal","pet"],
    "animal":    ["dog","cat","bird","creature"],
    "debris":    ["object","item","equipment","landmark"],
    "vehicle":   ["car","truck","object"],
    "equipment": ["gear","object","debris"],
}
CLUE_FLOOR = 0.45; CLUE_CEIL = 0.64; TGT_FLOOR = 0.65


class SemanticScorer:
    name = "SemanticScorer"
    DEFAULT_WEIGHTS = {
        "label_match":0.35,"description_match":0.25,"state_match":0.10,
        "confidence":0.10,"size":0.10,"temporal":0.05,"position":0.05,
    }

    def __init__(self, weights=None, temporal_window=10, frame_width=800):
        raw = weights or self.DEFAULT_WEIGHTS.copy()
        t = sum(raw.values())
        self._w = {k: v/t for k, v in raw.items()}
        self._fw = frame_width
        self._hist: Dict[str, Deque[float]] = {}
        self._win = temporal_window

    def score(self, detections: List[Detection], target: dict) -> List[Detection]:
        assoc = {a.lower() for a in target.get("associated_objects", [])}
        for det in detections:
            cls = self._classify(det, target, assoc)
            raw = sum(self._w[k] * v for k, v in self._components(det, target).items())
            if   cls == "target": score = TGT_FLOOR  + raw * (1.0 - TGT_FLOOR)
            elif cls == "clue":   score = CLUE_FLOOR + self._clue_q(det, target) * (CLUE_CEIL - CLUE_FLOOR)
            else:                 score = raw * CLUE_FLOOR
            k = self._tkey(det)
            if k not in self._hist: self._hist[k] = deque(maxlen=self._win)
            self._hist[k].append(score)
            det.relevance_score = round(sum(self._hist[k])/len(self._hist[k]), 4)
        return detections

    def _classify(self, det, target, assoc):
        dl = det.label.lower()
        tl = (target.get("object") or "").lower()
        label_ok = (dl == tl or dl in LABEL_ASSOC.get(tl,[]) or tl in LABEL_ASSOC.get(dl,[]))
        tcolors  = self._tcolors(target)

        if label_ok:
            if tcolors:
                # Color REQUIRED — no fallback to size
                return "target" if self._color_ok(det, tcolors) else "clue"
            else:
                return "target" if self._size_ok(det, target) else "clue"

        if dl in assoc:
            return "clue"
        return "irrelevant"

    @staticmethod
    def _tcolors(target):
        c = {x.lower() for x in target.get("colors", [])}
        for d in target.get("descriptions", []):
            for w in d.lower().split():
                if w in KNOWN_COLORS: c.add(w)
        return c

    @staticmethod
    def _color_ok(det, tcolors):
        for a in det.attributes:
            if a.startswith("color:"):
                return a.split(":",1)[1] in tcolors
        return False

    @staticmethod
    def _size_ok(det, target):
        hint = (target.get("size_hint") or "").lower()
        if not hint: return True
        bucket = next((a.split(":",1)[1] for a in det.attributes if a.startswith("size:")), None)
        if not bucket: return False
        order = ["tiny","small","medium","large"]
        hmap  = {"tiny":"tiny","small":"small","little":"small",
                 "medium":"medium","large":"large","big":"large","huge":"large"}
        norm = hmap.get(hint)
        return bool(norm and bucket in order and abs(order.index(norm)-order.index(bucket))<=1)

    def _clue_q(self, det, target):
        tcolors = self._tcolors(target)
        color_s = 1.0 if (tcolors and self._color_ok(det, tcolors)) else 0.2
        dl = det.label.lower(); tl = (target.get("object") or "").lower()
        label_s = 0.4 if (dl==tl or dl in LABEL_ASSOC.get(tl,[]) or tl in LABEL_ASSOC.get(dl,[])) else 0.0
        return min(1.0, 0.5*color_s + 0.3*det.confidence + 0.2*label_s*2)

    def _components(self, det, target):
        return {
            "label_match":       self._lm(det, target),
            "description_match": self._dm(det, target),
            "state_match":       self._sm(det, target),
            "confidence":        det.confidence,
            "size":              self._sz(det),
            "temporal":          self._tmp(det),
            "position":          self._pos(det),
        }

    def _lm(self, det, target):
        tl = (target.get("object") or "").lower(); dl = det.label.lower()
        asc = [a.lower() for a in target.get("associated_objects", [])]
        if not tl:                              return 0.5
        if dl == tl:                            return 1.0
        if dl in LABEL_ASSOC.get(tl,[]):        return 0.8
        if tl in LABEL_ASSOC.get(dl,[]):        return 0.75
        if dl in asc:                           return 0.5
        for cat in ({"person","child","human","adult","man","woman","kid","boy","girl","figure"},
                    {"animal","dog","cat","bird","creature","pet"},
                    {"debris","object","item","equipment","gear","landmark"}):
            if tl in cat and dl in cat: return 0.2
        return 0.0

    def _dm(self, det, target):
        sub = []
        tc = self._tcolors(target)
        if tc:
            c = next((a.split(":",1)[1] for a in det.attributes if a.startswith("color:")), None)
            sub.append((1.0 if c in tc else 0.0) if (c and c!="unknown") else 0.3)
        else: sub.append(0.5)
        hint = (target.get("size_hint") or "").lower()
        if hint:
            b = next((a.split(":",1)[1] for a in det.attributes if a.startswith("size:")), None)
            order=["tiny","small","medium","large"]
            hmap={"tiny":"tiny","small":"small","little":"small","medium":"medium",
                  "large":"large","big":"large","huge":"large"}
            n=hmap.get(hint)
            sub.append(1.0 if(n and b==n) else max(0.0,1.0-abs(order.index(n)-order.index(b))*0.4)
                       if(n and b and b in order) else 0.3)
        attrs=[a.lower() for a in target.get("attributes",[])]
        if attrs:
            s=" ".join(det.attributes).lower()
            sub.append(sum(1 for a in attrs if a in s)/len(attrs))
        return sum(sub)/len(sub) if sub else 0.5

    def _sm(self, det, target):
        ts=[s.lower() for s in target.get("state",[])]
        if not ts: return 0.5
        ds=[s.lower() for s in det.state]; sc=0.0; m=0
        for t in ts:
            if t=="lost":
                if "stationary" in ds or "possibly_lying" in ds: m+=1
                elif "moving" in ds: sc-=0.1
            elif t=="injured":
                if "possibly_lying" in ds and "stationary" in ds: m+=1
                elif "moving" in ds: sc-=0.2
            elif t=="moving":
                if "moving" in ds: m+=1
                elif "stationary" in ds: sc-=0.1
            else:
                if any(t in d for d in ds): m+=1
        return max(0.0, min(1.0, m/len(ts)+sc))

    def _sz(self, det):
        b=next((a.split(":",1)[1] for a in det.attributes if a.startswith("size:")),None)
        return {"tiny":0.1,"small":0.4,"medium":0.7,"large":1.0}.get(b,0.3)

    def _tmp(self, det):
        h=self._hist.get(self._tkey(det))
        return sum(h)/len(h) if h else 0.5

    def _pos(self, det):
        cx=det.bbox["x"]+det.bbox["width"]/2
        return round(math.cos(abs(cx/self._fw-0.5)*2*math.pi/2),4)

    @staticmethod
    def _tkey(det):
        return f"{det.label}@{(det.bbox['x']+det.bbox['width']//2)//100}"
