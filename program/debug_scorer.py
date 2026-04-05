"""debug_scorer.py — wraps any ScoringAlgorithm and prints per-frame decisions."""
from __future__ import annotations
from typing import List
from vision.frame_processor import Detection
from scoring_interface import ScoringAlgorithm

class DebugScorer:
    def __init__(self, inner: ScoringAlgorithm, print_every: int = 30):
        self._inner=inner; self._every=print_every; self._f=0
        self.name=f"DEBUG({inner.name})"

    def score(self, detections: List[Detection], target: dict) -> List[Detection]:
        detections=self._inner.score(detections,target)
        if self._f%self._every==0 and detections:
            print(f"\n  --- frame {self._f} ---")
            for d in sorted(detections,key=lambda x:x.relevance_score or 0,reverse=True):
                c=next((a.split(":",1)[1] for a in d.attributes if a.startswith("color:")),"?")
                s=next((a.split(":",1)[1] for a in d.attributes if a.startswith("size:")),"?")
                print(f"    {d.label:<12} color={c:<8} size={s:<8} conf={d.confidence:.2f} score={d.relevance_score:.4f}")
        self._f+=1
        return detections
